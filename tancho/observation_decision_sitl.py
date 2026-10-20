"""One closed-loop planning cycle, using injected inference and simulation transport."""
from copy import deepcopy

from .contracts import require
from .mission_context import canonical_sha256
from .observation_catalog_v06 import CATALOG_V06, acceptance_v06, replay_decision_v06
from .observation_compiler import compile_program
from .observation_decision import replay_decision
from .observation_decision_calibration import acceptance
from .observation_decision_runtime import DecisionProgramRuntime, infer_decision
from .observation_flight import prepare_px4_mission
from .observation_program import validate_observation_program
from .observation_program_inference import infer_role


def observe_shadow(baseline, frames, *, binding, invoke, now_ms, hold, timeout_ms,
                   calibration=None, execution_kind='native_cosmos', order=(0, 1, 2)):
    """Sidecar for the existing JSON runtime, without taking ownership of flight."""
    baseline._at('INTENTS_VERIFIED')
    original_audit = baseline.audit_record()
    sidecar = DecisionProgramRuntime(baseline.context, frames, now_ms=now_ms(), mode='shadow', binding=binding)
    sidecar.intents = deepcopy(baseline.intents)
    sidecar.state = 'INTENTS_VERIFIED'
    sidecar._event('baseline_intents', {'baseline_audit_head': original_audit['audit_head'],
                                       'intent_revision': baseline.intents['intent_revision']})
    decision = infer_decision(sidecar, frames, invoke=invoke, now_ms=now_ms, hold=hold,
                              timeout_ms=timeout_ms, calibration=calibration,
                              execution_kind=execution_kind, order=order)
    require(baseline.audit_record() == original_audit and baseline.state == 'INTENTS_VERIFIED',
            'Shadow inference changed the baseline runtime')
    return {'baseline_audit_head': original_audit['audit_head'], 'decision': decision,
            'shadow_audit': sidecar.audit_record(), 'baseline_mutated': False}


def observe_shadow_order_consensus(baseline, frames, *, binding, invoke, now_ms, hold, timeout_ms,
                                   calibration=None, execution_kind='native_cosmos'):
    """Compare two candidate orders in shadow without changing the JSON baseline."""
    original_audit = baseline.audit_record()
    first = observe_shadow(baseline, frames, binding=binding, invoke=invoke, now_ms=now_ms,
                           hold=hold, timeout_ms=timeout_ms, calibration=calibration,
                           execution_kind=execution_kind, order=(0, 1, 2))
    second = observe_shadow(baseline, frames, binding=binding, invoke=invoke, now_ms=now_ms,
                            hold=hold, timeout_ms=timeout_ms, calibration=calibration,
                            execution_kind=execution_kind, order=(2, 1, 0))
    a, b = first['decision'], second['decision']
    left, right = a['request'], b['request']
    require(left['context_sha256'] == right['context_sha256']
            and left['intents_sha256'] == right['intents_sha256']
            and left['frame_sha256'] == right['frame_sha256']
            and left['evidence_order'] == right['evidence_order']
            and left['model_revision'] == right['model_revision']
            and {row['candidate_id'] for row in left['candidates']}
            == {row['candidate_id'] for row in right['candidates']},
            'Shadow candidate registry changed between orders')
    first_id = a['receipt']['selected_candidate_id']
    second_id = b['receipt']['selected_candidate_id']
    if first_id != second_id:
        outcome = 'disagreement'
    elif a['receipt']['outcome'] == 'rejected' or b['receipt']['outcome'] == 'rejected':
        outcome = 'agreed_rejection'
    elif not a['acceptance']['accepted'] or not b['acceptance']['accepted']:
        outcome = 'not_accepted'
    else:
        outcome = 'agreed_candidate'
    require(baseline.audit_record() == original_audit and baseline.state == 'INTENTS_VERIFIED',
            'Consensus shadow changed the JSON baseline')
    return {'schema_version': 'tancho-shadow-order-consensus-1.0', 'outcome': outcome,
            'candidate_id': first_id if outcome == 'agreed_candidate' else None,
            'orders': [first, second], 'baseline_audit_head': original_audit['audit_head'],
            'baseline_mutated': False, 'combined_deadline_measured': False,
            'real_flight_authorized': False, 'release_qualified': False}


def run_cycle(runtime, frames, scene, vehicle, *, reasoner, decision, now_ms, hold,
              timeout_ms=120000, calibration=None, json_generator=None, execution_kind='native_cosmos'):
    require(isinstance(runtime, DecisionProgramRuntime), 'Explicit decision runtime required')
    infer_role(runtime, frames, role='reasoner', invoke=reasoner, now_ms=now_ms,
               hold=hold, timeout_ms=timeout_ms)
    require(runtime.intents['outcome'] == 'inspect', 'Reasoner did not request an observation')
    record = infer_decision(runtime, frames, invoke=decision, now_ms=now_ms, hold=hold,
                            timeout_ms=timeout_ms, calibration=calibration, execution_kind=execution_kind)
    flight, counterfactual = None, None
    if runtime.mode == 'shadow':
        require(json_generator is not None, 'Shadow comparison requires the JSON baseline')
        # The candidate decision cannot mutate the baseline program or authorize it.
        if record['acceptance']['accepted']:
            try:
                raw = replay_decision(record['request'], record['receipt'], runtime.context,
                                      runtime.intents, frames, now_ms=now_ms())['program']
                verified = validate_observation_program(raw, runtime.context, runtime.intents, now_ms=now_ms())
                counterfactual = {'compiled': compile_program(verified, runtime.context, scene, vehicle,
                                                              now_ms=now_ms()), 'compilable': True}
            except ValueError as exc:
                counterfactual = {'compilable': False, 'reason': str(exc)}
        infer_role(runtime, frames, role='generator', invoke=json_generator, now_ms=now_ms,
                   hold=hold, timeout_ms=timeout_ms)
        runtime.compile(scene, vehicle, now_ms=now_ms())
    elif runtime.state == 'PROGRAM_VERIFIED':
        runtime.compile(scene, vehicle, now_ms=now_ms())
        flight = prepare_px4_mission(runtime, scene['envelope'], start_position_m=vehicle['position_m'],
                                     now_ms=now_ms())
    body = {'schema_version': 'tancho-decision-sitl-preflight-1.0', 'mode': runtime.mode,
            'context_sha256': canonical_sha256(runtime.context), 'scene_sha256': canonical_sha256(scene),
            'vehicle_sha256': canonical_sha256(vehicle), 'binding': runtime.binding,
            'execution_kind': execution_kind, 'decision': record, 'program': runtime.program,
            'intents': runtime.intents,
            'calibration': calibration, 'prepared_at_ms': now_ms(),
            'compiled': runtime.compiled, 'prepared_flight': flight, 'counterfactual': counterfactual,
            'audit': runtime.audit_record(), 'px4_sent': False, 'simulator_executed': False,
            'inference_replayed': False,
            'real_flight_authorized': False, 'release_qualified': False}
    return {**deepcopy(body), 'sha256': canonical_sha256(body)}


def run_sitl_attempt(runtime, frames, scene, vehicle, *, reasoner, decision, now_ms, hold,
                     timeout_ms=120000, calibration=None, execution_kind='native_cosmos'):
    """Record a failed or rejected CPU/SITL preflight without counting it as execution."""
    require(isinstance(runtime, DecisionProgramRuntime) and runtime.mode == 'sitl',
            'SITL attempt requires an explicit simulation runtime')
    preflight, failure = None, None
    try:
        preflight = run_cycle(runtime, frames, scene, vehicle, reasoner=reasoner,
                              decision=decision, now_ms=now_ms, hold=hold,
                              timeout_ms=timeout_ms, calibration=calibration,
                              execution_kind=execution_kind)
    except Exception as exc:
        if runtime.state not in ('ABORTED', 'COMPLETE', 'DETECTOR_HANDOFF'):
            runtime.abort('sitl_preflight_failed')
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    if failure is not None:
        outcome = 'failed'
    elif runtime.state == 'ABORTED':
        outcome = 'rejected'
    else:
        require(runtime.state == 'EXECUTING' and preflight['prepared_flight'] is not None,
                'Prepared SITL attempt has no simulation plan')
        outcome = 'prepared'
    body = {'schema_version': 'tancho-decision-sitl-attempt-1.0',
            'outcome': outcome, 'failure': failure, 'preflight': preflight,
            'audit': runtime.audit_record(), 'px4_sent': False,
            'simulator_executed': False, 'real_flight_authorized': False,
            'counts_as_model_success': False}
    return {**deepcopy(body), 'sha256': canonical_sha256(body)}


def validate_preflight(preflight):
    require(preflight['sha256'] == canonical_sha256({k: v for k, v in preflight.items() if k != 'sha256'}),
            'Decision preflight changed')
    require(preflight['schema_version'] == 'tancho-decision-sitl-preflight-1.0'
            and preflight['mode'] == 'sitl' and preflight['real_flight_authorized'] is False
            and preflight['px4_sent'] is False, 'Expected unsent simulation preflight')
    flight = preflight['prepared_flight']
    require(flight is not None and preflight['decision']['acceptance']['accepted'], 'No accepted simulation plan')
    authorization = flight['authorization']
    require(authorization['schema_version'] == 'tancho-simulation-authorization-1.0'
            and authorization['simulation_only'] is True and authorization['real_flight_authorized'] is False,
            'Not a simulation authorization')
    require(authorization['authorization_sha256'] == canonical_sha256(
        {k: v for k, v in authorization.items() if k not in ('schema_version', 'authorization_sha256')})
        and authorization['compiled_revision'] == preflight['compiled']['compiled_revision'],
        'Simulation authorization changed')
    require(flight['rfl']['passed'] is True
            and flight['rfl']['compiled_revision'] == preflight['compiled']['compiled_revision']
            and flight['px4_mission_sha256'] == canonical_sha256(flight['px4_mission']), 'Simulation plan binding changed')
    return preflight


def verify_case(preflight, case, frames, *, now_ms):
    validate_preflight(preflight)
    for key in ('context', 'scene', 'vehicle'):
        require(preflight[key + '_sha256'] == canonical_sha256(case[key]), 'SITL case changed')
    record = preflight['decision']
    if record['request']['catalog_revision'] == CATALOG_V06:
        require(preflight['calibration'] is None, 'Catalog 0.6 has no fitted calibration')
        gate = acceptance_v06(record['request'], record['receipt'], model_sha256=preflight['binding']['export_sha256'],
                              readout_head_sha256=preflight['binding'].get('readout_head_sha256'),
                              readout_projection=preflight['binding'].get('readout_projection'))
        replay = replay_decision_v06
    else:
        gate = acceptance(record['request'], record['receipt'], model_sha256=preflight['binding']['export_sha256'],
                          calibration=preflight['calibration'],
                          readout_head_sha256=preflight['binding'].get('readout_head_sha256'),
                          readout_projection=preflight['binding'].get('readout_projection'))
        replay = replay_decision
    require(gate == record['acceptance'] and gate['accepted'], 'SITL acceptance policy changed')
    raw = replay(record['request'], record['receipt'], case['context'], preflight['intents'],
                 frames, now_ms=now_ms)['program']
    require(raw is not None, 'SITL candidate was rejected')
    program = validate_observation_program(raw, case['context'], preflight['intents'], now_ms=now_ms)
    compiled = compile_program(program, case['context'], case['scene'], case['vehicle'], now_ms=now_ms)
    require(program == preflight['program'] and compiled == preflight['compiled'], 'SITL compiled plan changed')
    return preflight
