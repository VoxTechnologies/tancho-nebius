"""Explicit shadow/SITL decision adapter; never emits a real-flight authorization."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

from .contracts import require
from .mission_context import canonical_sha256, validate_context_v1
from .observation_decision import CATALOG, build_request, make_receipt, replay_decision
from .observation_catalog_v06 import (CATALOG_V06, acceptance_v06, build_request_v06, make_receipt_v06,
                                      replay_decision_v06)
from .observation_decision_calibration import acceptance
from .observation_decision_native import RESPONSE
from .observation_program import validate_observation_program
from .observation_program_runtime import ObservationProgramRuntime


class DecisionProgramRuntime(ObservationProgramRuntime):
    def __init__(self, context, evidence, *, now_ms, mode, binding, catalog=CATALOG):
        require(mode in ('shadow', 'sitl'), 'Decision runtime is simulation/shadow only')
        # Catalog 0.6 runs only with a readout head fitted for it (docs/r2-catalog06-binding-design-2026-09-30.md).
        require(catalog == CATALOG or (catalog == CATALOG_V06 and 'readout_head_sha256' in binding),
                'Road preview catalog has no bound runtime checkpoint')
        required = {'model_revision', 'export_sha256', 'base_sha256',
                    'framework_revision', 'precision', 'label_token_ids'}
        require(set(binding) in (required, required | {'readout_head_sha256'},
                                 required | {'readout_head_sha256', 'readout_projection'}),
                'Invalid backend binding')
        require(binding['model_revision'] == context['model_revision'], 'Backend model differs from context')
        for name, size in (('export_sha256', 64), ('base_sha256', 64), ('framework_revision', 40)):
            value = binding[name]
            require(isinstance(value, str) and len(value) == size
                    and all(c in '0123456789abcdef' for c in value), 'Invalid backend fingerprint')
        require(binding['precision'] in ('bf16', 'fp16') and len(binding['label_token_ids']) == 3
                and len(set(binding['label_token_ids'])) == 3
                and all(type(i) is int and i >= 0 for i in binding['label_token_ids']), 'Invalid backend token binding')
        if 'readout_head_sha256' in binding:
            value = binding['readout_head_sha256']
            require(isinstance(value, str) and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value), 'Invalid readout head fingerprint')
        if 'readout_projection' in binding:
            require(binding['readout_projection'] == 'three_row', 'Invalid backend projection')
        super().__init__(context, evidence, now_ms=now_ms)
        self.mode, self.binding, self.catalog = mode, deepcopy(binding), catalog
        self.decision = None
        self.completion_verified = False

    def accept(self, request, receipt, frames, *, now_ms, calibration=None, provider=None):
        self._at('INTENTS_VERIFIED')
        require(self.decision is None, 'Decision already recorded for this context')
        require(request['catalog_revision'] == self.catalog,
                'Road preview catalog has no bound runtime checkpoint')
        v06 = self.catalog == CATALOG_V06
        replay = replay_decision_v06 if v06 else replay_decision
        converted = replay(request, receipt, self.context, self.intents, frames, now_ms=now_ms)
        require(receipt['label_token_ids'] == self.binding['label_token_ids'], 'Decision tokenizer changed')
        if v06:
            require(calibration is None, 'Catalog 0.6 has no fitted calibration')
            gate = acceptance_v06(request, receipt, model_sha256=self.binding['export_sha256'],
                                  readout_head_sha256=self.binding['readout_head_sha256'],
                                  readout_projection=self.binding.get('readout_projection'))
        else:
            gate = acceptance(request, receipt, model_sha256=self.binding['export_sha256'],
                              calibration=calibration,
                              readout_head_sha256=self.binding.get('readout_head_sha256'),
                              readout_projection=self.binding.get('readout_projection'))
        self.decision = {'request': deepcopy(request), 'receipt': deepcopy(receipt),
                         'acceptance': gate, 'conversion': converted['conversion'],
                         'provider': deepcopy(provider), 'mode': self.mode,
                         'real_flight_authorized': False}
        self._event('candidate_decision', self.decision)
        if self.mode == 'shadow':
            return self.decision
        if not gate['accepted']:
            self.abort('decision_' + gate['reason'])
            return self.decision
        self.program = validate_observation_program(converted['program'], self.context,
                                                    self.intents, now_ms=now_ms)
        self.state = 'PROGRAM_VERIFIED'
        self._event('program_verified', {'program_revision': self.program['program_revision'],
                                         'decision_source': 'cosmos_candidate_readout',
                                         'model_generated_program': False})
        return self.decision

    def generate(self, raw_text, *, now_ms):
        require(self.mode == 'shadow', 'SITL decision mode requires candidate readout')
        return super().generate(raw_text, now_ms=now_ms)

    def authorize_execution(self):
        require(self.mode == 'sitl' and self.decision is not None, 'Shadow mode cannot authorize execution')
        self._at('SAFETY_VALIDATED')
        require(self.decision['acceptance']['accepted'], 'Rejected decision cannot execute')
        self.state = 'EXECUTING'
        body = {'context_id': self.context['context_id'],
                'compiled_revision': self.compiled['compiled_revision'],
                'audit_head': self.audit[-1]['event_sha256'],
                'decision_receipt_sha256': self.decision['receipt']['receipt_sha256'],
                'simulation_only': True, 'real_flight_authorized': False}
        result = {'schema_version': 'tancho-simulation-authorization-1.0', **body,
                  'authorization_sha256': canonical_sha256(body)}
        self._event('simulation_execution_authorized', result)
        return result

    def reobserve(self, context, evidence, *, now_ms):
        self._at('RESULT_VERIFIED')
        require(not self.completion_verified, 'Completed mission cannot be reobserved')
        require(context['model_revision'] == self.binding['model_revision'], 'Reobservation checkpoint changed')
        require(self.cycles < self.context['policy']['max_reobservation_cycles'], 'Reobservation cycle limit reached')
        require(context['mission_id'] == self.context['mission_id']
                and context['context_id'] != self.context['context_id'], 'Reobservation requires fresh context')
        validate_context_v1(context, now_ms=now_ms, evidence=evidence)
        self.cycles += 1
        self.context = context
        self.intents = self.program = self.compiled = self.result = None
        self.state = 'CONTEXT_VALIDATED'
        self.decision = None
        self._event('reobservation_context_validated', {'context_id': context['context_id'], 'cycle': self.cycles})

    def verify_result(self, images, *, now_ms):
        from .observation_result import verify_observation_result
        self._at('RESULT_CAPTURED')
        self.result = verify_observation_result(self.result, self.compiled, self.context, images, now_ms=now_ms)
        self.state = 'RESULT_VERIFIED'
        self._event('result_verified', {'result_revision': self.result['result_revision'],
                                       'capture_cross_check': self.result['completed'],
                                       'completion_review_required': True})
        return self.result

    def review_completion(self, context, images, *, invoke, now_ms, hold, timeout_ms):
        import hashlib
        from .observation_program_inference import infer_role
        self._at('RESULT_VERIFIED')
        require(context['mission_id'] == self.context['mission_id']
                and context['context_id'] != self.context['context_id']
                and context['model_revision'] == self.context['model_revision']
                and context['mission_profile'] == self.context['mission_profile'], 'Completion context changed mission')
        require([{k: v for k, v in row.items() if k != 'completed_questions'} for row in context['targets']]
                == [{k: v for k, v in row.items() if k != 'completed_questions'} for row in self.context['targets']],
                'Completion review changed target obligations')
        require({key: hashlib.sha256(value).hexdigest() for key, value in images.items()}
                == self.result['image_sha256'], 'Completion review needs the verified capture bytes')
        require(context['captured_at_ms'] >= max(row['captured_at_ms'] for row in self.result['observations']),
                'Completion context predates the verified capture')
        require(all(context[key] == self.context[key] for key in ('map_revision', 'calibration_id', 'envelope_revision')),
                'Completion review changed the observation geometry')
        review = ObservationProgramRuntime(context, images, now_ms=now_ms())
        intents, receipt = infer_role(review, images, role='reasoner', invoke=invoke,
                                      now_ms=now_ms, hold=hold, timeout_ms=timeout_ms)
        self.completion_verified = self.result['completed'] and intents['outcome'] == 'satisfied'
        self._event('completion_review', {'review_audit': review.audit_record(),
                                         'capture_cross_check': self.result['completed'],
                                         'reasoner_terminates': intents['outcome'] == 'satisfied',
                                         'completed': self.completion_verified})
        if self.completion_verified:
            self.state = 'COMPLETE'
            self._event('complete', {'decision_source': 'reasoner_and_capture_cross_check',
                                     'generator_source': 'cosmos_candidate_readout'})
        return self.completion_verified

    def review_geometric_completion(self, record):
        """Completion for catalog 0.6 model scenes from the geometric judgement
        (docs/road-geometric-completion-runtime-design-2026-09-30.md); the v8 Reasoner cannot judge road questions."""
        self._at('RESULT_VERIFIED')
        require(self.catalog == CATALOG_V06, 'Geometric completion is for catalog 0.6 road scenes only')
        require(record.get('model_scene_only') is True, 'Geometric completion needs a model scene judgement')
        require(record['outcome'] in ('complete', 'incomplete', 'undecidable'), 'Unknown geometric outcome')
        compiled = [{'position_m': v['position_m'], 'target_position_m': v['target_position_m']}
                    for v in self.compiled['viewpoints']]
        require(record['program_viewpoints'] == compiled, 'Geometric judgement is for another compiled program')
        # Route sequence 0 is the initial view; the verified capture holds the program viewpoints only.
        program_frames = [value for key, value in record['frame_sha256'].items() if key != '0']
        require(sorted(program_frames) == sorted(self.result['image_sha256'].values()),
                'Geometric judgement used another capture')
        body = {'outcome': record['outcome'], 'missing_items': record['missing_items'],
                'pixel_check': record['pixel_check'], 'capture_cross_check': self.result['completed']}
        self._event('geometric_completion_review', body)
        if record['outcome'] == 'undecidable':
            self.abort('completion_undecidable')
            return False
        self.completion_verified = record['outcome'] == 'complete' and self.result['completed']
        if self.completion_verified:
            self.state = 'COMPLETE'
            self._event('complete', {'decision_source': 'geometric_items_and_pixel_check',
                                     'generator_source': 'cosmos_candidate_readout', 'model_scene_only': True})
        return self.completion_verified


def infer_decision(runtime, frames, *, invoke, now_ms, hold, timeout_ms, calibration=None,
                   execution_kind='native_cosmos', order=(0, 1, 2)):
    require(execution_kind in ('native_cosmos', 'cpu_fixture'), 'Unknown execution source')
    runtime._at('INTENTS_VERIFIED')
    require(runtime.decision is None, 'Decision already recorded for this context')
    require(type(timeout_ms) is int and 0 < timeout_ms <= 120000, 'Invalid decision timeout')
    started = now_ms()
    build, receipt_for = ((build_request_v06, make_receipt_v06) if runtime.catalog == CATALOG_V06
                          else (build_request, make_receipt))
    request = build(runtime.context, runtime.intents, frames, now_ms=started, order=order)
    deadline = time.monotonic() + timeout_ms / 1000
    previous = started
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(invoke, deepcopy(request), deepcopy(frames))
    try:
        while not future.done():
            current = now_ms()
            require(type(current) is int and current >= previous, 'Decision clock moved backwards')
            previous = current
            require(current - started < timeout_ms and current < runtime.context['expires_at_ms']
                    and time.monotonic() < deadline, 'Decision deadline exceeded')
            hold()
            time.sleep(.005)
        completed = now_ms()
        require(type(completed) is int and completed >= previous
                and completed < runtime.context['expires_at_ms'] and time.monotonic() < deadline,
                'Late decision discarded')
        response = future.result()
        require(response['schema_version'] == RESPONSE
                and response['execution_kind'] == execution_kind
                and response['request_sha256'] == request['request_sha256']
                and response['binding'] == runtime.binding, 'Decision provider binding changed')
        proof = response['processor_input_sha256']
        require(isinstance(proof, str) and len(proof) == 64
                and all(c in '0123456789abcdef' for c in proof), 'Missing processor proof')
        receipt = receipt_for(request, response['logits'], runtime.binding['label_token_ids'],
                               started_ms=started, completed_ms=completed, timeout_ms=timeout_ms)
        return runtime.accept(request, receipt, frames, now_ms=completed,
                              calibration=calibration, provider=response)
    except Exception:
        if runtime.state not in ('ABORTED', 'COMPLETE', 'DETECTOR_HANDOFF'):
            runtime.abort('decision_inference_failed')
        raise
    finally:
        # Native transport may still finish; its late output cannot mutate this runtime.
        pool.shutdown(wait=False, cancel_futures=True)
