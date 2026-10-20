"""RFL-gated conversion of a compiled plan into an unsent PX4 mission."""
from .contracts import require
from .mission_context import canonical_sha256
from .px4_mission import check_envelope, to_mission, validate_viewpoints
from .viewpoint_adapter import adaptation_record, adapt_to_viewpoints_v1


def prepare_px4_mission(runtime, envelope, *, start_position_m, now_ms):
    """Prepare and authorize; the caller still owns the actual vehicle transport."""
    require(runtime.state == 'COMPILED', 'Flight preparation requires a compiled plan')
    context = runtime.context
    viewpoints = adapt_to_viewpoints_v1(runtime.compiled, context)
    legacy_context = {
        'schema_version': '1.0', 'observation_id': context['mission_id'],
        'captured_at_ms': context['captured_at_ms'], 'expires_at_ms': context['expires_at_ms'],
        'input_sha256': context['input_sha256'], 'model_revision': context['model_revision'],
        'goal_revision': runtime.compiled['program_revision'],
        'envelope_revision': context['envelope_revision'],
        'evidence_ids': list(context['evidence_ids']),
        'target_ids': [target['target_id'] for target in context['targets']],
    }
    validate_viewpoints(viewpoints, legacy_context, now_ms=now_ms,
                        goal_revision=runtime.compiled['program_revision'],
                        envelope_revision=context['envelope_revision'])
    envelope_check = check_envelope(viewpoints, envelope, start_position_m=start_position_m)
    rfl_body = {'compiled_revision': runtime.compiled['compiled_revision'],
                'viewpoints_sha256': canonical_sha256(viewpoints),
                'envelope_check_sha256': canonical_sha256(envelope_check),
                'inside_envelope': envelope_check['inside_envelope']}
    rfl = {'schema_version': 'tancho-rfl-report-1.0',
           'compiled_revision': runtime.compiled['compiled_revision'], 'passed': True,
           'report_sha256': canonical_sha256(rfl_body)}
    # Mission conversion repeats the envelope check. It is still an unsent artifact.
    mission = to_mission(viewpoints, envelope, start_position_m=start_position_m)
    runtime.approve_safety(rfl)
    authorization = runtime.authorize_execution()
    mission_sha256 = canonical_sha256(mission)
    runtime.record_px4_prepared(mission_sha256)
    return {'schema_version': 'tancho-prepared-observation-flight-1.0',
            'viewpoints': viewpoints, 'adaptation': adaptation_record(runtime.compiled, viewpoints),
            'rfl': rfl, 'authorization': authorization, 'px4_mission': mission,
            'px4_mission_sha256': mission_sha256, 'sent_to_vehicle': False}
