"""Version 1 application contracts. Validation does not authorize flight."""
import math


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, names):
    require(isinstance(value, dict) and set(value) == set(names.split()), 'Unexpected or missing fields')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text(value):
    return isinstance(value, str) and 0 < len(value) <= 4096


def strings(value):
    return isinstance(value, list) and all(text(v) for v in value) and len(set(value)) == len(value)


def validate_context(c, now_ms):
    fields(c, 'schema_version observation_id captured_at_ms expires_at_ms input_sha256 model_revision goal_revision envelope_revision evidence_ids target_ids')
    require(c['schema_version'] == '1.0', 'Unknown context version')
    for key in ('observation_id', 'model_revision', 'goal_revision', 'envelope_revision'):
        require(text(c[key]), 'Missing identity or revision')
    require(isinstance(c['input_sha256'], str) and len(c['input_sha256']) == 64 and all(v in '0123456789abcdef' for v in c['input_sha256']), 'Invalid input hash')
    require(all(type(v) is int for v in (now_ms, c['captured_at_ms'], c['expires_at_ms'])), 'Timestamps must be integer milliseconds')
    require(0 <= c['captured_at_ms'] <= now_ms < c['expires_at_ms'], 'Future or expired observation')
    require(strings(c['evidence_ids']) and strings(c['target_ids']), 'Invalid evidence or target registry')


def validate_reasoner(output, context, *, now_ms):
    """Context is constructed by the application, never trusted from model output."""
    validate_context(context, now_ms)
    fields(output, 'schema_version observation_id target_id evidence_ids missing_information goal view_request priority abstain')
    require(output['schema_version'] == '1.0', 'Unknown output version')
    require(output['observation_id'] == context['observation_id'], 'Observation mismatch')
    require(type(output['abstain']) is bool, 'abstain must be boolean')
    require(type(output['priority']) is int and 0 <= output['priority'] <= 3, 'Invalid priority')
    require(isinstance(output['missing_information'], str) and len(output['missing_information']) <= 4096, 'Invalid explanation')
    require(strings(output['evidence_ids']) and set(output['evidence_ids']) <= set(context['evidence_ids']), 'Unknown or duplicate evidence')
    require(output['goal'] in ('inspect', 'no_action', 'abstain'), 'Unknown goal')
    target = output['target_id']
    require(target is None or (text(target) and target in context['target_ids']), 'Unknown target')
    require(output['abstain'] == (output['goal'] == 'abstain'), 'Inconsistent abstention')
    if output['goal'] == 'inspect':
        require(target is not None and bool(output['evidence_ids']), 'Inspection requires target and evidence')
        require(text(output['missing_information']), 'Inspection requires missing information')
        fields(output['view_request'], 'side preferred_pitch_deg')
        view = output['view_request']
        require(view['side'] in ('north','northeast','east','southeast','south','southwest','west','northwest','any'), 'Unknown view side')
        require(number(view['preferred_pitch_deg']) and -90 <= view['preferred_pitch_deg'] <= 0, 'Invalid pitch request')
        require(output['priority'] >= 1, 'Inspection priority must be positive')
    else:
        require(output['view_request'] is None and output['priority'] == 0, 'Non-action must not request a view')
        if output['goal'] == 'abstain':
            require(text(output['missing_information']), 'Abstention needs a reason')
    if output['goal'] == 'no_action':
        require(bool(output['evidence_ids']), 'No-action decision requires evidence')
    return output


def rotation_columns(six):
    """Gram-Schmidt, column-major 6D; returns columns, not rows."""
    require(isinstance(six, list) and len(six) == 6 and all(number(x) for x in six), 'Invalid rotation6d')
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    def unit(v):
        n = math.sqrt(dot(v,v))
        require(math.isfinite(n) and n > 1e-8, 'Degenerate rotation6d')
        return [x/n for x in v]
    a = unit(six[:3]); b0 = six[3:]; proj = dot(a,b0)
    b = unit([b0[i]-proj*a[i] for i in range(3)])
    c = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    return [a,b,c]


def validate_trajectory(output, context, *, now_ms, goal_revision, envelope_revision):
    validate_context(context, now_ms)
    fields(output, 'schema_version observation_id goal_revision envelope_revision calibration_id coordinate_frame sample_period_ms poses')
    require(output['schema_version'] == '1.0', 'Unknown trajectory version')
    require(output['observation_id'] == context['observation_id'], 'Observation mismatch')
    require(output['goal_revision'] == context['goal_revision'] == goal_revision, 'Goal changed')
    require(output['envelope_revision'] == context['envelope_revision'] == envelope_revision, 'Envelope changed')
    require(text(output['calibration_id']), 'Calibration reference missing')
    require(output['coordinate_frame'] == 'camera_at_observation', 'Unknown frame')
    require(type(output['sample_period_ms']) is int and output['sample_period_ms'] in (50, 500), 'Unsupported cadence')
    poses = output['poses']
    require(isinstance(poses, list) and len(poses) == 32, 'Expected 32 poses')
    for pose in poses:
        require(isinstance(pose, list) and len(pose) == 9 and all(number(x) for x in pose), 'Expected finite 9D pose')
        rotation_columns(pose[3:])
    return output


def validate_scene_splits(records):
    seen = {}
    sample_ids = set()
    for row in records:
        fields(row, 'sample_id leakage_group_id split')
        require(text(row['sample_id']) and text(row['leakage_group_id']), 'Missing dataset identity')
        require(row['sample_id'] not in sample_ids, 'Duplicate sample')
        sample_ids.add(row['sample_id'])
        require(row['split'] in ('train','validation','test'), 'Invalid split')
        group = row['leakage_group_id']
        require(group not in seen or seen[group] == row['split'], 'Scene leakage across splits')
        seen[group] = row['split']
