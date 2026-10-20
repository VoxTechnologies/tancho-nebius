"""Deterministic intercept and search staging for a drifting instrument; tracking starts after handoff."""
import math

from .contracts import fields, number, require
from .mission_context import canonical_sha256, validate_context_v1
from .observation_compiler import (CompileError, intent_target_registry,
                                   validate_vehicle_snapshot)
from .px4_mission import FRAME, check_envelope, inside, validate_envelope

SCHEMA = 'tancho-compiled-dynamic-search-1.0'
SCENE_SCHEMA = 'tancho-dynamic-search-snapshot-1.0'


def _point(value, message):
    require(isinstance(value, list) and len(value) == 3 and all(number(v) for v in value), message)


def validate_dynamic_scene(scene, context, *, now_ms):
    fields(scene, 'schema_version map_revision calibration_id envelope_revision envelope '
                  'track river camera policy blocked_regions unknown_regions')
    require(scene['schema_version'] == SCENE_SCHEMA, 'Unknown dynamic scene version')
    require(scene['map_revision'] == context['map_revision'], 'Dynamic map revision changed')
    require(scene['calibration_id'] == context['calibration_id'], 'Dynamic calibration changed')
    require(scene['envelope_revision'] == context['envelope_revision'], 'Dynamic envelope changed')
    validate_envelope(scene['envelope'])
    require(scene['envelope']['revision'] == context['envelope_revision'], 'Envelope revision changed')
    track = scene['track']
    require(isinstance(track, list) and 2 <= len(track) <= 32, 'Dynamic track needs 2 to 32 points')
    previous = None
    for row in track:
        fields(row, 'time_ms position_m')
        require(type(row['time_ms']) is int and (previous is None or row['time_ms'] > previous),
                'Track times must increase')
        _point(row['position_m'], 'Invalid track position')
        previous = row['time_ms']
    require(track[-1]['time_ms'] <= now_ms and now_ms - track[-1]['time_ms'] <= 5000,
            'Dynamic track is stale or future')
    river = scene['river']
    fields(river, 'center_m flow_direction_xy width_m search_length_m water_alt_m')
    _point(river['center_m'], 'Invalid river center')
    flow = river['flow_direction_xy']
    require(isinstance(flow, list) and len(flow) == 2 and all(number(v) for v in flow)
            and math.hypot(*flow) > 1e-6, 'Invalid river flow')
    require(number(river['width_m']) and river['width_m'] > 0
            and number(river['search_length_m']) and river['search_length_m'] > 0
            and number(river['water_alt_m']), 'Invalid river geometry')
    camera = scene['camera']
    fields(camera, 'horizontal_fov_deg image_width_px target_size_m required_pixels')
    require(number(camera['horizontal_fov_deg']) and 1 < camera['horizontal_fov_deg'] < 180
            and type(camera['image_width_px']) is int and camera['image_width_px'] > 0
            and number(camera['target_size_m']) and camera['target_size_m'] > 0
            and number(camera['required_pixels']) and camera['required_pixels'] > 0,
            'Invalid dynamic camera requirement')
    policy = scene['policy']
    fields(policy, 'intercept_lead_ms search_alt_m sweep_spacing_m dwell_ms')
    require(type(policy['intercept_lead_ms']) is int and 0 < policy['intercept_lead_ms'] <= 60000
            and number(policy['search_alt_m'])
            and number(policy['sweep_spacing_m']) and policy['sweep_spacing_m'] > 0
            and type(policy['dwell_ms']) is int and 1000 <= policy['dwell_ms'] <= 60000,
            'Invalid dynamic search policy')
    for key in ('blocked_regions', 'unknown_regions'):
        require(isinstance(scene[key], list) and len(scene[key]) <= 64, 'Invalid dynamic regions')
        for region in scene[key]:
            fields(region, 'center_m radius_m min_alt_m max_alt_m')
            _point(region['center_m'], 'Invalid dynamic region center')
            require(number(region['radius_m']) and region['radius_m'] > 0
                    and number(region['min_alt_m']) and number(region['max_alt_m'])
                    and region['min_alt_m'] <= region['max_alt_m'], 'Invalid dynamic region')
    return scene


def _program_integrity(program, context):
    fields(program, 'schema_version context_id mission_id intent_revision outcome observations '
                    'program_revision source_output_sha256')
    require(program['schema_version'] == 'tancho-verified-observation-program-1.0',
            'Unverified observation program')
    require(program['context_id'] == context['context_id']
            and program['mission_id'] == context['mission_id'], 'Program is bound elsewhere')
    body = {key: program[key] for key in ('context_id', 'mission_id', 'intent_revision',
                                          'outcome', 'observations')}
    require(program['program_revision'] == canonical_sha256(body),
            'Observation program changed after verification')
    require(program['outcome'] == 'observe', 'Only observable programs can be compiled')


def _predicted(scene):
    first, last = scene['track'][-2:]
    seconds = (last['time_ms'] - first['time_ms']) / 1000
    lead = scene['policy']['intercept_lead_ms'] / 1000
    return [last['position_m'][axis]
            + (last['position_m'][axis] - first['position_m'][axis]) / seconds * lead
            for axis in range(3)]


def _max_range(camera):
    gsd = camera['target_size_m'] / camera['required_pixels']
    return gsd * camera['image_width_px'] / (2 * math.tan(math.radians(camera['horizontal_fov_deg']) / 2))


def _river_coordinates(point, river):
    length = math.hypot(*river['flow_direction_xy'])
    flow = (river['flow_direction_xy'][0] / length, river['flow_direction_xy'][1] / length)
    cross = (-flow[1], flow[0])
    offset = (point[0] - river['center_m'][0], point[1] - river['center_m'][1])
    return offset[0] * flow[0] + offset[1] * flow[1], offset[0] * cross[0] + offset[1] * cross[1]


def _inside_search_region(point, river):
    along, across = _river_coordinates(point, river)
    return (abs(along) <= river['search_length_m'] / 2
            and abs(across) <= river['width_m'] / 2)


def _crosses_region(start, end, region):
    if max(start[2], end[2]) < region['min_alt_m'] or min(start[2], end[2]) > region['max_alt_m']:
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    t = 0 if denominator == 0 else max(0, min(1, ((region['center_m'][0] - start[0]) * dx
                                                  + (region['center_m'][1] - start[1]) * dy)
                                                 / denominator))
    closest = (start[0] + t * dx, start[1] + t * dy)
    return math.dist(closest, region['center_m'][:2]) <= region['radius_m']


def _search_points(scene):
    river, policy = scene['river'], scene['policy']
    length = math.hypot(*river['flow_direction_xy'])
    flow = (river['flow_direction_xy'][0] / length, river['flow_direction_xy'][1] / length)
    cross = (-flow[1], flow[0])
    steps = max(1, math.ceil(river['search_length_m'] / policy['sweep_spacing_m']))
    result = []
    for index in range(steps + 1):
        along = -river['search_length_m'] / 2 + river['search_length_m'] * index / steps
        sides = (-.4, .4) if index % 2 == 0 else (.4, -.4)
        for side in sides:
            result.append([river['center_m'][0] + flow[0] * along + cross[0] * river['width_m'] * side,
                           river['center_m'][1] + flow[1] * along + cross[1] * river['width_m'] * side,
                           river['water_alt_m']])
    return result


def compile_dynamic_search(program, context, scene, vehicle, *, now_ms):
    validate_context_v1(context, now_ms=now_ms)
    require(context['mission_profile'] == 'float_recovery_v1',
            'Dynamic compiler only accepts float_recovery_v1')
    validate_dynamic_scene(scene, context, now_ms=now_ms)
    validate_vehicle_snapshot(vehicle, context, now_ms=now_ms)
    _program_integrity(program, context)
    registry = intent_target_registry(context)
    targets = {target['target_id']: target for target in context['targets']}
    polygon = [tuple(point) for point in scene['envelope']['polygon_m']]
    altitude = scene['policy']['search_alt_m']
    require(scene['envelope']['min_alt_m'] < altitude < scene['envelope']['max_alt_m'],
            'Dynamic search altitude is outside the envelope')
    viewpoints = []
    previous_position = vehicle['position_m']
    unsafe_regions = scene['blocked_regions'] + scene['unknown_regions']
    for observation in program['observations']:
        require(observation['intent_id'] in registry, 'Unknown dynamic intent')
        target_id = registry[observation['intent_id']]
        require(targets[target_id]['category'] in ('float_track', 'river_search_region'),
                'Invalid dynamic target')
        aims = ([_predicted(scene)] if observation['view_relation'] == 'intercept'
                else _search_points(scene))
        for aim in aims:
            aim = [aim[0], aim[1], scene['river']['water_alt_m']]
            require(_inside_search_region(aim, scene['river']),
                    'Dynamic aim leaves the declared river search region')
            position = [aim[0], aim[1], altitude]
            require(inside(polygon, position[:2]), 'Dynamic viewpoint leaves the envelope')
            require(math.dist(position, aim) <= _max_range(scene['camera']),
                    'Dynamic viewpoint misses the detection pixel requirement')
            require(not any(_crosses_region(previous_position, position, region)
                            or _crosses_region(position, aim, region)
                            for region in unsafe_regions),
                    'Dynamic path intersects blocked or unknown space')
            viewpoints.append({'sequence': len(viewpoints),
                               'observation_sequence': observation['sequence'],
                               'intent_id': observation['intent_id'], 'target_id': target_id,
                               'position_m': [round(value, 2) for value in position],
                               'target_position_m': [round(value, 2) for value in aim],
                               'dwell_ms': scene['policy']['dwell_ms'],
                               'speed_mps': min(vehicle['cruise_speed_mps'],
                                                scene['envelope']['max_speed_mps']),
                               'sensor': observation['sensor'],
                               'stop_condition': observation['stop_condition']})
            require(len(viewpoints) <= context['policy']['max_viewpoints'],
                    'Compiled viewpoint limit exceeded')
            previous_position = position
    body = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
            'program_revision': program['program_revision'],
            'envelope_revision': context['envelope_revision'],
            'calibration_id': context['calibration_id'], 'coordinate_frame': FRAME,
            'viewpoints': viewpoints}
    result = {'schema_version': SCHEMA, **body, 'compiled_revision': canonical_sha256(body)}
    from .viewpoint_adapter import adapt_to_viewpoints_v1
    check_envelope(adapt_to_viewpoints_v1(result, context), scene['envelope'],
                   start_position_m=vehicle['position_m'])
    return result
