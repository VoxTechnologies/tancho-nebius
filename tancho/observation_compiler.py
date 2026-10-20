"""Deterministic compilation of verified observation semantics into ENU viewpoints.

The Generator selects relations and coverage.  This module alone selects metric
positions from trusted geometry, calibration, envelope, and vehicle state.
"""
from __future__ import annotations

import math

from .contracts import ContractError, fields, number, require, text
from .mission_context import canonical_sha256, validate_context_v1
from .mission_profiles import profile
from .px4_mission import FRAME, check_envelope, inside, validate_envelope
from .viewpoint_plan import ALTITUDE_MARGIN_M

SCHEMA = 'tancho-compiled-viewplan-1.0'
SCENE_SCHEMA = 'tancho-scene-snapshot-1.0'
VEHICLE_SCHEMA = 'tancho-vehicle-snapshot-1.0'

# Ordered clockwise from north.  Coordinates are quantized only after a safe
# candidate is selected, so quantization can never pull a point into bounds.
DIRECTIONS = ((0.0, 1.0), (math.sqrt(.5), math.sqrt(.5)), (1.0, 0.0),
              (math.sqrt(.5), -math.sqrt(.5)), (0.0, -1.0),
              (-math.sqrt(.5), -math.sqrt(.5)), (-1.0, 0.0),
              (-math.sqrt(.5), math.sqrt(.5)))


# Opt-in view contract: scenes without view_contract_revision compile exactly as before.
VIEW_CONTRACT = 'tancho-view-contract-1.0'
CLOSE_RELATIONS = {'nadir': -90.0, 'oblique': -30.0, 'cross_section': -30.0, 'along_axis': -30.0}
# Wide relations frame this many target radii across the image instead of approaching.
WIDE_RELATIONS = {'context': (-50.0, 16.0), 'overview': (-70.0, 8.0)}
CLOSE_SCALES = (1.0, .8, .6)
NOVEL_ANGLE_DEG, NOVEL_DISTANCE_RATIO = 30.0, .8
MIN_CLEARANCE_M, PATH_SAMPLE_M = 10.0, 2.0
# Opt-in aim contract: whole-boundary observations look at the target centre from the nearest
# distance at which every boundary point projects inside the capture window margins. Scenes without aim_contract
# compile exactly as before.
AIM_CONTRACT = 'tancho-aim-contract-1.1'
CAPTURE_ASPECT = 3 / 2  # the capture check measures the central 3:2 window


class CompileError(ContractError):
    """A valid semantic program has no safe geometric realization."""


def _point(value, message):
    require(isinstance(value, list) and len(value) == 3 and all(number(v) for v in value), message)


def _positive(value, message):
    require(number(value) and value > 0, message)


def view_contract(scene):
    return scene.get('view_contract_revision') == VIEW_CONTRACT


def aim_contract(scene):
    return 'aim_contract' in scene


def validate_scene_snapshot(scene, context):
    contracted = isinstance(scene, dict) and 'view_contract_revision' in scene
    aimed = isinstance(scene, dict) and 'aim_contract' in scene
    fields(scene, 'schema_version map_revision calibration_id envelope_revision envelope '
                  'camera geometries blocked_regions unknown_regions'
                  + (' view_contract_revision surface_obstacles' if contracted else '')
                  + (' aim_contract' if aimed else ''))
    if aimed:
        aim = scene['aim_contract']
        fields(aim, 'revision image_height_px margin_fraction')
        require(aim['revision'] == AIM_CONTRACT, 'Unknown aim contract')
        require(type(aim['image_height_px']) is int and aim['image_height_px'] > 0, 'Invalid image height')
        require(number(aim['margin_fraction']) and 0 <= aim['margin_fraction'] < .5, 'Invalid image margin')
    require(not contracted or scene['view_contract_revision'] == VIEW_CONTRACT,
            'Unknown view contract')
    require(scene['schema_version'] == SCENE_SCHEMA, 'Unknown scene snapshot version')
    require(scene['map_revision'] == context['map_revision'], 'Scene map revision changed')
    require(scene['calibration_id'] == context['calibration_id'], 'Scene calibration changed')
    require(scene['envelope_revision'] == context['envelope_revision'], 'Scene envelope changed')
    validate_envelope(scene['envelope'])
    require(scene['envelope']['revision'] == context['envelope_revision'], 'Envelope revision changed')
    camera = scene['camera']
    fields(camera, 'horizontal_fov_deg image_width_px quality_requirements')
    require(number(camera['horizontal_fov_deg']) and 1 <= camera['horizontal_fov_deg'] < 180,
            'Invalid camera field of view')
    require(type(camera['image_width_px']) is int and camera['image_width_px'] > 0,
            'Invalid camera image width')
    quality = camera['quality_requirements']
    require(isinstance(quality, dict), 'Invalid quality requirements')
    for key, value in quality.items():
        require(key in ('identify', 'state', 'damage'), 'Unknown detail quality requirement')
        fields(value, 'required_pixels dwell_ms' + (' max_gsd_m' if contracted else ''))
        _positive(value['required_pixels'], 'Invalid pixel requirement')
        if contracted:
            _positive(value['max_gsd_m'], 'Invalid ground sample distance requirement')
        require(type(value['dwell_ms']) is int and 1000 <= value['dwell_ms'] <= 60000,
                'Invalid dwell requirement')
    geometries = scene['geometries']
    require(isinstance(geometries, dict) and geometries, 'Target geometry is required')
    expected = {target['geometry_ref'] for target in context['targets']}
    require(expected <= set(geometries), 'Missing target geometry')
    for ref, geometry in geometries.items():
        require(text(ref), 'Invalid geometry reference')
        fields(geometry, 'center_m radius_m axis_xy boundary_points_m connection_points_m')
        _point(geometry['center_m'], 'Invalid geometry center')
        _positive(geometry['radius_m'], 'Invalid geometry radius')
        axis = geometry['axis_xy']
        require(isinstance(axis, list) and len(axis) == 2 and all(number(v) for v in axis)
                and math.hypot(*axis) > 1e-6, 'Invalid geometry axis')
        for key in ('boundary_points_m', 'connection_points_m'):
            require(isinstance(geometry[key], list) and len(geometry[key]) <= 32,
                    'Invalid geometry point set')
            for point in geometry[key]:
                _point(point, 'Invalid geometry point')
    for key in ('blocked_regions', 'unknown_regions'):
        require(isinstance(scene[key], list) and len(scene[key]) <= 64, 'Invalid scene regions')
        for region in scene[key]:
            fields(region, 'center_m radius_m min_alt_m max_alt_m')
            _point(region['center_m'], 'Invalid region center')
            _positive(region['radius_m'], 'Invalid region radius')
            require(number(region['min_alt_m']) and number(region['max_alt_m'])
                    and region['min_alt_m'] <= region['max_alt_m'], 'Invalid region altitude')
    if contracted:
        obstacles = scene['surface_obstacles']
        require(isinstance(obstacles, list) and len(obstacles) <= 256, 'Invalid surface obstacles')
        for obstacle in obstacles:
            fields(obstacle, 'center_m radius_m top_m')
            require(isinstance(obstacle['center_m'], list) and len(obstacle['center_m']) == 2
                    and all(number(v) for v in obstacle['center_m']), 'Invalid obstacle center')
            _positive(obstacle['radius_m'], 'Invalid obstacle radius')
            require(number(obstacle['top_m']), 'Invalid obstacle height')
    return scene


def _clear_of_surface(position, obstacles):
    """Conservative clearance: circular footprints with a flat top, plus a vertical margin."""
    return all(math.dist(position[:2], obstacle['center_m']) > obstacle['radius_m'] + MIN_CLEARANCE_M
               or position[2] >= obstacle['top_m'] + MIN_CLEARANCE_M for obstacle in obstacles)


def _path_clear_of_surface(start, end, obstacles):
    steps = max(1, math.ceil(math.dist(start, end) / PATH_SAMPLE_M))
    return all(_clear_of_surface([start[i] + (end[i] - start[i]) * k / steps for i in range(3)],
                                 obstacles) for k in range(steps + 1))


def _novel(position, aim, start):
    """A view adds information when it looks from a new side or from clearly closer."""
    a = (position[0] - aim[0], position[1] - aim[1])
    b = (start[0] - aim[0], start[1] - aim[1])
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < 1e-6 or nb < 1e-6:
        return True
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))))
    return angle >= NOVEL_ANGLE_DEG or math.dist(position, aim) <= NOVEL_DISTANCE_RATIO * math.dist(start, aim)


def validate_vehicle_snapshot(vehicle, context, *, now_ms):
    fields(vehicle, 'schema_version captured_at_ms position_m cruise_speed_mps '
                    'battery_remaining_wh reserve_wh')
    require(vehicle['schema_version'] == VEHICLE_SCHEMA, 'Unknown vehicle snapshot version')
    require(type(vehicle['captured_at_ms']) is int and vehicle['captured_at_ms'] <= now_ms,
            'Invalid vehicle timestamp')
    require(now_ms - vehicle['captured_at_ms'] <= 2000, 'Stale vehicle snapshot')
    _point(vehicle['position_m'], 'Invalid vehicle position')
    require(number(vehicle['cruise_speed_mps']) and .5 <= vehicle['cruise_speed_mps'] <= 15,
            'Invalid vehicle speed')
    require(number(vehicle['battery_remaining_wh']) and number(vehicle['reserve_wh'])
            and vehicle['battery_remaining_wh'] > vehicle['reserve_wh'] >= 0,
            'Insufficient battery reserve')
    require(context['captured_at_ms'] <= vehicle['captured_at_ms'],
            'Vehicle snapshot predates the mission context')
    return vehicle


def _unit(axis):
    length = math.hypot(*axis)
    return axis[0] / length, axis[1] / length


def _directions(relation, geometry):
    axis = _unit(geometry['axis_xy'])
    perpendicular = (-axis[1], axis[0])
    if relation == 'along_axis':
        return (axis, (-axis[0], -axis[1]))
    if relation == 'cross_section':
        return (perpendicular, (-perpendicular[0], -perpendicular[1]))
    return DIRECTIONS


def intent_target_registry(context):
    """Rebuild the bounded app-issued intent ID registry without model target fields."""
    targets = {target['target_id']: target for target in context['targets']}
    p = profile(context['mission_profile'])
    result = {}
    for sequence in range(context['policy']['max_intents']):
        for target_id, target in targets.items():
            for question in p.category_questions[target['category']]:
                intent_id = canonical_sha256({'context_id': context['context_id'],
                                               'sequence': sequence, 'target_id': target_id,
                                               'question': question})
                result[intent_id] = target_id
    return result


def _aim_points(observation, geometry):
    coverage = observation['coverage']
    if coverage in ('connection', 'facade_foot', 'approach_corridor'):
        points = geometry['connection_points_m']
        if not points:
            raise CompileError('missing_connection_geometry')
        return points
    if coverage in ('boundary', 'upper_lower_edges', 'upstream_downstream'):
        points = geometry['boundary_points_m']
        if len(points) < 2:
            raise CompileError('missing_boundary_geometry')
        # Stable extrema provide the minimum two-sided boundary witness.
        return [min(points), max(points)]
    return [geometry['center_m']]


def _intersects_region(position, regions):
    for region in regions:
        if region['min_alt_m'] <= position[2] <= region['max_alt_m']:
            if math.dist(position[:2], region['center_m'][:2]) <= region['radius_m']:
                return True
    return False


def _line_intersects_region(start, end, region):
    # Conservative 2D closest-point test plus overlapping altitude intervals.
    if max(start[2], end[2]) < region['min_alt_m'] or min(start[2], end[2]) > region['max_alt_m']:
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    t = 0 if denominator == 0 else max(0, min(1, ((region['center_m'][0] - start[0]) * dx
                                                  + (region['center_m'][1] - start[1]) * dy)
                                                 / denominator))
    closest = (start[0] + t * dx, start[1] + t * dy)
    return math.dist(closest, region['center_m'][:2]) <= region['radius_m']


def _frames_target(position, aim, geometry, scene):
    """Every boundary point projects inside the capture window margins of a camera at position looking at aim."""
    contract = scene['aim_contract']
    width, height = scene['camera']['image_width_px'], contract['image_height_px']
    focal = width / 2 / math.tan(math.radians(scene['camera']['horizontal_fov_deg']) / 2)
    crop = min(width, height * CAPTURE_ASPECT)
    margin = contract['margin_fraction']
    left, right = (width - crop) / 2 + margin * crop, (width + crop) / 2 - margin * crop
    top, bottom = margin * height, height - margin * height
    forward = [a - p for a, p in zip(aim, position, strict=True)]
    norm = math.sqrt(sum(v * v for v in forward))
    forward = [v / norm for v in forward]
    right_axis = [forward[1], -forward[0], 0.0]
    side = math.hypot(right_axis[0], right_axis[1])
    right_axis = [v / side for v in right_axis]
    up = [right_axis[1] * forward[2] - right_axis[2] * forward[1],
          right_axis[2] * forward[0] - right_axis[0] * forward[2],
          right_axis[0] * forward[1] - right_axis[1] * forward[0]]
    for point in geometry['boundary_points_m']:
        d = [q - p for q, p in zip(point, position, strict=True)]
        depth = sum(a * b for a, b in zip(d, forward, strict=True))
        if depth <= 0:
            return False
        x = width / 2 + focal * sum(a * b for a, b in zip(d, right_axis, strict=True)) / depth
        y = height / 2 - focal * sum(a * b for a, b in zip(d, up, strict=True)) / depth
        if not (left < x < right and top < y < bottom):
            return False
    return True


def _candidates(aim, geometry, observation, scene):
    quality = scene['camera']['quality_requirements'].get(observation['detail_level'])
    if quality is None:
        raise CompileError('missing_quality_requirement')
    half_fov = math.radians(scene['camera']['horizontal_fov_deg']) / 2
    wanted = ((geometry['radius_m'] * 2 / quality['required_pixels'])
              * scene['camera']['image_width_px'] / (2 * math.tan(half_fov))) / 1.05
    envelope = scene['envelope']
    polygon = [tuple(p) for p in envelope['polygon_m']]
    relation = observation['view_relation']
    contracted = view_contract(scene)
    if contracted and relation in WIDE_RELATIONS:
        pitch_deg, radii = WIDE_RELATIONS[relation]
        base, scales = geometry['radius_m'] * radii / (2 * math.tan(half_fov)), (1.0,)
    elif contracted:
        cap = quality['max_gsd_m'] * scene['camera']['image_width_px'] / (2 * math.tan(half_fov))
        pitch_deg, base, scales = CLOSE_RELATIONS[relation], min(wanted, cap), CLOSE_SCALES
    else:
        pitch_deg, base, scales = None, wanted, (1.0, .8, .6, .45, .3)
    fit = aim_contract(scene) and observation['stop_condition'] == 'full_boundary_visible'
    if fit:
        scales = (1.0,)
    pitch = pitch_deg if contracted else (-90 if relation == 'nadir' else -35)

    def place(direction, slant):
        ground = slant * math.cos(math.radians(-pitch))
        height = slant * math.sin(math.radians(-pitch))
        altitude = min(max(aim[2] + height, envelope['min_alt_m'] + ALTITUDE_MARGIN_M),
                       envelope['max_alt_m'] - ALTITUDE_MARGIN_M)
        return [aim[0] + direction[0] * ground, aim[1] + direction[1] * ground, altitude]

    rows = []
    for direction_index, direction in enumerate(_directions(relation, geometry)):
        for scale_index, scale in enumerate(scales):
            slant = base * scale
            if fit:
                # The nearest distance, from the resolution distance outwards, that frames the whole target.
                steps = [base * (1 + .01 * k) for k in range(301)]
                slant = next((s for s in steps if _frames_target(place(direction, s), aim, geometry, scene)), None)
                if slant is None:
                    continue
            position = place(direction, slant)
            if not inside(polygon, position[:2]) or position[2] <= aim[2]:
                continue
            if _intersects_region(position, scene['blocked_regions'] + scene['unknown_regions']):
                continue
            if any(_line_intersects_region(position, aim, r)
                   for r in scene['blocked_regions'] + scene['unknown_regions']):
                continue
            if contracted and not _clear_of_surface(position, scene['surface_obstacles']):
                continue
            # The first tuple fields represent the four mandatory zero penalties.
            key = (0, 0, 0, 0, 1, round(slant, 6),
                   tuple(round(v, 2) for v in position), direction_index, scale_index)
            rows.append((key, position, quality['dwell_ms']))
    if not rows:
        raise CompileError('no_safe_viewpoint')
    return sorted(rows, key=lambda row: row[0])


def compile_observations(program, context, scene, vehicle, *, now_ms):
    """Compile a verified static-target program; no model-produced metric is accepted."""
    validate_context_v1(context, now_ms=now_ms)
    require(context['mission_profile'] != 'float_recovery_v1',
            'Dynamic mission requires compile_dynamic_search')
    validate_scene_snapshot(scene, context)
    validate_vehicle_snapshot(vehicle, context, now_ms=now_ms)
    fields(program, 'schema_version context_id mission_id intent_revision outcome observations '
                    'program_revision source_output_sha256')
    require(program['schema_version'] == 'tancho-verified-observation-program-1.0',
            'Unverified observation program')
    require(program['context_id'] == context['context_id']
            and program['mission_id'] == context['mission_id'], 'Program is bound elsewhere')
    require(program['outcome'] == 'observe', 'Only observable programs can be compiled')
    program_body = {key: program[key] for key in ('context_id', 'mission_id', 'intent_revision',
                                                   'outcome', 'observations')}
    require(program['program_revision'] == canonical_sha256(program_body),
            'Observation program changed after verification')
    targets = {target['target_id']: target for target in context['targets']}
    intent_targets = intent_target_registry(context)
    geometry_by_target = {target_id: scene['geometries'][target['geometry_ref']]
                          for target_id, target in targets.items()}
    viewpoints = []
    previous_position = vehicle['position_m']
    unsafe_regions = scene['blocked_regions'] + scene['unknown_regions']
    for observation in program['observations']:
        require(observation['intent_id'] in intent_targets, 'Unknown compiled intent')
        target_id = intent_targets[observation['intent_id']]
        geometry = geometry_by_target[target_id]
        aims = _aim_points(observation, geometry)
        if aim_contract(scene) and observation['stop_condition'] == 'full_boundary_visible':
            aims = [geometry['center_m']]
        required_sides = 2 if observation['coverage'] == 'both_sides' else 1
        for aim_index, aim in enumerate(aims):
            candidates = _candidates(aim, geometry, observation, scene)
            contracted = view_contract(scene)
            if contracted:
                obstacles = scene['surface_obstacles']
                candidates = [c for c in candidates
                              if _path_clear_of_surface(previous_position, c[1], obstacles)]
                # Novel views rank before travel; within each group the original order holds.
                start = vehicle['position_m']
                candidates = sorted(candidates, key=lambda c: (not _novel(c[1], aim, start), c[0]))
            if required_sides == 1:
                options = [((0 if not contracted or _novel(candidate[1], aim, vehicle['position_m']) else 1),
                            math.dist(previous_position, candidate[1])
                            + math.dist(candidate[1], vehicle['position_m']), candidate)
                           for candidate in candidates
                           if not any(_line_intersects_region(previous_position, candidate[1], region)
                                      for region in unsafe_regions)]
                chosen = [min(options, key=lambda value: (value[0], value[1], value[2][0]))[2]] if options else []
            else:
                pairs = []
                for first in candidates:
                    if any(_line_intersects_region(previous_position, first[1], region)
                           for region in unsafe_regions):
                        continue
                    for second in candidates:
                        a = (first[1][0] - aim[0], first[1][1] - aim[1])
                        b = (second[1][0] - aim[0], second[1][1] - aim[1])
                        if a[0] * b[0] + a[1] * b[1] >= 0:
                            continue
                        if any(_line_intersects_region(first[1], second[1], region)
                               for region in unsafe_regions):
                            continue
                        if contracted and not _path_clear_of_surface(first[1], second[1],
                                                                     scene['surface_obstacles']):
                            continue
                        travel = (math.dist(previous_position, first[1])
                                  + math.dist(first[1], second[1])
                                  + math.dist(second[1], vehicle['position_m']))
                        rank = (0 if not contracted else
                                (not _novel(first[1], aim, vehicle['position_m']))
                                + (not _novel(second[1], aim, vehicle['position_m'])))
                        pairs.append((rank, travel, first[0], second[0], first, second))
                chosen = list(min(pairs, key=lambda value: value[:4])[4:]) if pairs else []
            if len(chosen) != required_sides:
                raise CompileError('coverage_has_no_safe_solution')
            for _, position, dwell_ms in chosen:
                viewpoints.append({
                    'sequence': len(viewpoints),
                    'observation_sequence': observation['sequence'],
                    'intent_id': observation['intent_id'],
                    'target_id': target_id,
                    'position_m': [round(v, 2) for v in position],
                    'target_position_m': [round(v, 2) for v in aim],
                    'dwell_ms': dwell_ms,
                    'speed_mps': min(vehicle['cruise_speed_mps'], scene['envelope']['max_speed_mps']),
                    'sensor': observation['sensor'],
                    'stop_condition': observation['stop_condition'],
                })
                require(len(viewpoints) <= context['policy']['max_viewpoints'],
                        'Compiled viewpoint limit exceeded')
                previous_position = position
    require(not any(_line_intersects_region(previous_position, vehicle['position_m'], region)
                    for region in unsafe_regions), 'Return path intersects blocked or unknown space')
    if view_contract(scene):
        require(_path_clear_of_surface(previous_position, vehicle['position_m'],
                                       scene['surface_obstacles']),
                'Return path lacks surface clearance')
    body = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
            'program_revision': program['program_revision'],
            'envelope_revision': context['envelope_revision'],
            'calibration_id': context['calibration_id'], 'coordinate_frame': FRAME,
            'viewpoints': viewpoints}
    result = {'schema_version': SCHEMA, **body, 'compiled_revision': canonical_sha256(body)}
    # Exercise the flight envelope including path and return budget before release.
    from .viewpoint_adapter import adapt_to_viewpoints_v1
    legacy = adapt_to_viewpoints_v1(result, context)
    check_envelope(legacy, scene['envelope'], start_position_m=vehicle['position_m'])
    return result


def compile_program(program, context, scene, vehicle, *, now_ms):
    """Route static and dynamic profiles to separate deterministic compilers."""
    if context.get('mission_profile') == 'float_recovery_v1':
        from .dynamic_search_compiler import compile_dynamic_search
        return compile_dynamic_search(program, context, scene, vehicle, now_ms=now_ms)
    return compile_observations(program, context, scene, vehicle, now_ms=now_ms)
