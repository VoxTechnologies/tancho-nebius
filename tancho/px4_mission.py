"""Viewpoint sequence to PX4 mission: the deterministic half of the flight path.

The model proposes where to look from, in metres inside the approved envelope's
local ENU frame. It never emits latitude, longitude or a MAVLink item. This
module validates that proposal, checks it against the envelope (RFL stage 2) and
converts the survivor into mission items and gimbal commands.

Horizons matter here. The camera9d contract in `tancho/contracts.py` covers
0.05 to 1.6 seconds (or 16 seconds at the slower cadence), which is far shorter
than a mission and, by its own spec, is not executable after a 60-second-old
prediction. A viewpoint sequence is the artifact that survives the 30-second
outer loop, so mission replanning is built on this contract, not on camera9d.

PX4 SITL settled the item shape on 2026-09-08. A bare waypoint is accepted, and
so is a waypoint followed by DO_SET_ROI_LOCATION. DO_CHANGE_SPEED,
DO_GIMBAL_MANAGER_PITCHYAW and DO_MOUNT_CONTROL each came back as
MAV_MISSION_UNSUPPORTED in either order, so none of them may appear in a
mission. The camera is aimed at a point rather than by an angle, which suits a
viewpoint contract: the model already names the target it wants to see.

Speed is therefore not a mission item. It belongs to the flight-plan parameters
set before the mission is uploaded, and the envelope check still clamps it.
"""
import math

from tancho.contracts import ContractError, fields, number, require, text, validate_context

SCHEMA = 'tancho-viewpoints-1.0'
FRAME = 'envelope_local_enu'
MAX_VIEWPOINTS = 8
MAX_HORIZON_MS = 300_000
MAX_LOCAL_OFFSET_M = 5000.0
# WGS84. The local tangent plane is only used inside MAX_LOCAL_OFFSET_M.
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
# Read from the upstream MAVLink common.xml and confirmed against PX4 SITL.
MAVLINK_DEFINITION_SHA256 = '72fc17bf1615b7403160530e09ea38acad953a9f42e0446905241a575e31dd19'
MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6
# Items that carry no position use the mission frame; a global frame with zero
# coordinates is what the refused six-item form sent for the return to launch.
MAV_FRAME_MISSION = 2
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_ROI_LOCATION = 195
# Refused by PX4 as mission items; kept here so nobody adds them back.
REFUSED_IN_MISSION = {'MAV_CMD_DO_CHANGE_SPEED': 178,
                      'MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW': 1000,
                      'MAV_CMD_DO_MOUNT_CONTROL': 205}


def validate_viewpoints(output, context, *, now_ms, goal_revision, envelope_revision):
    """Accept only a bounded, ordered viewpoint sequence in the envelope frame."""
    validate_context(context, now_ms)
    fields(output, 'schema_version observation_id goal_revision envelope_revision '
                   'calibration_id coordinate_frame viewpoints')
    require(output['schema_version'] == SCHEMA, 'Unknown viewpoint version')
    require(output['observation_id'] == context['observation_id'], 'Observation mismatch')
    require(output['goal_revision'] == context['goal_revision'] == goal_revision, 'Goal changed')
    require(output['envelope_revision'] == context['envelope_revision'] == envelope_revision,
            'Envelope changed')
    require(text(output['calibration_id']), 'Calibration reference missing')
    require(output['coordinate_frame'] == FRAME, 'Unknown frame')
    rows = output['viewpoints']
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_VIEWPOINTS, 'Unbounded viewpoint list')
    total = 0
    for index, row in enumerate(rows):
        fields(row, 'sequence position_m target_position_m dwell_ms speed_mps '
                    'target_id evidence_ids')
        require(row['sequence'] == index, 'Viewpoint order is not the list order')
        position, target = row['position_m'], row['target_position_m']
        for point in (position, target):
            require(isinstance(point, list) and len(point) == 3 and all(number(v) for v in point)
                    and all(abs(v) <= MAX_LOCAL_OFFSET_M for v in point), 'Invalid local position')
        # PX4 aims the camera at a point, so the contract names the point rather
        # than an angle. Looking up is refused as before, now geometrically.
        require(target[2] <= position[2], 'The viewpoint must look at or below the horizon')
        require(math.dist(position[:2], target[:2]) >= 1.0 or position[2] - target[2] >= 1.0,
                'Viewpoint and target are the same place')
        require(type(row['dwell_ms']) is int and 1000 <= row['dwell_ms'] <= 60_000, 'Invalid dwell')
        require(number(row['speed_mps']) and 0.5 <= row['speed_mps'] <= 15.0, 'Invalid speed')
        require(text(row['target_id']), 'Viewpoint without a target')
        require(row['target_id'] in context['target_ids'], 'Unknown target reference')
        evidence = row['evidence_ids']
        require(isinstance(evidence, list) and evidence
                and all(text(e) and e in context['evidence_ids'] for e in evidence),
                'Viewpoint without resolvable evidence')
        total += row['dwell_ms']
    require(total <= MAX_HORIZON_MS, 'Dwell alone exceeds the planning horizon')
    return output


def validate_envelope(envelope):
    """The approved volume: a surveyed origin, a polygon in metres and an altitude band."""
    fields(envelope, 'revision origin polygon_m min_alt_m max_alt_m max_speed_mps '
                     'max_mission_seconds return_margin_seconds')
    origin = envelope['origin']
    fields(origin, 'latitude_deg longitude_deg')
    require(number(origin['latitude_deg']) and -85.0 <= origin['latitude_deg'] <= 85.0,
            'Envelope origin latitude out of range')
    require(number(origin['longitude_deg']) and -180.0 <= origin['longitude_deg'] <= 180.0,
            'Envelope origin longitude out of range')
    polygon = envelope['polygon_m']
    require(isinstance(polygon, list) and len(polygon) >= 3, 'Envelope polygon needs three corners')
    for point in polygon:
        require(isinstance(point, list) and len(point) == 2 and all(number(v) for v in point)
                and all(abs(v) <= MAX_LOCAL_OFFSET_M for v in point), 'Invalid envelope corner')
    require(number(envelope['min_alt_m']) and number(envelope['max_alt_m'])
            and 0.0 <= envelope['min_alt_m'] < envelope['max_alt_m'] <= 150.0,
            'Envelope altitude band must sit inside the 150 m limit')
    require(number(envelope['max_speed_mps']) and 0.5 <= envelope['max_speed_mps'] <= 15.0,
            'Invalid envelope speed limit')
    require(type(envelope['max_mission_seconds']) is int and 0 < envelope['max_mission_seconds'] <= 1800,
            'Invalid mission time limit')
    require(type(envelope['return_margin_seconds']) is int and envelope['return_margin_seconds'] > 0,
            'Return margin required')
    return envelope


def inside(polygon, point):
    """Ray casting; a point on the boundary counts as outside, which is the safe side."""
    x, y = point
    result = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if math.isclose(crossing, x, abs_tol=1e-9):
                return False
            if x < crossing:
                result = not result
        elif math.isclose(y1, y) and math.isclose(y2, y) and min(x1, x2) <= x <= max(x1, x2):
            return False
    return result


def segment_samples(start, end, step=2.0):
    """Sample the straight leg so a path cannot leave and re-enter between corners."""
    distance = math.dist(start, end)
    count = max(1, math.ceil(distance / step))
    return [tuple(start[axis] + (end[axis] - start[axis]) * k / count for axis in range(3))
            for k in range(count + 1)]


def check_envelope(viewpoints, envelope, *, start_position_m):
    """RFL stage 2. Everything the aircraft would fly must sit inside the approved volume."""
    validate_envelope(envelope)
    polygon = [tuple(point) for point in envelope['polygon_m']]
    require(isinstance(start_position_m, (list, tuple)) and len(start_position_m) == 3
            and all(number(v) for v in start_position_m), 'Invalid current position')
    legs, seconds, position = [], 0.0, tuple(start_position_m)
    for row in viewpoints['viewpoints']:
        target = tuple(row['position_m'])
        speed = min(row['speed_mps'], envelope['max_speed_mps'])
        for sample in segment_samples(position, target):
            require(inside(polygon, (sample[0], sample[1])), 'Path leaves the approved polygon')
            require(envelope['min_alt_m'] <= sample[2] <= envelope['max_alt_m'],
                    'Path leaves the approved altitude band')
        distance = math.dist(position, target)
        seconds += distance / speed + row['dwell_ms'] / 1000.0
        legs.append({'sequence': row['sequence'], 'distance_m': distance, 'speed_mps': speed})
        position = target
    # The way home has to fit inside the same budget as the way out.
    home = math.dist(position, tuple(start_position_m)) / envelope['max_speed_mps']
    require(seconds + home + envelope['return_margin_seconds'] <= envelope['max_mission_seconds'],
            'Mission plus return margin exceeds the approved time')
    return {'legs': legs, 'flight_seconds': seconds, 'return_seconds': home,
            'return_margin_seconds': envelope['return_margin_seconds'],
            'envelope_revision': envelope['revision'], 'inside_envelope': True}


def local_to_geodetic(origin, east_m, north_m):
    """Local tangent plane, only valid inside MAX_LOCAL_OFFSET_M of the origin."""
    require(abs(east_m) <= MAX_LOCAL_OFFSET_M and abs(north_m) <= MAX_LOCAL_OFFSET_M,
            'Local offset exceeds the tangent plane limit')
    latitude = math.radians(origin['latitude_deg'])
    e2 = WGS84_F * (2 - WGS84_F)
    denominator = math.sqrt(1 - e2 * math.sin(latitude) ** 2)
    meridian = WGS84_A * (1 - e2) / denominator ** 3
    normal = WGS84_A / denominator
    return (origin['latitude_deg'] + math.degrees(north_m / meridian),
            origin['longitude_deg'] + math.degrees(east_m / (normal * math.cos(latitude))))


def to_mission(viewpoints, envelope, *, start_position_m, wrap_flight=True):
    """Convert a checked viewpoint sequence into mission items.

    `wrap_flight` adds a takeoff and a return to launch, and a flight needs both:
    without the takeoff, SITL armed and sat at 2 m. The six-item form was once
    refused, but that was a gap in the sequence numbers rather than an unsupported
    command, and PX4 1.15.4 accepted all five shapes tried on 2026-09-08 and flew
    this one to the commanded 60 m.
    """
    report = check_envelope(viewpoints, envelope, start_position_m=start_position_m)
    origin = envelope['origin']
    items, sequence = [], 0

    def add(command, params, *, geodetic=None, frame=None):
        """Parameters 5 to 7 travel as x, y and z in a MISSION_ITEM_INT.

        For a navigation command those three are latitude, longitude and
        altitude; for a DO command they carry the remaining parameters, which
        would otherwise never reach the autopilot.
        """
        nonlocal sequence
        require(len(params) == 7, 'A mission item carries exactly seven parameters')
        if geodetic is None:
            x, y, z = int(params[4]), int(params[5]), float(params[6])
        else:
            latitude, longitude, altitude = geodetic
            params = list(params[:4]) + [latitude, longitude, altitude]
            x, y, z = round(latitude * 1e7), round(longitude * 1e7), float(altitude)
        items.append({'seq': sequence, 'command': command,
                      'frame': frame or MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 'autocontinue': 1,
                      'params': [float(value) for value in params], 'x': x, 'y': y, 'z': z})
        sequence += 1

    if wrap_flight:
        first = viewpoints['viewpoints'][0]
        start_latitude, start_longitude = local_to_geodetic(origin, start_position_m[0],
                                                            start_position_m[1])
        add(MAV_CMD_NAV_TAKEOFF, [0, 0, 0, 0, 0, 0, 0],
            geodetic=(start_latitude, start_longitude, first['position_m'][2]))
    legs = []
    for row in viewpoints['viewpoints']:
        east, north, up = row['position_m']
        target_east, target_north, target_up = row['target_position_m']
        latitude, longitude = local_to_geodetic(origin, east, north)
        target_latitude, target_longitude = local_to_geodetic(origin, target_east, target_north)
        # PX4 stores the waypoint yaw in [0, 360), so emit it that way and the
        # round trip stays exact. Sending -90 came back as 270 in SITL.
        bearing = math.degrees(math.atan2(target_east - east, target_north - north)) % 360.0
        depression = math.degrees(math.atan2(up - target_up,
                                             math.dist((east, north), (target_east, target_north))
                                             or 1e-9))
        # param4 is the vehicle yaw: face the target so the gimbal has the least to do.
        add(MAV_CMD_NAV_WAYPOINT, [row['dwell_ms'] / 1000.0, 0, 0, bearing, 0, 0, 0],
            geodetic=(latitude, longitude, up))
        # PX4 refuses gimbal commands in a mission but accepts a region of interest.
        add(MAV_CMD_DO_SET_ROI_LOCATION, [0, 0, 0, 0, 0, 0, 0],
            geodetic=(target_latitude, target_longitude, target_up))
        legs.append({'sequence': row['sequence'], 'vehicle_yaw_deg': round(bearing, 3),
                     'camera_depression_deg': round(depression, 3),
                     'speed_mps': min(row['speed_mps'], envelope['max_speed_mps'])})
    if wrap_flight:
        add(MAV_CMD_NAV_RETURN_TO_LAUNCH, [0, 0, 0, 0, 0, 0, 0], frame=MAV_FRAME_MISSION)
    return {'schema_version': 'tancho-px4-mission-1.0',
            'observation_id': viewpoints['observation_id'],
            'goal_revision': viewpoints['goal_revision'],
            'envelope_revision': viewpoints['envelope_revision'],
            'origin': dict(origin), 'items': items, 'envelope_check': report,
            'model_emitted_geodetic_position': False,
            'mavlink_definition_sha256': MAVLINK_DEFINITION_SHA256,
            'mavlink_ids_verified_against_upstream_xml': True,
            'mission_shape_accepted_by_px4_sitl': True,
            'commands_refused_by_px4_in_a_mission': dict(REFUSED_IN_MISSION),
            # Speed is not a mission item here; it belongs to the flight plan.
            'flight_plan_parameters': {'legs': legs},
            'camera_aimed_by_region_of_interest': True,
            'takeoff_and_return_included': wrap_flight,
            # Legs alone upload cleanly and leave the vehicle on the ground.
            'flyable_as_uploaded': wrap_flight,
            # PX4 1.15.4 SITL accepted this shape, armed and flew it to the
            # commanded altitude on 2026-09-08. No real aircraft has flown it.
            'mission_execution_verified': 'px4-sitl-1.15.4' if wrap_flight else False,
            'vehicle_yaw_faces_the_target': True,
            'flown_by_a_real_aircraft': False}
