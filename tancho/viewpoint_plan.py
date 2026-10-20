"""The deterministic planner that the model's viewpoint sequence is measured against.

Two jobs, one geometry. It is the fallback the design already promises: when the
model's proposal fails RFL stage 2, a planner has to produce a sequence for the
same goal. And it is the baseline for the comparison, because a viewpoint
sequence is only worth a model if the model beats what a rule can compute from
the target position, the envelope and the requested side.

The camera is the one in the flight plan: SIYI A8 mini, 4K at 3840x2160 with a
81 degree horizontal field of view (User Manual v1.10, recorded 2026-09-10).
Nothing here touches latitude or longitude; positions are metres in the
envelope's local ENU frame, the same frame `px4_mission` validates.
"""
import math

from tancho.contracts import ContractError, require
from tancho.px4_mission import FRAME, MAX_VIEWPOINTS, SCHEMA, inside

HORIZONTAL_FOV_DEG = 81.0
# Margin held back from the edge of the altitude band, so that an interpolated path does
# not leave the band through rounding.
ALTITUDE_MARGIN_M = 0.5
IMAGE_WIDTH_PX = 3840
IMAGE_HEIGHT_PX = 2160
SIDES = {'north': (0.0, 1.0), 'northeast': (0.7071, 0.7071), 'east': (1.0, 0.0),
         'southeast': (0.7071, -0.7071), 'south': (0.0, -1.0), 'southwest': (-0.7071, -0.7071),
         'west': (-1.0, 0.0), 'northwest': (-0.7071, 0.7071)}


def ground_sample_distance_m(slant_range_m):
    """Metres per pixel across the frame centre at that distance."""
    width = 2.0 * slant_range_m * math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
    return width / IMAGE_WIDTH_PX


def pixels_across(size_m, slant_range_m):
    return size_m / ground_sample_distance_m(slant_range_m)


def range_for_pixels(size_m, pixels):
    """How close the camera has to be for an object of that size to span that many pixels."""
    require(size_m > 0 and pixels > 0, 'Object size and pixel demand must be positive')
    gsd = size_m / pixels
    return gsd * IMAGE_WIDTH_PX / (2.0 * math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0))


def plan(target_position_m, envelope, *, side='any', pitch_deg=-35.0, target_size_m=4.5,
         required_pixels=120.0, target_id='t1', evidence_ids=('e1',), dwell_ms=6000,
         speed_mps=5.0, observation_id='obs', goal_revision='g', envelope_revision='e',
         calibration_id='cal', count=1):
    """One viewpoint per requested side, at the distance the pixel demand implies.

    `pitch_deg` is the optical axis elevation from the horizon, negative downward,
    the same convention as the Reasoner's `view_request`. The planner keeps the
    requested angle when the envelope allows it and lowers the altitude, then
    shortens the range, when it does not.
    """
    require(isinstance(target_position_m, (list, tuple)) and len(target_position_m) == 3,
            'Target position must be a local ENU triple')
    require(-90.0 <= pitch_deg <= -5.0, 'Pitch must look downward and not straight down')
    require(0 < count <= MAX_VIEWPOINTS, 'Viewpoint count out of range')
    # Placing the viewpoint at exactly the required distance lands on the scoring boundary;
# move 5% closer to leave room.
    wanted = range_for_pixels(target_size_m, required_pixels) / 1.05
    # The side is a preference, not an instruction (`view_request.side` in contracts-v1). Try
    # the preferred side first and fall back to the others when it does not fit the approved
    # volume. Capturing from another side and scoring "side not satisfied" reads better than
# returning nothing.
    ordered = list(SIDES.values())
    if side in SIDES:
        ordered = [SIDES[side]] + [v for k, v in SIDES.items() if k != side]
    directions = ordered
    rows = []
    for unit in directions:
        if len(rows) >= count:
            break
        row = _one(target_position_m, envelope, unit, pitch_deg, wanted)
        if row is None:
            continue
        row.update(sequence=len(rows), dwell_ms=dwell_ms, speed_mps=speed_mps,
                   target_id=target_id, evidence_ids=list(evidence_ids))
        rows.append(row)
    if not rows:
        raise ContractError('No viewpoint inside the envelope satisfies the request')
    return {'schema_version': SCHEMA, 'observation_id': observation_id,
            'goal_revision': goal_revision, 'envelope_revision': envelope_revision,
            'calibration_id': calibration_id, 'coordinate_frame': FRAME, 'viewpoints': rows}


def _one(target, envelope, unit, pitch_deg, wanted_range):
    """Place one viewpoint on that side, pulled inside the envelope if it has to be."""
    polygon = [tuple(p) for p in envelope['polygon_m']]
    for scale in (1.0, 0.8, 0.6, 0.45, 0.3):
        slant = wanted_range * scale
        height = slant * math.sin(math.radians(-pitch_deg))
        ground = slant * math.cos(math.radians(-pitch_deg))
        altitude = target[2] + height
        # Never sit exactly on the edge of the altitude band. The path is interpolated to its end
        # points and checked point by point, so an edge placement leaves the band through rounding
        # (measured 2026-09-12: one viewpoint that satisfied the pixel, pitch and side requirements
        # was rejected for "path leaves the altitude band"). The pixel margin above exists for the
# same reason.
        altitude = min(max(altitude, envelope['min_alt_m'] + ALTITUDE_MARGIN_M),
                       envelope['max_alt_m'] - ALTITUDE_MARGIN_M)
        position = [target[0] + unit[0] * ground, target[1] + unit[1] * ground, altitude]
        if not inside(polygon, (position[0], position[1])):
            continue
        if position[2] <= target[2]:
            continue
        return {'position_m': [round(v, 2) for v in position],
                'target_position_m': [round(v, 2) for v in target]}
    return None
