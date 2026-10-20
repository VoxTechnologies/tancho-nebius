"""Versioned mission vocabularies for the Tancho observation-program contracts.

Profiles constrain references and semantics.  They deliberately do not prescribe a
single observation program for a question; doing so would reduce the Generator task
to a non-visual lookup table.
"""
from dataclasses import dataclass

from .contracts import ContractError, require


@dataclass(frozen=True)
class MissionProfile:
    name: str
    category_questions: dict
    category_sources: dict
    map_relations: frozenset
    view_relations: frozenset
    coverages: frozenset
    detail_levels: frozenset
    sensors: frozenset
    question_stop_conditions: dict

    @property
    def categories(self):
        return frozenset(self.category_questions)

    @property
    def questions(self):
        return frozenset(q for values in self.category_questions.values() for q in values)

    @property
    def stop_conditions(self):
        return frozenset(s for values in self.question_stop_conditions.values() for s in values)


SLOPE_FAILURE = MissionProfile(
    name='slope_failure_v1',
    category_questions={
        'landslide_area': frozenset({
            'landslide_extent', 'landslide_road_connection', 'landslide_river_connection',
            'landslide_occluded_region', 'slope_precursor_detail', 'surface_exposure',
        }),
    },
    category_sources={'landslide_area': frozenset({'area_classifier', 'map_difference'})},
    map_relations=frozenset({'intersects_road', 'intersects_river', 'near_building'}),
    view_relations=frozenset({'overview', 'nadir', 'oblique', 'cross_section', 'context'}),
    coverages=frozenset({'full_target', 'boundary', 'upper_lower_edges', 'connection',
                         'target_and_context'}),
    detail_levels=frozenset({'identify', 'state', 'damage'}),
    sensors=frozenset({'rgb'}),
    question_stop_conditions={
        'landslide_extent': frozenset({'full_boundary_visible', 'upper_lower_edges_visible'}),
        'landslide_road_connection': frozenset({'connection_visible'}),
        'landslide_river_connection': frozenset({'connection_visible'}),
        'landslide_occluded_region': frozenset({'occluded_region_resolved'}),
        'slope_precursor_detail': frozenset({'precursor_detail_visible'}),
        'surface_exposure': frozenset({'surface_exposure_visible'}),
    },
)

ACCESS_CLEARANCE = MissionProfile(
    name='access_clearance_v1',
    category_questions={
        'road_segment': frozenset({'corridor_continuity', 'roadbed_loss',
                                   'road_edge_support_loss', 'approach_access'}),
        'rail_segment': frozenset({'corridor_continuity', 'approach_access'}),
        'obstruction_area': frozenset({'obstruction_material', 'obstruction_extent'}),
        'bridge': frozenset({'bridge_damage', 'approach_access'}),
    },
    category_sources={
        'road_segment': frozenset({'map', 'map_difference'}),
        'rail_segment': frozenset({'map', 'map_difference'}),
        'obstruction_area': frozenset({'area_classifier', 'map_difference'}),
        'bridge': frozenset({'map', 'map_difference'}),
    },
    map_relations=frozenset({'on_road', 'on_rail', 'road_missing', 'bridge_segment',
                             'intersects_obstruction'}),
    view_relations=frozenset({'overview', 'nadir', 'oblique', 'cross_section', 'along_axis',
                              'context'}),
    coverages=frozenset({'full_target', 'boundary', 'both_sides', 'connection',
                         'target_and_context', 'approach_corridor'}),
    detail_levels=frozenset({'identify', 'state', 'damage'}),
    sensors=frozenset({'rgb'}),
    question_stop_conditions={
        'corridor_continuity': frozenset({'corridor_continuity_visible'}),
        'obstruction_material': frozenset({'obstruction_type_and_extent_visible'}),
        'obstruction_extent': frozenset({'obstruction_type_and_extent_visible'}),
        'roadbed_loss': frozenset({'roadbed_edges_visible'}),
        'road_edge_support_loss': frozenset({'road_edge_support_boundary_visible'}),
        'bridge_damage': frozenset({'bridge_damage_context_visible'}),
        'approach_access': frozenset({'approach_corridor_visible'}),
    },
)

FLOOD_RIVER = MissionProfile(
    name='flood_river_v1',
    category_questions={
        'flood_area': frozenset({'flood_extent', 'flood_boundary', 'hidden_ground'}),
        'road_segment': frozenset({'road_flooding'}),
        'river_segment': frozenset({'channel_blockage', 'debris_accumulation'}),
        'embankment': frozenset({'embankment_damage'}),
        'building': frozenset({'building_waterline'}),
        'debris_area': frozenset({'debris_accumulation', 'channel_blockage'}),
    },
    category_sources={
        'flood_area': frozenset({'area_classifier', 'map_difference'}),
        'road_segment': frozenset({'map', 'map_difference'}),
        'river_segment': frozenset({'map', 'map_difference'}),
        'embankment': frozenset({'map', 'map_difference'}),
        'building': frozenset({'object_detector', 'map', 'map_difference'}),
        'debris_area': frozenset({'area_classifier', 'map_difference'}),
    },
    map_relations=frozenset({'intersects_road', 'intersects_river', 'intersects_building',
                             'on_embankment', 'upstream', 'downstream'}),
    view_relations=frozenset({'overview', 'nadir', 'oblique', 'cross_section', 'along_axis',
                              'context'}),
    coverages=frozenset({'full_target', 'boundary', 'connection', 'upstream_downstream',
                         'target_and_context', 'facade_foot', 'both_sides'}),
    detail_levels=frozenset({'identify', 'state', 'damage'}),
    sensors=frozenset({'rgb'}),
    question_stop_conditions={
        'flood_extent': frozenset({'flood_boundary_visible'}),
        'flood_boundary': frozenset({'flood_boundary_visible'}),
        'hidden_ground': frozenset({'flood_occlusion_resolved'}),
        'road_flooding': frozenset({'road_water_boundary_visible'}),
        'building_waterline': frozenset({'facade_waterline_visible'}),
        'embankment_damage': frozenset({'bank_damage_context_visible'}),
        'channel_blockage': frozenset({'channel_cross_section_visible'}),
        'debris_accumulation': frozenset({'upstream_downstream_context_visible'}),
    },
)


STRUCTURE_DAMAGE = MissionProfile(
    name='structure_damage_v1',
    category_questions={
        'masonry_wall': frozenset({'damage_boundary', 'building_connection',
                                   'masonry_damage'}),
        'debris_area': frozenset({'debris_extent'}),
    },
    category_sources={
        'masonry_wall': frozenset({'object_detector', 'map', 'map_difference'}),
        'debris_area': frozenset({'area_classifier', 'map_difference'}),
    },
    map_relations=frozenset({'near_building', 'intersects_road'}),
    view_relations=frozenset({'overview', 'oblique', 'context'}),
    coverages=frozenset({'full_target', 'boundary', 'connection', 'target_and_context'}),
    detail_levels=frozenset({'identify', 'state', 'damage'}),
    sensors=frozenset({'rgb'}),
    question_stop_conditions={
        'damage_boundary': frozenset({'damage_boundary_visible'}),
        'building_connection': frozenset({'building_connection_visible'}),
        'masonry_damage': frozenset({'masonry_damage_visible'}),
        'debris_extent': frozenset({'debris_extent_visible'}),
    },
)

RESCUE = MissionProfile(
    name='rescue_v1',
    category_questions={
        'person': frozenset({'person_state', 'person_surrounding_hazard', 'rescue_access',
                             'notification_frame'}),
        'vehicle': frozenset({'vehicle_state', 'vehicle_occupant_evidence', 'rescue_access',
                              'notification_frame'}),
    },
    category_sources={
        'person': frozenset({'object_detector'}),
        'vehicle': frozenset({'object_detector'}),
    },
    map_relations=frozenset({'on_road', 'near_building', 'near_hazard', 'isolated'}),
    view_relations=frozenset({'overview', 'nadir', 'oblique', 'along_axis', 'context'}),
    coverages=frozenset({'full_target', 'target_and_context', 'approach_corridor'}),
    detail_levels=frozenset({'identify', 'state'}),
    sensors=frozenset({'rgb', 'thermal'}),
    question_stop_conditions={
        'person_state': frozenset({'person_state_and_context_visible'}),
        'person_surrounding_hazard': frozenset({'person_state_and_context_visible'}),
        'rescue_access': frozenset({'rescue_approach_visible'}),
        'notification_frame': frozenset({'notification_frame_sufficient'}),
        'vehicle_state': frozenset({'vehicle_state_and_context_visible'}),
        'vehicle_occupant_evidence': frozenset({'vehicle_state_and_context_visible'}),
    },
)

BUILDING_EARTHQUAKE = MissionProfile(
    name='building_earthquake_v1',
    category_questions={
        'building': frozenset({'building_collapse_extent', 'facade_damage', 'entrance_access',
                               'road_encroachment'}),
        'fire_region': frozenset({'fire_context'}),
        'person': frozenset({'entrance_access'}),
    },
    category_sources={
        'building': frozenset({'object_detector', 'map', 'map_difference'}),
        'fire_region': frozenset({'area_classifier', 'map_difference'}),
        'person': frozenset({'object_detector'}),
    },
    map_relations=frozenset({'on_road', 'near_building', 'intersects_road', 'at_entrance',
                             'inside_fire_perimeter'}),
    view_relations=frozenset({'overview', 'nadir', 'oblique', 'cross_section', 'context'}),
    coverages=frozenset({'full_target', 'boundary', 'target_and_context', 'facade_foot',
                         'approach_corridor'}),
    detail_levels=frozenset({'identify', 'state', 'damage'}),
    sensors=frozenset({'rgb', 'thermal'}),
    question_stop_conditions={
        'building_collapse_extent': frozenset({'roof_and_collapse_extent_visible'}),
        'facade_damage': frozenset({'required_facades_visible', 'facade_foot_visible'}),
        'entrance_access': frozenset({'entrance_access_visible'}),
        'road_encroachment': frozenset({'facade_foot_visible'}),
        'fire_context': frozenset({'fire_perimeter_context_visible'}),
    },
)

FLOAT_RECOVERY = MissionProfile(
    name='float_recovery_v1',
    category_questions={
        'float_track': frozenset({'intercept_region', 'detector_handoff'}),
        'river_search_region': frozenset({'cross_river_search', 'detector_handoff'}),
    },
    category_sources={
        'float_track': frozenset({'telemetry'}),
        'river_search_region': frozenset({'map', 'telemetry'}),
    },
    map_relations=frozenset({'in_river', 'downstream', 'search_corridor'}),
    view_relations=frozenset({'intercept', 'search_sweep'}),
    coverages=frozenset({'search_region'}),
    detail_levels=frozenset({'detect', 'identify'}),
    sensors=frozenset({'rgb'}),
    question_stop_conditions={
        'intercept_region': frozenset({'intercept_region_covered'}),
        'cross_river_search': frozenset({'search_region_covered'}),
        'detector_handoff': frozenset({'float_detector_handoff'}),
    },
)

PROFILES = {p.name: p for p in (SLOPE_FAILURE, ACCESS_CLEARANCE, FLOOD_RIVER,
                                 STRUCTURE_DAMAGE, RESCUE, BUILDING_EARTHQUAKE,
                                 FLOAT_RECOVERY)}


def profile(name):
    require(isinstance(name, str) and name in PROFILES, 'Unknown mission profile')
    return PROFILES[name]


def validate_profile_relation(profile_value, *, category, source=None, question=None,
                              map_relations=(), view_relation=None, coverage=None,
                              detail_level=None, sensor=None, stop_condition=None):
    """Validate supplied profile dimensions without choosing a gold observation set."""
    p = profile_value if isinstance(profile_value, MissionProfile) else profile(profile_value)
    require(category in p.categories, 'Category is not allowed by the mission profile')
    if source is not None:
        require(source in p.category_sources[category], 'Target source is not allowed for category')
    require(isinstance(map_relations, (list, tuple)), 'Map relations must be a list')
    require(len(map_relations) == len(set(map_relations)), 'Duplicate map relation')
    require(set(map_relations) <= p.map_relations, 'Map relation is not allowed by the mission profile')
    if question is not None:
        require(question in p.category_questions[category], 'Question is not allowed for target category')
    if view_relation is not None:
        require(view_relation in p.view_relations, 'View relation is not allowed by the mission profile')
    if coverage is not None:
        require(coverage in p.coverages, 'Coverage is not allowed by the mission profile')
    if detail_level is not None:
        require(detail_level in p.detail_levels, 'Detail level is not allowed by the mission profile')
    if sensor is not None:
        require(sensor in p.sensors, 'Sensor is not allowed by the mission profile')
    if stop_condition is not None:
        require(question is not None, 'Stop condition requires a question')
        require(stop_condition in p.question_stop_conditions[question],
                'Stop condition does not match the intent question')
    return p
