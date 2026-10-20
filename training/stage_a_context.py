"""Mission context of a stage A scene, shared by the geometry run and the stage C dataset.

Kept apart from training/view_outcome_geometry.py so the stage C worker needs only the context
and intent modules, not the compiler."""
from tancho.mission_context import build_context
from tancho.observation_intent import validate_and_bind_intents

NOW = 2000
# Profile, target category and map relation per question, as the weak-road review uses them.
SETTINGS = {
    'roadbed_loss': ('access_clearance_v1', 'road_segment', 'road_missing'),
    'bridge_damage': ('access_clearance_v1', 'bridge', 'bridge_segment'),
    'obstruction_material': ('access_clearance_v1', 'obstruction_area', 'intersects_obstruction'),
    'obstruction_extent': ('access_clearance_v1', 'obstruction_area', 'intersects_obstruction'),
    'road_flooding': ('flood_river_v1', 'road_segment', 'intersects_road'),
    'road_edge_support_loss': ('access_clearance_v1', 'road_segment', 'on_road'),
}


def stage_a_context(question, index, frame_bytes, model_revision='geometry-only'):
    """Mission context and verified intents of a stage A scene, bound to the given frame bytes.

    Stage A binds a placeholder frame; the stage C dataset binds the rendered initial image."""
    profile, category, relation = SETTINGS[question]
    context = build_context(
        context_id=f'{question}-{index}', mission_id='view-outcome-stage-a', mission_profile=profile,
        captured_at_ms=1000, expires_at_ms=100000, model_revision=model_revision,
        profile_revision=profile, map_revision='procedural-map', calibration_id='procedural-camera',
        envelope_revision='env-stage-a', evidence={'frame-1': frame_bytes},
        sensor_capabilities=['rgb'],
        targets=[{'target_id': 'road-target', 'category': category, 'source': 'map_difference',
                  'geometry_ref': 'geometry-1', 'map_relations': [relation],
                  'required_questions': [question], 'completed_questions': []}],
        policy={'max_intents': 4, 'max_observations': 8, 'max_viewpoints': 8,
                'max_reobservation_cycles': 1})
    intents = validate_and_bind_intents({
        'schema_version': 'tancho-observation-intent-1.0', 'context_id': context['context_id'],
        'mission_id': context['mission_id'], 'outcome': 'inspect', 'evidence_ids': ['frame-1'],
        'intents': [{'sequence': 0, 'target_id': 'road-target', 'question': question,
                     'priority': 1, 'evidence_ids': ['frame-1']}],
        'reason_code': 'unanswered_question'}, context, now_ms=NOW)
    return context, intents
