"""Exact crosswalk and proof validator for reviewed legacy Reasoner labels."""
from copy import deepcopy

from tancho.contracts import fields, require


SCHEMA = 'tancho-observation-program-legacy-reasoner-migration-proof-1.0'
RULE_REVISION = 'tancho-release1-goal-to-question-crosswalk-1.0'
LEGACY_BINDINGS = {
    'inspect': ('inspect', 'unanswered_question'),
    'no_action': ('satisfied', 'all_required_questions_completed'),
    'abstain': ('insufficient_evidence', 'evidence_unusable'),
}
MAPPINGS = (
    (('道路',), {
        'mission_profile': 'access_clearance_v1', 'target_category': 'road_segment',
        'target_source': 'map_difference', 'map_relations': ['road_missing'],
        'question': 'corridor_continuity',
        'observation': {'view_relation': 'along_axis', 'coverage': 'both_sides',
                        'detail_level': 'damage', 'sensor': 'rgb',
                        'stop_condition': 'corridor_continuity_visible'},
    }),
    (('流路', '水面'), {
        'mission_profile': 'slope_failure_v1', 'target_category': 'landslide_area',
        'target_source': 'map_difference', 'map_relations': ['intersects_river'],
        'question': 'landslide_river_connection',
        'observation': {'view_relation': 'oblique', 'coverage': 'connection',
                        'detail_level': 'damage', 'sensor': 'rgb',
                        'stop_condition': 'connection_visible'},
    }),
    (('森林の縁', '稜線', '樹木'), {
        'mission_profile': 'slope_failure_v1', 'target_category': 'landslide_area',
        'target_source': 'area_classifier', 'map_relations': [],
        'question': 'slope_precursor_detail',
        'observation': {'view_relation': 'oblique', 'coverage': 'target_and_context',
                        'detail_level': 'damage', 'sensor': 'rgb',
                        'stop_condition': 'precursor_detail_visible'},
    }),
    (('土砂の帯', '崩壊面', '地表の露出'), {
        'mission_profile': 'slope_failure_v1', 'target_category': 'landslide_area',
        'target_source': 'area_classifier', 'map_relations': [],
        'question': 'landslide_extent',
        'observation': {'view_relation': 'overview', 'coverage': 'full_target',
                        'detail_level': 'damage', 'sensor': 'rgb',
                        'stop_condition': 'full_boundary_visible'},
    }),
)


def mapping_for_item(item):
    if item['source_leakage_group_id'] != 'kumamoto-2016-sannoudani':
        return None
    for words, mapping in MAPPINGS:
        if any(word in item['observation_goal'] for word in words):
            return deepcopy(mapping)
    return None


def exact_mapping(goal, leakage_group_id):
    value = mapping_for_item({'observation_goal': goal,
                              'source_leakage_group_id': leakage_group_id})
    require(value is not None, 'Legacy goal is outside the exact Release 1 crosswalk')
    return {key: value[key] for key in ('mission_profile', 'target_category',
                                        'target_source', 'map_relations', 'question')}


def expected_target(proof):
    legacy = proof['legacy_label']
    mapping = exact_mapping(legacy['observation_goal'], legacy['leakage_group_id'])
    require(mapping == proof['mapping'], 'Legacy migration mapping differs from crosswalk')
    outcome, reason_code = LEGACY_BINDINGS[legacy['decision']]
    intents = []
    if outcome == 'inspect':
        intents = [{'sequence': 0, 'target_id': 'target-1',
                    'question': mapping['question'], 'priority': 1,
                    'evidence_ids': proof['evidence_ids']}]
    return {'schema_version': 'tancho-observation-intent-1.0',
            'context_id': proof['context_id'], 'mission_id': proof['mission_id'],
            'outcome': outcome, 'evidence_ids': proof['evidence_ids'],
            'intents': intents, 'reason_code': reason_code}


def validate_legacy_proof(proof):
    fields(proof, 'schema_version rule_revision source_manifest_sha256 '
                  'source_manifest_schema labels_confirmed_by_a_person legacy_label_scope '
                  'legacy_label mapping evidence_ids source_media_sha256 split profile role '
                  'context_id mission_id expected_target')
    require(proof['schema_version'] == SCHEMA and proof['rule_revision'] == RULE_REVISION,
            'Unknown legacy migration rule')
    require(proof['source_manifest_schema'] ==
            'tancho-observation-program-release1-real-train-source-1.0'
            and proof['labels_confirmed_by_a_person'] is True
            and proof['legacy_label_scope'] ==
            'decision and prose only; Observation Program labels unreviewed',
            'Legacy source does not assert the reviewed label scope')
    require(proof['split'] == 'train' and proof['role'] == 'reasoner'
            and proof['profile'] == proof['mapping']['mission_profile'],
            'Legacy migration can authorize only real training Reasoner labels')
    require(proof['legacy_label']['decision'] in LEGACY_BINDINGS,
            'Unknown reviewed legacy decision')
    require(proof['expected_target'] == expected_target(proof),
            'Legacy migration target differs from the deterministic crosswalk')
    return proof
