"""Deterministic, role-separated prompts for the shared Cosmos 3 Edge checkpoint."""
import json

from .contracts import fields, require
from .mission_context import canonical_sha256
from .mission_profiles import profile

VERSION = 'tancho-observation-program-prompts-1.1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def _reasoner_profile(context):
    p = profile(context['mission_profile'])
    return {
        'name': p.name,
        'category_questions': {key: sorted(value) for key, value in p.category_questions.items()},
        'sensors': sorted(p.sensors),
    }


def _generator_profile(context):
    p = profile(context['mission_profile'])
    return {
        'name': p.name,
        'view_relations': sorted(p.view_relations), 'coverages': sorted(p.coverages),
        'detail_levels': sorted(p.detail_levels), 'sensors': sorted(p.sensors),
        'question_stop_conditions': {key: sorted(value)
                                     for key, value in p.question_stop_conditions.items()},
    }


def reasoner_prompt(context):
    """Prompt only for selecting unanswered operational questions."""
    trusted = {key: context[key] for key in (
        'schema_version', 'context_id', 'mission_id', 'mission_profile', 'evidence_ids',
        'sensor_capabilities', 'targets', 'policy')}
    text = (
        'ROLE: TANCHO_REASONER\n'
        'Inspect the supplied images and trusted context. Select only unanswered operational '
        'questions. Return exactly one JSON object with fields: schema_version, context_id, '
        'mission_id, outcome, evidence_ids, intents, reason_code. Each intent has exactly: '
        'sequence, target_id, question, priority, evidence_ids. Do not emit observation '
        'relations, coordinates, angles, distances, speed, dwell time, geodetic positions, '
        'routes, or flight commands. Do not explain or repair uncertain input.\n'
        'Allowed profile vocabulary:\n' + _json(_reasoner_profile(context)) + '\n'
        'Trusted context:\n' + _json(trusted))
    return {'version': VERSION, 'role': 'reasoner', 'text': text,
            'sha256': canonical_sha256({'role': 'reasoner', 'text': text})}


def generator_prompt(context, intents):
    """Prompt only for composing semantic observations from verified intents."""
    fields(intents, 'schema_version context_id mission_id outcome evidence_ids intents '
                    'intent_revision source_output_sha256')
    require(intents['schema_version'] == 'tancho-verified-intents-1.0',
            'Generator prompt requires verified intents')
    require(intents['context_id'] == context['context_id']
            and intents['mission_id'] == context['mission_id'], 'Intent set is bound elsewhere')
    safe_intents = [{key: row[key] for key in
                     ('sequence', 'target_id', 'question', 'priority', 'evidence_ids')}
                    for row in intents['intents']]
    trusted = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
               'mission_profile': context['mission_profile'],
               'evidence_ids': context['evidence_ids'],
               'sensor_capabilities': context['sensor_capabilities'],
               'intents': safe_intents, 'max_observations': context['policy']['max_observations']}
    text = (
        'ROLE: TANCHO_GENERATOR\n'
        'Inspect the supplied images and compose the minimum sufficient ordered observation '
        'requirements for every verified intent. Return exactly one JSON object with fields: '
        'schema_version, outcome, observations, reason_code. Each observation has exactly: '
        'sequence, intent_sequence, view_relation, '
        'coverage, detail_level, sensor, stop_condition. Do not add, remove, rename, or answer '
        'intents. Do not emit context IDs, mission IDs, revisions, opaque hashes, targets, '
        'coordinates, angles, distances, speed, dwell time, '
        'geodetic positions, routes, or flight commands. Do not explain or repair output.\n'
        'Allowed profile vocabulary:\n' + _json(_generator_profile(context)) + '\n'
        'Trusted verified input:\n' + _json(trusted))
    return {'version': VERSION, 'role': 'generator', 'text': text,
            'sha256': canonical_sha256({'role': 'generator', 'text': text})}
