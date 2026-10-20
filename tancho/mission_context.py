"""Trusted MissionContext construction and validation for observation programs."""
from copy import deepcopy
import hashlib
import json
from typing import Any

from .contracts import fields, require, strings, text
from .mission_profiles import profile, validate_profile_relation

MissionContext = dict[str, Any]
SCHEMA = 'tancho-mission-context-1.1'
SOURCES = frozenset({'object_detector', 'area_classifier', 'map', 'map_difference', 'telemetry'})
SENSORS = frozenset({'rgb', 'thermal'})


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hex64(value, message):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), message)


def _integer(value, low, high, message):
    require(type(value) is int and low <= value <= high, message)


def _frame_hashes(evidence):
    require(isinstance(evidence, dict) and evidence, 'Evidence bytes are required')
    require(all(text(k) and isinstance(v, bytes) and v for k, v in evidence.items()),
            'Invalid evidence bytes')
    return {key: hashlib.sha256(value).hexdigest() for key, value in sorted(evidence.items())}


def context_input_sha256(context, evidence):
    bound = deepcopy(context)
    bound.pop('input_sha256', None)
    return canonical_sha256({'context': bound, 'evidence_sha256': _frame_hashes(evidence)})


def build_context(*, context_id, mission_id, mission_profile, captured_at_ms, expires_at_ms,
                  model_revision, profile_revision, map_revision, calibration_id,
                  envelope_revision, evidence, sensor_capabilities, targets, policy):
    context = {
        'schema_version': SCHEMA, 'context_id': context_id, 'mission_id': mission_id,
        'mission_profile': mission_profile, 'captured_at_ms': captured_at_ms,
        'expires_at_ms': expires_at_ms, 'input_sha256': '0' * 64,
        'model_revision': model_revision, 'profile_revision': profile_revision,
        'map_revision': map_revision, 'calibration_id': calibration_id,
        'envelope_revision': envelope_revision, 'evidence_ids': list(evidence),
        'sensor_capabilities': list(sensor_capabilities), 'targets': deepcopy(targets),
        'policy': deepcopy(policy),
    }
    context['input_sha256'] = context_input_sha256(context, evidence)
    return validate_context_v1(context, now_ms=captured_at_ms, evidence=evidence)


def validate_context_v1(context, *, now_ms, evidence=None):
    fields(context, 'schema_version context_id mission_id mission_profile captured_at_ms '
                    'expires_at_ms input_sha256 model_revision profile_revision map_revision '
                    'calibration_id envelope_revision evidence_ids sensor_capabilities targets policy')
    require(context['schema_version'] == SCHEMA, 'Unknown mission context version')
    for key in ('context_id', 'mission_id', 'model_revision', 'profile_revision', 'map_revision',
                'calibration_id', 'envelope_revision'):
        require(text(context[key]), 'Missing context identity or revision')
    p = profile(context['mission_profile'])
    require(type(now_ms) is int and type(context['captured_at_ms']) is int
            and type(context['expires_at_ms']) is int, 'Timestamps must be integer milliseconds')
    require(0 <= context['captured_at_ms'] <= now_ms < context['expires_at_ms'],
            'Future or expired mission context')
    _hex64(context['input_sha256'], 'Invalid context input hash')
    require(strings(context['evidence_ids']) and 1 <= len(context['evidence_ids']) <= 8,
            'Invalid evidence registry')
    sensors = context['sensor_capabilities']
    require(isinstance(sensors, list) and 1 <= len(sensors) <= 8 and len(sensors) == len(set(sensors))
            and set(sensors) <= SENSORS, 'Invalid sensor capabilities')
    targets = context['targets']
    require(isinstance(targets, list) and 1 <= len(targets) <= 32, 'Invalid target registry')
    target_ids = set()
    for target in targets:
        fields(target, 'target_id category source geometry_ref map_relations required_questions '
                       'completed_questions')
        require(text(target['target_id']) and target['target_id'] not in target_ids,
                'Invalid or duplicate target ID')
        target_ids.add(target['target_id'])
        require(target['source'] in SOURCES and text(target['geometry_ref']), 'Invalid target source')
        require(isinstance(target['required_questions'], list)
                and 1 <= len(target['required_questions']) <= 32
                and len(target['required_questions']) == len(set(target['required_questions'])),
                'Invalid required question registry')
        require(isinstance(target['completed_questions'], list)
                and len(target['completed_questions']) <= 32
                and len(target['completed_questions']) == len(set(target['completed_questions'])),
                'Invalid completed question registry')
        validate_profile_relation(p, category=target['category'], source=target['source'],
                                  map_relations=target['map_relations'])
        require(set(target['required_questions']) <= p.category_questions[target['category']],
                'Required question is not allowed for target category')
        require(set(target['completed_questions']) <= set(target['required_questions']),
                'Completed question is not required for target')
    policy = context['policy']
    fields(policy, 'max_intents max_observations max_viewpoints max_reobservation_cycles')
    _integer(policy['max_intents'], 1, 4, 'Invalid intent limit')
    _integer(policy['max_observations'], 1, 8, 'Invalid observation limit')
    _integer(policy['max_viewpoints'], 1, 8, 'Invalid viewpoint limit')
    _integer(policy['max_reobservation_cycles'], 0, 8, 'Invalid reobservation limit')
    if evidence is not None:
        require(set(evidence) == set(context['evidence_ids']), 'Evidence registry differs from bytes')
        require(context['input_sha256'] == context_input_sha256(context, evidence),
                'Context or evidence bytes changed')
    return context
