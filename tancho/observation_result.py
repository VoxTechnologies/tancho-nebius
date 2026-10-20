"""Validation of captures actually made from a compiled observation plan."""
import hashlib
import math

from .contracts import fields, number, require, strings
from .mission_context import canonical_sha256, validate_context_v1

SCHEMA = 'tancho-observation-result-1.0'
VERIFIED_SCHEMA = 'tancho-verified-observation-result-1.0'
ASSESSMENT_SOURCES = frozenset({'detector', 'area_classifier', 'quality_metric', 'human_review'})


def verify_observation_result(result, compiled, context, images, *, now_ms,
                              position_tolerance_m=.75, aim_tolerance_m=.75):
    validate_context_v1(context, now_ms=now_ms)
    require(number(position_tolerance_m) and position_tolerance_m > 0, 'Invalid position tolerance')
    require(number(aim_tolerance_m) and aim_tolerance_m > 0, 'Invalid aim tolerance')
    fields(result, 'schema_version context_id mission_id compiled_revision observations')
    require(result['schema_version'] == SCHEMA, 'Unknown observation result version')
    require(result['context_id'] == context['context_id']
            and result['mission_id'] == context['mission_id'], 'Result is bound elsewhere')
    require(result['compiled_revision'] == compiled['compiled_revision'], 'Compiled plan changed')
    require(isinstance(images, dict) and images, 'Captured image bytes are required')
    rows = result['observations']
    require(isinstance(rows, list) and len(rows) == len(compiled['viewpoints']),
            'Result must cover every compiled viewpoint')
    image_ids, verified = set(), []
    for expected, row in zip(compiled['viewpoints'], rows):
        fields(row, 'sequence intent_id captured_at_ms position_m target_position_m sensor '
                    'image_ids stop_condition assessment')
        require(row['sequence'] == expected['sequence'], 'Result order changed')
        require(row['intent_id'] == expected['intent_id'], 'Result intent changed')
        require(type(row['captured_at_ms']) is int
                and context['captured_at_ms'] <= row['captured_at_ms'] <= now_ms,
                'Invalid result capture time')
        for key, tolerance in (('position_m', position_tolerance_m),
                               ('target_position_m', aim_tolerance_m)):
            point = row[key]
            require(isinstance(point, list) and len(point) == 3 and all(number(v) for v in point),
                    'Invalid measured result pose')
            require(math.dist(point, expected[key]) <= tolerance, 'Measured result pose missed the plan')
        require(row['sensor'] == expected['sensor'], 'Result sensor changed')
        require(row['stop_condition'] == expected['stop_condition'], 'Result stop condition changed')
        require(strings(row['image_ids']) and bool(row['image_ids']), 'Result needs captured images')
        require(not image_ids.intersection(row['image_ids']), 'Captured image reused across viewpoints')
        image_ids.update(row['image_ids'])
        assessment = row['assessment']
        fields(assessment, 'source passed evidence_ids')
        require(assessment['source'] in ASSESSMENT_SOURCES, 'Untrusted result assessment source')
        require(type(assessment['passed']) is bool, 'Assessment passed must be boolean')
        require(strings(assessment['evidence_ids']) and bool(assessment['evidence_ids'])
                and set(assessment['evidence_ids']) <= set(row['image_ids']),
                'Assessment evidence is not a captured image')
        verified.append(dict(row))
    require(set(images) == image_ids, 'Captured image registry differs from result')
    hashes = {}
    for image_id, value in sorted(images.items()):
        require(isinstance(value, bytes) and value, 'Invalid captured image bytes')
        hashes[image_id] = hashlib.sha256(value).hexdigest()
    completed = all(row['assessment']['passed'] for row in rows)
    body = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
            'compiled_revision': compiled['compiled_revision'], 'image_sha256': hashes,
            'observations': verified, 'completed': completed}
    return {'schema_version': VERIFIED_SCHEMA, **body,
            'result_revision': canonical_sha256(body)}
