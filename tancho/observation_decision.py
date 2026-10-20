"""Offline candidate decisions; this module never authorizes flight."""
from copy import deepcopy
import hashlib
import json
import math

from .contracts import fields, number, require
from .mission_context import canonical_sha256, validate_context_v1
from .observation_intent import validate_and_bind_intents
from .observation_program import SCHEMA, validate_observation_program

CATALOG = 'tancho-observation-candidates-0.1'
ROAD_CATALOG = 'tancho-observation-candidates-0.2-road-preview'
ROAD_SEQUENCE_CATALOG = 'tancho-observation-candidates-0.3-road-sequence-preview'
ROAD_HAZARD_CATALOG = 'tancho-observation-candidates-0.4-road-hazard-preview'
ROAD_SUPPORT_CATALOG = 'tancho-observation-candidates-0.5-road-support-preview'
REQUEST = 'tancho-observation-decision-request-1.1'
RECEIPT = 'tancho-observation-decision-receipt-1.0'
LABELS = 'ABC'
CATALOG_ROWS = {
    ('slope_failure_v1', 'landslide_extent'): (
        ('overview', 'full_target', 'full_boundary_visible'),
        ('oblique', 'boundary', 'full_boundary_visible')),
    ('access_clearance_v1', 'corridor_continuity'): (
        ('along_axis', 'both_sides', 'corridor_continuity_visible'),
        ('context', 'target_and_context', 'corridor_continuity_visible')),
}
ROAD_CATALOG_ROWS = {
    **CATALOG_ROWS,
    ('access_clearance_v1', 'roadbed_loss'): (
        ('oblique', 'boundary', 'roadbed_edges_visible'),
        ('along_axis', 'both_sides', 'roadbed_edges_visible')),
    ('access_clearance_v1', 'bridge_damage'): (
        ('oblique', 'connection', 'bridge_damage_context_visible'),
        ('context', 'target_and_context', 'bridge_damage_context_visible')),
}
ROAD_SEQUENCE_CATALOG_ROWS = {
    **ROAD_CATALOG_ROWS,
    ('access_clearance_v1', 'roadbed_loss'): (
        (('cross_section', 'boundary', 'roadbed_edges_visible'),
         ('along_axis', 'both_sides', 'roadbed_edges_visible')),
        (('cross_section', 'boundary', 'roadbed_edges_visible'),
         ('context', 'target_and_context', 'roadbed_edges_visible'))),
}
ROAD_HAZARD_CATALOG_ROWS = {
    **ROAD_SEQUENCE_CATALOG_ROWS,
    ('access_clearance_v1', 'obstruction_material'): (
        ('oblique', 'target_and_context', 'obstruction_type_and_extent_visible'),
        ('nadir', 'full_target', 'obstruction_type_and_extent_visible')),
    ('access_clearance_v1', 'obstruction_extent'): (
        ('overview', 'full_target', 'obstruction_type_and_extent_visible'),
        ('along_axis', 'both_sides', 'obstruction_type_and_extent_visible')),
    ('flood_river_v1', 'road_flooding'): (
        ('nadir', 'boundary', 'road_water_boundary_visible'),
        ('oblique', 'both_sides', 'road_water_boundary_visible')),
}
ROAD_SUPPORT_CATALOG_ROWS = {
    **ROAD_HAZARD_CATALOG_ROWS,
    ('access_clearance_v1', 'road_edge_support_loss'): (
        (('oblique', 'boundary', 'road_edge_support_boundary_visible'),
         ('context', 'target_and_context', 'road_edge_support_boundary_visible')),
        (('cross_section', 'boundary', 'road_edge_support_boundary_visible'),
         ('context', 'target_and_context', 'road_edge_support_boundary_visible'))),
}
CATALOG_REGISTRY = {
    CATALOG: CATALOG_ROWS,
    ROAD_CATALOG: ROAD_CATALOG_ROWS,
    ROAD_SEQUENCE_CATALOG: ROAD_SEQUENCE_CATALOG_ROWS,
    ROAD_HAZARD_CATALOG: ROAD_HAZARD_CATALOG_ROWS,
    ROAD_SUPPORT_CATALOG: ROAD_SUPPORT_CATALOG_ROWS,
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def validate_intents(intents, context, now_ms):
    fields(intents, 'schema_version context_id mission_id outcome evidence_ids intents '
                    'intent_revision source_output_sha256')
    require(intents['schema_version'] == 'tancho-verified-intents-1.0', 'Unverified intents')
    require(intents['outcome'] == 'inspect', 'Decision requires inspect')
    require(isinstance(intents['intents'], list) and len(intents['intents']) == 1,
            'D0 requires one intent')
    row = intents['intents'][0]
    fields(row, 'sequence target_id question priority evidence_ids intent_id')
    require(type(row['sequence']) is int and row['sequence'] == 0, 'Invalid intent sequence')
    raw = {key: intents[key] for key in ('context_id', 'mission_id', 'outcome', 'evidence_ids')}
    raw.update(schema_version='tancho-observation-intent-1.0',
               reason_code='unanswered_question',
               intents=[{key: value for key, value in row.items() if key != 'intent_id'}])
    rebuilt = validate_and_bind_intents(raw, context, now_ms=now_ms)
    require(all(rebuilt[key] == intents[key] for key in rebuilt
                if key != 'source_output_sha256'), 'Intent binding changed')
    source = intents['source_output_sha256']
    require(isinstance(source, str) and len(source) == 64
            and all(c in '0123456789abcdef' for c in source), 'Invalid intent source hash')


def candidates(context, intents, *, now_ms, order=(0, 1, 2), catalog_revision=CATALOG):
    validate_context_v1(context, now_ms=now_ms)
    validate_intents(intents, context, now_ms)
    require(isinstance(order, (tuple, list)) and len(order) == 3
            and all(type(i) is int for i in order) and sorted(order) == [0, 1, 2],
            'Candidate order must be a permutation')
    require(catalog_revision in CATALOG_REGISTRY, 'Unknown decision catalog')
    key = (context['mission_profile'], intents['intents'][0]['question'])
    rows = CATALOG_REGISTRY[catalog_revision]
    require(key in rows, 'Unsupported D0 question')
    result = []
    for specification in rows[key]:
        steps = ((specification,) if isinstance(specification[0], str)
                 else specification)
        program = {'schema_version': SCHEMA, 'outcome': 'observe',
                   'reason_code': 'program_created', 'observations': [
                       {'sequence': index, 'intent_sequence': 0,
                        'view_relation': relation, 'coverage': coverage,
                        'detail_level': 'damage', 'sensor': 'rgb',
                        'stop_condition': stop}
                       for index, (relation, coverage, stop) in enumerate(steps)]}
        validate_observation_program(program, context, intents, now_ms=now_ms)
        result.append({'candidate_id': canonical_sha256(program), 'program': program})
    result.append({'candidate_id': 'reject', 'program': None})
    return [deepcopy(result[i]) for i in order]


def build_request(context, intents, frames, *, now_ms, order=(0, 1, 2),
                  catalog_revision=CATALOG):
    validate_context_v1(context, now_ms=now_ms, evidence=frames)
    options = candidates(context, intents, now_ms=now_ms, order=order,
                         catalog_revision=catalog_revision)
    intent = intents['intents'][0]
    target = next(row for row in context['targets'] if row['target_id'] == intent['target_id'])
    visible = {'profile': context['mission_profile'], 'category': target['category'],
               'question': intent['question'], 'priority': intent['priority'],
               'evidence_ids': context['evidence_ids'],
               'options': [{'label': label, 'observation_program': row['program']}
                           for label, row in zip(LABELS, options, strict=True)]}
    prompt = ('Inspect the supplied images. Choose the minimum sufficient observation '
              'program for the verified question. A null program means that none of the '
              'listed programs is appropriate or the evidence is insufficient. '
              'Do not declare mission completion. Respond with only the option letter.\n'
              + _json(visible))
    body = {'schema_version': REQUEST, 'catalog_revision': catalog_revision,
            'context_sha256': canonical_sha256(context),
            'intents_sha256': canonical_sha256(intents),
            'frame_sha256': {key: hashlib.sha256(value).hexdigest()
                             for key, value in sorted(frames.items())},
            'evidence_order': list(context['evidence_ids']),
            'model_revision': context['model_revision'], 'order': list(order),
            'candidates': options, 'labels': list(LABELS), 'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    return {**body, 'request_sha256': canonical_sha256(body)}


def _check_request(request):
    fields(request, 'schema_version catalog_revision context_sha256 intents_sha256 frame_sha256 '
                    'model_revision evidence_order order candidates labels prompt prompt_sha256 request_sha256')
    require(request['schema_version'] == REQUEST
            and request['catalog_revision'] in CATALOG_REGISTRY,
            'Unknown decision request')
    require(request['labels'] == list(LABELS) and len(request['candidates']) == 3,
            'Invalid option registry')
    require(request['request_sha256'] == canonical_sha256(
        {k: v for k, v in request.items() if k != 'request_sha256'}), 'Request changed')


def label_token_ids(tokenizer):
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    require(all(len(row) == 1 and type(row[0]) is int and row[0] >= 0 for row in ids),
            'Each label must be one token')
    result = [row[0] for row in ids]
    require(len(set(result)) == len(result), 'Label tokens must be distinct')
    return result


def make_receipt(request, logits, token_ids, *, started_ms, completed_ms, timeout_ms):
    _check_request(request)
    require(isinstance(logits, list) and len(logits) == 3
            and all(number(v) and math.isfinite(v) for v in logits), 'Invalid logits')
    require(isinstance(token_ids, list) and len(token_ids) == 3
            and all(type(v) is int and v >= 0 for v in token_ids)
            and len(set(token_ids)) == 3, 'Invalid label token IDs')
    require(all(type(v) is int for v in (started_ms, completed_ms, timeout_ms))
            and 0 <= started_ms <= completed_ms and 0 < timeout_ms <= 120000
            and completed_ms - started_ms < timeout_ms, 'Decision deadline exceeded')
    maximum = max(logits)
    weights = [math.exp(v - maximum) for v in logits]
    probabilities = [v / sum(weights) for v in weights]
    tied = logits.count(maximum) != 1
    index = None if tied else logits.index(maximum)
    selected = None if tied else request['candidates'][index]['candidate_id']
    outcome = 'rejected' if tied or selected == 'reject' else 'selected'
    body = {'schema_version': RECEIPT, 'request_sha256': request['request_sha256'],
            'logits': list(logits), 'label_token_ids': list(token_ids),
            'probabilities': probabilities, 'temperature': 1.0,
            'calibration_revision': 'uncalibrated', 'selected_index': index,
            'selected_candidate_id': selected, 'outcome': outcome,
            'reason': 'tie' if tied else ('no_candidate' if outcome == 'rejected' else 'argmax'),
            'started_ms': started_ms, 'completed_ms': completed_ms, 'timeout_ms': timeout_ms}
    return {**body, 'receipt_sha256': canonical_sha256(body)}


def replay_decision(request, receipt, context, intents, frames, *, now_ms):
    rebuilt = build_request(context, intents, frames, now_ms=now_ms,
                            order=request['order'],
                            catalog_revision=request['catalog_revision'])
    require(request == rebuilt, 'Decision request binding changed')
    expected = make_receipt(request, receipt['logits'], receipt['label_token_ids'],
                            started_ms=receipt['started_ms'], completed_ms=receipt['completed_ms'],
                            timeout_ms=receipt['timeout_ms'])
    require(receipt == expected, 'Decision receipt changed')
    require(context['captured_at_ms'] <= receipt['started_ms']
            <= receipt['completed_ms'] <= now_ms < context['expires_at_ms'],
            'Decision outside context lifetime')
    if receipt['outcome'] == 'rejected':
        return {'program': None, 'conversion': None}
    program = deepcopy(request['candidates'][receipt['selected_index']]['program'])
    validate_observation_program(program, context, intents, now_ms=now_ms)
    conversion = {'decision_source': 'cosmos_candidate_readout', 'model_generated_program': False,
                  'receipt_sha256': receipt['receipt_sha256'],
                  'candidate_id': receipt['selected_candidate_id'],
                  'application_program_sha256': canonical_sha256(program),
                  'flight_authorized': False}
    return {'program': program, 'conversion': conversion}


def candidate_coverage(request, gold):
    """Exact coverage only; a missing gold is never silently relabelled as reject."""
    return any(row['program'] == gold for row in request['candidates'] if row['program'] is not None)
