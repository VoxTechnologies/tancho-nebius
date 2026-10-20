"""Candidate catalog 0.6: each question offers a close detail view and a wide context view.

Kept outside tancho/observation_decision.py because the unopened road-clearance holdout
draft binds that file's bytes. The programs only differ in practice when the scene opts into
the view contract (tancho.observation_compiler.VIEW_CONTRACT).
"""
import hashlib
import math
from copy import deepcopy

from .contracts import fields, number, require
from .mission_context import canonical_sha256, validate_context_v1
from .observation_decision import (
    LABELS,
    RECEIPT,
    REQUEST,
    _check_request,
    _json,
    validate_intents,
)
from .observation_program import SCHEMA, validate_observation_program

CATALOG_V06 = 'tancho-observation-candidates-0.6-view-contract-preview'
# (view_relation, coverage, detail_level, stop_condition); A approaches, B widens.
ROWS_V06 = {
    ('access_clearance_v1', 'roadbed_loss'): (
        ('cross_section', 'boundary', 'damage', 'roadbed_edges_visible'),
        ('context', 'target_and_context', 'state', 'roadbed_edges_visible')),
    ('access_clearance_v1', 'bridge_damage'): (
        ('oblique', 'connection', 'damage', 'bridge_damage_context_visible'),
        ('context', 'target_and_context', 'state', 'bridge_damage_context_visible')),
    ('access_clearance_v1', 'obstruction_material'): (
        ('oblique', 'full_target', 'damage', 'obstruction_type_and_extent_visible'),
        ('overview', 'full_target', 'state', 'obstruction_type_and_extent_visible')),
    ('access_clearance_v1', 'obstruction_extent'): (
        ('along_axis', 'both_sides', 'state', 'obstruction_type_and_extent_visible'),
        ('overview', 'full_target', 'state', 'obstruction_type_and_extent_visible')),
    ('flood_river_v1', 'road_flooding'): (
        ('oblique', 'boundary', 'damage', 'road_water_boundary_visible'),
        ('context', 'target_and_context', 'state', 'road_water_boundary_visible')),
    ('access_clearance_v1', 'road_edge_support_loss'): (
        ('cross_section', 'boundary', 'damage', 'road_edge_support_boundary_visible'),
        ('context', 'target_and_context', 'state', 'road_edge_support_boundary_visible')),
}


def candidates_v06(context, intents, *, now_ms, order=(0, 1, 2)):
    validate_context_v1(context, now_ms=now_ms)
    validate_intents(intents, context, now_ms)
    require(isinstance(order, (tuple, list)) and sorted(order) == [0, 1, 2]
            and all(type(i) is int for i in order), 'Candidate order must be a permutation')
    key = (context['mission_profile'], intents['intents'][0]['question'])
    require(key in ROWS_V06, 'Unsupported catalog 0.6 question')
    result = []
    for relation, coverage, detail, stop in ROWS_V06[key]:
        program = {'schema_version': SCHEMA, 'outcome': 'observe', 'reason_code': 'program_created',
                   'observations': [{'sequence': 0, 'intent_sequence': 0, 'view_relation': relation,
                                     'coverage': coverage, 'detail_level': detail, 'sensor': 'rgb',
                                     'stop_condition': stop}]}
        validate_observation_program(program, context, intents, now_ms=now_ms)
        result.append({'candidate_id': canonical_sha256(program), 'program': program})
    result.append({'candidate_id': 'reject', 'program': None})
    return [deepcopy(result[i]) for i in order]


def build_request_v06(context, intents, frames, *, now_ms, order=(0, 1, 2)):
    """The decision request of tancho.observation_decision.build_request, with catalog 0.6 options."""
    validate_context_v1(context, now_ms=now_ms, evidence=frames)
    options = candidates_v06(context, intents, now_ms=now_ms, order=order)
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
    body = {'schema_version': REQUEST, 'catalog_revision': CATALOG_V06,
            'context_sha256': canonical_sha256(context),
            'intents_sha256': canonical_sha256(intents),
            'frame_sha256': {key: hashlib.sha256(value).hexdigest()
                             for key, value in sorted(frames.items())},
            'evidence_order': list(context['evidence_ids']),
            'model_revision': context['model_revision'], 'order': list(order),
            'candidates': options, 'labels': list(LABELS), 'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    return {**body, 'request_sha256': canonical_sha256(body)}


def check_request(request):
    """_check_request for the registered catalogs; the same checks for catalog 0.6, which the frozen
    registry cannot list."""
    if request.get('catalog_revision') != CATALOG_V06:
        return _check_request(request)
    fields(request, 'schema_version catalog_revision context_sha256 intents_sha256 frame_sha256 '
                    'model_revision evidence_order order candidates labels prompt prompt_sha256 request_sha256')
    require(request['schema_version'] == REQUEST, 'Unknown decision request')
    require(request['labels'] == list(LABELS) and len(request['candidates']) == 3, 'Invalid option registry')
    require(request['request_sha256'] == canonical_sha256(
        {k: v for k, v in request.items() if k != 'request_sha256'}), 'Request changed')


# The receipt, replay and acceptance of tancho.observation_decision, for catalog 0.6 requests, which
# the frozen request check cannot register. The arithmetic is the same; only the request check and
# the request rebuild differ.
def make_receipt_v06(request, logits, token_ids, *, started_ms, completed_ms, timeout_ms):
    require(request.get('catalog_revision') == CATALOG_V06, 'Not a catalog 0.6 request')
    check_request(request)
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


def replay_decision_v06(request, receipt, context, intents, frames, *, now_ms):
    rebuilt = build_request_v06(context, intents, frames, now_ms=now_ms, order=request['order'])
    require(request == rebuilt, 'Decision request binding changed')
    expected = make_receipt_v06(request, receipt['logits'], receipt['label_token_ids'],
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


def acceptance_v06(request, receipt, *, model_sha256, readout_head_sha256, readout_projection=None):
    """Uncalibrated acceptance: catalog 0.6 has no fitted calibration, so any selected candidate passes."""
    from .observation_decision_calibration import probabilities
    require(receipt == make_receipt_v06(request, receipt['logits'], receipt['label_token_ids'],
                                        started_ms=receipt['started_ms'], completed_ms=receipt['completed_ms'],
                                        timeout_ms=receipt['timeout_ms']), 'Acceptance receipt changed')
    require(isinstance(readout_head_sha256, str) and len(readout_head_sha256) == 64
            and all(c in '0123456789abcdef' for c in readout_head_sha256),
            'Catalog 0.6 requires a bound readout head')
    require(readout_projection in (None, 'three_row'), 'Invalid readout projection binding')
    p = probabilities(receipt['logits'], 1.0)
    selected = receipt['selected_index']
    accepted = receipt['outcome'] == 'selected' and selected is not None
    body = {'schema_version': 'tancho-decision-acceptance-1.0',
            'receipt_sha256': receipt['receipt_sha256'], 'model_sha256': model_sha256,
            'calibration_sha256': None, 'probabilities': p, 'min_probability': 0.0, 'accepted': accepted,
            'reason': 'accepted' if accepted else receipt['reason']}
    return {**body, 'sha256': canonical_sha256(body)}
