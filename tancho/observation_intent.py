"""Strict Reasoner output parsing, validation and trusted intent binding."""
import hashlib
import json
import re

from .contracts import ContractError, fields, require, strings
from .mission_context import canonical_sha256, validate_context_v1
from .mission_profiles import profile, validate_profile_relation

SCHEMA = 'tancho-observation-intent-1.0'
VERIFIED_SCHEMA = 'tancho-verified-intents-1.0'
OUTCOMES = {
    'inspect': 'unanswered_question',
    'satisfied': 'all_required_questions_completed',
    'unsupported_sensor': 'required_sensor_unavailable',
}
NON_ACTION_REASONS = frozenset({'target_unresolved', 'evidence_unusable'})


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def parse_json_object(raw):
    require(isinstance(raw, str) and 0 < len(raw) <= 32768, 'Invalid model output')
    source = raw.strip()
    if source.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', source, re.S)
        require(match is not None, 'Expected exactly one fenced JSON object')
        source = match.group(1)
    try:
        value = json.loads(source, object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ContractError('Non-finite JSON')))
    except json.JSONDecodeError as error:
        raise ContractError('Invalid JSON') from error
    require(isinstance(value, dict), 'Expected a JSON object')
    return value


def parse_intent_output(raw):
    return parse_json_object(raw)


def validate_and_bind_intents(output, context, *, now_ms, raw_text=None):
    validate_context_v1(context, now_ms=now_ms)
    fields(output, 'schema_version context_id mission_id outcome evidence_ids intents reason_code')
    require(output['schema_version'] == SCHEMA, 'Unknown intent output version')
    require(output['context_id'] == context['context_id'] and output['mission_id'] == context['mission_id'],
            'Intent output is bound to another context or mission')
    require(strings(output['evidence_ids']) and set(output['evidence_ids']) <= set(context['evidence_ids']),
            'Unknown or duplicate intent evidence')
    require(isinstance(output['intents'], list), 'Intents must be a list')
    outcome = output['outcome']
    if outcome == 'inspect':
        require(output['reason_code'] == OUTCOMES[outcome], 'Intent outcome/reason mismatch')
        require(1 <= len(output['intents']) <= context['policy']['max_intents'], 'Invalid intent count')
        require(bool(output['evidence_ids']), 'Inspection requires evidence')
    elif outcome in ('satisfied', 'unsupported_sensor'):
        require(output['reason_code'] == OUTCOMES[outcome] and not output['intents'],
                'Non-inspection outcome must not contain intents')
        require(bool(output['evidence_ids']), 'Outcome requires evidence')
    elif outcome == 'insufficient_evidence':
        require(output['reason_code'] in NON_ACTION_REASONS and not output['intents'],
                'Insufficient-evidence outcome is inconsistent')
    else:
        raise ContractError('Unknown intent outcome')
    p = profile(context['mission_profile'])
    targets = {target['target_id']: target for target in context['targets']}
    # satisfied means every required question is completed or answered by the supplied
    # evidence (docs/completion-review-design-2026-09-22.md). Ending the mission is
    # enforced elsewhere: completion also needs the deterministic capture cross-check.
    verified, pairs = [], set()
    for index, row in enumerate(output['intents']):
        fields(row, 'sequence target_id question priority evidence_ids')
        require(row['sequence'] == index, 'Intent sequence must be contiguous')
        require(row['target_id'] in targets, 'Unknown intent target')
        require(type(row['priority']) is int and 1 <= row['priority'] <= 3, 'Invalid intent priority')
        require(strings(row['evidence_ids']) and set(row['evidence_ids']) <= set(context['evidence_ids']),
                'Unknown or duplicate intent evidence')
        target = targets[row['target_id']]
        validate_profile_relation(p, category=target['category'], question=row['question'])
        require(row['question'] in target['required_questions'],
                'Intent question is not required for target')
        require(row['question'] not in target['completed_questions'], 'Completed question was requested again')
        pair = (row['target_id'], row['question'])
        require(pair not in pairs, 'Duplicate target question')
        pairs.add(pair)
        bound = dict(row)
        bound['intent_id'] = canonical_sha256({'context_id': context['context_id'],
                                                'sequence': index, 'target_id': row['target_id'],
                                                'question': row['question']})
        verified.append(bound)
    revision_body = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
                     'outcome': outcome, 'evidence_ids': output['evidence_ids'], 'intents': verified}
    source = raw_text.encode() if raw_text is not None else json.dumps(
        output, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return {'schema_version': VERIFIED_SCHEMA, **revision_body,
            'intent_revision': canonical_sha256(revision_body),
            'source_output_sha256': hashlib.sha256(source).hexdigest()}
