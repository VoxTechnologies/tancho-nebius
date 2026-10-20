"""Strict Generator observation-program contract."""
import hashlib
import json

from .contracts import ContractError, fields, require
from .mission_context import canonical_sha256, validate_context_v1
from .mission_profiles import profile, validate_profile_relation
from .observation_intent import parse_json_object

SCHEMA = 'tancho-observation-program-1.1'
LEGACY_SCHEMA = 'tancho-observation-program-1.0'
VERIFIED_SCHEMA = 'tancho-verified-observation-program-1.0'


def parse_program_output(raw):
    return parse_json_object(raw)


def validate_observation_program(output, context, intents, *, now_ms, raw_text=None):
    validate_context_v1(context, now_ms=now_ms)
    fields(intents, 'schema_version context_id mission_id outcome evidence_ids intents '
                    'intent_revision source_output_sha256')
    require(intents['schema_version'] == 'tancho-verified-intents-1.0', 'Unverified intent input')
    require(intents['context_id'] == context['context_id']
            and intents['mission_id'] == context['mission_id'], 'Intent set is bound elsewhere')
    schema = output.get('schema_version')
    if schema == SCHEMA:
        fields(output, 'schema_version outcome observations reason_code')
    elif schema == LEGACY_SCHEMA:
        fields(output, 'schema_version context_id mission_id intent_revision outcome observations reason_code')
        require(output['context_id'] == context['context_id']
                and output['mission_id'] == context['mission_id'],
                'Observation program is bound to another context or mission')
        require(output['intent_revision'] == intents['intent_revision'], 'Intent revision changed')
    else:
        raise ContractError('Unknown observation program version')
    require(intents['outcome'] == 'inspect' and bool(intents['intents']),
            'Generator requires an inspect intent set')
    require(isinstance(output['observations'], list), 'Observations must be a list')
    if output['outcome'] == 'observe':
        require(output['reason_code'] == 'program_created'
                and 1 <= len(output['observations']) <= context['policy']['max_observations'],
                'Invalid observation program outcome')
    elif output['outcome'] == 'uncompilable_intent':
        require(output['reason_code'] == 'unsupported_observation_primitive'
                and not output['observations'], 'Uncompilable outcome must be empty')
    else:
        raise ContractError('Unknown observation program outcome')
    p = profile(context['mission_profile'])
    intent_rows = {row['intent_id']: row for row in intents['intents']}
    intent_sequences = {row['sequence']: row for row in intents['intents']}
    targets = {target['target_id']: target for target in context['targets']}
    used, signatures, verified = set(), set(), []
    for index, row in enumerate(output['observations']):
        if schema == SCHEMA:
            fields(row, 'sequence intent_sequence view_relation coverage detail_level sensor stop_condition')
            require(type(row['intent_sequence']) is int
                    and row['intent_sequence'] in intent_sequences,
                    'Unknown observation intent sequence')
            intent = intent_sequences[row['intent_sequence']]
            intent_id = intent['intent_id']
        else:
            fields(row, 'sequence intent_id view_relation coverage detail_level sensor stop_condition')
            require(row['intent_id'] in intent_rows, 'Unknown observation intent')
            intent_id = row['intent_id']; intent = intent_rows[intent_id]
        require(row['sequence'] == index, 'Observation sequence must be contiguous')
        target = targets[intent['target_id']]
        validate_profile_relation(p, category=target['category'], question=intent['question'],
                                  view_relation=row['view_relation'], coverage=row['coverage'],
                                  detail_level=row['detail_level'], sensor=row['sensor'],
                                  stop_condition=row['stop_condition'])
        require(row['sensor'] in context['sensor_capabilities'], 'Requested sensor is unavailable')
        signature = (intent_id,) + tuple(row[key] for key in
                                         ('view_relation', 'coverage', 'detail_level',
                                          'sensor', 'stop_condition'))
        require(signature not in signatures, 'Duplicate observation requirement')
        bound = {key: row[key] for key in
                 ('sequence', 'view_relation', 'coverage', 'detail_level', 'sensor',
                  'stop_condition')}
        bound['intent_id'] = intent_id
        signatures.add(signature); used.add(intent_id); verified.append(bound)
    if output['outcome'] == 'observe':
        require(used == set(intent_rows), 'Every intent must have an observation')
    body = {'context_id': context['context_id'], 'mission_id': context['mission_id'],
            'intent_revision': intents['intent_revision'], 'outcome': output['outcome'],
            'observations': verified}
    source = raw_text.encode() if raw_text is not None else json.dumps(
        output, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return {'schema_version': VERIFIED_SCHEMA, **body,
            'program_revision': canonical_sha256(body),
            'source_output_sha256': hashlib.sha256(source).hexdigest()}
