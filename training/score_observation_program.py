"""Strict semantic scoring and an image-free rule baseline for Tancho roles."""
from collections import Counter, defaultdict

from tancho.observation_intent import parse_intent_output, validate_and_bind_intents
from tancho.observation_program import parse_program_output, validate_observation_program


def _labels(record, output):
    if record['role'] == 'reasoner':
        if output['outcome'] != 'inspect':
            return [output['outcome']]
        return sorted(f"{row['target_id']}|{row['question']}" for row in output['intents'])
    questions = {row['intent_id']: row['question'] for row in record['verified_intents']['intents']}
    sequences = {row['sequence']: row['question'] for row in record['verified_intents']['intents']}
    return sorted('|'.join(((questions[row['intent_id']] if 'intent_id' in row
                            else sequences[row['intent_sequence']]), row['view_relation'], row['coverage'],
                            row['detail_level'], row['sensor'], row['stop_condition']))
                  for row in output['observations'])


def _summary(rows):
    tp = fp = fn = exact = valid = 0
    for row in rows:
        gold, predicted = set(row['gold']), set(row['predicted'])
        tp += len(gold & predicted); fp += len(predicted - gold); fn += len(gold - predicted)
        exact += gold == predicted and row['valid']
        valid += row['valid']
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {'cases': len(rows), 'schema_valid_rate': valid / len(rows) if rows else 0.0,
            'exact_accuracy': exact / len(rows) if rows else 0.0,
            'micro_precision': precision, 'micro_recall': recall,
            'micro_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def score_predictions(records, raw_predictions):
    """One raw response per record; invalid responses stay invalid and are never repaired."""
    rows = []
    for record in records:
        raw = raw_predictions.get(record['sample_id'])
        valid, predicted, error = False, [], None
        try:
            if record['role'] == 'reasoner':
                parsed = parse_intent_output(raw)
                validate_and_bind_intents(parsed, record['context'],
                                          now_ms=record['context']['captured_at_ms'], raw_text=raw)
            else:
                parsed = parse_program_output(raw)
                validate_observation_program(parsed, record['context'], record['verified_intents'],
                                             now_ms=record['context']['captured_at_ms'], raw_text=raw)
            predicted, valid = _labels(record, parsed), True
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        rows.append({'sample_id': record['sample_id'], 'role': record['role'],
                     'mission_profile': record['mission_profile'],
                     'visual_pair_id': record['visual_pair_id'],
                     'gold': _labels(record, record['target']), 'predicted': predicted,
                     'valid': valid, 'error': error})
    groups = defaultdict(list)
    for row in rows:
        groups['role:' + row['role']].append(row)
        groups['profile:' + row['mission_profile']].append(row)
    return {'schema_version': 'tancho-observation-program-score-1.0',
            'overall': _summary(rows),
            'groups': {key: _summary(value) for key, value in sorted(groups.items())},
            'paired_visual_sensitivity': _paired_sensitivity(rows), 'cases': rows}


def _paired_sensitivity(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['visual_pair_id']:
            groups[row['visual_pair_id']].append(row)
    eligible = [value for value in groups.values()
                if len({tuple(row['gold']) for row in value}) > 1]
    changed = sum(len({tuple(row['predicted']) for row in value}) > 1 for value in eligible)
    return {'pairs': len(eligible), 'prediction_changed': changed,
            'rate': changed / len(eligible) if eligible else 0.0}


def image_free_rule_baseline(manifest, *, train_split='train', eval_split='test'):
    """Memorize the most common label set for non-image inputs only."""
    train = [row for row in manifest['samples'] if row['split'] == train_split]
    evaluate = [row for row in manifest['samples'] if row['split'] == eval_split]
    by_key = defaultdict(Counter); global_counts = defaultdict(Counter)
    for row in train:
        label = tuple(row['semantic_labels'])
        by_key[row['blind_key']][label] += 1
        global_counts[(row['role'], row['mission_profile'])][label] += 1
    rows = []
    for row in evaluate:
        counter = by_key.get(row['blind_key']) or global_counts[(row['role'], row['mission_profile'])]
        predicted = list(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]) if counter else []
        rows.append({'sample_id': row['sample_id'], 'role': row['role'],
                     'mission_profile': row['mission_profile'],
                     'visual_pair_id': row['visual_pair_id'], 'gold': row['semantic_labels'],
                     'predicted': predicted, 'valid': True, 'error': None})
    profiles = defaultdict(list)
    for row in rows:
        profiles[row['mission_profile']].append(row)
    summaries = {key: _summary(value) for key, value in sorted(profiles.items())}
    return {'schema_version': 'tancho-image-free-baseline-1.0', 'overall': _summary(rows),
            'profiles': summaries, 'paired_visual_sensitivity': _paired_sensitivity(rows),
            'generator_task_valid': all(value['exact_accuracy'] < 1.0
                                        for key, value in summaries.items()
                                        if any(row['role'] == 'generator'
                                               for row in profiles[key])),
            'cases': rows}
