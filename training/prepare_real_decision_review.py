"""Prepare and validate human review of real-image candidate decisions."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

from tancho.mission_context import canonical_sha256
from tancho.observation_decision import build_request
from tancho.observation_intent import validate_and_bind_intents
from training.observation_program_loader import verify_pack

SCHEMA = 'tancho-real-decision-review-1.0'
REVIEW_SCHEMA = 'tancho-real-decision-review-answers-1.0'


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(source, pack, output):
    source, pack, output = (Path(p).resolve() for p in (source, pack, output))
    if output.exists():
        raise ValueError('Use a fresh review directory')
    source_path = source / 'manifest.json'
    source_manifest = json.loads(source_path.read_text())
    if (source_manifest.get('schema_version') !=
            'tancho-observation-program-release1-real-train-source-1.0'
            or source_manifest.get('legacy_label_scope') !=
            'decision and prose only; Observation Program labels unreviewed'
            or source_manifest.get('manifest_sha256') != canonical_sha256({
                key: value for key, value in source_manifest.items()
                if key != 'manifest_sha256'})):
        raise ValueError('Unverified real training source or changed label scope')
    manifest = verify_pack(pack, purpose='training')
    source_rows = {row['id']: row for row in source_manifest['samples']}
    if len(source_rows) != len(source_manifest['samples']):
        raise ValueError('Duplicate real source sample identity')
    rows = []
    exclusions = []
    pending_copies = []
    for row in manifest['samples']:
        if row['provenance']['source_kind'] != 'real':
            continue
        sample_id = row['sample_id']
        source_id = sample_id.removesuffix('.reasoner')
        original = source_rows.get(source_id)
        dimensions = row['leakage_dimensions']
        canonical_frames = {frame['id']: frame['sha256']
                            for frame in original['canonical_frames']['frames']} if original else None
        if (row['split'] != 'train' or row['role'] != 'reasoner'
                or original is None or original['split'] != 'train'
                or original['source_id'] != row['provenance']['source_id']
                or original['leakage_group_id'] != row['leakage_group_id']
                or any(original[key] != dimensions[key] for key in
                       ('world_id', 'event_id', 'site_id', 'related_flight_id'))
                or len(canonical_frames) != len(original['canonical_frames']['frames'])
                or canonical_frames != row['frames_sha256']):
            raise ValueError('Real source and pack identities differ')
        target = json.loads(row['target'])
        if target['outcome'] != 'inspect' or len(target['intents']) != 1:
            exclusions.append({'sample_id': sample_id, 'reason': 'not_single_inspect_intent'})
            continue
        frames = {}
        for evidence_id, relative in zip(row['frames_sha256'], row['frames'], strict=True):
            path = (pack / relative).resolve()
            if not path.is_relative_to(pack) or _sha(path) != row['frames_sha256'][evidence_id]:
                raise ValueError('Real frame binding changed')
            frames[evidence_id] = path.read_bytes()
        intents = validate_and_bind_intents(target, row['context'], now_ms=0)
        try:
            request = build_request(row['context'], intents, frames, now_ms=0)
        except ValueError as exc:
            if str(exc) != 'Unsupported D0 question':
                raise
            exclusions.append({'sample_id': sample_id, 'reason': 'unsupported_catalog_question',
                               'question': target['intents'][0]['question']})
            continue
        if (original['decision'] != 'inspect' or
                request['frame_sha256'] != row['frames_sha256']):
            raise ValueError('Legacy decision or frame binding differs')
        image_rows = []
        for evidence_id, relative in zip(row['frames_sha256'], row['frames'], strict=True):
            target_path = 'frames/' + sample_id + '/' + evidence_id + '.png'
            image_rows.append({'evidence_id': evidence_id, 'path': target_path,
                               'sha256': row['frames_sha256'][evidence_id]})
            pending_copies.append((pack / relative, output / target_path))
        task = {'sample_id': sample_id, 'source_id': source_id,
                'site_id': original['site_id'], 'event_id': original['event_id'],
                'related_flight_id': original['related_flight_id'],
                'scene_id': original['scene_id'], 'observation_goal': original['observation_goal'],
                'question_origin': 'legacy_reasoner_migration_not_program_gold',
                'question': target['intents'][0]['question'],
                'frames': image_rows, 'request': request}
        task['task_sha256'] = canonical_sha256(task)
        rows.append(task)
    if not rows:
        raise ValueError('No real examples match the current candidate catalog')
    manifest_body = {'schema_version': SCHEMA, 'source_manifest_sha256': _sha(source_path),
                     'pack_manifest_sha256': manifest['manifest_sha256'],
                     'split': 'development_train_only', 'label_status': 'pending_human_review',
                     'flight_authorized': False, 'training_allowed': False,
                     'independent_evaluation': False, 'tasks': rows, 'excluded': exclusions}
    review_manifest = {**manifest_body, 'manifest_sha256': canonical_sha256(manifest_body)}
    output.mkdir(parents=True)
    for source_file, dest in pending_copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, dest)
    _write(output / 'tasks.json', review_manifest)
    _write(output / 'answers-draft.json', {
        'schema_version': REVIEW_SCHEMA, 'manifest_sha256': review_manifest['manifest_sha256'],
        'reviewer': '', 'answers': [{'sample_id': row['sample_id'],
                                   'task_sha256': row['task_sha256'],
                                   'question_confirmed': None, 'choice': None,
                                   'reject_reason': None, 'note': ''} for row in rows]})
    lines = ['# Real-image decision review', '',
             'These are development-train examples from one site and flight. The legacy decision',
             'and question mapping are not Observation Program gold labels. Inspect every frame,',
             'confirm the question, then edit `answers-draft.json`. The null option requires an',
             'explicit reason; a catalog gap is separate from insufficient visual evidence.',
             'For a program choice, enter its candidate ID in `choice` and leave `reject_reason` null.',
             'For null, set `choice` to `reject` and `reject_reason` to',
             '`insufficient_evidence` or `no_candidate_appropriate`. If the right program',
             'is missing from the catalog, set `choice` to null and `reject_reason` to',
             '`catalog_gap`; this does not count as a reject label. Set `question_confirmed`',
             'to false if the mapped question is wrong. Every completed row needs a rationale.', '']
    for task in rows:
        lines += ['## ' + task['sample_id'], '',
                  'Goal: ' + task['observation_goal'],
                  'Mapped question (verify against images): ' + task['question'], '']
        for frame in task['frames']:
            lines.append('![' + frame['evidence_id'] + '](' + frame['path'] + ')')
        lines += ['', 'Candidates (fixed order):', '']
        for label, candidate in zip(task['request']['labels'], task['request']['candidates'], strict=True):
            lines.append('- ' + label + ' / `' + candidate['candidate_id'] + '`: `'
                         + json.dumps(candidate['program'], ensure_ascii=False,
                                      sort_keys=True) + '`')
        lines.append('')
    (output / 'review.md').write_text('\n'.join(lines) + '\n')
    return review_manifest


def validate(directory, answers):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'tasks.json').read_text())
    if (manifest.get('schema_version') != SCHEMA or manifest.get('training_allowed') is not False
            or manifest.get('independent_evaluation') is not False
            or manifest.get('manifest_sha256') != canonical_sha256({
                key: value for key, value in manifest.items() if key != 'manifest_sha256'})):
        raise ValueError('Review task manifest changed')
    for task in manifest['tasks']:
        if task['task_sha256'] != canonical_sha256({
                key: value for key, value in task.items() if key != 'task_sha256'}):
            raise ValueError('Review task changed')
        for frame in task['frames']:
            path = (directory / frame['path']).resolve()
            if not path.is_relative_to(directory) or _sha(path) != frame['sha256']:
                raise ValueError('Review frame changed')
    if (answers.get('schema_version') != REVIEW_SCHEMA
            or answers.get('manifest_sha256') != manifest['manifest_sha256']
            or type(answers.get('reviewer')) is not str
            or len(answers.get('answers', [])) != len(manifest['tasks'])):
        raise ValueError('Review answers do not match the task manifest')
    counts = {'pending': 0, 'choice': 0, 'insufficient_evidence': 0,
              'no_candidate_appropriate': 0, 'catalog_gap': 0, 'question_mismatch': 0}
    for task, answer in zip(manifest['tasks'], answers['answers'], strict=True):
        if (set(answer) != {'sample_id', 'task_sha256', 'question_confirmed',
                            'choice', 'reject_reason', 'note'}
                or answer['sample_id'] != task['sample_id']
                or answer['task_sha256'] != task['task_sha256']
                or type(answer['note']) is not str):
            raise ValueError('Review answer binding changed')
        question, choice, reason = (answer[key] for key in
                                    ('question_confirmed', 'choice', 'reject_reason'))
        if question is None and choice is None and reason is None and not answer['note']:
            counts['pending'] += 1
            continue
        if not answers['reviewer'].strip() or type(question) is not bool or not answer['note'].strip():
            raise ValueError('Reviewed answer needs reviewer, question judgment, and rationale')
        if not question:
            if choice is not None or reason is not None:
                raise ValueError('Question mismatch cannot produce a candidate label')
            counts['question_mismatch'] += 1
            continue
        option_ids = [row['candidate_id'] for row in task['request']['candidates'][:-1]]
        if choice in option_ids and reason is None:
            counts['choice'] += 1
        elif choice == 'reject' and reason in ('insufficient_evidence', 'no_candidate_appropriate'):
            counts[reason] += 1
        elif choice is None and reason == 'catalog_gap':
            counts['catalog_gap'] += 1
        else:
            raise ValueError('Invalid decision or rejection authority')
    return {'counts': counts, 'complete': counts['pending'] == 0,
            'training_allowed': False, 'independent_evaluation': False,
            'flight_authorized': False}


def build_reviewed_manifest(directory, answers):
    directory = Path(directory).resolve()
    summary = validate(directory, answers)
    if not summary['complete']:
        raise ValueError('Review is incomplete')
    tasks = json.loads((directory / 'tasks.json').read_text())
    rows, excluded = [], []
    for task, answer in zip(tasks['tasks'], answers['answers'], strict=True):
        if not answer['question_confirmed']:
            excluded.append({'sample_id': task['sample_id'], 'reason': 'question_mismatch'})
            continue
        if answer['reject_reason'] == 'catalog_gap':
            excluded.append({'sample_id': task['sample_id'], 'reason': 'catalog_gap'})
            continue
        for order in itertools.permutations(range(3)):
            original = task['request']['candidates']
            candidates = [original[index] for index in order]
            index = next(i for i, candidate in enumerate(candidates)
                         if candidate['candidate_id'] == answer['choice'])
            rows.append({'sample_id': task['sample_id'], 'source_id': task['source_id'],
                         'task_sha256': task['task_sha256'], 'request_sha256': task['request']['request_sha256'],
                         'frame_sha256': task['request']['frame_sha256'],
                         'choice': answer['choice'], 'reject_reason': answer['reject_reason'],
                         'order': list(order), 'target_index': index,
                         'scene_id': task['scene_id'], 'site_id': task['site_id'],
                         'event_id': task['event_id'], 'related_flight_id': task['related_flight_id']})
    if not rows:
        raise ValueError('Review contains no catalog-covered training labels')
    body = {'schema_version': 'tancho-reviewed-real-decision-train-1.0',
            'review_manifest_sha256': tasks['manifest_sha256'],
            'answers_sha256': canonical_sha256(answers),
            'pack_manifest_sha256': tasks['pack_manifest_sha256'],
            'reviewer': answers['reviewer'],
            'review_status': 'self_reported_complete',
            'split': 'train', 'development_training_eligible': True,
            'independent_evaluation': False, 'release_qualified': False,
            'flight_authorized': False, 'rows': rows, 'excluded': excluded}
    artifact = {**body, 'sha256': canonical_sha256(body)}
    return artifact


def export_reviewed(directory, answers, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a fresh reviewed training manifest')
    artifact = build_reviewed_manifest(directory, answers)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_cmd = commands.add_parser('prepare')
    prepare_cmd.add_argument('--source', type=Path, required=True)
    prepare_cmd.add_argument('--pack', type=Path, required=True)
    prepare_cmd.add_argument('--output', type=Path, required=True)
    validate_cmd = commands.add_parser('validate')
    validate_cmd.add_argument('--directory', type=Path, required=True)
    validate_cmd.add_argument('--answers', type=Path, required=True)
    export_cmd = commands.add_parser('export')
    export_cmd.add_argument('--directory', type=Path, required=True)
    export_cmd.add_argument('--answers', type=Path, required=True)
    export_cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source, args.pack, args.output)
        print(json.dumps({'tasks': len(result['tasks']), 'excluded': len(result['excluded']),
                          'manifest_sha256': result['manifest_sha256']}, ensure_ascii=False))
    else:
        answers = json.loads(args.answers.read_text())
        if args.command == 'validate':
            print(json.dumps(validate(args.directory, answers), ensure_ascii=False))
        else:
            result = export_reviewed(args.directory, answers, args.output)
            print(json.dumps({'rows': len(result['rows']), 'excluded': len(result['excluded']),
                              'sha256': result['sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
