"""Derive candidate supervision without leaking labels into the input prompt."""
import itertools
import hashlib
import io
import json
import random
from copy import deepcopy
from pathlib import Path

from tancho.contracts import ContractError, require
from tancho.mission_context import canonical_sha256, context_input_sha256
from tancho.observation_decision import CATALOG, CATALOG_REGISTRY, build_request, candidates
from tancho.observation_intent import validate_and_bind_intents
from training.observation_program_loader import ObservationProgramDataset

REJECT_MODES = ('none', 'uniform_occlusion_v1')
# Disjoint intensities avoid identical synthetic blank frames across splits.
OCCLUSION_COLORS = {'train': (0, 255), 'validation': (96, 160), 'test': (64, 192)}


class DecisionDataset:
    def __init__(self, pack, split, *, training=False, limit_samples=None, reject_mode='none',
                 all_orders=False, catalog_revision=CATALOG, weak_review=None):
        require(split in ('train', 'validation', 'test'), 'Unknown split')
        require(not training or split == 'train', 'Training may only use the train split')
        require(reject_mode in REJECT_MODES, 'Unknown reject supervision mode')
        require(type(all_orders) is bool, 'Invalid all-orders setting')
        require(catalog_revision in CATALOG_REGISTRY, 'Unknown decision catalog')
        require(weak_review is None or (training and split == 'train' and reject_mode == 'none'),
                'Weak review requires clean training supervision')
        self.catalog_revision = catalog_revision
        self.reject_mode = reject_mode
        require(limit_samples is None or type(limit_samples) is int and 1 <= limit_samples <= 100000,
                'Invalid supervision sample limit')
        self.source = ObservationProgramDataset(
            pack, split, purpose='compatibility' if weak_review else 'training' if training else 'evaluation')
        if weak_review is not None:
            review = json.loads(Path(weak_review).read_text())
            require(review['manifest_sha256'] == canonical_sha256({
                key: value for key, value in review.items() if key != 'manifest_sha256'})
                and review['schema_version'] == 'tancho-road-hazard-weak-decision-review-1.0'
                and review['catalog_revision'] == catalog_revision
                and review['review_origin'] == 'ai_visual_review_weak_hypothesis'
                and review['independent_evaluation'] is False
                and review['flight_authorized'] is False
                and review['physical_view_improvement_proven'] is False,
                'Invalid weak review authority')
            require(len({row['source_id'] for row in review['rows']}) == len(review['rows'])
                    and all(row['next_view_action_gold'] is False
                            and ((row['choice'] in ('A', 'B')) ==
                                 (row['selected_candidate_id'] is not None))
                            for row in review['rows']),
                    'Invalid weak review labels')
            labeled = {row['source_id']: row for row in review['rows']
                       if row['selected_candidate_id'] is not None}
            require(len(labeled) == sum(row['selected_candidate_id'] is not None
                                        for row in review['rows'])
                    and len(labeled) == len(self.source.rows),
                    'Weak review does not cover the pack exactly')
            for row in self.source.rows:
                provenance = row['provenance']
                source_id, separator, review_sha = provenance['source_id'].rpartition('@')
                label = labeled.get(source_id)
                require(separator and review_sha == review['manifest_sha256']
                        and label is not None and row['role'] == 'generator'
                        and provenance['source_kind'] == 'synthetic'
                        and provenance['human_reviewed'] is False
                        and provenance['training_allowed'] is True
                        and provenance['public_allowed'] is False
                        and row['frames_sha256'] == {'frame-1': label['image_sha256']}
                        and row['verified_intents']['intents'][0]['question'] == label['question'],
                        'Weak review source binding changed')
                options = candidates(row['context'], row['verified_intents'],
                                     now_ms=row['context']['captured_at_ms'],
                                     catalog_revision=catalog_revision)
                matches = [option for option in options
                           if option['candidate_id'] == label['selected_candidate_id']]
                selected_index = ord(label['choice']) - ord('A')
                require(len(matches) == 1
                        and options[selected_index]['candidate_id'] == label['selected_candidate_id']
                        and matches[0]['program'] == json.loads(row['target']),
                        'Weak review candidate binding changed')
        self.rows, self.excluded = [], []
        orders = list(itertools.permutations(range(3))) if training or all_orders else [(0, 1, 2), (2, 1, 0)]
        selected_count = 0
        for index, row in enumerate(self.source.rows):
            if row['role'] != 'generator':
                self.excluded.append({'sample_id': row['sample_id'], 'reason': 'not_generator'})
                continue
            try:
                options = candidates(row['context'], row['verified_intents'],
                                     now_ms=row['context']['captured_at_ms'],
                                     catalog_revision=catalog_revision)
            except ContractError as exc:
                if str(exc) not in ('D0 requires one intent', 'Unsupported D0 question'):
                    raise
                self.excluded.append({'sample_id': row['sample_id'], 'reason': str(exc)})
                continue
            gold = json.loads(row['target'])
            matches = [i for i, option in enumerate(options)
                       if option['program'] is not None and option['program'] == gold]
            if len(matches) != 1:
                self.excluded.append({'sample_id': row['sample_id'], 'reason': 'gold_not_in_catalog'})
                continue
            if limit_samples is not None and selected_count >= limit_samples:
                self.excluded.append({'sample_id': row['sample_id'], 'reason': 'sample_limit'})
                continue
            selected_count += 1
            for variant in (('clean',) if reject_mode == 'none' else ('clean', reject_mode)):
                reject = variant != 'clean'
                for order in orders:
                    self.rows.append({'source_index': index,
                                      'sample_id': row['sample_id'] + ('::' + variant if reject else ''),
                                      'parent_sample_id': row['sample_id'], 'variant': variant,
                                      'order': list(order), 'target_index': order.index(2 if reject else matches[0])})
        self.manifest = {'schema_version': 'tancho-decision-supervision-1.1',
                         'source_manifest_sha256': self.source.manifest['manifest_sha256'],
                         'catalog_revision': catalog_revision, 'split': split, 'training': training,
                         'rows': self.rows, 'excluded': self.excluded,
                         'reject_mode': reject_mode, 'selected_parent_count': selected_count,
                         'reject_supervision_count': sum(r['variant'] != 'clean' for r in self.rows),
                         'occlusion_colors': OCCLUSION_COLORS if reject_mode != 'none' else None,
                         'reject_authority': 'synthetic_complete_visual_information_removal' if reject_mode != 'none' else None,
                         'independent_final': False,
                         'missing_gold_is_reject': False}
        if weak_review is not None:
            self.manifest['label_origin'] = 'ai_visual_review_weak_hypothesis'
            self.manifest['weak_review_sha256'] = review['manifest_sha256']
            self.manifest['release_qualified'] = False
            self.manifest['sampling_policy'] = 'parent_once_per_epoch_order_rotated_v1'
        if all_orders:
            self.manifest['all_orders'] = True
        self.fingerprint = canonical_sha256(self.manifest)
        self._weak_parent_rows = {}
        if weak_review is not None:
            for index, row in enumerate(self.rows):
                self._weak_parent_rows.setdefault(row['parent_sample_id'], []).append(index)
        self._weak_parent_rank = {parent: index for index, parent in enumerate(self._weak_parent_rows)}

    def __len__(self):
        return len(self.rows)

    def sample_index(self, step, seed):
        if not self._weak_parent_rows:
            order = list(range(len(self.rows)))
            random.Random(seed + step // len(self.rows)).shuffle(order)
            return order[step % len(order)]
        parents = list(self._weak_parent_rows)
        epoch, position = divmod(step, len(parents))
        shuffled = list(parents)
        random.Random(seed + epoch).shuffle(shuffled)
        parent = shuffled[position]
        rows = self._weak_parent_rows[parent]
        return rows[(self._weak_parent_rank[parent] + epoch) % len(rows)]

    def __getitem__(self, index):
        selected = self.rows[index]
        row = self.source.rows[selected['source_index']]
        item = self.source[selected['source_index']]
        frames = {key: (self.source.root / name).read_bytes()
                  for key, name in zip(row['frames_sha256'], row['frames'], strict=True)}
        context = deepcopy(row['context'])
        proof = {'parent_sample_id': row['sample_id'], 'variant': selected['variant'],
                 'source_manifest_sha256': self.source.manifest['manifest_sha256'],
                 'original_frame_sha256': dict(row['frames_sha256']), 'human_reviewed': False}
        if selected['variant'] != 'clean':
            from PIL import Image
            color = OCCLUSION_COLORS[self.manifest['split']][
                int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % 2]
            images, canonical = [], []
            for key in context['evidence_ids']:
                with Image.open(io.BytesIO(frames[key])) as image:
                    blank = Image.new('RGB', image.size, (color, color, color))
                stream = io.BytesIO(); blank.save(stream, format='PNG')
                frames[key] = stream.getvalue()
                images.append(blank)
                canonical.append({'id': key, 'sha256': hashlib.sha256(frames[key]).hexdigest(),
                                  'rgb_sha256': hashlib.sha256(blank.tobytes()).hexdigest(),
                                  'size': list(blank.size)})
            context['input_sha256'] = context_input_sha256(context, frames)
            item['texts'][0]['content'][0]['video'] = images
            item['canonical_frames']['frames'] = canonical
            proof.update(reason='evidence_insufficient', transformation='replace_every_pixel_in_every_frame',
                         rgb=[color] * 3, split=self.manifest['split'],
                         authority='synthetic_complete_visual_information_removal')
        request = build_request(context, row['verified_intents'], frames,
                                now_ms=context['captured_at_ms'], order=selected['order'],
                                catalog_revision=self.catalog_revision)
        proof['derived_frame_sha256'] = request['frame_sha256']
        proof['sha256'] = canonical_sha256(proof)
        # Remove the old assistant answer before any processor sees this item.
        item['texts'] = item['texts'][:1]
        item['texts'][0]['content'][1]['text'] = request['prompt']
        return {**item, **selected, '__key__': selected['sample_id'], 'request': request, 'context': context,
                'intents': row['verified_intents'], 'frames': frames, 'supervision_proof': proof}


def write_manifest(dataset, output):
    output = Path(output)
    require(not output.exists(), 'Use a fresh supervision manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**dataset.manifest, 'sha256': dataset.fingerprint}, indent=2) + '\n')


class ReviewedRealDecisionDataset:
    """Load only fully reviewed development-train labels from real frames."""

    def __init__(self, pack, review_directory, answers_path, reviewed_manifest):
        from training.prepare_real_decision_review import build_reviewed_manifest

        answers = json.loads(Path(answers_path).read_text())
        expected = build_reviewed_manifest(review_directory, answers)
        actual = json.loads(Path(reviewed_manifest).read_text())
        require(actual == expected, 'Reviewed real decision manifest differs from review answers')
        self.source = ObservationProgramDataset(pack, 'train', purpose='training')
        require(self.source.manifest['manifest_sha256'] == actual['pack_manifest_sha256'],
                'Reviewed labels are bound to a different training pack')
        self.source_indices = {row['sample_id']: index for index, row in enumerate(self.source.rows)}
        require(len(self.source_indices) == len(self.source.rows), 'Duplicate source sample IDs')
        self.rows = actual['rows']
        require(self.rows and all(row['sample_id'] in self.source_indices for row in self.rows),
                'Reviewed sample is absent from the source pack')
        self.manifest = {'schema_version': 'tancho-reviewed-real-decision-dataset-1.0',
                         'source_manifest_sha256': self.source.manifest['manifest_sha256'],
                         'review_manifest_sha256': actual['review_manifest_sha256'],
                         'answers_sha256': actual['answers_sha256'],
                         'reviewed_manifest_sha256': actual['sha256'],
                         'split': 'train', 'training': True,
                         'independent_final': False, 'synthetic': False,
                         'rows': self.rows}
        self.fingerprint = canonical_sha256(self.manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        selected = self.rows[index]
        source_index = self.source_indices[selected['sample_id']]
        row = self.source.rows[source_index]
        require(row['role'] == 'reasoner' and row['split'] == 'train'
                and row['provenance']['source_kind'] == 'real'
                and row['leakage_dimensions']['site_id'] == selected['site_id']
                and row['leakage_dimensions']['event_id'] == selected['event_id']
                and row['leakage_dimensions']['related_flight_id'] == selected['related_flight_id'],
                'Reviewed sample provenance changed')
        item = self.source[source_index]
        frames = {key: (self.source.root / name).read_bytes()
                  for key, name in zip(row['frames_sha256'], row['frames'], strict=True)}
        intents = validate_and_bind_intents(json.loads(row['target']), row['context'], now_ms=0)
        request = build_request(row['context'], intents, frames, now_ms=0, order=selected['order'])
        canonical_request = build_request(row['context'], intents, frames, now_ms=0)
        require(canonical_request['request_sha256'] == selected['request_sha256']
                and request['frame_sha256'] == selected['frame_sha256']
                and request['candidates'][selected['target_index']]['candidate_id'] == selected['choice'],
                'Reviewed answer no longer binds this candidate request')
        proof = {'review_manifest_sha256': self.manifest['review_manifest_sha256'],
                 'answers_sha256': self.manifest['answers_sha256'],
                 'sample_id': selected['sample_id'], 'task_sha256': selected['task_sha256'],
                 'reason': selected['reject_reason'], 'human_reviewed': True,
                 'review_authentication': 'self_reported'}
        proof['sha256'] = canonical_sha256(proof)
        item['texts'] = item['texts'][:1]
        item['texts'][0]['content'][1]['text'] = request['prompt']
        return {**item, 'sample_id': selected['sample_id'],
                'parent_sample_id': selected['sample_id'], 'variant': 'clean',
                'order': selected['order'], 'target_index': selected['target_index'],
                'request': request, 'context': row['context'], 'intents': intents,
                'frames': frames, 'supervision_proof': proof}
