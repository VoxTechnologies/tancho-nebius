"""Stage C inputs: rendered stage A initial views as catalog 0.6 decision requests, in all six orders.

Each labelled scene (A or B, both programs flyable) becomes one parent with six candidate
orders, so training/decision_feature_cache.py can capture last-token features and
training/fit_decision_readout.py can fit the three-row readout leave-one-image-out. The label
comes from stage A geometry; the prompt never contains it. Scenes are split into shards so one
bounded Modal job captures one shard.
"""
import hashlib
import itertools
import json
from pathlib import Path

from tancho.contracts import require
from tancho.mission_context import canonical_sha256
from tancho.observation_catalog_v06 import CATALOG_V06, build_request_v06
from training.stage_a_context import NOW, stage_a_context

ORDERS = list(itertools.permutations(range(3)))


class StageADecisionDataset:
    def __init__(self, images, *, shard=0, shards=1, model_revision='geometry-only'):
        images = Path(images)
        index = json.loads((images / 'index.json').read_text())
        require(index['sha256'] == canonical_sha256({k: v for k, v in index.items() if k != 'sha256'}),
                'Image index changed')
        require(type(shards) is int and shards >= 1 and type(shard) is int and 0 <= shard < shards, 'Invalid shard')
        labelled = [row for row in index['rows'] if row['both_flyable'] and row['label'] in ('A', 'B')]
        self.root, self.parents, self.model_revision = images, labelled[shard::shards], model_revision
        require(self.parents, 'Empty shard')
        self.rows = []
        for position, row in enumerate(self.parents):
            parent = f"{row['question']}-{row['index']:03d}"
            for order in ORDERS:
                self.rows.append({'source_index': position, 'sample_id': f'{parent}::{"".join(map(str, order))}',
                                  'parent_sample_id': parent, 'variant': 'clean', 'order': list(order),
                                  'target_index': order.index('AB'.index(row['label']))})
        self.manifest = {'schema_version': 'tancho-stage-a-decision-supervision-1.0',
                         'source_manifest_sha256': index['sha256'], 'catalog_revision': CATALOG_V06,
                         'split': 'train', 'training': False, 'shard': shard, 'shards': shards,
                         'palette': index.get('palette', 'roles'), 'rows': self.rows,
                         'model_revision': model_revision,
                         'label_origin': 'stage_a_geometry_items', 'release_qualified': False}
        self.fingerprint = canonical_sha256(self.manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        selected = self.rows[index]
        row = self.parents[selected['source_index']]
        path = self.root / row['image']
        frame = path.read_bytes()
        require(hashlib.sha256(frame).hexdigest() == row['sha256'], 'Rendered image changed')
        context, intents = stage_a_context(row['question'], row['index'], frame, self.model_revision)
        frames = {'frame-1': frame}
        request = build_request_v06(context, intents, frames, now_ms=NOW, order=selected['order'])
        proof = {'parent_sample_id': selected['parent_sample_id'], 'variant': 'clean',
                 'source_manifest_sha256': self.manifest['source_manifest_sha256'],
                 'original_frame_sha256': {'frame-1': row['sha256']}, 'human_reviewed': False,
                 'derived_frame_sha256': request['frame_sha256']}
        proof['sha256'] = canonical_sha256(proof)
        return {**selected, '__key__': selected['sample_id'], 'request': request, 'context': context,
                'intents': intents, 'frames': frames, 'supervision_proof': proof}
