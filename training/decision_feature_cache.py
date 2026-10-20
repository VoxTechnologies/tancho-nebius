"""Pinned final-token features for offline FP32 candidate readout experiments."""
import hashlib
import itertools
import json
from pathlib import Path

from tancho.contracts import require
from tancho.mission_context import canonical_sha256
from tancho.observation_catalog_v06 import check_request

SCHEMA = 'tancho-decision-feature-cache-1.0'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_cache(meta, tensors):
    import torch
    require(meta['schema_version'] == SCHEMA, 'Unknown feature cache schema')
    require(meta['execution_kind'] in ('native_cosmos', 'cpu_fixture'), 'Unknown feature origin')
    require(meta['dataset_sha256'] == canonical_sha256(meta['dataset']), 'Feature dataset changed')
    split = meta['dataset']['split']
    require(split in ('train', 'validation'), 'Feature diagnostics exclude test split')
    rows, source = meta['rows'], meta['dataset']['rows']
    require(len(rows) == len(source) > 0, 'Incomplete feature rows')
    require(set(tensors) == {'features', 'initial_weight', 'native_logits'}, 'Unexpected feature tensors')
    x, w, z = (tensors[k] for k in ('features', 'initial_weight', 'native_logits'))
    require(all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.dtype == torch.float32
                and bool(torch.isfinite(t).all()) for t in (x, w, z)), 'Expected finite CPU FP32 tensors')
    require(x.ndim == 2 and x.shape[0] == len(rows) and x.shape[1] > 0
            and w.shape == (3, x.shape[1]) and z.shape == (len(rows), 3), 'Invalid feature shapes')
    slots = meta['binding']['label_token_ids']
    require(len(slots) == 3 and len(set(slots)) == 3
            and all(type(i) is int and i >= 0 for i in slots), 'Invalid feature label tokens')
    require(meta['binding']['precision'] == 'bf16', 'Feature diagnostics require BF16 backbone')
    groups = {}
    for row, expected in zip(rows, source, strict=True):
        require(all(row[k] == expected[k] for k in
                    ('sample_id', 'parent_sample_id', 'variant', 'order', 'target_index')),
                'Feature supervision differs from dataset')
        require(type(row['target_index']) is int and 0 <= row['target_index'] < 3, 'Invalid feature target')
        request = row['request']
        check_request(request)
        require(request['order'] == row['order']
                and request['model_revision'] == meta['binding']['model_revision'], 'Feature request binding changed')
        supervision = row['supervision_proof']
        require(supervision['parent_sample_id'] == row['parent_sample_id']
                and supervision['variant'] == row['variant']
                and supervision['source_manifest_sha256'] == meta['dataset']['source_manifest_sha256']
                and supervision['derived_frame_sha256'] == request['frame_sha256']
                and supervision['sha256'] == canonical_sha256({k: v for k, v in supervision.items() if k != 'sha256'}),
                'Feature supervision proof changed')
        proof = row['processor_input_sha256']
        require(isinstance(proof, str) and len(proof) == 64
                and all(c in '0123456789abcdef' for c in proof), 'Missing feature tensor proof')
        reject = request['candidates'][row['target_index']]['program'] is None
        require(reject == (row['variant'] == 'uniform_occlusion_v1')
                and row['variant'] in ('clean', 'uniform_occlusion_v1'), 'Invalid feature reject supervision')
        key = (row['parent_sample_id'], row['variant'])
        groups.setdefault(key, []).append(tuple(row['order']))
    all_orders = set(itertools.permutations(range(3)))
    require(all(len(v) == 6 and set(v) == all_orders for v in groups.values()),
            'Feature cache requires all six distinct candidate orders')
    return meta, tensors


def save_cache(output, dataset, binding, rows, features, weight, logits, *, execution_kind):
    import torch
    output = Path(output)
    require(not output.exists(), 'Use a fresh feature output')
    meta = dict(schema_version=SCHEMA, dataset=dataset.manifest, dataset_sha256=dataset.fingerprint,
                binding=binding, rows=rows, execution_kind=execution_kind,
                feature_location='lm_head_input_last_token', release_qualified=False)
    tensors = {'features': features.detach().cpu().float().contiguous(),
               'initial_weight': weight.detach().cpu().float().contiguous(),
               'native_logits': logits.detach().cpu().float().contiguous()}
    validate_cache(meta, tensors)
    output.mkdir(parents=True)
    torch.save(tensors, output / 'tensors.pt')
    meta['tensors_sha256'] = digest(output / 'tensors.pt')
    meta['sha256'] = canonical_sha256(meta)
    write_json(output / 'manifest.json', meta)
    return load_cache(output)


def load_cache(directory):
    import torch
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())
    require(meta['sha256'] == canonical_sha256({k: v for k, v in meta.items() if k != 'sha256'}),
            'Feature manifest changed')
    require(digest(directory / 'tensors.pt') == meta['tensors_sha256'], 'Feature tensor bytes changed')
    tensors = torch.load(directory / 'tensors.pt', map_location='cpu', weights_only=True)
    return validate_cache(meta, tensors)


def read_export_rows(export, slots):
    """Read the saved rows before a whole-model BF16 loader can round them."""
    import torch
    from safetensors import safe_open
    require(len(slots) == 3 and len(set(slots)) == 3
            and all(type(i) is int and i >= 0 for i in slots), 'Invalid exported token slots')
    export = Path(export)
    index = export / 'model.safetensors.index.json'
    name = json.loads(index.read_text())['weight_map']['lm_head.weight'] if index.exists() else 'model.safetensors'
    path = export / name
    require(path.resolve().is_relative_to(export.resolve()), 'Head shard escapes export')
    with safe_open(path, framework='pt', device='cpu') as reader:
        tensor = reader.get_slice('lm_head.weight')
        require(len(tensor.get_shape()) == 2 and max(slots) < tensor.get_shape()[0], 'Invalid exported head shape')
        return torch.cat([tensor[i:i + 1].float() for i in slots], dim=0)


def capture(dataset, backend, initial_weight, output, *, execution_kind='native_cosmos'):
    import torch
    require(not Path(output).exists(), 'Use a fresh feature output')
    require(not backend.model.training, 'Feature backbone must be in evaluation mode')
    loaded_rows = backend.model.lm_head.weight[backend.binding['label_token_ids']].detach().cpu().float()
    require(initial_weight.dtype == torch.float32 and torch.equal(initial_weight.bfloat16().float(), loaded_rows),
            'Saved FP32 rows differ from loaded BF16 model')
    features, logits, rows = [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        captured = []

        def remember(module, args):
            require(len(args) == 1 and args[0].ndim == 3 and args[0].shape[0] == 1,
                    'Expected one sequence at the final head')
            captured.append(args[0][:, -1, :].detach().cpu().float().clone())

        hook = backend.model.lm_head.register_forward_pre_hook(remember)
        try:
            response = backend(item['request'], item['frames'])
        finally:
            hook.remove()
        require(len(captured) == 1, 'Expected exactly one final-head feature')
        require(response['binding'] == backend.binding and response['execution_kind'] == execution_kind
                and response['request_sha256'] == item['request']['request_sha256'], 'Feature provider binding changed')
        features.append(captured[0])
        logits.append(response['logits'])
        rows.append({**{k: item[k] for k in ('sample_id', 'parent_sample_id', 'variant', 'order', 'target_index')},
                     'request': item['request'], 'processor_input_sha256': response['processor_input_sha256'],
                     'supervision_proof': item['supervision_proof']})
    require(features, 'No features captured')
    return save_cache(output, dataset, backend.binding, rows, torch.cat(features), initial_weight,
                      torch.tensor(logits, dtype=torch.float32), execution_kind=execution_kind)
