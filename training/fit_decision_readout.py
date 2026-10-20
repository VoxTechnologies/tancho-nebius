"""Capture pinned features explicitly, then fit and evaluate a three-row head on CPU."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tancho.contracts import require
from tancho.mission_context import canonical_sha256
from training.decision_feature_cache import capture, digest, load_cache, read_export_rows, write_json
from training.observation_decision_data import DecisionDataset

HEAD = 'tancho-decision-fp32-readout-1.0'


def summarize(cache, logits):
    import torch
    meta, _ = cache
    require(logits.shape == (len(meta['rows']), 3) and bool(torch.isfinite(logits).all()), 'Invalid offline logits')
    records = []
    for row, values in zip(meta['rows'], logits.tolist(), strict=True):
        target = row['target_index']
        top = max(values)
        tied = values.count(top) > 1
        selected = None if tied else values.index(top)
        rejected = tied or row['request']['candidates'][selected]['program'] is None
        records.append(dict(parent_sample_id=row['parent_sample_id'], variant=row['variant'],
                            order=row['order'], target_index=target, logits=values, selected_index=selected,
                            correct=not tied and selected == target, rejected=rejected,
                            margin=values[target] - max(v for i, v in enumerate(values) if i != target)))
    def group(rows):
        clean = [r for r in rows if r['variant'] == 'clean']
        reject = [r for r in rows if r['variant'] != 'clean']
        return dict(records=len(rows), exact_accuracy=sum(r['correct'] for r in rows) / len(rows),
                    clean_accuracy=sum(r['correct'] for r in clean) / len(clean) if clean else None,
                    clean_false_rejection_rate=sum(r['rejected'] for r in clean) / len(clean) if clean else None,
                    reject_explicit_recall=sum(r['correct'] for r in reject) / len(reject) if reject else None,
                    reject_including_ties=sum(r['rejected'] for r in reject) / len(reject) if reject else None)
    targets = torch.tensor([r['target_index'] for r in meta['rows']])
    parents = sorted({r['parent_sample_id'] for r in records})
    return dict(**group(records), nll=float(torch.nn.functional.cross_entropy(logits, targets)),
                parents=len(parents), per_parent={p: group([r for r in records if r['parent_sample_id'] == p]) for p in parents},
                per_order={''.join(map(str, order)): group([r for r in records if tuple(r['order']) == order])
                           for order in sorted({tuple(r['order']) for r in records})}, rows=records)


def evaluate(cache, weight):
    import torch
    meta, tensors = cache
    require(weight.dtype == torch.float32 and weight.device.type == 'cpu'
            and weight.shape == tensors['initial_weight'].shape and bool(torch.isfinite(weight).all()),
            'Expected finite CPU FP32 readout')
    with torch.no_grad():
        x = tensors['features']
        return dict(cache_sha256=meta['sha256'], split=meta['dataset']['split'],
                    native_baseline=summarize(cache, tensors['native_logits']),
                    initial_fp32=summarize(cache, x @ tensors['initial_weight'].T),
                    trained_fp32=summarize(cache, x @ weight.T),
                    rounded_weight_fp32_matmul=summarize(cache, x @ weight.bfloat16().float().T),
                    quantization_note='BF16 weight rounding only; CPU FP32 matmul, not native BF16 kernel equivalence.',
                    release_qualified=False)


def load_head(directory):
    import torch
    directory = Path(directory)
    meta = json.loads((directory / 'head.json').read_text())
    require(meta['schema_version'] == HEAD and meta['sha256'] == canonical_sha256(
        {k: v for k, v in meta.items() if k != 'sha256'}), 'Readout manifest changed')
    require(digest(directory / 'weight.pt') == meta['weight_sha256'], 'Readout weight bytes changed')
    weight = torch.load(directory / 'weight.pt', map_location='cpu', weights_only=True)
    require(isinstance(weight, torch.Tensor) and weight.dtype == torch.float32 and weight.ndim == 2
            and weight.shape[0] == 3 and bool(torch.isfinite(weight).all()), 'Invalid saved FP32 readout')
    return meta, weight


def audit_precision(cache, head):
    import torch
    meta, tensors = cache
    head_meta, weight = head
    require(meta['dataset']['split'] == 'validation' and meta['execution_kind'] == 'native_cosmos',
            'Precision audit requires native validation features')
    require(head_meta['binding'] == meta['binding'] and weight.shape == tensors['initial_weight'].shape,
            'Readout and feature model binding differs')
    x = tensors['features']
    with torch.no_grad():
        fp32 = x @ weight.T
        rounded_weight = x @ weight.bfloat16().float().T
        bf16_cpu = (x.bfloat16() @ weight.bfloat16().T).float()
        initial_bf16_cpu = (x.bfloat16() @ tensors['initial_weight'].bfloat16().T).float()
    summaries = {name: summarize(cache, logits) for name, logits in
                 (('trained_fp32', fp32), ('rounded_weight_fp32_matmul', rounded_weight),
                  ('bf16_input_weight_cpu_matmul', bf16_cpu))}
    changed = []
    for index, (a, b) in enumerate(zip(summaries['trained_fp32']['rows'],
                                       summaries['bf16_input_weight_cpu_matmul']['rows'], strict=True)):
        if a['selected_index'] != b['selected_index']:
            changed.append({'row_index': index, 'parent_sample_id': a['parent_sample_id'],
                            'variant': a['variant'], 'order': a['order'],
                            'target_index': a['target_index'],
                            'fp32_selected_index': a['selected_index'],
                            'bf16_cpu_selected_index': b['selected_index'],
                            'fp32_logits': a['logits'], 'bf16_cpu_logits': b['logits']})
    native = tensors['native_logits']
    return {'schema_version': 'tancho-decision-readout-precision-audit-1.0',
            'validation_cache_sha256': meta['sha256'], 'head_sha256': head_meta['sha256'],
            'rows': len(meta['rows']),
            'scores': {name: {key: summary[key] for key in
                              ('exact_accuracy', 'clean_accuracy', 'clean_false_rejection_rate',
                               'reject_explicit_recall')} |
                       {'ties': sum(row['selected_index'] is None for row in summary['rows'])}
                       for name, summary in summaries.items()},
            'selection_changes': changed,
            'initial_head_cpu_bf16_vs_native': {
                'max_abs_logit_difference': float((initial_bf16_cpu - native).abs().max()),
                'mean_abs_logit_difference': float((initial_bf16_cpu - native).abs().mean()),
                'different_argmax_count': int((initial_bf16_cpu.argmax(1) != native.argmax(1)).sum())},
            'interpretation': 'CPU BF16 matmul diagnostic; trained-head native GPU BF16 was not run.',
            'native_equivalence_claim': False, 'release_qualified': False}


def audit_order_consensus(cache, head):
    import torch
    meta, tensors = cache
    head_meta, weight = head
    require(meta['dataset']['split'] == 'validation' and meta['execution_kind'] == 'native_cosmos'
            and head_meta['binding'] == meta['binding']
            and weight.shape == tensors['initial_weight'].shape,
            'Order audit requires bound native validation features')
    with torch.no_grad():
        logits = (tensors['features'].bfloat16() @ weight.bfloat16().T).float()
    require(bool(torch.isfinite(logits).all()), 'Order audit logits are not finite')
    groups = defaultdict(list)
    for row, scores in zip(meta['rows'], logits.tolist(), strict=True):
        highest = max(scores)
        selected = None if scores.count(highest) != 1 else scores.index(highest)
        candidates = row['request']['candidates']
        groups[(row['parent_sample_id'], row['variant'])].append({
            'order': row['order'],
            'selected_candidate_id': None if selected is None else candidates[selected]['candidate_id'],
            'target_candidate_id': candidates[row['target_index']]['candidate_id']})
    expected_orders = set(itertools.permutations(range(3)))
    rows = []
    for (parent, variant), choices in sorted(groups.items()):
        by_order = {tuple(choice['order']): choice for choice in choices}
        require(len(choices) == 6 and set(by_order) == expected_orders,
                'Order audit requires all six candidate permutations')
        target_ids = {choice['target_candidate_id'] for choice in choices}
        require(len(target_ids) == 1, 'Order audit target changed across permutations')
        first, second = by_order[(0, 1, 2)], by_order[(2, 1, 0)]
        a, b = first['selected_candidate_id'], second['selected_candidate_id']
        pair_agrees = a is not None and a == b
        all_ids = [choice['selected_candidate_id'] for choice in choices]
        rows.append({'parent_sample_id': parent, 'variant': variant,
                     'target_candidate_id': first['target_candidate_id'],
                     'pair_selected_candidate_ids': [a, b],
                     'pair_agrees': pair_agrees,
                     'pair_correct': pair_agrees and a == first['target_candidate_id'],
                     'pair_agrees_wrong': pair_agrees and a != first['target_candidate_id'],
                     'pair_error_count': sum(value != first['target_candidate_id'] for value in (a, b)),
                     'all_six_agree': all_ids[0] is not None and len(set(all_ids)) == 1,
                     'all_six_correct': all(value == first['target_candidate_id'] for value in all_ids)})
    def summary(variant):
        selected = [row for row in rows if row['variant'] == variant]
        return {'parents': len(selected), 'pair_agreement': sum(r['pair_agrees'] for r in selected),
                'pair_correct': sum(r['pair_correct'] for r in selected),
                'pair_agrees_wrong': sum(r['pair_agrees_wrong'] for r in selected),
                'pair_disagreement': sum(not r['pair_agrees'] for r in selected),
                'single_order_errors': sum(r['pair_error_count'] for r in selected),
                'all_six_agreement': sum(r['all_six_agree'] for r in selected),
                'all_six_correct': sum(r['all_six_correct'] for r in selected)}
    return {'schema_version': 'tancho-decision-order-consensus-audit-1.0',
            'validation_cache_sha256': meta['sha256'], 'head_sha256': head_meta['sha256'],
            'projection': 'cpu_bf16_matmul', 'pair_orders': [[0, 1, 2], [2, 1, 0]],
            'summary': {variant: summary(variant) for variant in ('clean', 'uniform_occlusion_v1')},
            'rows': rows, 'native_equivalence_claim': False,
            'release_qualified': False, 'flight_authorized': False}


def compatible(train, validation):
    import torch
    a, ta = train
    b, tb = validation
    require(a['dataset']['split'] == 'train' and b['dataset']['split'] == 'validation', 'Expected train and validation caches')
    require(a['binding'] == b['binding'] and a['execution_kind'] == b['execution_kind']
            and a['dataset']['source_manifest_sha256'] == b['dataset']['source_manifest_sha256']
            and torch.equal(ta['initial_weight'], tb['initial_weight']), 'Feature model or source binding differs')
    require(not {r['parent_sample_id'] for r in a['rows']} & {r['parent_sample_id'] for r in b['rows']},
            'Training and validation parents overlap')


def _optimize(x, targets, initial, *, steps, learning_rate, target_nll):
    import torch
    weight = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([weight], lr=learning_rate, weight_decay=0)
    history, reached = [], False
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        before = torch.nn.functional.cross_entropy(x @ weight.T, targets)
        require(bool(torch.isfinite(before)), 'Nonfinite readout loss')
        before.backward()
        norm = torch.nn.utils.clip_grad_norm_([weight], 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            logits = x @ weight.T
            after = torch.nn.functional.cross_entropy(logits, targets)
            require(bool(torch.isfinite(weight).all()) and bool(torch.isfinite(after)), 'Nonfinite readout update')
            correct = bool((logits.argmax(dim=1) == targets).all())
            competitors = logits.clone().scatter_(1, targets[:, None], -torch.inf)
            margin = (logits.gather(1, targets[:, None]).squeeze(1) - competitors.max(dim=1).values).min()
            history.append(dict(step=step + 1, nll_before=float(before.detach()), nll_after=float(after),
                                gradient_norm_before_clip=float(norm), minimum_target_margin=float(margin),
                                update_l2=float((weight - initial).norm())))
            reached = correct and float(margin) > 0 and float(after) <= target_nll
            if reached:
                break
    return weight.detach().clone(), history, reached


def _order_decisions(rows, logits):
    """Per-forward choices plus a per-image choice summed over orders by candidate identity."""
    names = ('A', 'B', 'reject')
    forwards, sums, labels = [], {}, {}
    for row, values in zip(rows, logits, strict=True):
        top = max(values)
        selected = values.index(top) if values.count(top) == 1 else None
        forwards.append({'parent_sample_id': row['parent_sample_id'], 'order': row['order'],
                         'target_index': row['target_index'], 'selected_index': selected,
                         'logits': values})
        total = sums.setdefault(row['parent_sample_id'], [0.0, 0.0, 0.0])
        for slot, candidate in enumerate(row['order']):
            total[candidate] += values[slot]
        labels[row['parent_sample_id']] = row['order'][row['target_index']]
    summed = {}
    for parent, total in sums.items():
        top = max(total)
        summed[parent] = total.index(top) if total.count(top) == 1 else None
    return {'per_forward_matches': sum(r['selected_index'] == r['target_index'] for r in forwards),
            'forwards': len(forwards),
            'ties': sum(r['selected_index'] is None for r in forwards),
            'order_summed_matches': sum(summed[p] == labels[p] for p in summed),
            'order_summed_ties': sum(value is None for value in summed.values()),
            'parents': len(summed),
            'selected_by_slot': [sum(r['selected_index'] == slot for r in forwards) for slot in range(3)],
            'selected_by_candidate': {name: sum(r['selected_index'] is not None
                                                and r['order'][r['selected_index']] == index
                                                for r in forwards)
                                      for index, name in enumerate(names)},
            'order_summed_by_parent': {p: {'selected': None if summed[p] is None else names[summed[p]],
                                           'label': names[labels[p]]} for p in sorted(summed)},
            'rows': forwards}


def merge_caches(caches):
    """Concatenate validated shard caches of one model binding into one train cache."""
    import torch
    metas = [meta for meta, _ in caches]
    require(len(caches) >= 2 and all(m['binding'] == metas[0]['binding'] for m in metas)
            and all(m['dataset']['split'] == 'train' for m in metas)
            and all(torch.equal(t['initial_weight'], caches[0][1]['initial_weight']) for _, t in caches),
            'Shard caches do not share one binding')
    parents = [r['parent_sample_id'] for m in metas for r in m['rows']]
    require(len({(r['parent_sample_id'], tuple(r['order'])) for m in metas for r in m['rows']}) == len(parents),
            'Shard caches overlap')
    rows = [r for m in metas for r in m['rows']]
    meta = {'dataset': {'split': 'train', 'rows': rows}, 'rows': rows, 'binding': metas[0]['binding'],
            'shard_cache_sha256': [m['sha256'] for m in metas]}
    meta['sha256'] = canonical_sha256(meta['shard_cache_sha256'])
    tensors = {key: torch.cat([t[key] for _, t in caches]) for key in ('features', 'native_logits')}
    tensors['initial_weight'] = caches[0][1]['initial_weight']
    return meta, tensors


def leave_one_parent_out(cache, *, steps=300, learning_rate=.001, target_nll=.02):
    """Fit the three-row readout without each image in turn and score only that image."""
    import torch
    meta, tensors = cache
    require(meta['dataset']['split'] == 'train' and {r['variant'] for r in meta['rows']} == {'clean'},
            'Leave-one-out uses clean train features only')
    require(type(steps) is int and 1 <= steps <= 10000 and math.isfinite(learning_rate)
            and 0 < learning_rate <= .1 and math.isfinite(target_nll) and 0 <= target_nll <= 1,
            'Invalid CPU fitting bounds')
    rows = meta['rows']
    parents = sorted({r['parent_sample_id'] for r in rows})
    require(len(parents) >= 2, 'Leave-one-out needs at least two images')
    x = tensors['features'].detach()
    targets = torch.tensor([r['target_index'] for r in rows])
    initial = tensors['initial_weight'].clone()
    held_logits = [None] * len(rows)
    folds = []
    for parent in parents:
        held = [i for i, r in enumerate(rows) if r['parent_sample_id'] == parent]
        train = [i for i, r in enumerate(rows) if r['parent_sample_id'] != parent]
        weight, history, reached = _optimize(x[train], targets[train], initial, steps=steps,
                                             learning_rate=learning_rate, target_nll=target_nll)
        with torch.no_grad():
            for index, values in zip(held, (x[held] @ weight.T).tolist(), strict=True):
                held_logits[index] = values
        folds.append({'held_out_parent': parent,
                      'train_parents': sorted({rows[i]['parent_sample_id'] for i in train}),
                      'held_out_rows': held, 'train_row_count': len(train), 'steps': len(history),
                      'stopping_reason': 'train_target_reached' if reached else 'step_limit',
                      'final_train_nll': history[-1]['nll_after'],
                      'update_l2': history[-1]['update_l2']})
    with torch.no_grad():
        zero_shot = (x @ initial.T).tolist()
    return {'schema_version': 'tancho-decision-readout-loo-1.0', 'cache_sha256': meta['sha256'],
            'algorithm': 'full_batch_adamw_from_exported_rows', 'steps': steps,
            'learning_rate': learning_rate, 'target_nll': target_nll,
            'probe': _order_decisions(rows, held_logits),
            'zero_shot': _order_decisions(rows, zero_shot),
            'native': _order_decisions(rows, tensors['native_logits'].tolist()),
            'folds': folds, 'gpu_executed': False, 'release_qualified': False}


def fit_clean(train, output, *, steps=300, learning_rate=.001, target_nll=.02):
    """Fit the three label rows on clean features only (stage C caches), for runtime binding.

    Without occluded pairs the reject row is never a target, so the head does not learn to
    reject; the head records that. Same optimizer and defaults as fit and leave-one-out."""
    import torch
    meta, tensors = train
    require(meta['dataset']['split'] == 'train' and {r['variant'] for r in meta['rows']} == {'clean'},
            'Clean fitting uses clean train features only')
    require(type(steps) is int and 1 <= steps <= 10000 and math.isfinite(learning_rate)
            and 0 < learning_rate <= .1 and math.isfinite(target_nll) and 0 <= target_nll <= 1,
            'Invalid CPU fitting bounds')
    output = Path(output)
    require(not output.exists(), 'Use a fresh readout output')
    x = tensors['features'].detach()
    targets = torch.tensor([r['target_index'] for r in meta['rows']])
    final, history, reached = _optimize(x, targets, tensors['initial_weight'].clone(), steps=steps,
                                        learning_rate=learning_rate, target_nll=target_nll)
    output.mkdir(parents=True)
    torch.save(final, output / 'weight.pt')
    catalogs = {r['request']['catalog_revision'] for r in meta['rows']}
    require(len(catalogs) == 1, 'Clean fitting needs one catalog')
    head = dict(schema_version=HEAD, binding=meta['binding'], train_cache_sha256=meta['sha256'],
                execution_kind='native_cosmos', weight_sha256=digest(output / 'weight.pt'),
                dtype='float32', bias=False, trainable_coefficients=final.numel(), release_qualified=False)
    head['sha256'] = canonical_sha256(head)
    write_json(output / 'head.json', head)
    _, restored = load_head(output)
    require(torch.equal(final, restored), 'FP32 readout reload changed weights')
    with torch.no_grad():
        chosen = (x @ restored.T).argmax(1)
    report = dict(schema_version='tancho-decision-cpu-fit-clean-1.0', algorithm='full_batch_adamw',
                  catalog_revision=catalogs.pop(), rows=len(meta['rows']),
                  parents=len({r['parent_sample_id'] for r in meta['rows']}),
                  steps=len(history), learning_rate=learning_rate, target_nll=target_nll,
                  stopping_reason='train_target_reached' if reached else 'step_limit',
                  final_train_nll=history[-1]['nll_after'],
                  train_row_accuracy=round(float((chosen == targets).float().mean()), 4),
                  reject_supervised=False, head_sha256=head['sha256'],
                  gpu_executed=False, release_qualified=False)
    write_json(output / 'report.json', report)
    return report


def fit(train, output, *, validation=None, steps=300, learning_rate=.001, target_nll=.02):
    import torch
    meta, tensors = train
    require(meta['dataset']['split'] == 'train', 'Readout fitting requires train split only')
    require({r['variant'] for r in meta['rows']} == {'clean', 'uniform_occlusion_v1'},
            'Readout fitting requires paired clean and reject examples')
    parents = {r['parent_sample_id'] for r in meta['rows']}
    require(all({r['variant'] for r in meta['rows'] if r['parent_sample_id'] == parent}
                == {'clean', 'uniform_occlusion_v1'} for parent in parents), 'Missing paired parent supervision')
    require(type(steps) is int and 1 <= steps <= 10000 and math.isfinite(learning_rate)
            and 0 < learning_rate <= .1 and math.isfinite(target_nll) and 0 <= target_nll <= 1,
            'Invalid CPU fitting bounds')
    output = Path(output)
    require(not output.exists(), 'Use a fresh readout output')
    if validation is not None:
        compatible(train, validation)
    x = tensors['features'].detach()
    targets = torch.tensor([r['target_index'] for r in meta['rows']])
    initial = tensors['initial_weight'].clone()
    final, history, reached = _optimize(x, targets, initial, steps=steps,
                                        learning_rate=learning_rate, target_nll=target_nll)
    output.mkdir(parents=True)
    torch.save(final, output / 'weight.pt')
    head = dict(schema_version=HEAD, binding=meta['binding'], train_cache_sha256=meta['sha256'],
                execution_kind=meta['execution_kind'], weight_sha256=digest(output / 'weight.pt'),
                dtype='float32', bias=False, trainable_coefficients=final.numel(), release_qualified=False)
    head['sha256'] = canonical_sha256(head)
    write_json(output / 'head.json', head)
    _, restored = load_head(output)
    require(torch.equal(final, restored) and torch.equal(x @ final.T, x @ restored.T), 'FP32 readout reload changed logits')
    report = dict(schema_version='tancho-decision-cpu-fit-1.0', algorithm='full_batch_adamw',
                  steps=len(history), learning_rate=learning_rate, target_nll=target_nll,
                  stopping_reason='train_target_reached' if reached else 'step_limit',
                  history=history, train=evaluate(train, restored),
                  validation=evaluate(validation, restored) if validation is not None else None,
                  reload_logits_exact=True, head_sha256=head['sha256'],
                  gpu_executed=False, release_qualified=False)
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('capture')
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument('--pack', type=Path)
    source.add_argument('--stage-a-images', type=Path)
    extract.add_argument('--stage-a-shard', type=int, default=0)
    extract.add_argument('--stage-a-shards', type=int, default=1)
    extract.add_argument('--stage-a-model-revision')
    extract.add_argument('--split', choices=('train', 'validation', 'both'), required=True)
    extract.add_argument('--limit-samples', type=int, default=1)
    extract.add_argument('--output', type=Path, required=True)
    extract.add_argument('--execute-native', action='store_true')
    extract.add_argument('--expected-data-sha256')
    extract.add_argument('--weak-review', type=Path)
    for key in ('base', 'export', 'framework', 'base-sha256', 'export-sha256', 'framework-revision'):
        extract.add_argument('--' + key)
    training = commands.add_parser('fit')
    training.add_argument('--train-cache', type=Path, required=True)
    training.add_argument('--validation-cache', type=Path)
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--steps', type=int, default=300)
    training.add_argument('--learning-rate', type=float, default=.001)
    training.add_argument('--target-nll', type=float, default=.02)
    precision = commands.add_parser('audit-precision')
    precision.add_argument('--validation-cache', type=Path, required=True)
    precision.add_argument('--head', type=Path, required=True)
    precision.add_argument('--output', type=Path, required=True)
    loo = commands.add_parser('loo')
    loo.add_argument('--cache', type=Path, action='append', required=True,
                     help='repeat for the shards of one capture; they are merged before fitting')
    loo.add_argument('--output', type=Path, required=True)
    loo.add_argument('--steps', type=int, default=300)
    loo.add_argument('--learning-rate', type=float, default=.001)
    loo.add_argument('--target-nll', type=float, default=.02)
    clean = commands.add_parser('fit-clean')
    clean.add_argument('--cache', type=Path, action='append', required=True,
                       help='repeat for the shards of one capture; they are merged before fitting')
    clean.add_argument('--output', type=Path, required=True)
    clean.add_argument('--steps', type=int, default=300)
    clean.add_argument('--learning-rate', type=float, default=.001)
    clean.add_argument('--target-nll', type=float, default=.02)
    order_audit = commands.add_parser('audit-order-consensus')
    order_audit.add_argument('--validation-cache', type=Path, required=True)
    order_audit.add_argument('--head', type=Path, required=True)
    order_audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit-clean':
        import torch
        torch.set_num_threads(1)
        caches = [load_cache(path) for path in args.cache]
        report = fit_clean(caches[0] if len(caches) == 1 else merge_caches(caches), args.output,
                           steps=args.steps, learning_rate=args.learning_rate, target_nll=args.target_nll)
        print(json.dumps({k: report[k] for k in ('rows', 'parents', 'steps', 'stopping_reason',
                                                 'final_train_nll', 'train_row_accuracy')}))
        return
    if args.command == 'loo':
        import torch
        torch.set_num_threads(1)
        require(not args.output.exists(), 'Use a fresh leave-one-out output')
        caches = [load_cache(path) for path in args.cache]
        report = leave_one_parent_out(caches[0] if len(caches) == 1 else merge_caches(caches), steps=args.steps,
                                      learning_rate=args.learning_rate, target_nll=args.target_nll)
        write_json(args.output, report)
        print(json.dumps({name: {key: report[name][key] for key in
                                 ('per_forward_matches', 'forwards', 'order_summed_matches', 'parents')}
                          for name in ('probe', 'zero_shot', 'native')}))
        return
    if args.command == 'audit-order-consensus':
        require(not args.output.exists(), 'Use a fresh order audit output')
        report = audit_order_consensus(load_cache(args.validation_cache), load_head(args.head))
        write_json(args.output, report)
        print(json.dumps(report['summary']))
        return
    if args.command == 'audit-precision':
        require(not args.output.exists(), 'Use a fresh precision audit output')
        report = audit_precision(load_cache(args.validation_cache), load_head(args.head))
        write_json(args.output, report)
        print(json.dumps({'rows': report['rows'], 'selection_changes': len(report['selection_changes']),
                          'native_executed': False}))
        return
    if args.command == 'fit':
        import torch
        torch.set_num_threads(1)
        report = fit(load_cache(args.train_cache), args.output,
                     validation=load_cache(args.validation_cache) if args.validation_cache else None,
                     steps=args.steps, learning_rate=args.learning_rate, target_nll=args.target_nll)
        print(json.dumps({'steps': report['steps'], 'stopping_reason': report['stopping_reason'],
                          'train_nll': report['train']['trained_fp32']['nll'], 'gpu_executed': False}))
        return
    if args.stage_a_images is not None:
        from training.stage_a_decision_data import StageADecisionDataset
        require(args.split == 'train' and args.weak_review is None and args.stage_a_model_revision,
                'Stage A features are one train split bound to a model revision')
        datasets = {'train': StageADecisionDataset(args.stage_a_images, shard=args.stage_a_shard,
                                                   shards=args.stage_a_shards,
                                                   model_revision=args.stage_a_model_revision)}
    elif args.weak_review is not None:
        from tancho.observation_decision import ROAD_SUPPORT_CATALOG
        require(args.split == 'train', 'Weak road features exist only for the reviewed train images')
        datasets = {'train': DecisionDataset(args.pack, 'train', training=True,
                                             catalog_revision=ROAD_SUPPORT_CATALOG,
                                             weak_review=args.weak_review)}
    else:
        splits = ('train', 'validation') if args.split == 'both' else (args.split,)
        datasets = {split: DecisionDataset(args.pack, split, limit_samples=args.limit_samples,
                                          reject_mode='uniform_occlusion_v1', all_orders=True) for split in splits}
    require(all(len(data) > 0 for data in datasets.values()), 'No covered feature samples')
    data_hashes = {split: data.fingerprint for split, data in datasets.items()}
    if args.expected_data_sha256 is not None:
        require(canonical_sha256(data_hashes) == args.expected_data_sha256, 'Feature datasets changed')
    require(not args.output.exists(), 'Use a fresh feature output')
    forward_count = sum(len(data) for data in datasets.values())
    if not args.execute_native:
        args.output.mkdir(parents=True)
        write_json(args.output / 'plan.json', dict(datasets={s: d.manifest for s, d in datasets.items()},
                   dataset_sha256=data_hashes, forward_count=forward_count,
                   execution_requested=False, cloud_created=False, release_qualified=False))
        print(json.dumps({'forward_count': forward_count, 'native_executed': False}))
        return
    from tancho.observation_decision_native import NativeDecisionBackend
    settings = {k: getattr(args, k) for k in ('base', 'export', 'framework', 'base_sha256', 'export_sha256', 'framework_revision')}
    require(all(settings.values()), 'Feature extraction requires pinned model settings')
    first = next(iter(datasets.values()))
    backend = NativeDecisionBackend(model_revision=first[0]['context']['model_revision'], **settings)
    weight = read_export_rows(args.export, backend.slots)
    caches = {split: capture(data, backend, weight, args.output / split) for split, data in datasets.items()}
    write_json(args.output / 'report.json', dict(schema_version='tancho-decision-feature-capture-1.0',
               binding=backend.binding, dataset_sha256=data_hashes, forward_count=forward_count,
               cache_sha256={s: c[0]['sha256'] for s, c in caches.items()},
               native_executed=True, release_qualified=False))
    print(json.dumps({'rows': forward_count, 'native_executed': True}))


if __name__ == '__main__':
    main()
