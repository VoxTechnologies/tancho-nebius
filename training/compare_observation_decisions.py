"""Diagnostic-only native Cosmos JSON/readout comparison; never a release gate."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tancho.contracts import ContractError, require
from tancho.mission_context import canonical_sha256
from tancho.observation_decision import (build_request, candidate_coverage, label_token_ids,
                                         make_receipt, replay_decision)
from tancho.observation_program import parse_program_output, validate_observation_program
from tancho.observation_program_prompts import generator_prompt
from training.observation_program_loader import ObservationProgramDataset


def bound_readout_head(path, expected_sha256, binding):
    from training.fit_decision_readout import load_head
    meta, weight = load_head(path)
    require(meta['sha256'] == expected_sha256 and meta['binding'] == binding
            and meta['execution_kind'] == 'native_cosmos' and meta['bias'] is False
            and meta['dtype'] == 'float32', 'Readout head binding changed')
    return weight


def project_readout(model, inputs, weight, torch):
    # Keep the export's lm_head untouched so JSON generation remains the original baseline.
    import torch.nn.functional as functional
    require('valid_input_len' not in inputs, 'Three-row projection does not support packed inputs')
    keys = ('input_ids', 'position_ids', 'attention_mask', 'pixel_values',
            'pixel_values_videos', 'image_grid_thw', 'video_grid_thw')
    backbone_inputs = {key: inputs[key] for key in keys if key in inputs}
    hidden = model.model(**backbone_inputs, use_cache=False)[0][:, -1, :]
    return functional.linear(hidden.to(weight.dtype), weight).float()


def directory_fingerprint(root):
    """Include model configuration as well as tensors; ignore HF download bookkeeping."""
    root = Path(root)
    rows = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(root).parts:
            with path.open('rb') as stream:
                rows[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(bool(rows), 'Empty model directory')
    return canonical_sha256(rows)


def select_rows(rows, *, limit):
    require(type(limit) is int and 1 <= limit <= 32, 'Diagnostic limit must be 1..32')
    eligible, skipped = [], []
    for index, row in enumerate(rows):
        if row['role'] != 'generator':
            skipped.append({'sample_id': row['sample_id'], 'reason': 'not_generator'})
            continue
        try:
            from tancho.observation_decision import candidates
            candidates(row['context'], row['verified_intents'],
                       now_ms=row['context']['captured_at_ms'])
        except ContractError as exc:
            # Only explicit scope exclusions may be skipped; corrupt inputs must stop the run.
            if str(exc) not in ('D0 requires one intent', 'Unsupported D0 question'):
                raise
            skipped.append({'sample_id': row['sample_id'], 'reason': str(exc)})
            continue
        if len(eligible) < limit:
            eligible.append(index)
        else:
            skipped.append({'sample_id': row['sample_id'], 'reason': 'diagnostic_limit'})
    require(bool(eligible), 'No supported generator rows')
    return eligible, skipped


def prepare_inputs(processor, conversation, prompt, torch, *, max_input_tokens):
    messages = deepcopy(conversation)
    require(len(messages) >= 1 and messages[0]['role'] == 'user', 'Unexpected conversation')
    messages[0]['content'][1]['text'] = prompt
    for content in messages[0]['content']:
        if content['type'] != 'video':
            continue
        frames = content['video']
        max_frame_pixels = 1920 * 1080 // max(1, len(frames))
        resized = []
        for frame in frames:
            if not hasattr(frame, 'width') or not hasattr(frame, 'height') \
                    or frame.width * frame.height <= max_frame_pixels:
                resized.append(frame)
                continue
            # Bound all video frames together because visual tokens share one decision budget.
            from PIL import Image
            scale = math.sqrt(max_frame_pixels / (frame.width * frame.height))
            frame.thumbnail((max(1, int(frame.width * scale)),
                             max(1, int(frame.height * scale))), Image.Resampling.LANCZOS)
            resized.append(frame)
        content['video'] = resized
    inputs = processor.apply_chat_template([messages[0]], tokenize=True,
                                           add_generation_prompt=True, return_tensors='pt')
    tensors = {key: (value.unsqueeze(0) if key in ('input_ids', 'attention_mask')
                    and value.ndim == 1 else value).to('cuda:0')
               for key, value in inputs.items() if torch.is_tensor(value)}
    ids = tensors['input_ids']
    require(ids.ndim == 2 and ids.shape[0] == 1 and 0 < ids.shape[-1] <= max_input_tokens,
            'Input token limit exceeded')
    require(any(key not in ('input_ids', 'attention_mask', 'position_ids') for key in tensors),
            'No multimodal processor tensors')
    mask = tensors.get('attention_mask')
    if mask is not None:
        require(bool((mask == 1).all().item()), 'D0 requires unpadded inputs')
    return tensors


def tensor_fingerprint(tensors, torch):
    rows = {}
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().contiguous().cpu()
        rows[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                      'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return {'tensors': rows, 'sha256': canonical_sha256(rows)}


def summarize(records):
    covered = [r for r in records if r['oracle_covered']]
    metrics = {}
    for kind in ('json', 'readout'):
        values = [r[kind]['inference_s'] for r in records]
        ordered = sorted(values)
        totals = [r[kind]['preprocessing_s'] + r[kind]['inference_s']
                  + (r['candidate_build_s'] + r[kind].get('conversion_s', 0)
                     if kind == 'readout' else r[kind].get('conversion_s', 0))
                  for r in records if 'preprocessing_s' in r[kind]]
        metrics[kind] = {'exact_accuracy': sum(r[kind]['exact'] for r in records) / len(records),
                         'p50_inference_s': statistics.median(values),
                         'p95_inference_s': ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
                         'max_inference_s': max(values),
                         'p50_decision_compute_s': statistics.median(totals) if totals else None}
    return {'record_count': len(records), 'oracle_covered_count': len(covered),
            'oracle_coverage': len(covered) / len(records),
            'readout_accuracy_when_covered': (sum(r['readout']['exact'] for r in covered) / len(covered)
                                               if covered else None),
            'reject_count': sum(r['readout']['rejected'] for r in records), 'metrics': metrics}


def run(args):
    require(not args.output.exists(), 'Use a fresh diagnostic output directory')
    require(1 <= args.max_input_tokens <= 16384 and 1 <= args.max_new_tokens <= 1536,
            'Invalid token bounds')
    require(1 <= args.repetitions <= 5, 'Diagnostic repetitions must be 1..5')
    require((args.readout_head is None) == (args.readout_head_sha256 is None),
            'Readout head path and digest are required together')
    dataset = ObservationProgramDataset(args.pack, args.split, purpose='evaluation')
    indices, skipped = select_rows(dataset.rows, limit=args.limit)
    require(all(dataset.rows[i]['context']['model_revision'] == args.model_revision for i in indices),
            'Context checkpoint differs from requested model revision')
    actual_export = directory_fingerprint(args.export)
    actual_base = directory_fingerprint(args.base)
    require(actual_export == args.export_sha256 and actual_base == args.base_sha256,
            'Model directory fingerprint changed')
    import subprocess
    actual_framework = subprocess.check_output(['git', '-C', str(args.framework), 'rev-parse', 'HEAD'],
                                                text=True).strip()
    require(actual_framework == args.framework_revision, 'Framework revision changed')
    require(not subprocess.check_output(['git', '-C', str(args.framework), 'status', '--porcelain',
                                         '--untracked-files=no'], text=True).strip(),
            'Framework has modified tracked files')
    args.output.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')

    import torch
    from transformers import AutoModelForImageTextToText
    import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401
    from cosmos_framework.data.generator.processors import build_processor
    from tancho.edge_chat import install

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one CUDA GPU required')
    install()
    torch.manual_seed(42)
    load_started = time.perf_counter()
    model, loading = AutoModelForImageTextToText.from_pretrained(
        str(args.export), torch_dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    require(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys',
                                            'error_msgs')), 'Checkpoint loading mismatch')
    model.eval()
    processor = build_processor(tokenizer_type=str(args.base), config_variant='hf')
    tokenizer = processor.processor.tokenizer
    slots = label_token_ids(tokenizer)
    readout_weight = None
    if args.readout_head is not None:
        binding = {'model_revision': args.model_revision, 'export_sha256': args.export_sha256,
                   'base_sha256': args.base_sha256, 'framework_revision': args.framework_revision,
                   'precision': 'bf16', 'label_token_ids': slots}
        weight = bound_readout_head(args.readout_head, args.readout_head_sha256, binding)
        require(weight.shape == (3, model.lm_head.weight.shape[1]),
                'Readout head dimensions changed')
        readout_weight = weight.to(device=model.lm_head.weight.device,
                                   dtype=model.lm_head.weight.dtype)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    records, warmups = [], []
    for index in indices:
        row = dataset.rows[index]
        item = dataset[index]
        frames = {key: (dataset.root / path).read_bytes()
                  for key, path in zip(row['frames_sha256'], row['frames'], strict=True)}
        context, intents = row['context'], row['verified_intents']
        baseline_prompt = generator_prompt(context, intents)['text']
        require(baseline_prompt == row['prompt'], 'Baseline prompt is not the current Generator contract')
        for order in ((0, 1, 2), (2, 1, 0)):
            candidate_started = time.perf_counter()
            request = build_request(context, intents, frames, now_ms=context['captured_at_ms'], order=order)
            candidate_build_s = time.perf_counter() - candidate_started
            gold = json.loads(row['target'])
            for repetition in range(args.repetitions):
                pair = {'sample_id': row['sample_id'], 'order': list(order), 'repetition': repetition,
                        'oracle_covered': candidate_coverage(request, gold),
                        'candidate_build_s': candidate_build_s,
                        'canonical_frames': item['canonical_frames'], 'request': request}
                kinds = ('json', 'readout') if len(records) % 2 == 0 else ('readout', 'json')
                for kind in kinds:
                    prompt = baseline_prompt if kind == 'json' else request['prompt']
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    inputs = prepare_inputs(processor, item['texts'], prompt, torch,
                                            max_input_tokens=args.max_input_tokens)
                    torch.cuda.synchronize()
                    preprocessing_s = time.perf_counter() - began
                    proof = tensor_fingerprint(inputs, torch)

                    def infer():
                        with torch.inference_mode():
                            if kind == 'json':
                                return model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                      do_sample=False, use_cache=False,
                                                      eos_token_id=11, pad_token_id=0)
                            if readout_weight is not None:
                                return project_readout(model, inputs, readout_weight, torch)
                            return model(**inputs, use_cache=False, return_dict=True).logits[:, -1, slots]

                    if kind not in {w['kind'] for w in warmups}:
                        torch.cuda.synchronize(); began = time.perf_counter()
                        warm = infer()
                        torch.cuda.synchronize()
                        warmups.append({'kind': kind, 'seconds': time.perf_counter() - began})
                        del warm
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); began = time.perf_counter()
                    output = infer()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - began
                    result = {'inference_s': elapsed, 'preprocessing_s': preprocessing_s,
                              'input_tokens': int(inputs['input_ids'].shape[-1]),
                              'processor_input': proof,
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                    if kind == 'json':
                        conversion_started = time.perf_counter()
                        tokens = output[0, inputs['input_ids'].shape[-1]:].detach().cpu().tolist()
                        raw = tokenizer.decode(tokens, skip_special_tokens=True)
                        result.update(raw_text=raw, output_token_ids=tokens, valid=False, exact=False)
                        try:
                            require(bool(tokens) and tokens[-1] == 11, 'Generation did not reach EOS')
                            parsed = parse_program_output(raw)
                            validate_observation_program(parsed, context, intents,
                                                         now_ms=context['captured_at_ms'], raw_text=raw)
                            result.update(valid=True, exact=parsed == gold)
                        except ContractError as exc:
                            result['validation_error'] = str(exc)
                        result['conversion_s'] = time.perf_counter() - conversion_started
                        result['decision_compute_s'] = (preprocessing_s + elapsed
                                                        + result['conversion_s'])
                    else:
                        conversion_started = time.perf_counter()
                        logits = output[0].float().cpu().tolist()
                        started = context['captured_at_ms']
                        completed = started + math.ceil((preprocessing_s + elapsed) * 1000)
                        receipt = make_receipt(request, logits, slots, started_ms=started,
                                               completed_ms=completed, timeout_ms=120000)
                        converted = replay_decision(request, receipt, context, intents, frames,
                                                    now_ms=completed)
                        result.update(receipt=receipt, conversion=converted,
                                      exact=converted['program'] == gold,
                                      rejected=receipt['outcome'] == 'rejected',
                                      clock='offline_elapsed_from_capture')
                        result['conversion_s'] = time.perf_counter() - conversion_started
                        result['decision_compute_s'] = (candidate_build_s + preprocessing_s + elapsed
                                                        + result['conversion_s'])
                    pair[kind] = result
                    del inputs, output
                records.append(pair)
                # Preserve each completed comparison even when a later native call fails.
                with (args.output / 'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(pair, allow_nan=False) + '\n')
    report = {'schema_version': 'tancho-decision-diagnostic-0.1', 'purpose': 'diagnostic',
              'release_qualified': False, 'flight_authorized': False,
              'readout_projection': 'three_row' if readout_weight is not None else 'full_vocab',
              'baseline_export_unpatched': True,
              'split_usage': 'regression' if args.split == 'test' else 'development',
              'pack_manifest_sha256': dataset.manifest['manifest_sha256'],
              'settings': settings, 'loading_info': loading, 'load_s': load_s,
              'device': torch.cuda.get_device_name(0), 'warmups': warmups,
              'skipped': skipped, 'selected_samples': len(indices),
              'summary': summarize(records)}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'base', 'export', 'framework', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('model-revision', 'framework-revision', 'base-sha256', 'export-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--readout-head', type=Path)
    parser.add_argument('--readout-head-sha256')
    args = parser.parse_args()
    print(json.dumps(run(args)['summary'], indent=2))


if __name__ == '__main__':
    main()
