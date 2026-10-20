"""Frozen, single-attempt evaluation of both roles on a context-bearing pack."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tancho.mission_context import canonical_sha256
from training.observation_program_loader import ObservationProgramDataset, verify_pack
from training.score_observation_program import score_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh evaluation output directory')
    manifest = verify_pack(args.pack, purpose='evaluation')
    thresholds = json.loads(args.thresholds.read_text())
    if thresholds.get('schema_version') != 'tancho-observation-program-eval-thresholds-1.0' \
            or thresholds.get('pack_manifest_sha256') != manifest['manifest_sha256']:
        raise ValueError('Evaluation thresholds are not bound to this frozen pack')
    args.output.mkdir(parents=True)
    import torch
    from transformers import AutoModelForImageTextToText
    import cosmos_framework.model.generator.reasoner.cosmos3_edge
    from cosmos_framework.data.generator.processors import build_processor
    from tancho.edge_chat import install
    from tancho.observation_intent import parse_intent_output, validate_and_bind_intents
    from tancho.observation_program import parse_program_output, validate_observation_program

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Frozen evaluation requires exactly one CUDA GPU')
    install(); torch.manual_seed(42)
    model, loading = AutoModelForImageTextToText.from_pretrained(
        str(args.export), torch_dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in
           ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Export reload changed tensors')
    model.eval(); processor = build_processor(tokenizer_type=str(args.base), config_variant='hf')
    dataset = ObservationProgramDataset(args.pack, 'test', purpose='evaluation')
    predictions, records, receipts = {}, [], []
    started = time.monotonic()
    for index, row in enumerate(dataset.rows):
        if 'context' not in row:
            raise ValueError('Evaluation row lacks validator context')
        item = dataset[index]
        inputs = processor.apply_chat_template([item['texts'][0]], tokenize=True,
                                               add_generation_prompt=True, return_tensors='pt')
        tensor_inputs = {key: (value.unsqueeze(0) if key in ('input_ids', 'attention_mask')
                               and value.ndim == 1 else value).to('cuda:0')
                         for key, value in inputs.items() if torch.is_tensor(value)}
        case_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**tensor_inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=False,
                                       eos_token_id=11, pad_token_id=0)
        tokens = generated[0, tensor_inputs['input_ids'].shape[-1]:].detach().cpu().tolist()
        raw = processor.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        predictions[row['sample_id']] = raw
        valid, error = False, None
        try:
            if row['role'] == 'reasoner':
                parsed = parse_intent_output(raw)
                validate_and_bind_intents(parsed, row['context'],
                                          now_ms=row['context']['captured_at_ms'], raw_text=raw)
            else:
                parsed = parse_program_output(raw)
                validate_observation_program(parsed, row['context'], row['verified_intents'],
                                             now_ms=row['context']['captured_at_ms'], raw_text=raw)
            valid = True
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        raw_path = args.output / (row['sample_id'] + '.txt'); raw_path.write_text(raw)
        receipts.append({'sample_id': row['sample_id'], 'role': row['role'],
                         'single_attempt': True, 'valid': valid, 'error': error,
                         'tokens': len(tokens), 'seconds': time.monotonic() - case_started,
                         'raw_sha256': hashlib.sha256(raw.encode()).hexdigest()})
        records.append({'sample_id': row['sample_id'], 'role': row['role'],
                        'mission_profile': row['mission_profile'],
                        'visual_pair_id': row['visual_pair_id'], 'context': row['context'],
                        'verified_intents': row['verified_intents'],
                        'target': json.loads(row['target'])})
        del tensor_inputs, generated
    score = score_predictions(records, predictions)
    gates = {
        'schema_valid_rate': score['overall']['schema_valid_rate'] >=
                             thresholds['minimum_schema_valid_rate'],
        'exact_accuracy': score['overall']['exact_accuracy'] >=
                          thresholds['minimum_exact_accuracy'],
        'paired_visual_sensitivity': score['paired_visual_sensitivity']['rate'] >=
                                     thresholds['minimum_paired_visual_sensitivity'],
        'no_invalid_outputs': all(row['valid'] for row in receipts),
    }
    raw_hash = canonical_sha256({row['sample_id']: row['raw_sha256'] for row in receipts})
    report = {'schema_version': 'tancho-observation-program-frozen-evaluation-1.0',
              'passed': all(gates.values()), 'pack_manifest_sha256': manifest['manifest_sha256'],
              'thresholds_sha256': hashlib.sha256(args.thresholds.read_bytes()).hexdigest(),
              'frozen_before_inference': True, 'single_attempt_no_repair': True,
              'loading_info': loading, 'score': score, 'gates': gates,
              'receipts': receipts, 'model_raw_outputs_sha256': raw_hash,
              'elapsed_seconds': time.monotonic() - started}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('TANCHO_FROZEN_EVAL '+json.dumps({'passed': report['passed'],
          'overall': score['overall'], 'paired': score['paired_visual_sensitivity']}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
