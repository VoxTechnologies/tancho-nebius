"""Resident native Cosmos candidate readout; GPU dependencies are loaded on demand."""
import hashlib
import io
from pathlib import Path
import subprocess
import threading
import time

from .contracts import require
from .observation_catalog_v06 import check_request
from .observation_decision import label_token_ids

RESPONSE = 'tancho-decision-provider-response-1.0'


def load_native(*, base, export, framework, base_sha256, export_sha256,
                framework_revision, precision='bf16'):
    from training.compare_observation_decisions import directory_fingerprint
    require(precision in ('bf16', 'fp16'), 'Unsupported native precision')
    require(directory_fingerprint(base) == base_sha256
            and directory_fingerprint(export) == export_sha256, 'Native model fingerprint changed')
    revision = subprocess.check_output(['git', '-C', str(framework), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(framework), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True).strip()
    require(revision == framework_revision and not dirty, 'Native Framework changed')
    import torch
    from transformers import AutoModelForImageTextToText
    import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401
    from cosmos_framework.data.generator.processors import build_processor
    from .edge_chat import install
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one CUDA GPU required')
    install()
    model, loading = AutoModelForImageTextToText.from_pretrained(
        str(export), torch_dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
        device_map='cuda:0', attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    require(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys',
                                            'error_msgs')), 'Native checkpoint load mismatch')
    model.eval()
    processor = build_processor(tokenizer_type=str(base), config_variant='hf')
    return model, processor, label_token_ids(processor.processor.tokenizer)


class NativeDecisionBackend:
    def __init__(self, *, model_revision, max_input_tokens=4096,
                 readout_head=None, readout_head_sha256=None,
                 readout_projection='full_vocab', **settings):
        require(type(max_input_tokens) is int and 1 <= max_input_tokens <= 16384, 'Invalid input bound')
        require(readout_projection in ('full_vocab', 'three_row'), 'Unknown readout projection')
        require((readout_head is None) == (readout_head_sha256 is None),
                'Readout head path and digest are required together')
        require(readout_projection != 'three_row' or readout_head is not None,
                'Three-row projection requires a bound readout head')
        head_meta = weight = None
        if readout_head is not None:
            from training.fit_decision_readout import load_head
            head_meta, weight = load_head(readout_head)
            expected = {'model_revision': model_revision, 'export_sha256': settings['export_sha256'],
                        'base_sha256': settings['base_sha256'],
                        'framework_revision': settings['framework_revision'],
                        'precision': settings.get('precision', 'bf16')}
            require(head_meta['sha256'] == readout_head_sha256
                    and all(head_meta['binding'].get(key) == value for key, value in expected.items())
                    and head_meta['execution_kind'] == 'native_cosmos'
                    and head_meta['bias'] is False and head_meta['dtype'] == 'float32',
                    'Readout head does not match native model')
        self.model, self.processor, self.slots = load_native(**settings)
        self.binding = {'model_revision': model_revision, 'export_sha256': settings['export_sha256'],
                        'base_sha256': settings['base_sha256'],
                        'framework_revision': settings['framework_revision'],
                        'precision': settings.get('precision', 'bf16'), 'label_token_ids': self.slots}
        if readout_head is not None:
            require(head_meta['binding'] == self.binding
                    and weight.shape == (3, self.model.lm_head.weight.shape[1]),
                    'Readout head does not match native model')
            input_embeddings = self.model.get_input_embeddings()
            require(input_embeddings is not None
                    and input_embeddings.weight.data_ptr() != self.model.lm_head.weight.data_ptr(),
                    'Readout patch would alter tied input embeddings')
            import torch
            slots = torch.tensor(self.slots, dtype=torch.long, device=self.model.lm_head.weight.device)
            with torch.no_grad():
                self.model.lm_head.weight.index_copy_(
                    0, slots, weight.to(device=slots.device, dtype=self.model.lm_head.weight.dtype))
            require(torch.equal(self.model.lm_head.weight.index_select(0, slots).cpu().float(),
                                weight.to(dtype=self.model.lm_head.weight.dtype).float()),
                    'Native readout patch did not match BF16-rounded weights')
            self.binding['readout_head_sha256'] = head_meta['sha256']
        self.readout_projection = readout_projection
        if readout_head is not None:
            self.projection_weight = self.model.lm_head.weight.index_select(0, slots).detach()
        if readout_projection == 'three_row':
            self.binding['readout_projection'] = readout_projection
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()

    def __call__(self, request, frames):
        require(self.lock.acquire(blocking=False), 'Backend busy; previous call has not finished')
        try:
            return self._readout(request, frames)
        finally:
            self.lock.release()

    def _candidate_logits(self, inputs, *, projection=None):
        import torch
        projection = self.readout_projection if projection is None else projection
        if projection == 'three_row':
            require(hasattr(self, 'projection_weight'), 'Three-row projection requires a bound readout head')
            require('valid_input_len' not in inputs, 'Three-row projection does not support packed inputs')
            # The pinned forward projects every token over the full vocabulary.
            backbone_keys = ('input_ids', 'position_ids', 'attention_mask', 'pixel_values',
                             'pixel_values_videos', 'image_grid_thw', 'video_grid_thw')
            backbone_inputs = {key: inputs[key] for key in backbone_keys if key in inputs}
            hidden = self.model.model(**backbone_inputs, use_cache=False)[0][:, -1, :]
            return torch.nn.functional.linear(
                hidden.to(self.projection_weight.dtype), self.projection_weight).float()
        require(projection == 'full_vocab', 'Unknown readout projection')
        return self.model(**inputs, use_cache=False, return_dict=True).logits[:, -1, self.slots]

    def _prepare(self, request, frames):
        import torch
        from PIL import Image
        from training.compare_observation_decisions import prepare_inputs, tensor_fingerprint
        check_request(request)
        require(request['model_revision'] == self.binding['model_revision'], 'Native checkpoint identity changed')
        require({key: hashlib.sha256(value).hexdigest() for key, value in frames.items()}
                == request['frame_sha256'], 'Native frame bytes changed')
        images = []
        require(len(request['evidence_order']) == len(frames)
                and set(request['evidence_order']) == set(frames), 'Native frame order changed')
        for key in request['evidence_order']:
            with Image.open(io.BytesIO(frames[key])) as source:
                require(source.mode == 'RGB' and source.format == 'PNG'
                        and getattr(source, 'n_frames', 1) == 1, 'Expected canonical RGB PNG')
                source.load()
                images.append(source.copy())
        require(len({image.size for image in images}) == 1, 'Native frame dimensions differ')
        conversation = [{'role': 'user', 'content': [
            {'type': 'video', 'video': images, 'fps': .5}, {'type': 'text', 'text': request['prompt']}]}]
        torch.cuda.synchronize()
        began = time.perf_counter()
        inputs = prepare_inputs(self.processor, conversation, request['prompt'], torch,
                                max_input_tokens=self.max_input_tokens)
        torch.cuda.synchronize()
        preprocessing = time.perf_counter() - began
        proof = tensor_fingerprint(inputs, torch)
        return inputs, proof['sha256'], preprocessing

    def _timed_logits(self, inputs, *, projection):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        with torch.inference_mode():
            logits = self._candidate_logits(inputs, projection=projection)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        values = logits[0].float().cpu().tolist()
        require(bool(torch.isfinite(logits).all().item()), 'Native logits are not finite')
        return {'logits': values, 'inference_s': elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}

    def _readout(self, request, frames):
        inputs, proof, preprocessing = self._prepare(request, frames)
        measured = self._timed_logits(inputs, projection=self.readout_projection)
        return {'schema_version': RESPONSE, 'request_sha256': request['request_sha256'],
                'binding': dict(self.binding), 'processor_input_sha256': proof,
                'execution_kind': 'native_cosmos', 'preprocessing_s': preprocessing, **measured}

    def compare_projections(self, request, frames, *, repetitions=2,
                            warmup=True, first_projection='full_vocab'):
        require(hasattr(self, 'projection_weight'), 'Projection comparison requires a bound readout head')
        require(self.readout_projection == 'full_vocab', 'Comparison requires the full-vocabulary binding')
        require(type(repetitions) is int and 1 <= repetitions <= 10, 'Invalid comparison repetitions')
        require(type(warmup) is bool and first_projection in ('full_vocab', 'three_row'),
                'Invalid comparison schedule')
        require(self.lock.acquire(blocking=False), 'Backend busy; previous call has not finished')
        try:
            inputs, proof, preprocessing = self._prepare(request, frames)
            samples = []
            other = 'three_row' if first_projection == 'full_vocab' else 'full_vocab'
            for index in range(repetitions + int(warmup)):
                order = (first_projection, other) if index % 2 == 0 else (other, first_projection)
                pair = {mode: self._timed_logits(inputs, projection=mode) for mode in order}
                if not warmup or index:
                    samples.append(pair)
            deltas = [max(abs(a - b) for a, b in zip(pair['full_vocab']['logits'],
                                                      pair['three_row']['logits'], strict=True))
                      for pair in samples]
            def selected(logits):
                top = max(logits)
                return logits.index(top) if logits.count(top) == 1 else None
            return {'schema_version': 'tancho-decision-projection-comparison-1.0',
                    'request_sha256': request['request_sha256'], 'binding': dict(self.binding),
                    'processor_input_sha256': proof, 'preprocessing_s': preprocessing,
                    'repetitions': repetitions, 'samples': samples,
                    'max_abs_logit_delta': max(deltas),
                    'logits_exact_all': all(delta == 0 for delta in deltas),
                    'selected_index_agreement_all': all(
                        selected(pair['full_vocab']['logits']) == selected(pair['three_row']['logits'])
                        for pair in samples),
                    'release_qualified': False, 'flight_authorized': False}
        finally:
            self.lock.release()
