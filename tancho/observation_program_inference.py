"""Single-attempt, checkpoint-bound inference for both Observation Program roles."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import time

from .contracts import fields, require
from .observation_program_prompts import generator_prompt, reasoner_prompt

RESPONSE_SCHEMA = 'tancho-observation-model-response-1.0'


def _frame_hashes(frames):
    require(isinstance(frames, dict) and frames, 'Inference frames are required')
    require(all(isinstance(key, str) and key and isinstance(value, bytes) and value
                for key, value in frames.items()), 'Invalid inference frames')
    return {key: hashlib.sha256(value).hexdigest() for key, value in sorted(frames.items())}


def _validate_response(response, *, role, prompt, context, frame_hashes):
    fields(response, 'schema_version role raw_text model_revision prompt_sha256 frame_sha256 '
                     'processor_input_sha256 generation provider_ok')
    require(response['schema_version'] == RESPONSE_SCHEMA, 'Unknown model response version')
    require(response['role'] == role, 'Model response role changed')
    require(isinstance(response['raw_text'], str) and response['raw_text'].strip(),
            'Model returned no raw output')
    require(response['model_revision'] == context['model_revision'], 'Wrong model checkpoint served')
    require(response['prompt_sha256'] == prompt['sha256'], 'Model prompt binding changed')
    require(response['frame_sha256'] == frame_hashes, 'Model frame binding changed')
    value = response['processor_input_sha256']
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Missing processor input proof')
    generation = response['generation']
    fields(generation, 'attempt do_sample temperature max_new_tokens finish_reason')
    require(generation['attempt'] == 1 and generation['do_sample'] is False
            and generation['temperature'] == 0
            and type(generation['max_new_tokens']) is int
            and 1 <= generation['max_new_tokens'] <= 1536
            and generation['finish_reason'] == 'stop', 'Non-final or non-greedy generation')
    require(response['provider_ok'] is True, 'Provider or loaded-weight proof failed')
    return response


def infer_role(runtime, frames, *, role, invoke, now_ms, hold, timeout_ms):
    """Invoke once while holding position; late results have no route into runtime."""
    require(role in ('reasoner', 'generator'), 'Unknown inference role')
    require(type(timeout_ms) is int and 0 < timeout_ms <= 120000, 'Invalid inference timeout')
    require(callable(now_ms) and callable(hold) and callable(invoke), 'Inference callbacks are required')
    if role == 'reasoner':
        require(runtime.state == 'CONTEXT_VALIDATED', 'Reasoner invoked from the wrong state')
        prompt = reasoner_prompt(runtime.context)
    else:
        require(runtime.state == 'INTENTS_VERIFIED', 'Generator invoked from the wrong state')
        prompt = generator_prompt(runtime.context, runtime.intents)
    hashes = _frame_hashes(frames)
    require(set(hashes) == set(runtime.context['evidence_ids']), 'Inference evidence registry changed')
    started = now_ms()
    require(type(started) is int and runtime.context['captured_at_ms'] <= started
            < runtime.context['expires_at_ms'], 'Inference starts outside context lifetime')
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(invoke, role, prompt['text'], copy.deepcopy(frames))
    wall_deadline = time.monotonic() + timeout_ms / 1000
    previous = started
    try:
        while not future.done():
            current = now_ms()
            require(type(current) is int and current >= previous, 'Inference clock moved backwards')
            previous = current
            require(current - started < timeout_ms and current < runtime.context['expires_at_ms']
                    and time.monotonic() < wall_deadline, 'Inference deadline exceeded')
            hold()
            time.sleep(.01)
        completed = now_ms()
        require(type(completed) is int and completed >= previous
                and completed - started < timeout_ms
                and completed < runtime.context['expires_at_ms'], 'Late inference result discarded')
        response = _validate_response(future.result(), role=role, prompt=prompt,
                                      context=runtime.context, frame_hashes=hashes)
        if role == 'reasoner':
            artifact = runtime.reason(response['raw_text'], now_ms=completed)
        else:
            artifact = runtime.generate(response['raw_text'], now_ms=completed)
        receipt = {'role': role, 'model_revision': response['model_revision'],
                   'prompt_sha256': prompt['sha256'], 'frame_sha256': hashes,
                   'processor_input_sha256': response['processor_input_sha256'],
                   'raw_output_sha256': hashlib.sha256(response['raw_text'].encode()).hexdigest(),
                   'started_ms': started, 'completed_ms': completed,
                   'elapsed_ms': completed - started, 'attempt': 1,
                   'do_sample': False, 'temperature': 0,
                   'provider_ok': True}
        runtime.record_model_receipt(receipt)
        return artifact, receipt
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
