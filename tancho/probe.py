"""Bounded Token Factory connectivity probes using synthetic inputs only."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
KEY_FILE = Path.home() / ".config/tancho/token_factory_api_key"
COSMOS = "nvidia/Cosmos3-Super-Reasoner"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def fixture_png():
    """64px RGB image: red square on white; deterministic, no external data."""
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + b''.join(
        b'\xff\0\0' if 16 <= x < 48 and 16 <= y < 48 else b'\xff\xff\xff'
        for x in range(64)) for y in range(64))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def payload(kind, max_tokens):
    if not 1 <= max_tokens <= 512:
        raise ValueError('max-tokens must be between 1 and 512')
    if kind == 'cosmos':
        content = [
            {'type': 'text', 'text': 'This is a synthetic connectivity test, not a real scene. Describe the central shape and its color in one short sentence.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(fixture_png()).decode()}}
        ]
        model = COSMOS
    elif kind == 'nemotron':
        content = ('Synthetic postflight log, not a real flight: observations=2; '
                   'accepted_updates=1; rejected_updates=1; injuries=unknown. '
                   'Write one short factual sentence summarizing these counts. Do not infer safety or issue commands.')
        model = NEMOTRON
    else:
        raise ValueError('unknown probe')
    return {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens, 'stream': False}


def read_key(path=KEY_FILE):
    key = os.environ.get('NEBIUS_TOKEN_FACTORY_API_KEY') or os.environ.get('NEBIUS_API_KEY')
    if not key:
        if path.stat().st_mode & 0o077:
            raise ValueError('Key file permissions must be 600 or stricter')
        key = path.read_text().strip()
    if not key.strip() or any(c.isspace() for c in key.strip()):
        raise ValueError('Invalid API key format')
    return key.strip()


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if isinstance(value, dict):
        return {redact(k, key): redact(v, key) for k, v in value.items()}
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward credentials to a different endpoint.


def request(path, key, body=None, opener=None):
    encoded = json.dumps(body, sort_keys=True).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=encoded, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    result = {'started_at': datetime.now(timezone.utc).isoformat(),
              'path': path, 'model': body.get('model') if body else None,
              'request_sha256': hashlib.sha256(encoded or path.encode()).hexdigest()}
    opener = opener or urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=60) as response:
            result['http_status'] = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        result['http_status'] = exc.code
        raw = exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        result.update(http_status=None, error='Transport failure; no automatic retry. A timed-out request may still be billed.')
        raw = b'{}'
    try:
        result['response'] = json.loads(raw)
        if not isinstance(result['response'], dict):
            result['error'] = 'Unexpected response: expected JSON object'
    except (ValueError, UnicodeDecodeError):
        result['error'] = 'Non-JSON response body omitted'
    result['latency_seconds'] = round(time.monotonic() - started, 3)
    return redact(result, key)


def save_result(result, directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.json')
    with path.open('x', encoding='utf-8') as f:
        os.chmod(path, 0o600)
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, default=KEY_FILE)
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    sub.add_parser('models')
    p = sub.add_parser('probe')
    p.add_argument('kind', choices=['cosmos', 'nemotron'])
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--dry-run', action='store_true', help='No key access and no network request')
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps({'key_file_exists': args.key_file.is_file(),
                          'key_env_set': bool(os.environ.get('NEBIUS_TOKEN_FACTORY_API_KEY') or os.environ.get('NEBIUS_API_KEY')),
                          'inference_performed': False}))
        return 0
    try:
        body = payload(args.kind, args.max_tokens) if args.command == 'probe' else None
        if args.command == 'probe' and args.dry_run:
            print(json.dumps({'dry_run': True, 'request': body}, indent=2))
            return 0
        key = read_key(args.key_file)
        result = request('/models?verbose=true' if args.command == 'models' else '/chat/completions', key, body)
        path = save_result(result, args.output_dir)
        summary = {k: result.get(k) for k in ['http_status', 'model', 'latency_seconds', 'error']}
        response = result.get('response', {})
        if isinstance(response, dict):
            summary['usage'] = response.get('usage')
            if args.command == 'models':
                summary['models'] = [m['id'] for m in response.get('data', [])]
            else:
                summary['choices'] = response.get('choices')
                summary['api_error'] = response.get('error') or response.get('detail')
        summary['record'] = str(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if result.get('http_status') == 200 and not result.get('error') and isinstance(response, dict) and not response.get('error') else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': type(exc).__name__, 'detail': 'Check key configuration, arguments, and output directory.'}))
        return 2
