"""Validation-only temperature fitting and separately recorded acceptance policy."""
import math

from .contracts import number, require
from .mission_context import canonical_sha256
from .observation_decision import CATALOG, _check_request, make_receipt


def probabilities(logits, temperature=1.0):
    require(number(temperature) and math.isfinite(temperature) and .05 <= temperature <= 20,
            'Temperature outside supported bounds')
    require(isinstance(logits, list) and len(logits) == 3
            and all(number(v) and math.isfinite(v) for v in logits), 'Invalid logits')
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    return [value / sum(weights) for value in weights]


def metrics(rows, temperature):
    require(bool(rows), 'No calibration samples')
    nll = brier = correct = 0.0
    bins = [[] for _ in range(10)]
    for row in rows:
        target = row['target_index']
        require(type(target) is int and 0 <= target < 3, 'Invalid calibration target')
        p = probabilities(row['logits'], temperature)
        best = max(range(3), key=p.__getitem__)
        hit = int(best == target)
        nll -= math.log(max(p[target], 1e-300))
        brier += sum((value - int(i == target)) ** 2 for i, value in enumerate(p))
        correct += hit
        bins[min(9, int(max(p) * 10))].append((max(p), hit))
    ece = sum(abs(sum(p for p, _ in bucket) - sum(hit for _, hit in bucket))
              for bucket in bins) / len(rows)
    return {'nll': nll / len(rows), 'brier': brier / len(rows),
            'accuracy': correct / len(rows), 'ece_10_bins': ece, 'records': len(rows)}


def fit_temperature(rows, *, model_sha256, validation_sha256, min_probability,
                    readout_head_sha256=None, readout_projection=None):
    require(rows and all(row['split'] == 'validation' for row in rows),
            'Calibration requires validation only')
    require(len({(r['sample_id'], tuple(r['order'])) for r in rows}) == len(rows),
            'Repeated calibration records would overweight samples')
    require(number(min_probability) and 0 <= min_probability <= 1, 'Invalid acceptance threshold')
    for value in (model_sha256, validation_sha256, readout_head_sha256):
        if value is None:
            continue
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Invalid calibration binding')
    require(readout_projection is None or
            (readout_projection == 'three_row' and readout_head_sha256 is not None),
            'Invalid calibration projection')
    grid = [math.exp(i * math.log(4) / 40) for i in range(-40, 41)]
    best = min(grid, key=lambda t: (metrics(rows, t)['nll'], abs(math.log(t))))
    body = {'schema_version': 'tancho-decision-calibration-1.0', 'catalog_revision': CATALOG,
            'model_sha256': model_sha256, 'validation_sha256': validation_sha256,
            'fit_records_sha256': canonical_sha256(rows), 'fit_split': 'validation',
            'temperature': best, 'min_probability': min_probability,
            'threshold_source': 'operator_predeclared',
            'before': metrics(rows, 1.0), 'after': metrics(rows, best),
            'release_qualified': False}
    if readout_head_sha256 is not None:
        body['readout_head_sha256'] = readout_head_sha256
    if readout_projection is not None:
        body['readout_projection'] = readout_projection
    return {**body, 'sha256': canonical_sha256(body)}


def acceptance(request, receipt, *, model_sha256, calibration=None,
               readout_head_sha256=None, readout_projection=None):
    _check_request(request)
    require(receipt == make_receipt(request, receipt['logits'], receipt['label_token_ids'],
                                   started_ms=receipt['started_ms'], completed_ms=receipt['completed_ms'],
                                   timeout_ms=receipt['timeout_ms']), 'Acceptance receipt changed')
    require(receipt['request_sha256'] == request['request_sha256'], 'Acceptance request changed')
    if readout_head_sha256 is not None:
        require(isinstance(readout_head_sha256, str) and len(readout_head_sha256) == 64
                and all(c in '0123456789abcdef' for c in readout_head_sha256), 'Invalid readout head binding')
    require(readout_projection is None or
            (readout_projection == 'three_row' and readout_head_sha256 is not None),
            'Invalid readout projection binding')
    temperature, threshold = 1.0, 0.0
    if calibration is not None:
        require(calibration['sha256'] == canonical_sha256(
            {k: v for k, v in calibration.items() if k != 'sha256'}), 'Calibration changed')
        require(calibration['schema_version'] == 'tancho-decision-calibration-1.0'
                and calibration['catalog_revision'] == CATALOG
                and calibration['model_sha256'] == model_sha256
                and calibration.get('readout_head_sha256') == readout_head_sha256
                and calibration.get('readout_projection') == readout_projection
                and calibration['fit_split'] == 'validation', 'Calibration belongs elsewhere')
        temperature, threshold = calibration['temperature'], calibration['min_probability']
        require(number(threshold) and 0 <= threshold <= 1, 'Invalid acceptance threshold')
    p = probabilities(receipt['logits'], temperature)
    selected = receipt['selected_index']
    accepted = receipt['outcome'] == 'selected' and selected is not None and p[selected] >= threshold
    body = {'schema_version': 'tancho-decision-acceptance-1.0',
            'receipt_sha256': receipt['receipt_sha256'], 'model_sha256': model_sha256,
            'calibration_sha256': calibration['sha256'] if calibration else None,
            'probabilities': p, 'min_probability': threshold, 'accepted': accepted,
            'reason': 'accepted' if accepted else ('low_confidence' if receipt['outcome'] == 'selected'
                                                  else receipt['reason'])}
    return {**body, 'sha256': canonical_sha256(body)}
