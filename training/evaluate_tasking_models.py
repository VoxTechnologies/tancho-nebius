"""Score Token Factory models on the pre-registered tasking-to-goal cases. One call per case."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tancho.tasking_to_goal import api_key, parse_goal, request, validate_goal

CASES = ROOT / 'training/tasking-goal-cases-v1.json'


def score_case(case, result):
    row = {'id': case['id'], 'seconds': result['seconds'], 'ok': result['ok']}
    if not result['ok']:
        return {**row, 'schema_valid': False, 'error': result['raw'][:200]}
    try:
        goal = parse_goal(result['raw'])
    except Exception as error:
        return {**row, 'schema_valid': False, 'error': 'unparsable: %r' % (error,)}
    problems = validate_goal(goal)
    expected = set(case['required_questions'])
    got = set(goal.get('required_questions') or []) if isinstance(
        goal.get('required_questions'), list) else set()
    overlap = len(expected & got)
    f1 = 0.0 if not overlap else 2 * overlap / (len(expected) + len(got))
    return {**row, 'schema_valid': not problems, 'problems': problems,
            'profile_exact': goal.get('mission_profile') == case['mission_profile'],
            'category_exact': goal.get('category') == case['category'],
            'question_f1': round(f1, 3), 'goal': goal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/tasking-model-comparison')
    args = parser.parse_args()
    cases = json.loads(CASES.read_text())
    frozen = hashlib.sha256(CASES.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    key = api_key()
    summary = {'schema_version': 'tancho-tasking-model-comparison-1.0',
               'cases_sha256': frozen, 'case_count': len(cases['cases']), 'models': {}}
    for model in args.model:
        rows = [score_case(case, request(case['tasking'], model, key=key))
                for case in cases['cases']]
        n = len(rows)
        summary['models'][model] = {
            'schema_valid': sum(r['schema_valid'] for r in rows) / n,
            'profile_exact': sum(r.get('profile_exact', False) for r in rows) / n,
            'category_exact': sum(r.get('category_exact', False) for r in rows) / n,
            'question_f1_mean': round(statistics.fmean(r.get('question_f1', 0.0) for r in rows), 3),
            'all_three_exact': sum(r['schema_valid'] and r.get('profile_exact') and
                                   r.get('category_exact') and r.get('question_f1') == 1.0
                                   for r in rows) / n,
            'median_seconds': round(statistics.median(r['seconds'] for r in rows), 2)}
        (args.out / (model.replace('/', '_') + '.json')).write_text(
            json.dumps({'model': model, 'cases_sha256': frozen, 'rows': rows}, indent=1) + '\n')
        print(model, json.dumps(summary['models'][model]))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
