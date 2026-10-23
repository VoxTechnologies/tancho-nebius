"""Show the ground-side tasking step running: plain language in, a validated goal out.

This is the step that keeps free text away from the aircraft. NVIDIA Nemotron runs on Nebius
Token Factory and may answer only with values from the mission profile catalog; `validate_goal`
reports every violation, and a goal with any violation is never handed to the on-board model.

    python3 tools/demo_tasking.py
"""
import argparse
import json
import sys
import time

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0])
from tancho.tasking_to_goal import catalog, parse_goal, request, validate_goal

TASKINGS = ('Check whether the road under the slope is still passable.',
            'We need to know how far the slide reached, and whether it stopped at the river.')
RULE = '-' * 78


def show(tasking, model):
    print(RULE)
    print('tasking   %s' % tasking)
    started = time.monotonic()
    reply = request(tasking, model)
    goal = parse_goal(reply['raw'])
    violations = validate_goal(goal)
    print('model     %s on Nebius Token Factory, %.1f s' % (model, time.monotonic() - started))
    print()
    print('observation goal')
    for key, value in goal.items():
        print('  %-20s %s' % (key, json.dumps(value)))
    print()
    allowed = catalog()[goal['mission_profile']]
    print('  checked against the catalog for %s: %d categories, %d questions, %d map relations'
          % (goal['mission_profile'], len(allowed['categories']), len(allowed['questions']),
             len(allowed['map_relations'])))
    print('  violations           %s' % (violations if violations else 'none, the goal may fly'))
    print()
    return not violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='nvidia/Nemotron-3_5-Lightning')
    parser.add_argument('--tasking', action='append', default=None)
    args = parser.parse_args()
    print('Ground side: free text -> one validated observation goal')
    ok = [show(tasking, args.model) for tasking in (args.tasking or TASKINGS)]
    print(RULE)
    print('%d of %d taskings produced a goal the aircraft may be given.' % (sum(ok), len(ok)))


if __name__ == '__main__':
    main()
