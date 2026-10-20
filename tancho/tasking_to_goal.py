"""Turn a free-text tasking into a structured observation goal on Nebius Token Factory.

This runs on the ground, before takeoff, where a hosted NVIDIA model is affordable and a
mistake is still correctable. The on-board model never sees free text: it receives the
validated goal that this step produces.
"""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .mission_profiles import PROFILES

BASE_URL = 'https://api.tokenfactory.nebius.com/v1'
KEY_FILE = Path.home() / '.config/tancho/token_factory_api_key'
PUBLIC_PROFILES = tuple(name for name in sorted(PROFILES) if 'recovery' not in name)


def catalog(profiles=PUBLIC_PROFILES):
    """The closed vocabulary the model must choose from; nothing else is accepted."""
    return {name: {'categories': sorted(PROFILES[name].categories),
                   'questions': sorted(PROFILES[name].questions),
                   'map_relations': sorted(PROFILES[name].map_relations)}
            for name in profiles}


def build_messages(tasking, profiles=PUBLIC_PROFILES):
    system = ('You convert a disaster-response tasking into one observation goal for an '
              'on-board drone model. Answer with a single JSON object and nothing else:\n'
              '{"mission_profile": <profile>, "category": <category>, '
              '"required_questions": [<question>, ...], "map_relations": [<relation>, ...]}\n'
              'Every value must come from the catalog for the profile you choose. '
              'Use the fewest questions that answer the tasking. '
              'map_relations may be an empty list.\n'
              'Catalog:\n' + json.dumps(catalog(profiles), separators=(',', ':')))
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': tasking}]


def api_key(path=KEY_FILE):
    key = os.environ.get('NEBIUS_TOKEN_FACTORY_API_KEY') or os.environ.get('NEBIUS_API_KEY')
    if key:
        return key.strip()
    return Path(path).read_text().strip()


def request(tasking, model, *, key=None, timeout=300, max_tokens=3000,
            profiles=PUBLIC_PROFILES):
    # Nemotron 3.5 reasons by default and spends max_tokens on reasoning tokens first, so a
    # small cap returns content=None with finish_reason=length. Measured 2026-10-21.
    body = json.dumps({'model': model, 'messages': build_messages(tasking, profiles),
                       'temperature': 0, 'max_tokens': max_tokens,
                       'response_format': {'type': 'json_object'}}).encode()
    call = urllib.request.Request(BASE_URL + '/chat/completions', data=body, method='POST')
    call.add_header('Authorization', 'Bearer ' + (key or api_key()))
    call.add_header('Content-Type', 'application/json')
    started = time.monotonic()
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            payload = json.loads(response.read())
        choice = payload['choices'][0]
        text = choice['message'].get('content')
        usage = payload.get('usage') or {}
        detail = usage.get('completion_tokens_details') or {}
        return {'ok': text is not None, 'raw': text if text is not None else
                'content was null; finish_reason=%s' % (choice.get('finish_reason'),),
                'finish_reason': choice.get('finish_reason'),
                'reasoning_tokens': detail.get('reasoning_tokens'),
                'completion_tokens': usage.get('completion_tokens'),
                'seconds': round(time.monotonic() - started, 2)}
    except urllib.error.HTTPError as error:
        return {'ok': False, 'status': error.code, 'raw': error.read().decode()[:400],
                'seconds': round(time.monotonic() - started, 2)}
    except Exception as error:  # transport failure; a timed-out request may still be billed
        return {'ok': False, 'status': None, 'raw': repr(error)[:400],
                'seconds': round(time.monotonic() - started, 2)}


def parse_goal(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('```')[1]
        text = text.split('\n', 1)[1] if text.lower().startswith('json') else text
    goal = json.loads(text)
    if not isinstance(goal, dict):
        raise ValueError('Goal is not an object')
    return goal


def validate_goal(goal):
    """Return the list of vocabulary violations; an empty list means the goal is usable."""
    problems = []
    name = goal.get('mission_profile')
    if name not in PROFILES or name not in PUBLIC_PROFILES:
        return ['unknown mission_profile: %r' % (name,)]
    profile = PROFILES[name]
    if goal.get('category') not in profile.categories:
        problems.append('unknown category: %r' % (goal.get('category'),))
    questions = goal.get('required_questions')
    if not isinstance(questions, list) or not questions:
        problems.append('required_questions must be a non-empty list')
    else:
        problems += ['unknown question: %r' % (q,) for q in questions if q not in profile.questions]
    relations = goal.get('map_relations', [])
    if not isinstance(relations, list):
        problems.append('map_relations must be a list')
    else:
        problems += ['unknown relation: %r' % (r,) for r in relations
                     if r not in profile.map_relations]
    return problems
