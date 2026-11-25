"""Build public observer snapshots from MaKTO event logs.

Output remains provisional pending human inspection. Never copy whole source events.
"""
from collections import Counter
import json

EXPECTED = Counter({'seer': 1, 'witch': 1, 'hunter': 1, 'werewolf': 3, 'simple_villager': 3})
RULESET = 'nine_seer_witch_hunter_v0'


def convert(events, game_id):
    roles = [e['content'] for e in events if e.get('event') == 'roles']
    if len(roles) != 9 or {r['player'] for r in roles} != set(range(1, 10)) or Counter(r['role'] for r in roles) != EXPECTED:
        raise ValueError('invalid role assignments')
    labels = [{'game_id': game_id, 'player_id': r['player'], 'true_role': 'villager' if r['role'] == 'simple_villager' else r['role']} for r in roles]
    public, snapshots, warnings = [], [], []
    day, in_day, votes, last_vote_tied, cutoff = None, False, 0, False, None

    def add(idx, typ, **fields):
        public.append({'game_id': game_id, 'event_id': f'{game_id}:e{idx:04}', 'day': day, 'phase': 'day', 'order': idx, 'type': typ, **fields})

    def finish(boundary):
        nonlocal votes, last_vote_tied, cutoff
        if not in_day or not votes:
            return
        if last_vote_tied:
            warnings.append({'day': day, 'reason': 'unresolved_tie', 'boundary': boundary})
            return
        if cutoff is None:
            raise ValueError('missing cutoff')
        snapshots.append({'snapshot_id': f'{game_id}:day{day}', 'game_id': game_id, 'ruleset_id': RULESET, 'day': day, 'cutoff_order': cutoff, 'public_events': [p.copy() for p in public if p['order'] <= cutoff]})

    for idx, item in enumerate(events):
        kind, content = item.get('event'), item.get('content')
        if kind == 'cycle_round':
            if not isinstance(content, dict) or content.get('status') not in ('day', 'night'):
                raise ValueError(f'invalid cycle_round at {idx}')
            finish('next_cycle')
            in_day = content['status'] == 'day'
            if in_day:
                day = content['round']
                votes, last_vote_tied, cutoff = 0, False, None
                if content.get('context'):
                    if not isinstance(content['context'], str):
                        raise ValueError('invalid day announcement')
                    add(idx, 'announcement', speaker=None, text=content['context'])
            continue
        if kind == 'end':
            finish('end')
            in_day = False
            continue
        if not in_day:
            continue
        if kind == 'speech':
            if not isinstance(content, dict) or content.get('player') not in range(1, 10) or not isinstance(content.get('context'), str):
                raise ValueError(f'invalid speech at {idx}')
            add(idx, 'speech', speaker=content['player'], text=content['context'])
        elif kind == 'vote_results':
            if not isinstance(content, dict) or not content:
                raise ValueError(f'invalid votes at {idx}')
            pairs = []
            for player, target in content.items():
                if not str(player).isdigit() or int(player) not in range(1, 10) or (target is not None and (type(target) is not int or target not in range(1, 10))):
                    raise ValueError(f'invalid vote entry at {idx}')
                pairs.append({'player': int(player), 'target': target})
            pairs.sort(key=lambda p: p['player'])
            counts = Counter(p['target'] for p in pairs if p['target'] is not None)
            if not counts:
                raise ValueError(f'no valid votes at {idx}')
            highest = max(counts.values())
            leaders = sorted(k for k, v in counts.items() if v == highest)
            last_vote_tied = len(leaders) != 1
            add(idx, 'vote_results', speaker=None, votes=pairs, eliminated_player=None if last_vote_tied else leaders[0], tied_players=leaders if last_vote_tied else [])
            votes += 1
        elif kind == 'shoot':
            if not isinstance(content, dict) or content.get('player') not in range(1, 10):
                raise ValueError(f'invalid shoot at {idx}')
            target = content.get('shoot_player')
            if target is not None:
                if type(target) is not int or target not in range(1, 10):
                    raise ValueError(f'invalid shoot target at {idx}')
                add(idx, 'shoot', speaker=content['player'], target=target)
        # The cutoff advances through the day, but never crosses a night or end.
        cutoff = idx
    if in_day:
        warnings.append({'day': day, 'reason': 'missing_terminal_boundary'})
    if not snapshots:
        warnings.append({'reason': 'no_eligible_snapshot'})
    return public, labels, snapshots, warnings


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
