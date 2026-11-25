"""Convert WolfGame public logs to provisional observer snapshots.

Numbers are stable pseudonyms assigned from the public roster, not seats.
Final state supplies labels and the public, fixed seat directory; it never
supplies event mappings or end-of-game alive status.
"""
from collections import Counter
import re

ROLES = Counter({'seer': 1, 'witch': 1, 'hunter': 1, 'werewolf': 3, 'villager': 3})
SPEECH = {'player_speech', 'sheriff_campaign_speech', 'sheriff_pk_speech', 'day_pk_speech', 'last_words'}
DAY_VOTES = {'day_vote_result', 'day_revote_result'}


def public_identity_masks(snapshots):
    """WolfGame reveals hunter identity even when the hunter doesn't shoot."""
    return {s['snapshot_id']: sorted({e['speaker'] for e in s['public_events']
                                     if e['type'] in ('identity_reveal', 'shoot')}) for s in snapshots}


def convert(events, final_state, game_id):
    if not events or any(type(e.get('sequence')) is not int for e in events):
        raise ValueError('missing public sequence')
    sequence = [e['sequence'] for e in events]
    if sequence != sorted(set(sequence)):
        raise ValueError('nonmonotonic public sequence')
    start = events[0]
    if start.get('metadata', {}).get('event_type') != 'game_started':
        raise ValueError('missing public roster')
    match = re.search(r'本局共有\s*9\s*名玩家[：:]\s*([^。]+)', start.get('content', ''))
    if not match:
        raise ValueError('public roster not recognized')
    names = [x.strip() for x in match.group(1).split('、')]
    if len(names) != 9 or len(set(names)) != 9 or any(not x for x in names):
        raise ValueError('invalid public roster')
    numbers = {name: i + 1 for i, name in enumerate(names)}
    players = final_state.get('players', [])
    if len(players) != 9 or {p.get('name') for p in players} != set(names) or Counter(p.get('role') for p in players) != ROLES:
        raise ValueError('invalid final labels or roster')
    seats = {p['name']: p.get('seat_ref') for p in players}
    if set(seats.values()) != {f'{i}号' for i in range(1, 10)}:
        raise ValueError('invalid public seat directory')
    labels = [{'game_id': game_id, 'player_id': numbers[p['name']], 'true_role': p['role']} for p in players]
    # Mapping is discovered only from public speech and public vote wording.
    id_to_name = {}
    public, snapshots = [], []
    active_day, had_vote, tied = None, False, False

    def associate(pid, name):
        if not isinstance(pid, str) or name not in numbers:
            raise ValueError('invalid public identity mapping')
        if pid in id_to_name and id_to_name[pid] != name:
            raise ValueError('contradictory public mapping')
        if name in id_to_name.values() and id_to_name.get(pid) != name:
            raise ValueError('duplicate public mapping')
        id_to_name[pid] = name

    def finish():
        if active_day is not None and had_vote and not tied and public:
            cutoff = public[-1]['order']
            snapshots.append({'snapshot_id': f'{game_id}:day{active_day}', 'game_id': game_id,
                              'ruleset_id': 'wolfgame_sheriff_v1', 'day': active_day, 'cutoff_order': cutoff,
                              'player_directory': [{'player_id': numbers[n], 'name': n, 'seat_ref': seats[n]} for n in names],
                              'public_events': [dict(e) for e in public]})

    for source in events:
        kind, phase, day = source.get('kind'), source.get('phase'), source.get('day')
        meta = source.get('metadata', {})
        event_type = meta.get('event_type')
        if kind == 'role_reveal' or event_type == 'game_finished' or phase == 'finished':
            finish()
            break
        if type(day) is not int or not isinstance(meta, dict):
            raise ValueError('invalid public event')
        if active_day is not None and day > active_day:
            finish()
            had_vote, tied = False, False
        if day > 0:
            active_day = day
        content = source.get('content')
        if not isinstance(content, str):
            raise ValueError('invalid public text')
        seq = source['sequence']
        base = {'game_id': game_id, 'event_id': f'{game_id}:e{seq:04}', 'day': day,
                'phase': phase, 'order': seq}
        if kind in SPEECH:
            matches = [n for n in names if content.startswith(n + '：') or content.startswith(n + '（')]
            if len(matches) != 1:
                raise ValueError('speech name not uniquely recognized')
            associate(source.get('actor_id'), matches[0])
            public.append({**base, 'type': 'speech', 'speaker': numbers[matches[0]], 'text': content})
        elif kind == 'gm_announcement':
            if event_type in DAY_VOTES or event_type in ('sheriff_vote_result', 'sheriff_revote_result'):
                records = meta.get('votes')
                if not isinstance(records, list):
                    raise ValueError('invalid votes')
                name_pattern = '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                pairs = re.findall(f'({name_pattern})投给({name_pattern})', content)
                for position, rec in enumerate(records):
                    candidates = [(a, b) for a, b in pairs if (rec['voter_id'] not in id_to_name or id_to_name[rec['voter_id']] == a) and (rec['target_id'] not in id_to_name or id_to_name[rec['target_id']] == b)]
                    # Vote records and publicly stated pairs have the same ordering.
                    if len(pairs) != len(records) or position >= len(pairs):
                        raise ValueError('cannot align public votes')
                    pair = pairs[position]
                    if pair not in candidates:
                        raise ValueError('public vote mapping mismatch')
                    associate(rec['voter_id'], pair[0]); associate(rec['target_id'], pair[1])
                if event_type in DAY_VOTES:
                    votes = [{'player': numbers[id_to_name[v['voter_id']]], 'target': numbers[id_to_name[v['target_id']]]} for v in records]
                    leaders = [numbers[id_to_name[p]] for p in meta.get('leaders', [])]
                    eliminated = meta.get('eliminated')
                    public.append({**base, 'type': 'vote_results', 'speaker': None, 'text': content, 'votes': votes,
                                   'eliminated_player': numbers[id_to_name[eliminated]] if eliminated else None,
                                   'tied_players': leaders if len(leaders) > 1 else []})
                    had_vote, tied = True, len(leaders) > 1 and eliminated is None
                    continue
            if event_type in ('hunter_revealed', 'hunter_shot'):
                found = [n for n in names if content.startswith(n + ' ')]
                if len(found) != 1:
                    raise ValueError('hunter name not recognized')
                associate(meta.get('hunter_id'), found[0])
                if event_type == 'hunter_revealed':
                    public.append({**base, 'type': 'identity_reveal', 'speaker': numbers[found[0]], 'text': content})
                    continue
                target_names = [n for n in names if f'带走了 {n}' in content]
                if len(target_names) != 1:
                    raise ValueError('hunter target not recognized')
                associate(meta.get('target_id'), target_names[0])
                public.append({**base, 'type': 'shoot', 'speaker': numbers[found[0]], 'target': numbers[target_names[0]], 'text': content})
            else:
                public.append({**base, 'type': 'announcement', 'speaker': None, 'text': content})
        else:
            raise ValueError(f'unknown public event kind: {kind}')
    else:
        raise ValueError('no public end boundary')
    return snapshots, labels
