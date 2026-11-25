"""Conservatively construct cyclic seat relabellings for manual review.

Rotation preserves circular seat adjacency, but cannot guarantee textual semantics.
Only training games without ambiguous numeric or Chinese-number seat references
are included. Never use candidates without further semantic audit.
"""
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import make_messages
from run_baseline import read_jsonl

AMBIGUOUS = re.compile(r'\d+(?!号|\d)|(?<!\d)(?:0|[1-9]\d+)号|[一二三四五六七八九十]号|[０-９]')
SEAT = re.compile(r'(?<!\d)([1-9])号')
FIELDS = {'announcement': {'day','event_id','game_id','order','phase','speaker','text','type'},
          'speech': {'day','event_id','game_id','order','phase','speaker','text','type'},
          'shoot': {'day','event_id','game_id','order','phase','speaker','target','type'},
          'vote_results': {'day','event_id','game_id','order','phase','speaker','type','eliminated_player','tied_players','votes'}}


def rotate(player, shift):
    if player is None:
        return None
    if type(player) is not int or player not in range(1, 10):
        raise ValueError('invalid player')
    return (player - 1 + shift) % 9 + 1


def transform(snapshot, shift):
    result = copy.deepcopy(snapshot)
    for e in result['public_events']:
        if e['type'] not in FIELDS or set(e) - FIELDS[e['type']]:
            raise ValueError('unknown event field')
        if 'text' in e:
            if AMBIGUOUS.search(e['text']):
                raise ValueError('ambiguous number')
            e['text'] = SEAT.sub(lambda m: f"{rotate(int(m[1]), shift)}号", e['text'])
        for key in ('speaker', 'target', 'eliminated_player'):
            if key in e:
                e[key] = rotate(e[key], shift)
        if 'tied_players' in e:
            e['tied_players'] = [rotate(p, shift) for p in e['tied_players']]
        if 'votes' in e:
            e['votes'] = [{'player': rotate(v['player'], shift), 'target': rotate(v['target'], shift)} for v in e['votes']]
    return result


def main():
    train = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'train'}
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in train]
    bad = {s['game_id'] for s in snapshots for e in s['public_events'] if AMBIGUOUS.search(e.get('text', ''))}
    labels = defaultdict(dict)
    for r in read_jsonl(ROOT / 'data/processed/labels.jsonl'):
        if r['game_id'] in train:
            labels[r['game_id']][r['player_id']] = r['true_role']
    path = ROOT / 'data/processed/sft_rotation_review_candidates_v1.jsonl'
    with path.open('x', encoding='utf8') as out:
        for s in snapshots:
            if s['game_id'] in bad:
                continue
            for shift in range(1, 5):
                transformed = transform(s, shift)
                assignment = sorted((rotate(p, shift), role) for p, role in labels[s['game_id']].items())
                messages = make_messages(transformed)
                messages.append({'role': 'assistant', 'content': json.dumps({'assignments': [{'player_id': p, 'role': role} for p, role in assignment]}, ensure_ascii=False)})
                out.write(json.dumps({'source_snapshot_id': s['snapshot_id'], 'rotation': shift, 'messages': messages}, ensure_ascii=False)+'\n')
    print(f'{len(train-bad)} eligible games; {len(bad)} excluded games; review candidates at {path}')


if __name__ == '__main__':
    main()
