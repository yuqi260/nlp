"""Train a character-bigram baseline on training games, evaluate validation.

Each training game contributes only its last available snapshot. No labels or
events from the validation split enter fitting. Role allocation uses quotas.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import ROLES
from werewolf_nlp.world_solver import solve
from run_baseline import read_jsonl


def features(snapshot, player):
    chunks = []
    for event in snapshot['public_events']:
        if event.get('type') != 'speech' or event.get('speaker') != player:
            continue
        text = re.sub(r'(?<!\d)([1-9])号', lambda m: '自己号' if int(m[1]) == player else '他人号', event.get('text', ''))
        chunks.append(re.sub(r'\s+', '', text))
    return Counter(a+b for chunk in chunks for a, b in zip(chunk, chunk[1:]))


def main():
    splits = {r['game_id']: r['split'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl')}
    snapshots = read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
    labels = defaultdict(dict)
    for row in read_jsonl(ROOT / 'data/processed/labels.jsonl'):
        if splits.get(row['game_id']) == 'train':
            labels[row['game_id']][row['player_id']] = row['true_role']
    latest = {}
    for snapshot in snapshots:
        if splits.get(snapshot['game_id']) == 'train':
            old = latest.get(snapshot['game_id'])
            if old is None or snapshot['cutoff_order'] > old['cutoff_order']:
                latest[snapshot['game_id']] = snapshot
    counts = {role: Counter() for role in ROLES}
    totals, documents, vocabulary = Counter(), Counter(), set()
    for game, snapshot in latest.items():
        if set(labels[game]) != set(range(1, 10)):
            raise ValueError('incomplete training labels')
        for player in range(1, 10):
            role = labels[game][player]
            f = features(snapshot, player)
            counts[role].update(f)
            totals[role] += sum(f.values())
            documents[role] += 1
            vocabulary.update(f)
    if not vocabulary:
        raise ValueError('empty vocabulary')
    output = ROOT / 'outputs/predictions/naive_bayes_validation_v2.jsonl'
    with output.open('x', encoding='utf8') as out:
        for snapshot in snapshots:
            if splits.get(snapshot['game_id']) != 'validation':
                continue
            scores = {}
            for player in range(1, 10):
                f = features(snapshot, player)
                scores[player] = {}
                for role in ROLES:
                    denominator = totals[role] + len(vocabulary) + 1
                    # One additional bucket covers unseen bigrams.
                    score = math.log(documents[role] / sum(documents.values()))
                    score += sum(n * math.log((counts[role].get(token, 0) + 1) / denominator) for token, n in f.items())
                    scores[player][role] = score
            result = solve(scores, snapshot)
            record = {'snapshot_id': snapshot['snapshot_id'], 'game_id': snapshot['game_id'], 'day': snapshot['day'], 'split': 'validation',
                      'status': 'ok', 'prompt_version': 'char_bigram_nb_v2', 'role_counts_valid': True,
                      'training_games': len(latest), 'optimal_assignments': result['optimal_assignments'],
                      'assignments': [{'player_id': p, 'role': role} for p, role in sorted(result['assignment'].items())]}
            out.write(json.dumps(record, ensure_ascii=False)+'\n')
    print('training games', len(latest), 'vocabulary', len(vocabulary), 'output', output)


if __name__ == '__main__':
    main()
