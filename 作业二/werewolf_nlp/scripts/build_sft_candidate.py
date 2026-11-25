"""Build train-only direct prediction candidates for later format review.

No generated rationales or augmented examples are included.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import COUNTS, make_messages
from run_baseline import read_jsonl


def main():
    train = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'train'}
    labels = defaultdict(dict)
    for row in read_jsonl(ROOT / 'data/processed/labels.jsonl'):
        if row['game_id'] in train:
            player = row['player_id']
            if player in labels[row['game_id']]:
                raise ValueError('duplicate label')
            labels[row['game_id']][player] = row['true_role']
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in train]
    path = ROOT / 'data/processed/sft_direct_train_candidate_v1.jsonl'
    with path.open('x', encoding='utf8') as out:
        for snapshot in snapshots:
            roles = labels[snapshot['game_id']]
            if set(roles) != set(range(1, 10)) or Counter(roles.values()) != COUNTS:
                raise ValueError('invalid labels')
            messages = make_messages(snapshot, 'direct_v1')
            target = {'assignments': [{'player_id': p, 'role': roles[p]} for p in sorted(roles)]}
            messages.append({'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)})
            out.write(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
    print(f'{len(snapshots)} examples from {len({s["game_id"] for s in snapshots})} games: {path}')


if __name__ == '__main__':
    main()
