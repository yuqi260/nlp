"""Partition candidate conversations by game for internal training validation."""
import json
import random
from pathlib import Path
from run_baseline import ROOT, read_jsonl


def main():
    train_games = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'train'}
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in train_games]
    rows = read_jsonl(ROOT / 'data/processed/sft_direct_train_candidate_v1.jsonl')
    if len(rows) != len(snapshots):
        raise ValueError('candidate/snapshot mismatch')
    games = sorted({s['game_id'] for s in snapshots})
    random.Random(42).shuffle(games)
    holdout = set(games[:11])
    partitions = {'training': [], 'internal_validation': []}
    for snapshot, row in zip(snapshots, rows):
        partitions['internal_validation' if snapshot['game_id'] in holdout else 'training'].append(row)
    for name, records in partitions.items():
        path = ROOT / f'data/processed/sft_direct_{name}_group_v1.jsonl'
        with path.open('x', encoding='utf8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(name, len(records))
    manifest = {'seed': 42, 'training_games': sorted(set(games)-holdout), 'internal_validation_games': sorted(holdout)}
    with (ROOT / 'data/processed/sft_direct_group_split_v1.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
