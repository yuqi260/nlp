"""Evaluate a complete frozen test ensemble and its earlier direct baseline."""
import json
from pathlib import Path
from run_baseline import ROOT, read_jsonl
from evaluate_baseline import evaluate, public_identity_masks


def main():
    games = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'test'}
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in games]
    expected = {s['snapshot_id'] for s in snapshots}
    labels = read_jsonl(ROOT / 'data/processed/labels.jsonl')
    masks = public_identity_masks(snapshots)
    files = {'direct_off': 'qwen3_5_27b_test_v1.jsonl', 'thinking_vote': 'qwen3_5_27b_test_thinking_vote_v1.jsonl'}
    result = {'note': 'Test set was previously viewed. Mask excludes actual public shooters only.', 'groups': {}}
    for name, filename in files.items():
        rows = read_jsonl(ROOT / 'outputs/predictions' / filename)
        if len(rows) != len(expected) or {r['snapshot_id'] for r in rows} != expected or any(r['status'] != 'ok' for r in rows):
            raise ValueError('test predictions are incomplete or failed')
        result['groups'][name] = {'masked': evaluate(rows, labels, masks), 'all_players': evaluate(rows, labels)}
    output = ROOT / 'outputs/metrics/test_thinking_vote_v1_comparison.json'
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(output)


if __name__ == '__main__':
    main()
