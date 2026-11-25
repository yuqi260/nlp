"""Merge the saved two-snapshot prefixes and remaining parallel batches."""
import json
from run_baseline import ROOT, read_jsonl


def main():
    test = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'test'}
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in test]
    expected = [s['snapshot_id'] for s in snapshots]
    prepared = {}
    for name in ('direct', 'hyp_a', 'hyp_b'):
        prefix = ROOT / f'outputs/predictions/qwen3_5_27b_test_thinking_{name}_v1.jsonl'
        paths = [prefix] + [ROOT / f'outputs/predictions/qwen3_5_27b_test_{name}_offset{offset}_v1.jsonl' for offset in (2, 9, 16)]
        rows = [row for path in paths for row in read_jsonl(path)]
        index = {r['snapshot_id']: r for r in rows}
        if len(rows) != len(expected) or len(index) != len(rows) or set(index) != set(expected) or any(r['status'] != 'ok' for r in rows):
            raise ValueError(f'{name} incomplete, duplicated, or failed')
        prepared[name] = [index[k] for k in expected]
    for name, rows in prepared.items():
        output = ROOT / f'outputs/predictions/qwen3_5_27b_test_thinking_{name}_complete_v1.jsonl'
        with output.open('x', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(output)


if __name__ == '__main__':
    main()
