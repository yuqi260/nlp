"""Screen teacher candidates against labels and snapshot event IDs.

Passing this screen does not verify explanations and does not make SFT data.
"""
import json
import re
from collections import defaultdict
from run_baseline import ROOT, read_jsonl


def main():
    predictions = read_jsonl(ROOT / 'outputs/predictions/qwen3_5_27b_train_hypotheses_teacher_pilot_v1.jsonl')
    train = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'train'}
    snapshots = {r['snapshot_id']: r for r in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if r['game_id'] in train}
    labels = defaultdict(dict)
    for row in read_jsonl(ROOT / 'data/processed/labels.jsonl'):
        if row['game_id'] in train:
            labels[row['game_id']][row['player_id']] = row['true_role']
    results = []
    for row in predictions:
        if row['snapshot_id'] not in snapshots:
            raise ValueError('non-training snapshot')
        snapshot = snapshots[row['snapshot_id']]
        result = {'snapshot_id': row['snapshot_id'], 'status': row['status']}
        if row['status'] == 'ok':
            pred = {x['player_id']: x['role'] for x in row['assignments']}
            result['fully_correct'] = pred == labels[row['game_id']]
            raw = row['raw_output'].strip()
            if raw.startswith('```') and raw.endswith('```'):
                raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            value = json.loads(raw)
            hypotheses = value.get('hypotheses', [])
            refs = re.findall(r'e\d{4}', json.dumps(hypotheses, ensure_ascii=False))
            present = {e['event_id'].rsplit(':', 1)[-1] for e in snapshot['public_events']}
            result.update(hypotheses=len(hypotheses), references=len(refs), invalid_references=[x for x in refs if x not in present])
        results.append(result)
    output = ROOT / 'outputs/metrics/train_teacher_pilot_screen_v1.json'
    with output.open('x', encoding='utf8') as f:
        json.dump({'note': 'Correct labels and event IDs do not prove explanations correct; manual semantic review required.', 'records': results}, f, ensure_ascii=False, indent=2)
    print(output)


if __name__ == '__main__':
    main()
