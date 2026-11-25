"""Apply reviewed extraction candidates to previously recorded float scores."""
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.world_solver import solve
from run_baseline import read_jsonl


def main():
    snapshots = {s['snapshot_id']: s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl')}
    claims = {r['snapshot_id']: r for r in read_jsonl(ROOT / 'outputs/predictions/validation_atomic_checks_v1.jsonl')}
    scores = read_jsonl(ROOT / 'outputs/predictions/qwen3_5_27b_validation_scores_float_v1.jsonl')
    path = ROOT / 'outputs/predictions/qwen3_5_27b_validation_float_checks_v1.jsonl'
    with path.open('x', encoding='utf8') as out:
        for row in scores:
            snapshot = snapshots[row['snapshot_id']]
            events = {e['event_id']: e for e in snapshot['public_events'] if e['type'] == 'speech'}
            candidates = json.loads(claims[row['snapshot_id']]['raw_output'])['checks']
            verified = []
            for claim in candidates:
                event = events[claim['event_id']]
                # Only whitespace may differ; require one unique matching span.
                pattern = r'\s*'.join(re.escape(c) for c in re.sub(r'\s+', '', claim['quote']))
                matches = list(re.finditer(pattern, event['text']))
                if len(matches) != 1:
                    raise ValueError('quote not uniquely located')
                verified.append({**claim, 'quote': matches[0].group()})
            result = solve({int(p): v for p, v in row['scores'].items()}, snapshot, verified)
            record = {**row, 'prompt_version': 'scores_float_v1_plus_checks',
                      'assignments': [{'player_id': p, 'role': role} for p, role in sorted(result['assignment'].items())],
                      'optimal_assignments': result['optimal_assignments'], 'feasible_assignments': result['feasible_assignments'],
                      'checks': verified, 'check_source': 'validation_atomic_checks_v1'}
            out.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(path)


if __name__ == '__main__':
    main()
