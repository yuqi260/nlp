"""Smoke-test inference on a synthetic game; not a formal evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import call_api, make_messages, parse_response
from werewolf_nlp.wolfgame_adapter import public_identity_masks
from evaluate_baseline import evaluate
from run_baseline import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--call', action='store_true')
    args = p.parse_args()
    data = ROOT / 'data/independent/wolfgame_llm_pilot_v3'
    snapshots = read_jsonl(data / 'snapshots.jsonl')
    audit = json.loads((data / 'audit.json').read_text(encoding='utf8'))
    if not snapshots or audit['fallback_public_event_sequences']:
        raise ValueError('empty or fallback-contaminated pilot')
    messages = [(s, make_messages(s, 'wolfgame_direct_v1')) for s in snapshots]
    if not args.call:
        print(f'{len(messages)} synthetic snapshots prepared; no API call')
        return
    output = ROOT / 'outputs/predictions/wolfgame_synthetic_pilot_direct_v1.jsonl'
    records = []
    with output.open('x', encoding='utf8') as f:
        for snapshot, prompt in messages:
            record = {'snapshot_id': snapshot['snapshot_id'], 'game_id': snapshot['game_id'], 'day': snapshot['day'], 'status': 'failed'}
            started = time.monotonic()
            try:
                raw, usage, model, finish = call_api(os.environ['WEREWOLF_API_BASE_URL'], os.environ['WEREWOLF_API_KEY'], 'qwen3.5-27b', prompt, 0.6, 4096, 120, False)
                pred, quota = parse_response(raw)
                record.update(status='ok', raw_output=raw, usage=usage, returned_model=model, finish_reason=finish,
                              assignments=[{'player_id': p, 'role': role} for p, role in sorted(pred.items())], role_counts_valid=quota)
            except Exception as exc:
                record['error_type'] = type(exc).__name__
            record['elapsed_seconds'] = round(time.monotonic()-started, 3)
            records.append(record); f.write(json.dumps(record, ensure_ascii=False)+'\n'); f.flush()
    metrics = evaluate(records, read_jsonl(data / 'labels.jsonl'), public_identity_masks(snapshots))
    metrics['scope'] = 'synthetic one-day pilot, excludes publicly revealed hunters; not a formal metric'
    with (ROOT / 'outputs/metrics/wolfgame_synthetic_pilot_direct_v1.json').open('x', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'{len(records)} synthetic snapshot calls; see pilot metrics')


if __name__ == '__main__':
    main()
