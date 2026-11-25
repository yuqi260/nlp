"""Extract candidate inspection claims; grounding is not semantic verification."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import call_api, make_messages
from run_baseline import read_jsonl

PROMPT = '''从公开发言中抽取发言者声称自己查验某人的结果，不进行身份预测。仅抽取明确表示自己查验目标为狼人或非狼的声明，不抽取他人转述、假设、猜测或投票立场。只输出JSON：{"checks":[{"event_id":"完整事件编号","speaker":1,"target":2,"result":"wolf","quote":"连续原文"}]}。result仅为wolf或nonwolf；保留原文，无法确定就省略。声明可能为假。'''


def validate(raw, snapshot):
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    rows = json.loads(raw)['checks']
    if not isinstance(rows, list):
        raise ValueError('checks must be a list')
    events = {e['event_id']: e for e in snapshot['public_events'] if e['type'] == 'speech'}
    for row in rows:
        event = events.get(row.get('event_id'))
        s, t, q = row.get('speaker'), row.get('target'), row.get('quote')
        if (not event or type(s) is not int or s != event.get('speaker') or type(t) is not int or
                t not in range(1, 10) or t == s or row.get('result') not in ('wolf', 'nonwolf') or
                not isinstance(q, str) or not q.strip() or q not in event.get('text', '')):
            raise ValueError('invalid claim grounding')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20)
    args = p.parse_args()
    split = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'validation'}
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in split][:args.limit]
    key = os.environ['WEREWOLF_API_KEY']
    base = os.environ['WEREWOLF_API_BASE_URL']
    with args.output.open('x', encoding='utf8') as out:
        for snapshot in snapshots:
            messages = make_messages(snapshot)
            messages[0]['content'] = PROMPT
            started = time.monotonic()
            record = {'snapshot_id': snapshot['snapshot_id'], 'status': 'failed'}
            try:
                raw, usage, model, finish = call_api(base, key, 'qwen3.5-27b', messages, 0.6, 4096, 120, False)
                record.update(raw_output=raw, usage=usage, returned_model=model, finish_reason=finish)
                record['checks'] = validate(raw, snapshot)
                record['status'] = 'ok'
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                record['error'] = type(exc).__name__
            record['elapsed_seconds'] = round(time.monotonic()-started, 3)
            out.write(json.dumps(record, ensure_ascii=False)+'\n'); out.flush()
            print(record['snapshot_id'], record['status'], flush=True)


if __name__ == '__main__':
    main()
