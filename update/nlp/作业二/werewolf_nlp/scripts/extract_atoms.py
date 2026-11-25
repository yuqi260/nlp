"""Extract traceable semantic candidates; validation does not prove meaning."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import call_api, make_messages
from run_baseline import read_jsonl

PROMPT = '''只从公开发言中抽取四类玩家明确声明，不预测身份：claim 自称角色，check 自己查验某人的结果，witch_info 自己的用药声明，stance 明确支持或反对某位玩家。声明可能是假。每项给 event_id、speaker、kind、target、value、quote。claim 的 target 为 null、value 是 seer/witch/hunter/werewolf/villager；check 的 value 是 wolf/nonwolf；witch_info 的 value 是 save/poison；stance 的 value 是 support/oppose；后三类 target 是1至9的玩家编号，不确定就不抽取。quote 为发言中连续原文，保留原样，不可增删空格；不抽取第三方转述、假设和不确定内容。只输出 JSON：{"atoms":[]}。'''


def check(raw, snapshot):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    atoms = json.loads(text)['atoms']
    if not isinstance(atoms, list):
        raise ValueError('atoms must be list')
    events = {e['event_id']: e for e in snapshot['public_events'] if e['type'] == 'speech'}
    allowed = {'claim': {'seer', 'witch', 'hunter', 'werewolf', 'villager'}, 'check': {'wolf', 'nonwolf'},
               'witch_info': {'save', 'poison'}, 'stance': {'support', 'oppose'}}
    results = []
    for atom in atoms:
        event, kind = events.get(atom.get('event_id')), atom.get('kind')
        speaker, target, quote = atom.get('speaker'), atom.get('target'), atom.get('quote')
        if (not event or type(speaker) is not int or speaker != event['speaker'] or kind not in allowed or
                atom.get('value') not in allowed.get(kind, set()) or not isinstance(quote, str) or not quote.strip() or
                (kind == 'claim' and target is not None) or
                (kind != 'claim' and (type(target) is not int or target not in range(1, 10))) or
                (kind in ('check', 'stance') and target == speaker)):
            raise ValueError('invalid atom fields')
        pattern = r'\s*'.join(re.escape(c) for c in re.sub(r'\s+', '', quote))
        matches = list(re.finditer(pattern, event['text']))
        if len(matches) != 1:
            raise ValueError('quote not uniquely grounded')
        source_quote = matches[0].group()
        flags = []
        if kind == 'witch_info' and atom['value'] == 'poison' and '毒' not in source_quote:
            flags.append('poison_not_explicit_in_quote')
        if re.search(r'(没(?:有)?|未|不|无)(?:使用|用|给|将|把|去|能|想|选择|成功|来|了|过|对|我|他|她|其|\d|号|玩家|\s){0,12}(?:救|毒)', source_quote):
            flags.append('possible_negation')
        if any(word in source_quote for word in ('可能', '也许', '或许', '暂时', '不确定')):
            flags.append('possible_uncertainty')
        results.append({**atom, 'quote': source_quote, 'original_quote': quote, 'review_flags': flags})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--offset', type=int, default=0)
    args = p.parse_args()
    games = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'validation'}
    if args.offset < 0 or args.limit < 1:
        p.error('offset must be nonnegative and limit positive')
    snapshots = [s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if s['game_id'] in games][args.offset:args.offset + args.limit]
    with args.output.open('x', encoding='utf8') as out:
        for snapshot in snapshots:
            messages = make_messages(snapshot)
            messages[0]['content'] = PROMPT
            record = {'snapshot_id': snapshot['snapshot_id'], 'status': 'failed'}
            started = time.monotonic()
            try:
                raw, usage, model, finish = call_api(os.environ['WEREWOLF_API_BASE_URL'], os.environ['WEREWOLF_API_KEY'], 'qwen3.5-27b', messages, 0.6, 8192, 120, False)
                record.update(raw_output=raw, usage=usage, returned_model=model, finish_reason=finish)
                record['atoms'] = check(raw, snapshot)
                record['status'] = 'ok'
            except Exception as exc:
                record['error'] = type(exc).__name__
            record['elapsed_seconds'] = round(time.monotonic()-started, 3)
            out.write(json.dumps(record, ensure_ascii=False)+'\n'); out.flush()
            print(record['snapshot_id'], record['status'], flush=True)


if __name__ == '__main__':
    main()
