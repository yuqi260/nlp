"""Convert completed synthetic WolfGame runs; report fallback and call quality."""
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.wolfgame_adapter import convert, public_identity_masks
from run_baseline import read_jsonl


def main():
    root = ROOT / 'data/raw/wolfgame_candidate/runs_llm_pilot_v3'
    paths = [p for p in root.glob('game-*') if p.is_dir()]
    if len(paths) != 1:
        raise ValueError('expected one completed pilot')
    path = paths[0]
    state = json.loads((path / 'shared/final_state.json').read_text(encoding='utf8'))
    events = read_jsonl(path / 'shared/public_events.jsonl')
    if not any(e.get('kind') == 'role_reveal' for e in events):
        raise ValueError('pilot is incomplete')
    calls, tokens = Counter(), Counter()
    for trace in path.glob('agents/*/logs/trace.jsonl'):
        for row in read_jsonl(trace):
            if row.get('event_type') != 'llm_interaction':
                continue
            calls[row.get('status', 'unknown')] += 1
            usage = row.get('payload', {}).get('usage') or {}
            for key in ('input_tokens', 'output_tokens', 'total_tokens', 'requests'):
                if type(usage.get(key)) is int:
                    tokens[key] += usage[key]
    fallbacks = [e['sequence'] for e in events if '模型输出异常' in e.get('content', '')]
    snapshots, labels = convert(events, state, 'synthetic_pilot_v3')
    masks = public_identity_masks(snapshots)
    roles = {r['player_id']: r['true_role'] for r in labels}
    if any(roles[p] != 'hunter' for players in masks.values() for p in players):
        raise ValueError('revealed hunter does not match label')
    output = ROOT / 'data/independent/wolfgame_llm_pilot_v3'
    if output.exists():
        raise ValueError('output exists')
    output.mkdir(parents=True)
    for name, values in [('snapshots.jsonl', snapshots), ('labels.jsonl', labels)]:
        with (output / name).open('x', encoding='utf8') as f:
            for value in values:
                f.write(json.dumps(value, ensure_ascii=False)+'\n')
    report = {'source': 'synthetic Qwen agents, one-day cap; not independent human gameplay',
              'game_id': path.name, 'snapshots': len(snapshots), 'call_status': dict(calls),
              'recorded_usage': dict(tokens), 'fallback_public_event_sequences': fallbacks,
              'public_masks': masks, 'formal_evaluation': False}
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
