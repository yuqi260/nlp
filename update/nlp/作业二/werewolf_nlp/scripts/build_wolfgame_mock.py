"""Build a separate offline Mock adapter fixture, not evaluation data."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.wolfgame_adapter import convert, public_identity_masks
from werewolf_nlp.baseline import make_messages
from run_baseline import read_jsonl


def main():
    runs = ROOT / 'data/raw/wolfgame_candidate/runs'
    output = ROOT / 'data/independent/wolfgame_mock_v2'
    if output.exists():
        raise SystemExit('output already exists')
    source = sorted(p for p in runs.glob('game-*') if p.is_dir())
    if not source:
        raise SystemExit('no Mock runs')
    all_snapshots, all_labels, inventory = [], [], []
    for path in source:
        with (path / 'shared/audit_trace.jsonl').open(encoding='utf8') as f:
            first = json.loads(f.readline())
        if first.get('event_type') != 'game_created' or first.get('payload', {}).get('model') != 'mock-werewolf-agent':
            raise ValueError('non-Mock run in Mock fixture directory')
        events = read_jsonl(path / 'shared/public_events.jsonl')
        state = json.loads((path / 'shared/final_state.json').read_text(encoding='utf8'))
        game_id = 'mock_' + path.name
        snapshots, labels = convert(events, state, game_id)
        for snapshot in snapshots:
            make_messages(snapshot, 'wolfgame_direct_v1')
        masks = public_identity_masks(snapshots)
        truth = {row['player_id']: row['true_role'] for row in labels}
        if any(truth[player] != 'hunter' for players in masks.values() for player in players):
            raise ValueError('public hunter and labels disagree')
        inventory.append({'game_id': game_id, 'snapshots': len(snapshots), 'public_confirmations': sum(bool(x) for x in masks.values())})
        all_snapshots.extend(snapshots); all_labels.extend(labels)
    output.mkdir(parents=True)
    for name, rows in [('snapshots.jsonl', all_snapshots), ('labels.jsonl', all_labels), ('inventory.jsonl', inventory)]:
        with (output / name).open('x', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(f'{len(source)} offline Mock games, {len(all_snapshots)} snapshots; no real evaluation')


if __name__ == '__main__':
    main()
