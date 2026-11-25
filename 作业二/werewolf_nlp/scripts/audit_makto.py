"""Inventory the downloaded Chinese target-board files; no training samples are produced."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/makto_source/raw'
OUT = ROOT / 'docs/audit'
EXPECTED = {'seer': 1, 'witch': 1, 'hunter': 1, 'werewolf': 3, 'simple_villager': 3}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, events, hashes = [], collections.Counter(), collections.defaultdict(list)
    for split in ('train', 'test'):
        base = SOURCE / split / '9_player_game/hunter_witch_seer'
        for game in sorted(base.glob('game_*')):
            path = game / 'event_zh.json'
            row = {'split': split, 'game': game.name, 'event_file': str(path.relative_to(ROOT)).replace('\\', '/'), 'note_present': (game / 'note_zh.json').is_file()}
            problems = []
            try:
                raw = path.read_bytes()
                hashes[hashlib.sha256(raw).hexdigest()].append(f'{split}/{game.name}')
                data = json.loads(raw)
                if not isinstance(data, list):
                    raise ValueError('top level is not a list')
                kinds = collections.Counter(x.get('event') for x in data)
                events.update(kinds)
                roles = [x['content'] for x in data if x.get('event') == 'roles']
                counts = collections.Counter(x.get('role') for x in roles)
                players = [x.get('player') for x in roles]
                if counts != EXPECTED or set(players) != set(range(1, 10)) or len(players) != 9:
                    problems.append('invalid_roles')
                for required in ('speech', 'vote_results', 'cycle_round', 'end'):
                    if not kinds[required]:
                        problems.append('missing_' + required)
                row.update(event_count=len(data), speech_count=kinds['speech'], vote_results_count=kinds['vote_results'], end_count=kinds['end'], role_counts=json.dumps(counts, ensure_ascii=False, sort_keys=True))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append('read_error:' + str(exc))
            row['structural_status'] = 'review' if problems else 'candidate'
            row['issues'] = ';'.join(problems)
            rows.append(row)
    keys = ['split','game','event_file','note_present','event_count','speech_count','vote_results_count','end_count','role_counts','structural_status','issues']
    with (OUT / 'game_inventory.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
    summary = {'games': len(rows), 'by_split': dict(collections.Counter(r['split'] for r in rows)), 'structural_status': dict(collections.Counter(r['structural_status'] for r in rows)), 'events': dict(events.most_common()), 'exact_duplicate_files': [v for v in hashes.values() if len(v) > 1], 'scope': 'target board Chinese event files; structural checks only'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
