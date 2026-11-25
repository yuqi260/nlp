"""Build provisional public snapshots while preserving original test games."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.preprocess import convert, jsonl


def main():
    source = ROOT / 'data/raw/makto_source/raw'
    games = [(s, p) for s in ('train', 'test') for p in sorted((source / s / '9_player_game/hunter_witch_seer').glob('game_*/event_zh.json'))]
    if not games:
        raise SystemExit('No raw target games found; download dataset first.')
    train = [p for s, p in games if s == 'train']
    random.Random(42).shuffle(train)
    val_set = set(train[:max(1, round(len(train) * 0.1))])
    public, labels, snapshots, splits, audit = [], [], [], [], []
    text_hashes = collections.defaultdict(list)
    for origin, path in games:
        game_id = f'{origin}_hunter_witch_seer_{path.parent.name}'
        split = 'test' if origin == 'test' else ('validation' if path in val_set else 'train')
        record = {'game_id': game_id, 'split': split, 'source': str(path.relative_to(ROOT)).replace('\\', '/'), 'status': 'review', 'snapshots': 0, 'warnings': ''}
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            p, l, s, warnings = convert(raw, game_id)
            if not s:
                raise ValueError('no complete snapshots')
            speech = '\n'.join(str(e.get('content', {}).get('context', '')) for e in raw if e.get('event') == 'speech')
            text_hashes[hashlib.sha256(speech.encode('utf-8')).hexdigest()].append((game_id, split))
            public.extend(p); labels.extend(l); snapshots.extend(s)
            splits.append({'game_id': game_id, 'split': split, 'original_split': origin})
            record.update(status='candidate', snapshots=len(s), warnings=json.dumps(warnings, ensure_ascii=False))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            record['warnings'] = str(exc)
        audit.append(record)
    duplicates = [v for v in text_hashes.values() if len(v) > 1]
    cross = [v for v in duplicates if len({x[1] for x in v}) > 1]
    if cross:
        raise SystemExit(f'Exact duplicate speech across splits, aborting: {cross}')
    output = ROOT / 'data/processed'
    for name, rows in [('public_events.jsonl', public), ('labels.jsonl', labels), ('snapshots.jsonl', snapshots)]:
        jsonl(output / name, rows)
    jsonl(ROOT / 'data/splits/games.jsonl', splits)
    report = ROOT / 'docs/audit'; report.mkdir(parents=True, exist_ok=True)
    with (report / 'preprocessing_inventory.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id','split','source','status','snapshots','warnings']); w.writeheader(); w.writerows(audit)
    summary = {'source_games': len(games), 'candidate_games': len(splits), 'failed_games': len(games)-len(splits), 'splits': dict(collections.Counter(x['split'] for x in splits)), 'snapshots': len(snapshots), 'snapshot_splits': dict(collections.Counter(next(x['split'] for x in splits if x['game_id']==s['game_id']) for s in snapshots)), 'public_events': len(public), 'labels': len(labels), 'game_warnings': sum(r['warnings'] not in ('', '[]') for r in audit), 'exact_speech_duplicates': duplicates, 'split_seed': 42, 'provisional': True}
    (report / 'preprocessing_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
