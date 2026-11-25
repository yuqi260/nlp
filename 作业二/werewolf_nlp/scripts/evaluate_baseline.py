"""Offline metrics for direct baseline; truth is never sent to the API."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import ROLES
from run_baseline import read_jsonl


def public_identity_masks(snapshots):
    return {
        snapshot['snapshot_id']: sorted({
            event['speaker'] for event in snapshot['public_events'] if event['type'] == 'shoot'
        })
        for snapshot in snapshots
    }


def evaluate(records, labels, excluded=None):
    if len({x['snapshot_id'] for x in records}) != len(records):
        raise ValueError('duplicate snapshot prediction')
    truth = defaultdict(dict)
    for row in labels:
        if row['player_id'] in truth[row['game_id']]:
            raise ValueError('duplicate label')
        truth[row['game_id']][row['player_id']] = row['true_role']
    confusion, n, correct, exact, successes = Counter(), 0, 0, 0, 0
    by_day = defaultdict(lambda: {'snapshots': 0, 'correct_players': 0, 'players': 0})
    quota_ok = 0
    scored_snapshots = 0
    by_game = defaultdict(lambda: [0, 0])
    for r in records:
        if r['status'] != 'ok':
            continue
        gold = truth[r['game_id']]
        pairs = r['assignments']
        pred = {x['player_id']: x['role'] for x in pairs}
        if len(pairs) != 9 or set(pred) != set(range(1, 10)) or set(gold) != set(range(1, 10)) or any(role not in ROLES for role in pred.values()):
            raise ValueError('invalid predictions or missing labels')
        successes += 1
        players = set(gold) - set((excluded or {}).get(r['snapshot_id'], []))
        matches = sum(pred[p] == gold[p] for p in players)
        correct += matches; n += len(players)
        scored_snapshots += bool(players)
        exact += int(bool(players) and matches == len(players))
        by_game[r['game_id']][0] += matches
        by_game[r['game_id']][1] += len(players)
        quota_ok += bool(r.get('role_counts_valid'))
        day = by_day[str(r['day'])]; day['snapshots'] += 1; day['correct_players'] += matches; day['players'] += len(players)
        for player in players:
            confusion[(gold[player], pred[player])] += 1
    per_role = {}
    for role in ROLES:
        tp = confusion[role, role]
        actual = sum(confusion[role, p] for p in ROLES)
        predicted = sum(confusion[t, role] for t in ROLES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_role[role] = {'support': actual, 'predicted': predicted, 'precision': precision, 'recall': recall, 'f1': 2*tp/(actual+predicted) if actual+predicted else 0.0}
    for day in by_day.values():
        day['accuracy'] = day['correct_players'] / day['players'] if day['players'] else None
    # Keep denominators visible when a public-identity mask is applied.
    extra = {'scored_players': n, 'correct_players': correct, 'scored_snapshots': scored_snapshots,
             'game_mean_accuracy': sum(c/t for c,t in by_game.values() if t)/sum(bool(t) for c,t in by_game.values()) if n else None}
    return {**extra, 'scope': 'excluding confirmed shooters only; provisional' if excluded is not None else 'all players including publicly revealed identities; provisional', 'requested_snapshots': len(records), 'successful_snapshots': successes, 'failed_snapshots': len(records)-successes, 'success_rate': successes/len(records) if records else None, 'role_count_valid_rate_among_successes': quota_ok/successes if successes else None, 'player_accuracy_among_successes': correct/n if n else None, 'role_macro_f1_among_successes': sum(x['f1'] for x in per_role.values())/len(ROLES) if successes else None, 'exact_snapshot_accuracy_among_successes': exact/scored_snapshots if scored_snapshots else None, 'per_role': per_role, 'by_day': dict(by_day), 'confusion': {t: {p: confusion[t,p] for p in ROLES} for t in ROLES}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('predictions', type=Path); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    records = read_jsonl(args.predictions)
    if not records or any(r['status'] == 'dry_run' for r in records):
        parser.error('no actual predictions: dry runs are not evaluation data')
    result = evaluate(records, read_jsonl(ROOT / 'data/processed/labels.jsonl'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
