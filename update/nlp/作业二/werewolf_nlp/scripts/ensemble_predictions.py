"""Combine three prediction runs by player-role votes under role quotas."""
import argparse
import json
from pathlib import Path

from run_baseline import ROOT, read_jsonl
from werewolf_nlp.baseline import ROLES, assign_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path, nargs=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for path in args.predictions:
        rows = read_jsonl(path)
        index = {row['snapshot_id']: row for row in rows}
        if len(index) != len(rows) or any(row['status'] != 'ok' for row in rows):
            raise ValueError(f'duplicate or failed prediction in {path}')
        runs.append(index)
    if not runs[0] or any(set(run) != set(runs[0]) for run in runs[1:]):
        raise ValueError('all runs must contain the same snapshots')
    snapshots = {row['snapshot_id']: row for row in read_jsonl(ROOT / 'data/processed/snapshots.jsonl')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        for snapshot_id, first in runs[0].items():
            predictions = [run[snapshot_id] for run in runs]
            snapshot = snapshots[snapshot_id]
            if any((row['game_id'], row['day'], row['split']) !=
                   (first['game_id'], first['day'], first['split']) for row in predictions):
                raise ValueError(f'mismatched metadata for {snapshot_id}')
            if (snapshot['game_id'], snapshot['day']) != (first['game_id'], first['day']):
                raise ValueError(f'mismatched snapshot for {snapshot_id}')
            votes = {p: {role: 0 for role in ROLES} for p in range(1, 10)}
            for row in predictions:
                assignments = row['assignments']
                if len(assignments) != 9 or {x['player_id'] for x in assignments} != set(votes):
                    raise ValueError(f'invalid assignment for {snapshot_id}')
                for item in assignments:
                    votes[item['player_id']][item['role']] += 1
            rows = [{'player_id': player, **{role: count - 1 for role, count in roles.items()}}
                    for player, roles in votes.items()]
            assignment, _, score, ties = assign_scores(json.dumps({'scores': rows}), snapshot)
            record = {
                'snapshot_id': snapshot_id, 'game_id': first['game_id'], 'day': first['day'],
                'split': first['split'], 'status': 'ok', 'prompt_version': 'three_run_vote_v1',
                'sources': [str(path) for path in args.predictions],
                'assignments': [{'player_id': player, 'role': assignment[player]} for player in sorted(assignment)],
                'role_counts_valid': True, 'optimal_assignments': ties, 'vote_score_shifted': score,
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
