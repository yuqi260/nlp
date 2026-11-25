"""Offline input audit and conservative public-identity evaluation."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
from collections import Counter
from evaluate_baseline import evaluate, public_identity_masks
from run_baseline import ROOT, read_jsonl
from werewolf_nlp.preprocess import convert


def main():
    p = argparse.ArgumentParser()
    p.add_argument('predictions', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    snapshots = read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
    labels = read_jsonl(ROOT / 'data/processed/labels.jsonl')
    records = read_jsonl(args.predictions)
    truth = {(x['game_id'], x['player_id']): x['true_role'] for x in labels}
    masks = public_identity_masks(snapshots)
    texts, findings = {}, []
    for s in snapshots:
        for player in masks[s['snapshot_id']]:
            if truth[s['game_id'], player] != 'hunter':
                raise ValueError('shoot event inconsistent with hunter label')
        if any(e['order'] > s['cutoff_order'] or e['game_id'] != s['game_id'] for e in s['public_events']):
            raise ValueError('snapshot boundary violation')
        texts[s['game_id']] = '\n'.join(e['text'] for e in s['public_events'] if e['type'] == 'speech')
    # Rebuild every source game and compare, without rewriting derived data.
    for path in sorted((ROOT / 'data/raw/makto_source/raw').glob('*/9_player_game/hunter_witch_seer/*/event_zh.json')):
        game_id = path.parents[3].name + '_hunter_witch_seer_' + path.parent.name
        _, rebuilt_labels, rebuilt, _ = convert(json.loads(path.read_text(encoding='utf-8')), game_id)
        existing = [s for s in snapshots if s['game_id'] == game_id]
        if rebuilt != existing or (existing and any(truth.get((x['game_id'], x['player_id'])) != x['true_role'] for x in rebuilt_labels)):
            raise ValueError('rebuild mismatch: ' + game_id)
    # Character similarity is a candidate screen, not a semantic duplicate audit.
    games = sorted(texts)
    for i, a in enumerate(games):
        for b in games[i+1:]:
            ratio = SequenceMatcher(None, texts[a], texts[b]).quick_ratio()
            if ratio >= .85:
                ratio = SequenceMatcher(None, texts[a], texts[b]).ratio()
                if ratio >= .85:
                    findings.append({'a': a, 'b': b, 'similarity': ratio})
    result = {'mask_policy': 'Only actual shoot actors; no claims, shot targets or inferred roles. Other reveal mechanisms remain pending audit.',
              'all_players': evaluate(records, labels), 'unknown_players': evaluate(records, labels, masks),
              'masked_predictions': sum(len(masks[r['snapshot_id']]) for r in records),
              'snapshots_checked': len(snapshots), 'games_checked': len(texts),
              'near_duplicate_candidates_threshold_0_85': findings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'all_players'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
