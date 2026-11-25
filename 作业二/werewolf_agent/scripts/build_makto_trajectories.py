"""Build offline first-person MaKTO trajectory datasets (train/val/test) for M3."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent.makto_trajectories import (
    load_game_splits,
    reconstruct_game_trajectories,
    save_split_npz,
)


def default_makto_raw_dir() -> Path:
    return ROOT.parent / "werewolf_nlp" / "data" / "raw" / "makto_source" / "raw"


def default_splits_file() -> Path:
    return ROOT.parent / "werewolf_nlp" / "data" / "splits" / "games.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MaKTO first-person trajectories.")
    parser.add_argument("--makto-raw-dir", type=Path, default=default_makto_raw_dir())
    parser.add_argument("--splits-file", type=Path, default=default_splits_file())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()

    split_map = load_game_splits(args.splits_file)
    belief_by_split = defaultdict(list)
    action_by_split = defaultdict(list)
    games_by_split = defaultdict(list)
    total_stats = Counter()
    unmatched_games = []

    for orig_split in ("train", "test"):
        pattern_dir = args.makto_raw_dir / orig_split / "9_player_game" / "hunter_witch_seer"
        for event_file in sorted(pattern_dir.glob("*/event_zh.json")):
            game_folder = event_file.parent.name
            game_id = f"{orig_split}_hunter_witch_seer_{game_folder}"
            split = split_map.get(game_id)
            if split is None:
                # Preserve strict split alignment with Phase 1 (e.g., excluded game_181 -> train only if desired, or skip)
                unmatched_games.append(game_id)
                continue
            events = json.loads(event_file.read_text(encoding="utf-8"))
            b_list, a_list, st = reconstruct_game_trajectories(events, game_id=game_id)
            belief_by_split[split].extend(b_list)
            action_by_split[split].extend(a_list)
            games_by_split[split].append(game_id)
            total_stats.update(st)

    # Verify zero game overlap across splits
    split_sets = {s: set(gs) for s, gs in games_by_split.items()}
    assert not (split_sets["train"] & split_sets["validation"]), "Train/Val game leakage!"
    assert not (split_sets["train"] & split_sets["test"]), "Train/Test game leakage!"
    assert not (split_sets["validation"] & split_sets["test"]), "Val/Test game leakage!"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dir": str(args.makto_raw_dir),
        "splits_file": str(args.splits_file),
        "games_per_split": {s: len(gs) for s, gs in sorted(games_by_split.items())},
        "skipped_unmatched_games": unmatched_games,
        "reconstruct_stats": dict(sorted(total_stats.items())),
        "samples_per_split": {},
    }

    for split in ("train", "validation", "test"):
        out_npz = args.out_dir / f"makto_trajectories_{split}.npz"
        counts = save_split_npz(out_npz, belief_by_split[split], action_by_split[split])
        summary["samples_per_split"][split] = counts

    audit_path = args.out_dir / "makto_trajectories_audit.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
