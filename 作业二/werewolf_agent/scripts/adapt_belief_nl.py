"""Adapt BeliefMLP to live qwen3.5-27b NL conversations using MaKTO train + disjoint NL replays (0 API cost)."""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent.features import (
    BELIEF_SEAT_FEAT_DIM,
    NUM_ROLES,
    ROLE_TO_IDX,
    build_belief_seat_features,
)
from werewolf_agent.models.belief import BeliefMLP, evaluate_belief_predictions, train_belief_mlp
from werewolf_agent.protocol import Phase, PlayerObservation, PublicEvent, Role


def extract_nl_belief_samples(game_files: List[Path]) -> Dict[str, np.ndarray]:
    """Extract first-person Belief training/eval samples from completed NL game JSONs."""
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ms: List[np.ndarray] = []

    for f in game_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        roles = {int(k): Role(v) for k, v in d["roles"].items()}
        wolf_set = {p for p, r in roles.items() if r == Role.WEREWOLF}
        true_vec = np.array([ROLE_TO_IDX[roles[p]] for p in range(1, 10)], dtype=np.int64)

        alive = {p: True for p in range(1, 10)}
        revealed: Dict[int, Role] = {}
        history: List[PublicEvent] = []

        for raw in d["public_events"]:
            et = raw["event_type"]
            if et == "night_announcement":
                for dp in raw["payload"].get("dead_players", []):
                    alive[int(dp)] = False
            elif et == "hunter_shoot":
                revealed[int(raw["payload"]["shooter"])] = Role.HUNTER
                alive[int(raw["payload"]["target"])] = False

            if et == "vote_result":
                # Sample BEFORE applying vote exile, matching live DAY_VOTE decision point
                alive_t = tuple(p for p in range(1, 10) if alive[p])
                dead_t = tuple(p for p in range(1, 10) if not alive[p])
                for pid in alive_t:
                    r = roles[pid]
                    obs = PlayerObservation(
                        game_id=f.stem,
                        mode="nl",
                        day=max(int(raw["day"]), 1),
                        phase=Phase.DAY_VOTE,
                        player_id=pid,
                        my_role=r,
                        player_count=9,
                        alive_players=alive_t,
                        dead_players=dead_t,
                        revealed_roles=dict(revealed),
                        public_events=tuple(history),
                        werewolf_teammates=tuple(sorted(wolf_set - {pid})) if r == Role.WEREWOLF else (),
                        vote_candidates=alive_t,
                    )
                    seat_x = build_belief_seat_features(obs)
                    unk = np.ones(9, dtype=bool)
                    unk[pid - 1] = False
                    if r == Role.WEREWOLF:
                        for tm in wolf_set:
                            unk[tm - 1] = False
                    for rp in revealed:
                        unk[rp - 1] = False
                    Xs.append(seat_x)
                    ys.append(true_vec)
                    ms.append(unk)

                ex = raw["payload"].get("exiled")
                if ex is not None:
                    alive[int(ex)] = False

            if et != "game_end":
                history.append(
                    PublicEvent(
                        event_id=raw["event_id"],
                        day=raw["day"],
                        phase=Phase(raw["phase"]),
                        event_type=et,
                        actor=raw["actor"],
                        payload=raw["payload"],
                    )
                )

    return {
        "belief_X": np.stack(Xs, axis=0) if Xs else np.zeros((0, 9, BELIEF_SEAT_FEAT_DIM), dtype=np.float32),
        "belief_y": np.stack(ys, axis=0) if ys else np.zeros((0, 9), dtype=np.int64),
        "belief_unknown_mask": np.stack(ms, axis=0) if ms else np.zeros((0, 9), dtype=bool),
    }


def main() -> None:
    t0 = time.perf_counter()
    # 1. Load MaKTO offline train/test
    with np.load(ROOT / "data" / "processed" / "makto_trajectories_train.npz") as f:
        makto_tr = {k: f[k] for k in ("belief_X", "belief_y", "belief_unknown_mask")}
    with np.load(ROOT / "data" / "processed" / "makto_trajectories_test.npz") as f:
        makto_te = {k: f[k] for k in ("belief_X", "belief_y", "belief_unknown_mask")}

    # 2. Extract disjoint live qwen3.5-27b NL replays:
    #    Train on ablation_live_qwen35_80g (80 games, seeds 5000..6009) + single_seat_live_qwen35_1v8_90g (90 games, seeds 9000..9405)
    #    Validate on dual_only_live_qwen35_dual_20g (20 games) + ours_only_live_qwen35_ours_v2_20g (20 games)
    #    Note: All future balanced 1v8 tests will use brand-new seeds 20000+ (zero leakage!).
    tr_files = sorted(
        [Path(p) for p in glob.glob(str(ROOT / "outputs" / "m5_eval" / "ablation_live_qwen35_80g" / "games" / "*.json"))]
        + [Path(p) for p in glob.glob(str(ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_1v8_90g" / "games" / "*.json"))]
    )
    va_files = sorted(
        [Path(p) for p in glob.glob(str(ROOT / "outputs" / "m5_eval" / "dual_only_live_qwen35_dual_20g" / "games" / "*.json"))]
        + [Path(p) for p in glob.glob(str(ROOT / "outputs" / "m5_eval" / "ours_only_live_qwen35_ours_v2_20g" / "games" / "*.json"))]
    )
    nl_tr = extract_nl_belief_samples(tr_files)
    nl_va = extract_nl_belief_samples(va_files)

    # Combine MaKTO train + live NL train (oversample live NL 1x since counts are comparable)
    comb_tr = {
        k: np.concatenate([makto_tr[k], nl_tr[k]], axis=0) for k in ("belief_X", "belief_y", "belief_unknown_mask")
    }

    # Evaluate old MaKTO-only BeliefMLP on held-out NL val
    old_model = BeliefMLP.load(ROOT / "outputs" / "models" / "belief_mlp.npz")
    old_probs = old_model.predict_probs_batch(nl_va["belief_X"].reshape(-1, BELIEF_SEAT_FEAT_DIM)).reshape(
        nl_va["belief_X"].shape[0], 9, NUM_ROLES
    )
    old_nl_eval = evaluate_belief_predictions(old_probs, nl_va["belief_y"], nl_va["belief_unknown_mask"])

    # Train domain-adapted BeliefMLP_NL
    new_model, hist = train_belief_mlp(comb_tr, nl_va, epochs=45, batch_size=256, lr=2e-3, seed=42)
    out_model = ROOT / "outputs" / "models" / "belief_mlp_nl.npz"
    new_model.save(out_model)

    new_probs_nl = new_model.predict_probs_batch(nl_va["belief_X"].reshape(-1, BELIEF_SEAT_FEAT_DIM)).reshape(
        nl_va["belief_X"].shape[0], 9, NUM_ROLES
    )
    new_nl_eval = evaluate_belief_predictions(new_probs_nl, nl_va["belief_y"], nl_va["belief_unknown_mask"])

    new_probs_makto = new_model.predict_probs_batch(makto_te["belief_X"].reshape(-1, BELIEF_SEAT_FEAT_DIM)).reshape(
        makto_te["belief_X"].shape[0], 9, NUM_ROLES
    )
    new_makto_eval = evaluate_belief_predictions(new_probs_makto, makto_te["belief_y"], makto_te["belief_unknown_mask"])

    summary = {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "train_samples": {"makto_snapshots": int(makto_tr["belief_X"].shape[0]), "nl_snapshots": int(nl_tr["belief_X"].shape[0])},
        "heldout_nl_snapshots": int(nl_va["belief_X"].shape[0]),
        "old_makto_belief_on_heldout_nl": old_nl_eval,
        "new_adapted_belief_on_heldout_nl": new_nl_eval,
        "new_adapted_belief_on_makto_test": new_makto_eval,
        "saved_model": str(out_model),
    }
    out_json = ROOT / "outputs" / "metrics" / "belief_nl_adaptation.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
