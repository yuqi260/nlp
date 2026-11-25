"""Full M3 pipeline: train BeliefMLP, BC Policy, Population Self-Play PPO, and run symbolic ablations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent.agents import BeliefRuleAgent, NeuralPolicyAgent, RandomAgent, RuleBasedAgent
from werewolf_agent.models.belief import BeliefMLP, evaluate_belief_predictions, train_belief_mlp
from werewolf_agent.models.pointer_policy import SeatEquivariantPolicyValueNet
from werewolf_agent.models.policy import ALL_DECISION_TYPES, MultiHeadPolicyValueMLP
from werewolf_agent.training import (
    evaluate_bc_policy,
    evaluate_matchup,
    refresh_policy_states_with_belief,
    train_bc_policy,
    train_self_play_ppo,
)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M3 Belief + BC + Self-Play PPO models on CPU.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "models")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs" / "metrics")
    parser.add_argument("--arch", choices=("pointer", "flat"), default="pointer")
    parser.add_argument("--belief-epochs", type=int, default=40)
    parser.add_argument("--bc-epochs", type=int, default=35)
    parser.add_argument("--ppo-iters", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=48)
    parser.add_argument("--ppo-lr", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=0.005)
    parser.add_argument("--eval-games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    train_data = load_npz_dict(args.data_dir / "makto_trajectories_train.npz")
    val_data = load_npz_dict(args.data_dir / "makto_trajectories_validation.npz")
    test_data = load_npz_dict(args.data_dir / "makto_trajectories_test.npz")

    # ---------------- 1. Train & Evaluate Belief Tracker ----------------
    print("[M3-1/4] Training seat-shared BeliefMLP on MaKTO train split...")
    t_b = time.perf_counter()
    belief_model, belief_hist = train_belief_mlp(
        train_data, val_data, epochs=args.belief_epochs, seed=args.seed
    )
    belief_sec = round(time.perf_counter() - t_b, 2)
    belief_model.save(args.model_dir / "belief_mlp.npz")

    # Compare trained BeliefMLP vs uniform prior on val & test unknown seats
    def eval_belief_split(split_d: dict[str, np.ndarray]) -> dict:
        X = split_d["belief_X"].astype(np.float32)
        y = split_d["belief_y"].astype(np.int64)
        m = split_d["belief_unknown_mask"]
        probs = belief_model.predict_probs_batch(X.reshape(-1, belief_model.in_dim)).reshape(
            X.shape[0], X.shape[1], 5
        )
        prior = np.tile(np.array([1 / 9, 1 / 9, 1 / 9, 3 / 9, 3 / 9], dtype=np.float32), (X.shape[0], X.shape[1], 1))
        return {
            "prior_baseline": evaluate_belief_predictions(prior, y, m),
            "belief_mlp": evaluate_belief_predictions(probs, y, m),
        }

    belief_eval = {
        "train_sec": belief_sec,
        "validation": eval_belief_split(val_data),
        "test": eval_belief_split(test_data),
    }
    print(f"  Done in {belief_sec}s | Test Unknown Acc: {belief_eval['test']['belief_mlp']['unknown_acc']} (Prior: {belief_eval['test']['prior_baseline']['unknown_acc']}), Macro-F1: {belief_eval['test']['belief_mlp']['macro_f1']}, Wolf-F1: {belief_eval['test']['belief_mlp']['wolf_f1']}")

    # ---------------- 2. Train & Evaluate Offline BC Policy ----------------
    print(f"[M3-2/4] Training {args.arch} Masked Policy via Behavior Cloning (BC)...", flush=True)
    t_bc = time.perf_counter()
    bc_policy, bc_hist = train_bc_policy(
        train_data, val_data, belief_model=belief_model, epochs=args.bc_epochs, arch=args.arch, seed=args.seed
    )
    bc_sec = round(time.perf_counter() - t_bc, 2)
    bc_policy.save(args.model_dir / f"policy_{args.arch}_bc.npz")

    test_refreshed = dict(test_data)
    for dt in ALL_DECISION_TYPES:
        k = f"act_{dt.value}_X"
        test_refreshed[k] = refresh_policy_states_with_belief(test_refreshed[k], belief_model)
    bc_test_eval = evaluate_bc_policy(bc_policy, test_refreshed)
    print(f"  Done in {bc_sec}s | Test Overall Action Acc: {bc_test_eval['overall_acc']} (Vote: {bc_test_eval['per_decision'].get('day_vote', {}).get('acc')}, Speech: {bc_test_eval['per_decision'].get('day_speech', {}).get('acc')})", flush=True)

    # ---------------- 3. Population-Based Self-Play Masked PPO ----------------
    print(f"[M3-3/4] Running Self-Play Masked PPO ({args.ppo_iters} iters x {args.games_per_iter} games = {args.ppo_iters * args.games_per_iter} games)...", flush=True)
    t_rl = time.perf_counter()
    rl_policy, ppo_hist = train_self_play_ppo(
        init_policy=bc_policy,
        belief_model=belief_model,
        iterations=args.ppo_iters,
        games_per_iter=args.games_per_iter,
        lr=args.ppo_lr,
        kl_coef=args.kl_coef,
        eval_every=max(1, args.ppo_iters // 10),
        eval_games=150,
        seed=args.seed,
    )
    rl_sec = round(time.perf_counter() - t_rl, 2)
    rl_policy.save(args.model_dir / f"policy_{args.arch}_bc_rl.npz")
    for row in ppo_hist:
        print(
            f"  iter={row['iter']:3d} games={row['games_seen']:5d} | "
            f"Good_vs_RuleW={row['eval_good_vs_rule_wolves']:.3f} "
            f"(VoteW={row['eval_good_vote_wolf_acc']:.3f}) | "
            f"Wolf_vs_RuleG={row['eval_wolf_vs_rule_good']:.3f} | "
            f"Combined={row['eval_combined_score']:.3f}",
            flush=True,
        )

    # Also train a pure-RL (from scratch, without BC init) ablation for completeness
    print(f"[M3-3b/4] Running Pure-RL ({args.arch}, no BC init) ablation...", flush=True)
    t_rl0 = time.perf_counter()
    scratch_policy = (
        SeatEquivariantPolicyValueNet(seed=args.seed + 7)
        if args.arch == "pointer"
        else MultiHeadPolicyValueMLP(seed=args.seed + 7)
    )
    rl_scratch_policy, ppo_scratch_hist = train_self_play_ppo(
        init_policy=scratch_policy,
        belief_model=belief_model,
        iterations=args.ppo_iters,
        games_per_iter=args.games_per_iter,
        lr=args.ppo_lr,
        eval_every=max(1, args.ppo_iters // 5),
        eval_games=120,
        kl_coef=0.0,
        seed=args.seed + 7,
    )
    rl0_sec = round(time.perf_counter() - t_rl0, 2)
    rl_scratch_policy.save(args.model_dir / f"policy_{args.arch}_rl_scratch.npz")

    # ---------------- 4. Final Standardized Symbolic Ablation Matrix ----------------
    print(f"[M3-4/4] Running standardized {args.eval_games}-game Symbolic Ablation Matrix...")
    agent_factories = {
        "Random": lambda s: RandomAgent(seed=s),
        "Rule": lambda s: RuleBasedAgent(seed=s),
        "Belief+Rule": lambda s: BeliefRuleAgent(belief_model, seed=s),
        "Belief+BC": lambda s: NeuralPolicyAgent(bc_policy, belief_model, seed=s, deterministic=False, temperature=0.7),
        "Belief+RL_scratch": lambda s: NeuralPolicyAgent(rl_scratch_policy, belief_model, seed=s, deterministic=False, temperature=0.7),
        "Belief+BC+RL": lambda s: NeuralPolicyAgent(rl_policy, belief_model, seed=s, deterministic=False, temperature=0.7),
    }

    ablation_matrix = {}
    for name, factory in agent_factories.items():
        as_good = evaluate_matchup(
            good_factory=factory,
            wolf_factory=lambda s: RuleBasedAgent(seed=s),
            games=args.eval_games,
            seed=50000,
        )
        as_wolf = evaluate_matchup(
            good_factory=lambda s: RuleBasedAgent(seed=s),
            wolf_factory=factory,
            games=args.eval_games,
            seed=60000,
        )
        mirror = evaluate_matchup(
            good_factory=factory,
            wolf_factory=factory,
            games=args.eval_games,
            seed=70000,
        )
        ablation_matrix[name] = {
            "as_good_vs_rule_wolves": as_good,
            "as_wolf_vs_rule_good": as_wolf,
            "mirror_9p": mirror,
            "combined_vs_rule": round(0.5 * (as_good["good_win_rate"] + as_wolf["wolf_win_rate"]), 4),
        }

    total_sec = round(time.perf_counter() - t0, 2)
    report = {
        "config": {
            "seed": args.seed,
            "belief_epochs": args.belief_epochs,
            "bc_epochs": args.bc_epochs,
            "ppo_iters": args.ppo_iters,
            "games_per_iter": args.games_per_iter,
            "eval_games": args.eval_games,
        },
        "timing_sec": {
            "belief": belief_sec,
            "bc": bc_sec,
            "ppo_bc_rl": rl_sec,
            "ppo_scratch": rl0_sec,
            "total": total_sec,
        },
        "belief_eval": belief_eval,
        "belief_history": belief_hist,
        "bc_test_eval": bc_test_eval,
        "bc_history": bc_hist,
        "ppo_history": ppo_hist,
        "ppo_scratch_history": ppo_scratch_hist,
        "symbolic_ablation_matrix": ablation_matrix,
    }

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.metrics_dir / f"m3_results_{args.arch}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Final Symbolic Ablation Summary (vs Fixed Rule Opponents) ===")
    print(f"{'Agent':<20} {'GoodWin':>9} {'VoteWolf':>9} {'WolfWin':>9} {'Combined':>9} {'MirrorGood':>11}")
    for name, row in ablation_matrix.items():
        gw = row["as_good_vs_rule_wolves"]["good_win_rate"]
        vw = row["as_good_vs_rule_wolves"]["good_vote_wolf_acc"]
        ww = row["as_wolf_vs_rule_good"]["wolf_win_rate"]
        cb = row["combined_vs_rule"]
        mg = row["mirror_9p"]["good_win_rate"]
        print(f"{name:<20} {gw:9.3f} {vw:9.3f} {ww:9.3f} {cb:9.3f} {mg:11.3f}")
    print(f"\nTotal M3 runtime: {total_sec}s | Saved report to {out_json}")


if __name__ == "__main__":
    main()
