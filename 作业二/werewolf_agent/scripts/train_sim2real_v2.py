"""Train Sim-to-Real aligned Pointer PPO (v2) from existing BeliefMLP + BC checkpoint."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import DecisionType, Role, SpeechIntent, WerewolfEngine
from werewolf_agent.agents import NeuralPolicyAgent, RuleBasedAgent
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.training import evaluate_matchup, train_self_play_ppo


def main() -> None:
    belief = BeliefMLP.load(ROOT / "outputs" / "models" / "belief_mlp.npz")
    bc = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc.npz")

    print("[Sim2Real-v2] Training Sim-to-Real aligned Pointer PPO (100 iters x 48 games)...", flush=True)
    t0 = time.perf_counter()
    v2, hist = train_self_play_ppo(
        init_policy=bc,
        belief_model=belief,
        iterations=100,
        games_per_iter=48,
        lr=0.001,
        kl_coef=0.008,  # 5x on DAY_SPEECH -> 0.04 speech anchor!
        eval_every=10,
        eval_games=150,
        seed=142,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    out_model = ROOT / "outputs" / "models" / "policy_pointer_bc_rl_v2.npz"
    v2.save(out_model)

    for row in hist:
        print(
            f"  iter={row['iter']:3d} games={row['games_seen']:5d} | "
            f"Good_vs_RuleW={row['eval_good_vs_rule_wolves']:.3f} "
            f"(VoteW={row['eval_good_vote_wolf_acc']:.3f}) | "
            f"Wolf_vs_RuleG={row['eval_wolf_vs_rule_good']:.3f} | "
            f"Combined={row['eval_combined_score']:.3f}",
            flush=True,
        )

    # Audit speech distribution of v2
    eng = WerewolfEngine(mode="symbolic")
    good_sp, wolf_sp = Counter(), Counter()
    for g in range(30):
        agents = {pid: NeuralPolicyAgent(v2, belief, seed=g * 17 + pid, temperature=0.7) for pid in range(1, 10)}
        res = eng.run_game(agents, seed=8000 + g)
        for st in res.trajectory:
            if st.decision_type == DecisionType.DAY_SPEECH.value:
                it = SpeechIntent.from_action_id(st.action_id).intent_type.value
                (wolf_sp if res.roles[st.player_id] == Role.WEREWOLF else good_sp)[it] += 1

    as_g = evaluate_matchup(
        lambda s: NeuralPolicyAgent(v2, belief, seed=s, temperature=0.7),
        lambda s: RuleBasedAgent(seed=s),
        games=300,
        seed=50000,
    )
    as_w = evaluate_matchup(
        lambda s: RuleBasedAgent(seed=s),
        lambda s: NeuralPolicyAgent(v2, belief, seed=s, temperature=0.7),
        games=300,
        seed=60000,
    )
    out_metrics = ROOT / "outputs" / "metrics" / "m3_sim2real_v2.json"
    out_metrics.write_text(
        json.dumps(
            {
                "elapsed_sec": elapsed,
                "history": hist,
                "as_good_vs_rule": as_g,
                "as_wolf_vs_rule": as_w,
                "combined_vs_rule": round(0.5 * (as_g["good_win_rate"] + as_w["wolf_win_rate"]), 4),
                "good_speeches_30g": dict(good_sp.most_common()),
                "wolf_speeches_30g": dict(wolf_sp.most_common()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Sim2Real-v2] Done in {elapsed}s | Good={as_g['good_win_rate']} Wolf={as_w['wolf_win_rate']} | Wolf speeches={dict(wolf_sp.most_common(5))}", flush=True)


if __name__ == "__main__":
    main()
