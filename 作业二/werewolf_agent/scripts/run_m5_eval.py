"""M5 Formal Natural-Language Controlled Ablation & Cross-Model Evaluation (parallel + resumable)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent import Camp, DecisionType, Role, WerewolfEngine
from werewolf_agent.agents import BaseAgent, BeliefLLMAgent, DecoupledLLMAgent, DualSystemLLMAgent, PureLLMAgent
from werewolf_agent.llm_client import BailianLLMClient, MockLLMClient, UsageTracker
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.nl_parser import HybridSpeechParser

BASE_ROLES = [
    Role.SEER,
    Role.WITCH,
    Role.HUNTER,
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
]


def deterministic_roles(seed: int) -> Dict[int, Role]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(9)
    return {pid: BASE_ROLES[int(perm[pid - 1])] for pid in range(1, 10)}


def make_agent(
    agent_type: str,
    client: Any,
    belief: BeliefMLP,
    bc_policy: SeatEquivariantPolicyValueNet,
    rl_policy: SeatEquivariantPolicyValueNet,
    seed: int,
) -> BaseAgent:
    if agent_type == "B0_PureLLM":
        return PureLLMAgent(client, seed=seed, name="B0_PureLLM")
    if agent_type == "B1_BeliefLLM":
        return BeliefLLMAgent(client, belief, seed=seed, name="B1_BeliefLLM")
    if agent_type == "B2_BeliefBC":
        return DecoupledLLMAgent(bc_policy, belief, client, seed=seed, temperature=0.7, name="B2_BeliefBC")
    if agent_type == "Ours_BeliefBCRL":
        return DecoupledLLMAgent(rl_policy, belief, client, seed=seed, temperature=0.7, name="Ours_BeliefBCRL")
    if agent_type == "Ours_DualSystem":
        return DualSystemLLMAgent(rl_policy, belief, client, rl_weight=0.55, seed=seed, name="Ours_DualSystem")
    raise ValueError(f"Unknown agent_type: {agent_type}")


def run_single_eval_game(
    system_name: str,
    side: str,  # "as_good" | "as_wolf"
    game_idx: int,
    seed: int,
    main_model: str,
    opp_model: str,
    execute: bool,
    belief: BeliefMLP,
    bc_policy: SeatEquivariantPolicyValueNet,
    rl_policy: SeatEquivariantPolicyValueNet,
    out_file: Path,
    per_game_budget_rmb: float = 1.0,
) -> Dict[str, Any]:
    if out_file.is_file():
        return json.loads(out_file.read_text(encoding="utf-8"))

    tracker_main = UsageTracker()
    tracker_opp = UsageTracker()
    if execute:
        main_client: Any = BailianLLMClient(
            model=main_model, enable_thinking=False, max_budget_rmb=per_game_budget_rmb, tracker=tracker_main
        )
        opp_client: Any = BailianLLMClient(
            model=opp_model, enable_thinking=False, max_budget_rmb=per_game_budget_rmb, tracker=tracker_opp
        )
    else:
        main_client = MockLLMClient(model=f"mock-{main_model}", tracker=tracker_main)
        opp_client = MockLLMClient(model=f"mock-{opp_model}", tracker=tracker_opp)

    parser_hook = HybridSpeechParser(llm_client=main_client, prefer_regex=True)
    engine = WerewolfEngine(mode="nl", speech_parser=parser_hook)
    fixed_roles = deterministic_roles(seed)

    agents: Dict[int, BaseAgent] = {}
    for pid, r in fixed_roles.items():
        s = seed * 37 + pid
        is_eval_seat = (r != Role.WEREWOLF) if side == "as_good" else (r == Role.WEREWOLF)
        if is_eval_seat:
            agents[pid] = make_agent(system_name, main_client, belief, bc_policy, rl_policy, s)
        else:
            agents[pid] = PureLLMAgent(opp_client, seed=s, name=f"Opp_{opp_model}")

    t0 = time.perf_counter()
    res = engine.run_game(
        agents=agents,
        seed=seed,
        game_id=f"m5_{system_name}_{side}_{game_idx}",
        fixed_roles=fixed_roles,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    # Compute tactical process stats for the evaluated side
    eval_won = (res.winner == Camp.GOOD) if side == "as_good" else (res.winner == Camp.WEREWOLF)
    good_vote_total = good_vote_wolf = 0
    wolf_vote_total = wolf_vote_good = 0
    witch_pois_total = witch_pois_wolf = 0
    hunter_shoot_total = hunter_shoot_wolf = 0
    eval_alive_end = 0
    eval_total_seats = 0

    for pid, r in res.roles.items():
        is_eval = (r != Role.WEREWOLF) if side == "as_good" else (r == Role.WEREWOLF)
        if is_eval:
            eval_total_seats += 1
            if pid in res.alive_players:
                eval_alive_end += 1

    for st in res.trajectory:
        r = res.roles[st.player_id]
        if st.decision_type == DecisionType.DAY_VOTE.value and st.action_id >= 1:
            if r != Role.WEREWOLF:
                good_vote_total += 1
                if res.roles[st.action_id] == Role.WEREWOLF:
                    good_vote_wolf += 1
            else:
                wolf_vote_total += 1
                if res.roles[st.action_id] != Role.WEREWOLF:
                    wolf_vote_good += 1
        elif st.decision_type == DecisionType.NIGHT_WITCH_ACT.value and st.action_id >= 2:
            witch_pois_total += 1
            if res.roles[st.action_id - 1] == Role.WEREWOLF:
                witch_pois_wolf += 1
        elif st.decision_type == DecisionType.HUNTER_SHOOT.value and st.action_id >= 1:
            hunter_shoot_total += 1
            if res.roles[st.action_id] == Role.WEREWOLF:
                hunter_shoot_wolf += 1

    # Collect guardrail stats for evaluated seats only
    eval_guardrail = {"passed": 0, "retried": 0, "fallback": 0}
    for pid, ag in agents.items():
        r = res.roles[pid]
        is_eval = (r != Role.WEREWOLF) if side == "as_good" else (r == Role.WEREWOLF)
        if is_eval and hasattr(ag, "guardrail_stats"):
            for k in eval_guardrail:
                eval_guardrail[k] += ag.guardrail_stats.get(k, 0)

    record = {
        "system": system_name,
        "side": side,
        "game_idx": game_idx,
        "seed": seed,
        "main_model": main_model,
        "opp_model": opp_model,
        "elapsed_sec": elapsed,
        "eval_won": eval_won,
        "winner": res.winner.value,
        "win_reason": res.win_reason,
        "total_days": res.total_days,
        "eval_survival_rate": round(eval_alive_end / max(eval_total_seats, 1), 4),
        "process_counts": {
            "good_vote_total": good_vote_total,
            "good_vote_wolf": good_vote_wolf,
            "wolf_vote_total": wolf_vote_total,
            "wolf_vote_good": wolf_vote_good,
            "witch_pois_total": witch_pois_total,
            "witch_pois_wolf": witch_pois_wolf,
            "hunter_shoot_total": hunter_shoot_total,
            "hunter_shoot_wolf": hunter_shoot_wolf,
        },
        "eval_guardrail": eval_guardrail,
        "parser_stats": dict(parser_hook.stats),
        "usage_main": tracker_main.to_dict(),
        "usage_opp": tracker_opp.to_dict(),
        "seat_agents": {str(pid): ag.name for pid, ag in sorted(agents.items())},
        "roles": {str(pid): r.value for pid, r in sorted(res.roles.items())},
        "death_causes": {str(pid): c for pid, c in sorted(res.death_causes.items())},
        "public_events": [ev.to_dict() for ev in res.public_events],
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sys: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in records:
        by_sys.setdefault(r["system"], {"as_good": [], "as_wolf": []})[r["side"]].append(r)

    summary: Dict[str, Any] = {}
    total_cost = 0.0
    total_tokens = 0

    for sys_name, sides in sorted(by_sys.items()):
        g_recs = sides["as_good"]
        w_recs = sides["as_wolf"]
        all_recs = g_recs + w_recs

        g_win = sum(1 for r in g_recs if r["eval_won"]) / max(len(g_recs), 1)
        w_win = sum(1 for r in w_recs if r["eval_won"]) / max(len(w_recs), 1)
        comb = 0.5 * (g_win + w_win) if (g_recs and w_recs) else (g_win if g_recs else w_win)

        # Tactical rates from the games where this system played that camp
        gv_tot = sum(r["process_counts"]["good_vote_total"] for r in g_recs)
        gv_hit = sum(r["process_counts"]["good_vote_wolf"] for r in g_recs)
        wv_tot = sum(r["process_counts"]["wolf_vote_total"] for r in w_recs)
        wv_hit = sum(r["process_counts"]["wolf_vote_good"] for r in w_recs)
        wp_tot = sum(r["process_counts"]["witch_pois_total"] for r in g_recs)
        wp_hit = sum(r["process_counts"]["witch_pois_wolf"] for r in g_recs)
        hs_tot = sum(r["process_counts"]["hunter_shoot_total"] for r in g_recs)
        hs_hit = sum(r["process_counts"]["hunter_shoot_wolf"] for r in g_recs)

        gr_pass = sum(r["eval_guardrail"]["passed"] for r in all_recs)
        gr_retry = sum(r["eval_guardrail"]["retried"] for r in all_recs)
        gr_fb = sum(r["eval_guardrail"]["fallback"] for r in all_recs)
        gr_tot = gr_pass + gr_fb

        pr_reg = sum(r["parser_stats"]["regex_hits"] for r in all_recs)
        pr_llm = sum(r["parser_stats"]["llm_calls"] for r in all_recs)

        cost = sum(r["usage_main"]["estimated_cost_rmb"] + r["usage_opp"]["estimated_cost_rmb"] for r in all_recs)
        toks = sum(r["usage_main"]["total_tokens"] + r["usage_opp"]["total_tokens"] for r in all_recs)
        total_cost += cost
        total_tokens += toks

        summary[sys_name] = {
            "games_as_good": len(g_recs),
            "games_as_wolf": len(w_recs),
            "good_win_rate": round(g_win, 4),
            "wolf_win_rate": round(w_win, 4),
            "combined_win_rate": round(comb, 4),
            "avg_days": round(sum(r["total_days"] for r in all_recs) / max(len(all_recs), 1), 3),
            "good_vote_wolf_acc": round(gv_hit / max(gv_tot, 1), 4),
            "wolf_vote_good_acc": round(wv_hit / max(wv_tot, 1), 4),
            "witch_poison_wolf_acc": round(wp_hit / max(wp_tot, 1), 4) if wp_tot else None,
            "hunter_shoot_wolf_acc": round(hs_hit / max(hs_tot, 1), 4) if hs_tot else None,
            "guardrail_pass_rate": round(gr_pass / max(gr_tot, 1), 4),
            "guardrail_retries": gr_retry,
            "guardrail_fallbacks": gr_fb,
            "parser_regex_hit_rate": round(pr_reg / max(pr_reg + pr_llm, 1), 4),
            "total_tokens": toks,
            "estimated_cost_rmb": round(cost, 4),
            "avg_cost_per_game_rmb": round(cost / max(len(all_recs), 1), 4),
        }

    return {
        "systems": summary,
        "grand_total_games": len(records),
        "grand_total_tokens": total_tokens,
        "grand_total_cost_rmb": round(total_cost, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5 Controlled NL Evaluation Suite.")
    parser.add_argument("--execute", action="store_true", help="Enable live paid API calls.")
    parser.add_argument(
        "--suite",
        choices=("ablation", "cross_model", "ours_only", "dual_only", "cross_dual"),
        default="ablation",
    )
    parser.add_argument("--main-model", type=str, default="qwen3.5-27b")
    parser.add_argument("--opp-model", type=str, default="qwen3.5-27b")
    parser.add_argument("--rl-model-path", type=Path, default=ROOT / "outputs" / "models" / "policy_pointer_bc_rl.npz")
    parser.add_argument("--games-per-side", type=int, default=10, help="Games as Good + games as Wolf per system.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--tag", type=str, default="v1")
    args = parser.parse_args()

    belief = BeliefMLP.load(ROOT / "outputs" / "models" / "belief_mlp.npz")
    bc_policy = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc.npz")
    rl_policy = SeatEquivariantPolicyValueNet.load(args.rl_model_path)

    if args.suite == "ablation":
        systems = ["B0_PureLLM", "B1_BeliefLLM", "B2_BeliefBC", "Ours_BeliefBCRL"]
    elif args.suite == "ours_only":
        systems = ["Ours_BeliefBCRL"]
    elif args.suite == "dual_only":
        systems = ["Ours_DualSystem"]
    elif args.suite == "cross_dual":
        systems = ["Ours_DualSystem"]
    else:
        systems = ["B0_PureLLM", "Ours_BeliefBCRL"]

    mode_str = "live" if args.execute else "mock"
    run_dir = ROOT / "outputs" / "m5_eval" / f"{args.suite}_{mode_str}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, int, int, Path]] = []
    for sys_name in systems:
        for side_offset, side in enumerate(("as_good", "as_wolf")):
            for g_idx in range(args.games_per_side):
                # Identical paired seeds across systems for low-variance comparison!
                seed = args.seed + side_offset * 1000 + g_idx
                out_f = run_dir / "games" / f"{sys_name}_{side}_{g_idx:03d}.json"
                tasks.append((sys_name, side, g_idx, seed, out_f))

    print(
        f"[M5] suite={args.suite} mode={mode_str} main={args.main_model} opp={args.opp_model} "
        f"total_games={len(tasks)} workers={args.workers}",
        flush=True,
    )
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {
            pool.submit(
                run_single_eval_game,
                sys_name,
                side,
                g_idx,
                seed,
                args.main_model,
                args.opp_model,
                args.execute,
                belief,
                bc_policy,
                rl_policy,
                out_f,
            ): (sys_name, side, g_idx)
            for sys_name, side, g_idx, seed, out_f in tasks
        }
        done = 0
        for fut in as_completed(fut_map):
            meta = fut_map[fut]
            done += 1
            try:
                rec = fut.result()
                records.append(rec)
                if done % 8 == 0 or done == len(tasks):
                    print(
                        f"  [{done}/{len(tasks)}] last={meta[0]} {meta[1]}#{meta[2]} "
                        f"won={rec['eval_won']} days={rec['total_days']} "
                        f"cost={rec['usage_main']['estimated_cost_rmb'] + rec['usage_opp']['estimated_cost_rmb']:.4f} RMB",
                        flush=True,
                    )
            except Exception as exc:
                err_msg = f"{meta}: {exc}"
                errors.append(err_msg)
                print(f"  [ERROR] {err_msg}", flush=True)

    agg = aggregate_records(records)
    agg["config"] = {
        "suite": args.suite,
        "mode": mode_str,
        "main_model": args.main_model,
        "opp_model": args.opp_model,
        "games_per_side": args.games_per_side,
        "seed": args.seed,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "errors": errors,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== M5 Natural Language Evaluation Summary ===")
    print(f"{'System':<18} {'GoodWin':>8} {'WolfWin':>8} {'Combined':>9} {'VoteWolf':>9} {'GR_Pass':>8} {'CostRMB':>8}")
    for sname, row in agg["systems"].items():
        print(
            f"{sname:<18} {row['good_win_rate']:8.3f} {row['wolf_win_rate']:8.3f} "
            f"{row['combined_win_rate']:9.3f} {row['good_vote_wolf_acc']:9.3f} "
            f"{row['guardrail_pass_rate']:8.3f} {row['estimated_cost_rmb']:8.3f}"
        )
    print(f"\nTotal games: {agg['grand_total_games']} | Total cost: {agg['grand_total_cost_rmb']:.4f} RMB | Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
