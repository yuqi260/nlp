"""1v8 Single-Seat Stratified Role-Assignment Evaluation (Symbolic & Live NL, parallel + resumable).

Assigns ONLY 1 seat to the evaluated agent (stratified evenly across all 5 roles:
seer, witch, hunter, werewolf, villager) while the other 8 seats are fixed opponents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent import Camp, DecisionType, Role, WerewolfEngine
from werewolf_agent.agents import (
    BaseAgent,
    DecoupledLLMAgent,
    DualSystemLLMAgent,
    DualSystemV2Agent,
    DualSystemV3Agent,
    NeuralPolicyAgent,
    PureLLMAgent,
    RandomAgent,
    RuleBasedAgent,
)
from werewolf_agent.llm_client import BailianLLMClient, MockLLMClient, UsageTracker
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.nl_parser import HybridSpeechParser

ALL_ROLES = [Role.SEER, Role.WITCH, Role.HUNTER, Role.WEREWOLF, Role.VILLAGER]
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


def stratified_assignment(target_role: Role, seed: int) -> Tuple[int, Dict[int, Role]]:
    """Deterministically shuffle all 9 roles and pick a seat holding target_role."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(9)
    roles = {pid: BASE_ROLES[int(perm[pid - 1])] for pid in range(1, 10)}
    matching_seats = [pid for pid, r in roles.items() if r == target_role]
    target_seat = int(rng.choice(matching_seats))
    return target_seat, roles


def run_symbolic_1v8(
    belief: BeliefMLP,
    bc_policy: SeatEquivariantPolicyValueNet,
    rl_policy: SeatEquivariantPolicyValueNet,
    games_per_role: int = 200,
    seed: int = 80000,
) -> Dict[str, Any]:
    """Fast 0-cost 1v8 Symbolic benchmark: 1 evaluated agent vs 8 RuleBasedAgents."""
    engine = WerewolfEngine(mode="symbolic")
    systems = {
        "Random": lambda s: RandomAgent(seed=s),
        "Rule": lambda s: RuleBasedAgent(seed=s),
        "Belief+BC": lambda s: NeuralPolicyAgent(bc_policy, belief, seed=s, temperature=0.7),
        "Belief+BC+RL": lambda s: NeuralPolicyAgent(rl_policy, belief, seed=s, temperature=0.7),
    }
    out: Dict[str, Any] = {}
    for sname, factory in systems.items():
        role_wins: Dict[str, int] = {r.value: 0 for r in ALL_ROLES}
        role_votes_ok: Dict[str, int] = {r.value: 0 for r in ALL_ROLES}
        role_votes_tot: Dict[str, int] = {r.value: 0 for r in ALL_ROLES}
        surv_sum = 0
        total_games = games_per_role * len(ALL_ROLES)

        for r_idx, role in enumerate(ALL_ROLES):
            for g in range(games_per_role):
                g_seed = seed + r_idx * 10000 + g
                t_seat, roles = stratified_assignment(role, g_seed)
                agents: Dict[int, BaseAgent] = {}
                for pid in range(1, 10):
                    s = g_seed * 17 + pid
                    agents[pid] = factory(s) if pid == t_seat else RuleBasedAgent(seed=s)
                res = engine.run_game(agents, seed=g_seed, game_id=f"sym1v8_{sname}_{role.value}_{g}", fixed_roles=roles)
                won = (res.winner == Camp.WEREWOLF) if role == Role.WEREWOLF else (res.winner == Camp.GOOD)
                if won:
                    role_wins[role.value] += 1
                if t_seat in res.alive_players:
                    surv_sum += 1
                for st in res.trajectory:
                    if st.player_id == t_seat and st.decision_type == DecisionType.DAY_VOTE.value and st.action_id >= 1:
                        role_votes_tot[role.value] += 1
                        tgt_is_wolf = res.roles[st.action_id] == Role.WEREWOLF
                        if (role == Role.WEREWOLF and not tgt_is_wolf) or (role != Role.WEREWOLF and tgt_is_wolf):
                            role_votes_ok[role.value] += 1

        per_role = {
            r.value: round(role_wins[r.value] / games_per_role, 4) for r in ALL_ROLES
        }
        good_avg = sum(role_wins[r.value] for r in ALL_ROLES if r != Role.WEREWOLF) / (4 * games_per_role)
        v_ok = sum(role_votes_ok.values())
        v_tot = sum(role_votes_tot.values())
        out[sname] = {
            "games_per_role": games_per_role,
            "overall_win_rate": round(sum(role_wins.values()) / total_games, 4),
            "good_roles_avg_win_rate": round(good_avg, 4),
            "per_role_win_rate": per_role,
            "target_vote_acc": round(v_ok / max(v_tot, 1), 4),
            "target_survival_rate": round(surv_sum / total_games, 4),
        }
    return out


def run_single_nl_1v8_game(
    system_name: str,
    target_role: Role,
    game_idx: int,
    seed: int,
    model: str,
    execute: bool,
    belief: BeliefMLP,
    rl_policy: SeatEquivariantPolicyValueNet,
    out_file: Path,
    per_game_budget_rmb: float = 1.0,
) -> Dict[str, Any]:
    if out_file.is_file():
        return json.loads(out_file.read_text(encoding="utf-8"))

    tracker_target = UsageTracker()
    tracker_opp = UsageTracker()
    if execute:
        target_client: Any = BailianLLMClient(
            model=model, enable_thinking=False, max_budget_rmb=per_game_budget_rmb, tracker=tracker_target
        )
        opp_client: Any = BailianLLMClient(
            model=model, enable_thinking=False, max_budget_rmb=per_game_budget_rmb, tracker=tracker_opp
        )
    else:
        target_client = MockLLMClient(model=f"mock-{model}", tracker=tracker_target)
        opp_client = MockLLMClient(model=f"mock-{model}", tracker=tracker_opp)

    parser_hook = HybridSpeechParser(llm_client=opp_client, prefer_regex=True)
    engine = WerewolfEngine(mode="nl", speech_parser=parser_hook)
    t_seat, fixed_roles = stratified_assignment(target_role, seed)

    agents: Dict[int, BaseAgent] = {}
    for pid in range(1, 10):
        s = seed * 37 + pid
        if pid == t_seat:
            if system_name == "B0_PureLLM":
                agents[pid] = PureLLMAgent(target_client, seed=s, name="B0_PureLLM")
            elif system_name == "Ours_Decoupled":
                agents[pid] = DecoupledLLMAgent(rl_policy, belief, target_client, seed=s, temperature=0.7, name="Ours_Decoupled")
            elif system_name == "Ours_DualSystem":
                agents[pid] = DualSystemLLMAgent(rl_policy, belief, target_client, rl_weight=0.55, seed=s, name="Ours_DualSystem")
            elif system_name == "Ours_DualSystem_v2":
                agents[pid] = DualSystemV2Agent(rl_policy, belief, target_client, rl_weight=0.50, seed=s, name="Ours_DualSystem_v2")
            elif system_name == "Ours_DualSystem_v3":
                agents[pid] = DualSystemV3Agent(rl_policy, belief, target_client, seed=s, name="Ours_DualSystem_v3")
            else:
                raise ValueError(f"Unknown system: {system_name}")
        else:
            agents[pid] = PureLLMAgent(opp_client, seed=s, name=f"Opp_{model}")

    t0 = time.perf_counter()
    res = engine.run_game(
        agents=agents,
        seed=seed,
        game_id=f"s1v8_{system_name}_{target_role.value}_{game_idx}",
        fixed_roles=fixed_roles,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    won = (res.winner == Camp.WEREWOLF) if target_role == Role.WEREWOLF else (res.winner == Camp.GOOD)
    survived = t_seat in res.alive_players

    t_vote_tot = t_vote_ok = 0
    for st in res.trajectory:
        if st.player_id == t_seat and st.decision_type == DecisionType.DAY_VOTE.value and st.action_id >= 1:
            t_vote_tot += 1
            tgt_wolf = res.roles[st.action_id] == Role.WEREWOLF
            if (target_role == Role.WEREWOLF and not tgt_wolf) or (target_role != Role.WEREWOLF and tgt_wolf):
                t_vote_ok += 1

    target_ag = agents[t_seat]
    target_gr = dict(getattr(target_ag, "guardrail_stats", {"passed": 0, "retried": 0, "fallback": 0}))

    rec = {
        "protocol": "single_seat_1v8",
        "system": system_name,
        "target_seat": t_seat,
        "target_role": target_role.value,
        "game_idx": game_idx,
        "seed": seed,
        "main_model": model,
        "opp_model": model,
        "side": f"1v8_{target_role.value}",
        "elapsed_sec": elapsed,
        "eval_won": won,
        "target_survived": survived,
        "winner": res.winner.value,
        "win_reason": res.win_reason,
        "total_days": res.total_days,
        "target_vote_total": t_vote_tot,
        "target_vote_ok": t_vote_ok,
        "target_guardrail": target_gr,
        "parser_stats": dict(parser_hook.stats),
        "usage_main": tracker_target.to_dict(),
        "usage_opp": tracker_opp.to_dict(),
        "seat_agents": {str(pid): ag.name for pid, ag in sorted(agents.items())},
        "roles": {str(pid): r.value for pid, r in sorted(res.roles.items())},
        "death_causes": {str(pid): c for pid, c in sorted(res.death_causes.items())},
        "public_events": [ev.to_dict() for ev in res.public_events],
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def aggregate_1v8(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sys: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sys[r["system"]].append(r)

    summary: Dict[str, Any] = {}
    grand_cost = 0.0
    grand_tokens = 0
    for sname, recs in sorted(by_sys.items()):
        by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_role[r["target_role"]].append(r)

        per_role = {
            role.value: {
                "games": len(by_role[role.value]),
                "win_rate": round(sum(1 for x in by_role[role.value] if x["eval_won"]) / max(len(by_role[role.value]), 1), 4),
            }
            for role in ALL_ROLES
        }
        good_recs = [r for r in recs if r["target_role"] != "werewolf"]
        wolf_recs = [r for r in recs if r["target_role"] == "werewolf"]

        v_ok = sum(r["target_vote_ok"] for r in recs)
        v_tot = sum(r["target_vote_total"] for r in recs)
        gr_p = sum(r["target_guardrail"]["passed"] for r in recs)
        gr_fb = sum(r["target_guardrail"]["fallback"] for r in recs)
        cost = sum(r["usage_main"]["estimated_cost_rmb"] + r["usage_opp"]["estimated_cost_rmb"] for r in recs)
        toks = sum(r["usage_main"]["total_tokens"] + r["usage_opp"]["total_tokens"] for r in recs)
        grand_cost += cost
        grand_tokens += toks

        good_wr = sum(1 for r in good_recs if r["eval_won"]) / max(len(good_recs), 1)
        wolf_wr = sum(1 for r in wolf_recs if r["eval_won"]) / max(len(wolf_recs), 1)
        # Standard 9-player natural deck weighting (MaKTO tournament metric: 1/9 each god, 3/9 villager, 3/9 wolf)
        deck_wr = (
            per_role["seer"]["win_rate"]
            + per_role["witch"]["win_rate"]
            + per_role["hunter"]["win_rate"]
            + 3.0 * per_role["villager"]["win_rate"]
            + 3.0 * per_role["werewolf"]["win_rate"]
        ) / 9.0
        summary[sname] = {
            "total_games": len(recs),
            "overall_win_rate": round(sum(1 for r in recs if r["eval_won"]) / max(len(recs), 1), 4),
            "deck_weighted_win_rate": round(deck_wr, 4),
            "camp_balanced_win_rate": round(0.5 * (good_wr + wolf_wr), 4),
            "good_roles_win_rate": round(good_wr, 4),
            "wolf_role_win_rate": round(wolf_wr, 4),
            "per_role": per_role,
            "target_vote_acc": round(v_ok / max(v_tot, 1), 4),
            "target_survival_rate": round(sum(1 for r in recs if r["target_survived"]) / max(len(recs), 1), 4),
            "guardrail_pass_rate": round(gr_p / max(gr_p + gr_fb, 1), 4),
            "total_tokens": toks,
            "estimated_cost_rmb": round(cost, 4),
            "avg_cost_per_game_rmb": round(cost / max(len(recs), 1), 4),
        }

    return {
        "systems": summary,
        "grand_total_games": len(records),
        "grand_total_tokens": grand_tokens,
        "grand_total_cost_rmb": round(grand_cost, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1v8 Single-Seat Stratified Evaluation.")
    parser.add_argument("--execute", action="store_true", help="Run live paid NL API games.")
    parser.add_argument("--model", type=str, default="qwen3.5-27b")
    parser.add_argument("--games-per-role", type=int, default=6, help="NL games per role (x5 roles per system).")
    parser.add_argument("--systems", type=str, default="B0_PureLLM,Ours_DualSystem_v2")
    parser.add_argument("--belief-model", type=Path, default=ROOT / "outputs" / "models" / "belief_mlp_nl.npz")
    parser.add_argument("--skip-symbolic", action="store_true")
    parser.add_argument("--sym-games-per-role", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--tag", type=str, default="qwen35_balanced_1v8_60g")
    args = parser.parse_args()

    belief = BeliefMLP.load(args.belief_model)
    bc_policy = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc.npz")
    rl_policy = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc_rl.npz")

    # 1. Fast 1,000-game Symbolic 1v8 benchmark (0 API cost)
    sym_res = None
    if not args.skip_symbolic:
        print(f"[1v8-Symbolic] Running {args.sym_games_per_role * 5} games/system vs 8 RuleBasedAgents...", flush=True)
        t_sym = time.perf_counter()
        sym_res = run_symbolic_1v8(belief, bc_policy, rl_policy, games_per_role=args.sym_games_per_role, seed=80000)
        print(f"[1v8-Symbolic] Done in {time.perf_counter() - t_sym:.2f}s:", flush=True)
        for k, v in sym_res.items():
            print(f"  {k:<14} Overall={v['overall_win_rate']:.3f} | Good4={v['good_roles_avg_win_rate']:.3f} | Roles={v['per_role_win_rate']} | VoteAcc={v['target_vote_acc']:.3f}", flush=True)

    # 2. Live/Mock 1v8 NL evaluation
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    mode_str = "live" if args.execute else "mock"
    run_dir = ROOT / "outputs" / "m5_eval" / f"single_seat_{mode_str}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sname in systems:
        for r_idx, role in enumerate(ALL_ROLES):
            for g in range(args.games_per_role):
                g_seed = args.seed + r_idx * 100 + g
                out_f = run_dir / "games" / f"{sname}_{role.value}_{g:02d}.json"
                tasks.append((sname, role, g, g_seed, out_f))

    print(f"[1v8-NL] mode={mode_str} model={args.model} total_games={len(tasks)} workers={args.workers}", flush=True)
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {
            pool.submit(
                run_single_nl_1v8_game,
                sname,
                role,
                g,
                g_seed,
                args.model,
                args.execute,
                belief,
                rl_policy,
                out_f,
            ): (sname, role.value, g)
            for sname, role, g, g_seed, out_f in tasks
        }
        done = 0
        for fut in as_completed(fut_map):
            meta = fut_map[fut]
            done += 1
            try:
                rec = fut.result()
                records.append(rec)
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"  [{done}/{len(tasks)}] last={meta[0]} role={meta[1]}#{meta[2]} "
                        f"seat={rec['target_seat']} won={rec['eval_won']} days={rec['total_days']} "
                        f"cost={rec['usage_main']['estimated_cost_rmb'] + rec['usage_opp']['estimated_cost_rmb']:.4f} RMB",
                        flush=True,
                    )
            except Exception as exc:
                errors.append(f"{meta}: {exc}")
                print(f"  [ERROR] {meta}: {exc}", flush=True)

    agg = aggregate_1v8(records)
    agg["symbolic_1v8"] = sym_res
    agg["config"] = {
        "mode": mode_str,
        "model": args.model,
        "games_per_role": args.games_per_role,
        "seed": args.seed,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "errors": errors,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 1v8 Single-Seat NL Evaluation Summary (1 Agent vs 8 PureLLM Opponents) ===")
    print(f"{'System':<19} {'5Role':>6} {'Deck9':>6} {'CampBal':>7} {'Good4':>6} {'Seer':>6} {'Witch':>6} {'Hunt':>6} {'Vill':>6} {'Wolf':>6} {'Vote':>6} {'Cost':>6}")
    for sname, row in agg["systems"].items():
        pr = row["per_role"]
        print(
            f"{sname:<19} {row['overall_win_rate']:6.3f} {row['deck_weighted_win_rate']:6.3f} {row['camp_balanced_win_rate']:7.3f} {row['good_roles_win_rate']:6.3f} "
            f"{pr['seer']['win_rate']:6.3f} {pr['witch']['win_rate']:6.3f} {pr['hunter']['win_rate']:6.3f} "
            f"{pr['villager']['win_rate']:6.3f} {pr['werewolf']['win_rate']:6.3f} "
            f"{row['target_vote_acc']:6.3f} {row['estimated_cost_rmb']:6.3f}"
        )
    print(f"\nTotal 1v8 NL games: {agg['grand_total_games']} | Total cost: {agg['grand_total_cost_rmb']:.4f} RMB | Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
