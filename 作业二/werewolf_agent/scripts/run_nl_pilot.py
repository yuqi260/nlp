"""M4 NL pilot runner: defaults to zero-cost Mock mode; requires --execute for live API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent import Role, WerewolfEngine
from werewolf_agent.agents import BeliefLLMAgent, DecoupledLLMAgent, PureLLMAgent
from werewolf_agent.llm_client import BailianLLMClient, MockLLMClient, UsageTracker
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.nl_parser import HybridSpeechParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M4 NL Werewolf pilot (Mock by default).")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable live paid API calls.")
    parser.add_argument("--model", type=str, default="qwen3.5-27b")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--matchup", choices=("ours_vs_pure", "pure_mirror", "ours_mirror"), default="ours_vs_pure")
    parser.add_argument("--max-budget-rmb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "replays")
    args = parser.parse_args()

    belief = BeliefMLP.load(ROOT / "outputs" / "models" / "belief_mlp.npz")
    rl_policy = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc_rl.npz")

    tracker = UsageTracker()
    if args.execute:
        client = BailianLLMClient(
            model=args.model,
            enable_thinking=False,
            max_budget_rmb=args.max_budget_rmb,
            tracker=tracker,
        )
        mode_tag = f"live_{args.model.replace('.', '_')}"
    else:
        client = MockLLMClient(model=f"mock-{args.model}", tracker=tracker)
        mode_tag = "mock_dry_run"

    parser_hook = HybridSpeechParser(llm_client=client, prefer_regex=True)
    engine = WerewolfEngine(mode="nl", speech_parser=parser_hook)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    guardrail_totals = {"passed": 0, "retried": 0, "fallback": 0}

    for g in range(args.games):
        agents = {}
        for pid in range(1, 10):
            s = args.seed * 31 + g * 17 + pid
            if args.matchup == "pure_mirror":
                agents[pid] = PureLLMAgent(client, seed=s)
            elif args.matchup == "ours_mirror":
                agents[pid] = DecoupledLLMAgent(rl_policy, belief, client, seed=s, temperature=0.7)
            else:
                # Odd seats Ours (Belief+BC+RL+LLM), even seats PureLLM (B0)
                if pid % 2 == 1:
                    agents[pid] = DecoupledLLMAgent(rl_policy, belief, client, seed=s, temperature=0.7)
                else:
                    agents[pid] = PureLLMAgent(client, seed=s)

        res = engine.run_game(agents=agents, seed=args.seed + g, game_id=f"nl_{mode_tag}_{g}")
        for ag in agents.values():
            for k in guardrail_totals:
                guardrail_totals[k] += ag.guardrail_stats.get(k, 0)

        replay_obj = {
            "summary": res.to_summary_dict(),
            "seat_agents": {str(pid): ag.name for pid, ag in sorted(agents.items())},
            "public_events": [ev.to_dict() for ev in res.public_events],
        }
        replay_path = args.out_dir / f"pilot_{mode_tag}_{args.matchup}_g{g}.json"
        replay_path.write_text(json.dumps(replay_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(res.to_summary_dict())

    report = {
        "mode": "live" if args.execute else "mock_dry_run",
        "model": args.model,
        "matchup": args.matchup,
        "games": args.games,
        "usage": tracker.to_dict(),
        "avg_cost_per_game_rmb": round(tracker.estimated_cost_rmb / max(args.games, 1), 5),
        "parser_stats": dict(parser_hook.stats),
        "guardrail_stats": guardrail_totals,
        "game_summaries": summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
