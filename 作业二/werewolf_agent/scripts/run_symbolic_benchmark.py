"""Stress-test and win-rate benchmark for Symbolic Mode WerewolfEngine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from werewolf_agent import Camp, Role, WerewolfEngine
from werewolf_agent.agents import RandomAgent, RuleBasedAgent


def make_agents(policy: str, seed: int) -> dict[int, object]:
    agents = {}
    for pid in range(1, 10):
        sub_seed = seed * 31 + pid
        if policy == "random":
            agents[pid] = RandomAgent(seed=sub_seed)
        elif policy == "rule":
            agents[pid] = RuleBasedAgent(seed=sub_seed)
        elif policy == "mixed":
            agents[pid] = RuleBasedAgent(seed=sub_seed) if pid % 2 == 1 else RandomAgent(seed=sub_seed)
        else:
            raise ValueError(f"Unknown policy: {policy}")
    return agents


def run_benchmark(games: int, policy: str, seed: int) -> dict:
    engine = WerewolfEngine(mode="symbolic")
    winners = Counter()
    reasons = Counter()
    role_wins = Counter()
    role_games = Counter()
    total_days = 0
    total_steps = 0
    t0 = time.perf_counter()

    for i in range(games):
        agents = make_agents(policy, seed + i)
        res = engine.run_game(agents=agents, seed=seed + i, game_id=f"{policy}_{i}")
        winners[res.winner.value] += 1
        reasons[res.win_reason] += 1
        total_days += res.total_days
        total_steps += len(res.trajectory)
        for pid, role in res.roles.items():
            role_games[role.value] += 1
            if (role == Role.WEREWOLF and res.winner == Camp.WEREWOLF) or (
                role != Role.WEREWOLF and res.winner == Camp.GOOD
            ):
                role_wins[role.value] += 1

    elapsed = time.perf_counter() - t0
    return {
        "policy": policy,
        "games": games,
        "elapsed_sec": round(elapsed, 4),
        "games_per_sec": round(games / max(elapsed, 1e-9), 1),
        "avg_days": round(total_days / games, 3),
        "avg_steps": round(total_steps / games, 2),
        "winner_counts": dict(winners),
        "good_win_rate": round(winners["good"] / games, 4),
        "werewolf_win_rate": round(winners["werewolf"] / games, 4),
        "draw_rate": round(winners["draw"] / games, 4),
        "win_reasons": dict(reasons),
        "role_win_rates": {
            r: round(role_wins[r] / role_games[r], 4) for r in sorted(role_games)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Symbolic Werewolf stress benchmark.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--policy", choices=("random", "rule", "mixed"), default="rule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_benchmark(args.games, args.policy, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
