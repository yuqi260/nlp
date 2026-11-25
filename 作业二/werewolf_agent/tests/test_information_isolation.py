"""Strict perspective isolation tests: no role may observe forbidden private info."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import ActionRequest, ActionResponse, DecisionType, PlayerObservation, Role, WerewolfEngine
from werewolf_agent.agents import RuleBasedAgent


class AuditingAgent(RuleBasedAgent):
    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self.observations: List[PlayerObservation] = []

    def act(self, request: ActionRequest) -> ActionResponse:
        self.observations.append(request.observation)
        return super().act(request)


class TestInformationIsolation(unittest.TestCase):
    def test_strict_role_isolation_across_multiple_games(self) -> None:
        engine = WerewolfEngine(mode="symbolic")
        for seed in range(20):
            agents = {pid: AuditingAgent(seed=seed * 17 + pid) for pid in range(1, 10)}
            res = engine.run_game(agents=agents, seed=seed, game_id=f"iso_{seed}")
            true_wolves = {p for p, r in res.roles.items() if r == Role.WEREWOLF}

            for pid, agent in agents.items():
                role = res.roles[pid]
                self.assertTrue(len(agent.observations) > 0)
                for obs in agent.observations:
                    self.assertEqual(obs.player_id, pid)
                    self.assertEqual(obs.my_role, role)

                    # Werewolf teammates strictly restricted to werewolves
                    if role == Role.WEREWOLF:
                        self.assertEqual(set(obs.werewolf_teammates), true_wolves - {pid})
                    else:
                        self.assertEqual(obs.werewolf_teammates, ())

                    # Seer checks strictly restricted to Seer
                    if role != Role.SEER:
                        self.assertEqual(obs.seer_checks, ())

                    # Witch potion & attack info strictly restricted to Witch
                    if role != Role.WITCH:
                        self.assertIsNone(obs.witch_has_antidote)
                        self.assertIsNone(obs.witch_has_poison)
                        self.assertIsNone(obs.witch_tonight_attacked)
                    else:
                        if obs.witch_has_antidote is False:
                            self.assertIsNone(obs.witch_tonight_attacked)

                    # No game_end event should ever appear during active decisions
                    self.assertTrue(all(ev.event_type != "game_end" for ev in obs.public_events))

                    # Night announcements must not leak death cause or true role
                    for ev in obs.public_events:
                        if ev.event_type == "night_announcement":
                            self.assertNotIn("cause", ev.payload)
                            self.assertNotIn("roles", ev.payload)


if __name__ == "__main__":
    unittest.main()
