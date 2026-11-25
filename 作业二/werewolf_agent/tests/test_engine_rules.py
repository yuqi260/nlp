"""Unit tests for nine-player Werewolf rules, edge cases, and win conditions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import (
    ActionRequest,
    ActionResponse,
    Camp,
    DecisionType,
    Role,
    SpeechIntent,
    SpeechIntentType,
    WerewolfEngine,
    load_game_config,
)
from werewolf_agent.agents import BaseAgent


class ScriptedAgent(BaseAgent):
    def __init__(self, fn: Callable[[ActionRequest], int]) -> None:
        super().__init__(name="scripted")
        self.fn = fn

    def act(self, request: ActionRequest) -> ActionResponse:
        return ActionResponse(action_id=self.fn(request))


STANDARD_ROLES: Dict[int, Role] = {
    1: Role.SEER,
    2: Role.WITCH,
    3: Role.HUNTER,
    4: Role.WEREWOLF,
    5: Role.WEREWOLF,
    6: Role.WEREWOLF,
    7: Role.VILLAGER,
    8: Role.VILLAGER,
    9: Role.VILLAGER,
}


class TestEngineRules(unittest.TestCase):
    def test_config_loads_and_validates(self) -> None:
        cfg = load_game_config()
        self.assertEqual(cfg.player_count, 9)
        self.assertEqual(cfg.end_rule, "side_kill")
        self.assertFalse(cfg.has_sheriff)
        self.assertFalse(cfg.witch_allow_both_same_night)

    def test_speech_intent_roundtrip(self) -> None:
        for aid in range(SpeechIntent.action_space_size(9)):
            intent = SpeechIntent.from_action_id(aid, 9)
            self.assertEqual(intent.to_action_id(9), aid)
            self.assertTrue(len(intent.render_template_zh(1)) > 0)

    def test_witch_poison_blocks_hunter_shoot(self) -> None:
        hunter_queried = {"count": 0}

        def policy(req: ActionRequest) -> int:
            dt = req.decision_type
            if dt == DecisionType.NIGHT_SEER_CHECK:
                return 3 if 3 in req.legal_action_ids else req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WOLF_KILL:
                return 6 if 6 in req.legal_action_ids else req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WITCH_ACT:
                # Day 1 poison Hunter (player 3 -> action 1 + 3 = 4)
                return 4 if 4 in req.legal_action_ids else 0
            if dt == DecisionType.HUNTER_SHOOT:
                hunter_queried["count"] += 1
                return 0
            if dt == DecisionType.DAY_SPEECH:
                return 0
            if dt == DecisionType.DAY_VOTE:
                # Vote out wolves 4, 5, 6 in order
                for target in (4, 5, 6):
                    if target in req.legal_action_ids:
                        return target
                return 0
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        res = engine.run_game(agents, seed=1, fixed_roles=STANDARD_ROLES)
        self.assertEqual(hunter_queried["count"], 0)
        self.assertEqual(res.death_causes.get(3), "witch_poison")
        self.assertNotIn(3, {ev.actor for ev in res.public_events if ev.event_type == "hunter_shoot"})

    def test_hunter_wolf_killed_can_shoot_and_reveal(self) -> None:
        def policy(req: ActionRequest) -> int:
            dt = req.decision_type
            if dt == DecisionType.NIGHT_SEER_CHECK:
                return 3 if 3 in req.legal_action_ids else req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WOLF_KILL:
                return 2 if 2 in req.legal_action_ids else req.legal_action_ids[0]  # kill Hunter on Day 1
            if dt == DecisionType.NIGHT_WITCH_ACT:
                return 0  # do not save
            if dt == DecisionType.HUNTER_SHOOT:
                return 4  # shoot Wolf 4
            if dt == DecisionType.DAY_SPEECH:
                return 0
            if dt == DecisionType.DAY_VOTE:
                for target in (5, 6):
                    if target in req.legal_action_ids:
                        return target
                return 0
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        res = engine.run_game(agents, seed=2, fixed_roles=STANDARD_ROLES)
        shoot_events = [ev for ev in res.public_events if ev.event_type == "hunter_shoot"]
        self.assertEqual(len(shoot_events), 1)
        self.assertEqual(shoot_events[0].actor, 3)
        self.assertEqual(shoot_events[0].payload["target"], 4)
        self.assertEqual(res.death_causes.get(4), "hunter_shoot")
        self.assertEqual(res.winner, Camp.GOOD)

    def test_side_kill_all_villagers_eliminated_wolf_wins(self) -> None:
        def policy(req: ActionRequest) -> int:
            dt = req.decision_type
            if dt == DecisionType.NIGHT_SEER_CHECK:
                return req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WOLF_KILL:
                for v in (7, 8, 9):
                    if (v - 1) in req.legal_action_ids:
                        return v - 1
                return req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WITCH_ACT:
                return 0
            if dt == DecisionType.DAY_SPEECH:
                return 0
            if dt == DecisionType.DAY_VOTE:
                # Exile villagers too
                for v in (7, 8, 9):
                    if v in req.legal_action_ids:
                        return v
                return 0
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        res = engine.run_game(agents, seed=3, fixed_roles=STANDARD_ROLES)
        self.assertEqual(res.winner, Camp.WEREWOLF)
        self.assertEqual(res.win_reason, "all_villagers_eliminated")

    def test_tie_vote_triggers_revote_and_caps_rounds(self) -> None:
        vote_rounds_seen: List[tuple[int, int]] = []

        def policy(req: ActionRequest) -> int:
            dt = req.decision_type
            obs = req.observation
            if dt == DecisionType.NIGHT_WOLF_KILL:
                return 6 if 6 in req.legal_action_ids else req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WITCH_ACT:
                return 1 if (obs.day == 1 and 1 in req.legal_action_ids) else 0
            if dt == DecisionType.DAY_SPEECH:
                return 0
            if dt == DecisionType.DAY_VOTE:
                # Day 1 force 4-vs-4 tie between seat 1 and seat 4 (seat 9 abstains)
                if obs.day == 1:
                    vote_rounds_seen.append((obs.player_id, len(obs.vote_candidates)))
                    if obs.player_id in (1, 2, 3, 7) and 4 in req.legal_action_ids:
                        return 4
                    if obs.player_id in (4, 5, 6, 8) and 1 in req.legal_action_ids:
                        return 1
                    return 0
                for target in (4, 5, 6):
                    if target in req.legal_action_ids:
                        return target
                return 0
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        res = engine.run_game(agents, seed=4, fixed_roles=STANDARD_ROLES)
        day1_votes = [ev for ev in res.public_events if ev.event_type == "vote_result" and ev.day == 1]
        self.assertEqual(len(day1_votes), 2)  # 1 initial + 1 revote
        self.assertEqual(day1_votes[0].payload["tied"], [1, 4])
        self.assertIsNone(day1_votes[1].payload["exiled"])

    def test_witch_self_save_night1_only(self) -> None:
        night2_can_heal = {"value": None}

        def policy(req: ActionRequest) -> int:
            dt = req.decision_type
            obs = req.observation
            if dt == DecisionType.NIGHT_WOLF_KILL:
                # Day 1 kill villager 7; Day 2 kill Witch (player 2 -> index 1)
                if obs.day == 1 and 6 in req.legal_action_ids:
                    return 6
                if obs.day == 2 and 1 in req.legal_action_ids:
                    return 1
                return req.legal_action_ids[0]
            if dt == DecisionType.NIGHT_WITCH_ACT:
                if obs.day == 2:
                    night2_can_heal["value"] = 1 in req.legal_action_ids
                return 0
            if dt == DecisionType.DAY_SPEECH:
                return 0
            if dt == DecisionType.DAY_VOTE:
                for target in (4, 5, 6):
                    if target in req.legal_action_ids:
                        return target
                return 0
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        engine.run_game(agents, seed=6, fixed_roles=STANDARD_ROLES)
        self.assertFalse(night2_can_heal["value"])

    def test_illegal_action_raises(self) -> None:
        def bad_policy(req: ActionRequest) -> int:
            if req.decision_type == DecisionType.NIGHT_SEER_CHECK:
                return req.observation.player_id - 1  # cannot check self
            return req.legal_action_ids[0]

        agents = {pid: ScriptedAgent(bad_policy) for pid in range(1, 10)}
        engine = WerewolfEngine(mode="symbolic")
        with self.assertRaises(ValueError):
            engine.run_game(agents, seed=5, fixed_roles=STANDARD_ROLES)


if __name__ == "__main__":
    unittest.main()
