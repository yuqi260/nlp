"""End-to-end tests for Random/Rule agents and NL mode speech hook."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import (
    ActionRequest,
    ActionResponse,
    Camp,
    DecisionType,
    SpeechIntent,
    SpeechIntentType,
    WerewolfEngine,
)
from werewolf_agent.agents import RandomAgent, RuleBasedAgent


class MockNLAgent(RuleBasedAgent):
    def act(self, request: ActionRequest) -> ActionResponse:
        resp = super().act(request)
        if request.decision_type == DecisionType.DAY_SPEECH:
            return ActionResponse(
                action_id=resp.action_id,
                utterance=f"[NL测试发言] {request.observation.player_id}号表水。",
            )
        return resp


class TestBenchmarkAgents(unittest.TestCase):
    def test_random_and_rule_games_complete_cleanly(self) -> None:
        engine = WerewolfEngine(mode="symbolic")
        for seed in range(30):
            agents = {
                pid: (RuleBasedAgent(seed=seed * 13 + pid) if pid <= 5 else RandomAgent(seed=seed * 13 + pid))
                for pid in range(1, 10)
            }
            res = engine.run_game(agents, seed=seed, game_id=f"mix_{seed}")
            self.assertIn(res.winner, (Camp.GOOD, Camp.WEREWOLF, Camp.DRAW))
            self.assertGreaterEqual(res.total_days, 1)
            self.assertEqual(res.public_events[-1].event_type, "game_end")

    def test_nl_mode_preserves_utterance_and_supports_parser_hook(self) -> None:
        parser_calls = {"count": 0}

        def dummy_parser(utterance: str, obs, legal_ids):
            parser_calls["count"] += 1
            # Map every NL utterance back to STAY_LOW (0) to verify hook override
            return 0 if 0 in legal_ids else legal_ids[0]

        engine = WerewolfEngine(mode="nl", speech_parser=dummy_parser)
        agents = {pid: MockNLAgent(seed=100 + pid) for pid in range(1, 10)}
        res = engine.run_game(agents, seed=99, game_id="nl_hook_test")
        speeches = [ev for ev in res.public_events if ev.event_type == "speech"]
        self.assertGreater(len(speeches), 0)
        self.assertEqual(parser_calls["count"], len(speeches))
        for sp in speeches:
            self.assertIn("[NL测试发言]", sp.payload["text"])
            self.assertEqual(sp.payload["intent"]["intent_type"], SpeechIntentType.STAY_LOW.value)
            self.assertEqual(SpeechIntent.from_action_id(sp.payload["action_id"]).intent_type, SpeechIntentType.STAY_LOW)


if __name__ == "__main__":
    unittest.main()
