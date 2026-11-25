"""Unit tests for M4 NL pipeline: Guardrail, HybridSpeechParser, MockLLM, and NL agents."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import Camp, Phase, PlayerObservation, PublicEvent, Role, SpeechIntent, SpeechIntentType, WerewolfEngine
from werewolf_agent.agents import BeliefLLMAgent, DecoupledLLMAgent, DualSystemLLMAgent, DualSystemV2Agent, DualSystemV3Agent, PureLLMAgent
from werewolf_agent.fact_tracker import build_cross_round_notes, build_verified_fact_board
from werewolf_agent.llm_client import BailianLLMClient, BudgetExceededError, MockLLMClient, UsageTracker
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.nl_parser import HybridSpeechParser, regex_parse_speech_intent
from werewolf_agent.nl_speech import check_speech_guardrail


class TestNLPipeline(unittest.TestCase):
    def test_guardrail_catches_wolf_leak_and_missing_intent(self) -> None:
        obs_wolf = PlayerObservation(
            game_id="g",
            mode="nl",
            day=1,
            phase=Phase.DAY_SPEECH,
            player_id=4,
            my_role=Role.WEREWOLF,
            player_count=9,
            alive_players=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            dead_players=(),
            revealed_roles={},
            public_events=(),
            werewolf_teammates=(5, 6),
        )
        r1 = check_speech_guardrail("我是狼人，5号是我队友，请大家投1号。", obs_wolf)
        self.assertFalse(r1.passed)
        self.assertIn("wolf_self_or_teammate_leak", r1.violations)

        r2 = check_speech_guardrail("游戏已经结束，好人已经获胜了！", obs_wolf)
        self.assertFalse(r2.passed)
        self.assertIn("premature_game_over_hallucination", r2.violations)

        intent = SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, 7)
        r3 = check_speech_guardrail("我是平民，今天先听听大家怎么说。", obs_wolf, expected_intent=intent)
        self.assertFalse(r3.passed)
        self.assertIn("missing_target_7", r3.violations)
        self.assertIn("missing_seer_claim", r3.violations)

        r4 = check_speech_guardrail("我跳预言家，昨晚验了7号是查杀狼人，请大家今天全票出7号！", obs_wolf, expected_intent=intent)
        self.assertTrue(r4.passed)

    def test_hybrid_parser_regex_fastpath_and_llm_fallback(self) -> None:
        legal = list(range(SpeechIntent.action_space_size(9)))
        hit = regex_parse_speech_intent("我跳预言家，昨晚查验了3号是查杀狼人！", speaker=1, legal_action_ids=legal)
        self.assertIsNotNone(hit)
        self.assertEqual(SpeechIntent.from_action_id(hit[0]), SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, 3))

        # Regression test from live qwen3.5-27b pilot
        hit2 = regex_parse_speech_intent("我是 4 号，真预言家。首夜验 2 号是狼人，今天全票出 2 号。", speaker=4, legal_action_ids=legal)
        self.assertIsNotNone(hit2)
        self.assertEqual(SpeechIntent.from_action_id(hit2[0]), SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, 2))

        # Regression test: third-person quote "2号跳预言家查杀4号" must NOT be parsed as speaker 3 claiming Seer!
        hit3 = regex_parse_speech_intent("3 号发言。我是好人，2 号跳预言家查杀 4 号，先听后面发言。", speaker=3, legal_action_ids=legal)
        if hit3 is not None:
            self.assertNotIn(
                SpeechIntent.from_action_id(hit3[0]).intent_type,
                (SpeechIntentType.CLAIM_SEER_GOOD, SpeechIntentType.CLAIM_SEER_WOLF),
            )

        mock = MockLLMClient()
        parser = HybridSpeechParser(llm_client=mock, prefer_regex=True)
        obs = PlayerObservation(
            game_id="g",
            mode="nl",
            day=1,
            phase=Phase.DAY_SPEECH,
            player_id=2,
            my_role=Role.VILLAGER,
            player_count=9,
            alive_players=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            dead_players=(),
            revealed_roles={},
            public_events=(),
        )
        # Ambiguous mention of 5号 without clear accuse/defend keywords -> triggers LLM fallback
        aid = parser("5号刚才那句话挺耐人寻味的，大家品一品。", obs, legal)
        self.assertIn(aid, legal)
        self.assertEqual(parser.stats["llm_calls"], 1)

    def test_budget_circuit_breaker_blocks_overspend(self) -> None:
        tracker = UsageTracker(estimated_cost_rmb=1.05)
        client = BailianLLMClient(api_key="dummy-key", max_budget_rmb=1.0, tracker=tracker)
        with self.assertRaises(BudgetExceededError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_end_to_end_nl_game_with_all_three_llm_agent_types(self) -> None:
        belief = BeliefMLP(seed=1)
        policy = SeatEquivariantPolicyValueNet(seed=2)
        tracker = UsageTracker()
        mock = MockLLMClient(tracker=tracker)
        parser = HybridSpeechParser(llm_client=mock, prefer_regex=True)
        engine = WerewolfEngine(mode="nl", speech_parser=parser)

        agents = {}
        for pid in range(1, 10):
            if pid <= 2:
                agents[pid] = PureLLMAgent(mock, seed=pid)
            elif pid <= 4:
                agents[pid] = BeliefLLMAgent(mock, belief, seed=pid)
            elif pid <= 6:
                agents[pid] = DecoupledLLMAgent(policy, belief, mock, seed=pid)
            elif pid <= 7:
                agents[pid] = DualSystemLLMAgent(policy, belief, mock, seed=pid)
            elif pid <= 8:
                agents[pid] = DualSystemV2Agent(policy, belief, mock, seed=pid)
            else:
                agents[pid] = DualSystemV3Agent(policy, belief, mock, seed=pid)

        res = engine.run_game(agents, seed=77, game_id="nl_e2e_mock")
        self.assertIn(res.winner, (Camp.GOOD, Camp.WEREWOLF, Camp.DRAW))
        speeches = [ev for ev in res.public_events if ev.event_type == "speech"]
        self.assertGreater(len(speeches), 0)
        self.assertGreater(tracker.calls, 0)
        # Regex fast-path should handle most structured speeches without calling parser LLM
        self.assertGreater(parser.stats["regex_hits"], 0)

    def test_fact_tracker_detects_fake_seer_accusing_me(self) -> None:
        ev1 = PublicEvent(
            1, 1, Phase.DAY_SPEECH, "speech", 3,
            {"intent": {"intent_type": "claim_seer_good", "target": 2}, "text": "我是 3 号预言家，首夜查验 2 号是好人。"},
        )
        ev2 = PublicEvent(
            2, 1, Phase.DAY_SPEECH, "speech", 7,
            {"intent": {"intent_type": "claim_seer_wolf", "target": 4}, "text": "我跳预言家，昨晚验 4 号是查杀狼人！"},
        )
        obs = PlayerObservation(
            game_id="ft", mode="nl", day=1, phase=Phase.DAY_VOTE, player_id=4,
            my_role=Role.VILLAGER, player_count=9, alive_players=tuple(range(1, 10)),
            dead_players=(), revealed_roles={}, public_events=(ev1, ev2),
        )
        fb = build_verified_fact_board(obs)
        self.assertIn(7, fb.proven_wolves)
        self.assertEqual(fb.likely_true_seer, 3)
        self.assertIn(2, fb.proven_good)

        # Verify cross-round notes detect say-vs-vote mismatch
        ev3 = PublicEvent(
            3, 1, Phase.DAY_VOTE, "vote_result", None,
            {"votes": {"7": 2, "4": 7}, "exiled": 7, "tied": []},
        )
        obs2 = PlayerObservation(
            game_id="ft", mode="nl", day=2, phase=Phase.DAY_SPEECH, player_id=4,
            my_role=Role.VILLAGER, player_count=9, alive_players=(1, 2, 3, 4, 5, 6, 8, 9),
            dead_players=(7,), revealed_roles={}, public_events=(ev1, ev2, ev3),
        )
        notes = build_cross_round_notes(obs2)
        self.assertIn("言行不一", notes)
        self.assertIn("7号(说踩4号却投2号)", notes)


if __name__ == "__main__":
    unittest.main()
