"""Tests for unified feature encoding and MaKTO first-person trajectory reconstruction."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import DecisionType, Phase, PlayerObservation, Role, SpeechIntentType, WerewolfEngine
from werewolf_agent.agents import RuleBasedAgent
from werewolf_agent.features import (
    BELIEF_SEAT_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    PLAYER_FEAT_DIM,
    ROLE_TO_IDX,
    apply_belief_hard_constraints,
    build_belief_seat_features,
    default_prior_belief,
    encode_observation_parts,
    encode_policy_state,
    policy_state_dim,
)
from werewolf_agent.makto_trajectories import map_speech_to_intent, reconstruct_game_trajectories


class TestFeaturesAndMakto(unittest.TestCase):
    def test_feature_shapes_and_hard_constraints(self) -> None:
        obs = PlayerObservation(
            game_id="f1",
            mode="symbolic",
            day=2,
            phase=Phase.DAY_VOTE,
            player_id=1,
            my_role=Role.SEER,
            player_count=9,
            alive_players=(1, 2, 3, 4, 5, 7, 8),
            dead_players=(6, 9),
            revealed_roles={6: Role.HUNTER},
            public_events=(),
            seer_checks=((1, 4, True), (2, 2, False)),
            vote_candidates=(1, 2, 3, 4, 5, 7, 8),
        )
        g, p = encode_observation_parts(obs)
        self.assertEqual(g.shape, (GLOBAL_FEAT_DIM,))
        self.assertEqual(p.shape, (9, PLAYER_FEAT_DIM))
        bf = build_belief_seat_features(obs)
        self.assertEqual(bf.shape, (9, BELIEF_SEAT_FEAT_DIM))
        ps = encode_policy_state(obs)
        self.assertEqual(ps.shape, (policy_state_dim(9),))

        # Verify hard constraints
        prior = default_prior_belief(obs)
        self.assertAlmostEqual(float(prior[0, ROLE_TO_IDX[Role.SEER]]), 1.0, places=5)  # self is Seer
        self.assertAlmostEqual(float(prior[3, ROLE_TO_IDX[Role.WEREWOLF]]), 1.0, places=5)  # checked wolf
        self.assertAlmostEqual(float(prior[1, ROLE_TO_IDX[Role.WEREWOLF]]), 0.0, places=5)  # checked good
        self.assertAlmostEqual(float(prior[5, ROLE_TO_IDX[Role.HUNTER]]), 1.0, places=5)  # revealed hunter
        # Nobody else can be Seer or Hunter
        for i in range(9):
            if i != 0:
                self.assertAlmostEqual(float(prior[i, ROLE_TO_IDX[Role.SEER]]), 0.0, places=5)
            if i != 5:
                self.assertAlmostEqual(float(prior[i, ROLE_TO_IDX[Role.HUNTER]]), 0.0, places=5)

    def test_speech_mapping_covers_core_tactics(self) -> None:
        alive = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        s1 = map_speech_to_intent(1, "我是预言家，4号是查杀！", {"self_present": ["seer"], "identity_label": {"4": ["werewolf"]}}, alive)
        self.assertEqual(s1.intent_type, SpeechIntentType.CLAIM_SEER_WOLF)
        self.assertEqual(s1.target, 4)

        s2 = map_speech_to_intent(5, "我是女巫，昨晚我救了2号银水。", {"self_present": ["witch"]}, alive)
        self.assertEqual(s2.intent_type, SpeechIntentType.CLAIM_WITCH_SAVE)
        self.assertEqual(s2.target, 2)

        s3 = map_speech_to_intent(7, "我是平民，先听大家发言。", {"self_present": ["simple_villager"]}, alive)
        self.assertEqual(s3.intent_type, SpeechIntentType.STAY_LOW)
        self.assertIsNone(s3.target)

    def test_reconstruct_real_makto_game_causal_and_legal(self) -> None:
        sample_file = (
            ROOT.parent
            / "werewolf_nlp"
            / "data"
            / "raw"
            / "makto_source"
            / "raw"
            / "train"
            / "9_player_game"
            / "hunter_witch_seer"
            / "game_161"
            / "event_zh.json"
        )
        if not sample_file.exists():
            self.skipTest("Local MaKTO raw dataset not present")
        events = json.loads(sample_file.read_text(encoding="utf-8"))
        b_list, a_list, stats = reconstruct_game_trajectories(events, game_id="game_161")
        self.assertGreater(len(b_list), 0)
        self.assertGreater(len(a_list), 0)

        # Every reconstructed action MUST satisfy its legal mask and have finite features
        for a in a_list:
            self.assertTrue(bool(a.legal_mask[a.action_id]))
            self.assertTrue(np.all(np.isfinite(a.policy_state)))
            self.assertEqual(a.policy_state.shape, (policy_state_dim(9),))

        # Verify strict causal check: Day 1 Night Seer check cannot already contain its own check result
        seer_night1 = [a for a in a_list if a.decision_type == DecisionType.NIGHT_SEER_CHECK and a.day == 1]
        self.assertEqual(len(seer_night1), 1)
        # Global features idx 15 & 16 are seer_found_wolf_count and seer_found_good_count
        self.assertEqual(float(seer_night1[0].policy_state[15]), 0.0)
        self.assertEqual(float(seer_night1[0].policy_state[16]), 0.0)

        # Verify every belief sample has unknown_mask=False for self
        for b in b_list:
            self.assertFalse(bool(b.unknown_mask[b.observer_id - 1]))
            self.assertEqual(b.seat_features.shape, (9, BELIEF_SEAT_FEAT_DIM))


if __name__ == "__main__":
    unittest.main()
