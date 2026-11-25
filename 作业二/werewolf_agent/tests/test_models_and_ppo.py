"""Unit tests for BeliefMLP, MultiHeadPolicyValueMLP analytical gradients, save/load, and PPO."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import DecisionType, WerewolfEngine
from werewolf_agent.agents import BeliefRuleAgent, NeuralPolicyAgent
from werewolf_agent.features import BELIEF_SEAT_FEAT_DIM, policy_state_dim
from werewolf_agent.models.belief import BeliefMLP
from werewolf_agent.models.pointer_policy import SeatEquivariantPolicyValueNet
from werewolf_agent.models.policy import MultiHeadPolicyValueMLP
from werewolf_agent.protocol import action_space_size
from werewolf_agent.training import train_self_play_ppo


class TestModelsAndPPO(unittest.TestCase):
    def test_belief_mlp_gradients_match_finite_difference(self) -> None:
        rng = np.random.default_rng(0)
        model = BeliefMLP(in_dim=12, h1=8, h2=6, out_dim=5, seed=1)
        X = rng.standard_normal((5, 12)).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        _, grads = model.loss_and_grads(X, y)

        eps = 1e-3
        for pname in ("w1", "b1", "w2", "w3", "b3"):
            param = model.params[pname]
            it = np.ndindex(param.shape)
            for idx in list(it)[:6]:
                orig = float(param[idx])
                param[idx] = orig + eps
                lp, _ = model.loss_and_grads(X, y)
                param[idx] = orig - eps
                lm, _ = model.loss_and_grads(X, y)
                param[idx] = orig
                num_g = (lp - lm) / (2.0 * eps)
                self.assertAlmostEqual(float(grads[pname][idx]), num_g, places=3)

    def test_policy_bc_and_ppo_gradients_match_finite_difference(self) -> None:
        rng = np.random.default_rng(7)
        policy = MultiHeadPolicyValueMLP(player_count=9, h1=16, h2=12, seed=2)
        dt = DecisionType.DAY_VOTE
        adim = action_space_size(dt, 9)
        X = rng.standard_normal((6, policy_state_dim(9))).astype(np.float32)
        mask = np.ones((6, adim), dtype=bool)
        mask[:, 2] = False
        actions = np.array([0, 1, 3, 4, 5, 6], dtype=np.int64)

        # 1. BC gradient check
        _, grads = policy.bc_loss_and_grads(dt, X, actions, mask)
        eps = 1e-3
        for pname in (f"w_{dt.value}", f"b_{dt.value}", "w2"):
            p = policy.params[pname]
            for idx in list(np.ndindex(p.shape))[:5]:
                orig = float(p[idx])
                p[idx] = orig + eps
                lp, _ = policy.bc_loss_and_grads(dt, X, actions, mask)
                p[idx] = orig - eps
                lm, _ = policy.bc_loss_and_grads(dt, X, actions, mask)
                p[idx] = orig
                self.assertAlmostEqual(float(grads[pname][idx]), (lp - lm) / (2.0 * eps), places=3)

        # 2. PPO + Value + Entropy gradient check (unclipped regime via small adv & old_lp = cur_lp)
        _, old_lp_mat, _, _ = policy.forward_head(X, dt, mask)
        old_lp = old_lp_mat[np.arange(6), actions].copy()
        adv = np.array([0.4, -0.3, 0.2, -0.5, 0.1, 0.3], dtype=np.float32)
        rets = np.array([1.0, -1.0, 0.5, -0.5, 0.2, 0.8], dtype=np.float32)
        anchor = policy.clone()
        anchor.params[f"w_{dt.value}"] += 0.05

        def total_obj() -> float:
            m, _ = policy.ppo_loss_and_grads(
                dt, X, actions, mask, old_lp, adv, rets,
                value_coef=0.5, entropy_coef=0.02, bc_anchor=anchor, kl_coef=0.03
            )
            return m["policy_loss"] + 0.5 * m["value_loss"] - 0.02 * m["entropy"] + 0.03 * m["kl_bc"]

        _, ppo_grads = policy.ppo_loss_and_grads(
            dt, X, actions, mask, old_lp, adv, rets,
            value_coef=0.5, entropy_coef=0.02, bc_anchor=anchor, kl_coef=0.03
        )
        for pname in (f"w_{dt.value}", "wv", "w2"):
            p = policy.params[pname]
            for idx in list(np.ndindex(p.shape))[:5]:
                orig = float(p[idx])
                p[idx] = orig + eps
                lp = total_obj()
                p[idx] = orig - eps
                lm = total_obj()
                p[idx] = orig
                self.assertAlmostEqual(float(ppo_grads[pname][idx]), (lp - lm) / (2.0 * eps), places=3)

    def test_save_load_and_ppo_smoke_run(self) -> None:
        belief = BeliefMLP(seed=3)
        policy = MultiHeadPolicyValueMLP(seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            bpath = Path(tmp) / "belief.npz"
            ppath = Path(tmp) / "policy.npz"
            belief.save(bpath)
            policy.save(ppath)
            b2 = BeliefMLP.load(bpath)
            p2 = MultiHeadPolicyValueMLP.load(ppath)
            for k in belief.params:
                self.assertTrue(np.allclose(belief.params[k], b2.params[k]))
            for k in policy.params:
                self.assertTrue(np.allclose(policy.params[k], p2.params[k]))

        # Verify BeliefRuleAgent and NeuralPolicyAgent complete full games
        engine = WerewolfEngine(mode="symbolic")
        agents = {
            pid: (BeliefRuleAgent(belief, seed=pid) if pid <= 4 else NeuralPolicyAgent(policy, belief, seed=pid))
            for pid in range(1, 10)
        }
        res = engine.run_game(agents, seed=10, game_id="neural_smoke")
        self.assertGreaterEqual(res.total_days, 1)

        # Fast 2-iteration PPO smoke test
        trained, hist = train_self_play_ppo(
            init_policy=policy,
            belief_model=belief,
            iterations=2,
            games_per_iter=8,
            eval_every=2,
            eval_games=10,
            seed=5,
        )
        self.assertEqual(len(hist), 2)
        self.assertIsInstance(trained, MultiHeadPolicyValueMLP)

    def test_pointer_policy_gradients_and_save_load(self) -> None:
        rng = np.random.default_rng(11)
        policy = SeatEquivariantPolicyValueNet(player_count=9, h_seat=12, h_mid=8, seed=9)
        X = rng.standard_normal((4, policy_state_dim(9))).astype(np.float32)

        for dt in (DecisionType.DAY_VOTE, DecisionType.DAY_SPEECH, DecisionType.NIGHT_WITCH_ACT):
            adim = action_space_size(dt, 9)
            mask = np.ones((4, adim), dtype=bool)
            mask[:, 1] = False
            actions = np.array([0, 2, 3, 4], dtype=np.int64)
            _, old_lp_mat, _, _ = policy.forward_head(X, dt, mask)
            old_lp = old_lp_mat[np.arange(4), actions].copy()
            adv = np.array([0.4, -0.3, 0.2, -0.2], dtype=np.float32)
            rets = np.array([1.0, -1.0, 0.3, -0.4], dtype=np.float32)
            anchor = policy.clone()
            anchor.params["w_ptr"] += 0.05

            def obj() -> float:
                m, _ = policy.ppo_loss_and_grads(
                    dt, X, actions, mask, old_lp, adv, rets,
                    value_coef=0.5, entropy_coef=0.02, bc_anchor=anchor, kl_coef=0.03
                )
                return m["policy_loss"] + 0.5 * m["value_loss"] - 0.02 * m["entropy"] + 0.03 * m["kl_bc"]

            _, grads = policy.ppo_loss_and_grads(
                dt, X, actions, mask, old_lp, adv, rets,
                value_coef=0.5, entropy_coef=0.02, bc_anchor=anchor, kl_coef=0.03
            )
            eps = 1e-3
            for pname in ("w_ptr", "w_gact", "wv", "ws2"):
                p = policy.params[pname]
                for idx in list(np.ndindex(p.shape))[:4]:
                    orig = float(p[idx])
                    p[idx] = orig + eps
                    lp = obj()
                    p[idx] = orig - eps
                    lm = obj()
                    p[idx] = orig
                    self.assertAlmostEqual(float(grads[pname][idx]), (lp - lm) / (2.0 * eps), places=3)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ptr.npz"
            policy.save(p)
            loaded = SeatEquivariantPolicyValueNet.load(p)
            for k in policy.params:
                self.assertTrue(np.allclose(policy.params[k], loaded.params[k]))


if __name__ == "__main__":
    unittest.main()
