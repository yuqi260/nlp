"""Multi-head Masked Policy & Value MLP supporting Behavior Cloning (BC) and PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..features import policy_state_dim
from ..nn import AdamOptimizer, init_linear, masked_log_softmax, relu, relu_grad
from ..protocol import DecisionType, action_space_size

ALL_DECISION_TYPES: Tuple[DecisionType, ...] = tuple(DecisionType)


class MultiHeadPolicyValueMLP:
    """Shared trunk MLP with 6 masked discrete action heads + 1 scalar Value head."""

    def __init__(
        self,
        player_count: int = 9,
        h1: int = 160,
        h2: int = 128,
        seed: int = 42,
    ) -> None:
        self.player_count = player_count
        self.in_dim = policy_state_dim(player_count)
        self.h1 = h1
        self.h2 = h2

        rng = np.random.default_rng(seed)
        w1, b1 = init_linear(rng, self.in_dim, h1)
        w2, b2 = init_linear(rng, h1, h2)
        wv, bv = init_linear(rng, h2, 1, scale=0.3)

        self.params: Dict[str, np.ndarray] = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "wv": wv,
            "bv": bv,
        }
        for dt in ALL_DECISION_TYPES:
            adim = action_space_size(dt, player_count)
            wh, bh = init_linear(rng, h2, adim, scale=0.2)
            self.params[f"w_{dt.value}"] = wh
            self.params[f"b_{dt.value}"] = bh

    def clone(self) -> "MultiHeadPolicyValueMLP":
        copy = MultiHeadPolicyValueMLP(self.player_count, self.h1, self.h2, seed=0)
        copy.params = {k: v.copy() for k, v in self.params.items()}
        return copy

    def forward_trunk(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.params["w1"] + self.params["b1"]
        a1 = relu(z1)
        z2 = a1 @ self.params["w2"] + self.params["b2"]
        a2 = relu(z2)
        return a2.astype(np.float32), {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def forward_head(
        self,
        x: np.ndarray,
        dt: DecisionType,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Forward trunk + specified head + value head.

        Returns:
            probs: [B, A]
            log_probs: [B, A]
            values: [B]
            cache: dict
        """
        a2, cache = self.forward_trunk(x)
        logits = a2 @ self.params[f"w_{dt.value}"] + self.params[f"b_{dt.value}"]
        probs, log_probs = masked_log_softmax(logits, mask)
        values = (a2 @ self.params["wv"] + self.params["bv"]).squeeze(-1)
        cache["logits"] = logits
        cache["probs"] = probs
        cache["log_probs"] = log_probs
        cache["values"] = values
        return probs, log_probs, values.astype(np.float32), cache

    def act_single(
        self,
        state: np.ndarray,
        dt: DecisionType,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[int, float, float]:
        """Single-step inference returning (action_id, log_prob, value)."""
        x = state[None, :]
        m = mask[None, :]
        a2, _ = self.forward_trunk(x)
        logits = (a2 @ self.params[f"w_{dt.value}"] + self.params[f"b_{dt.value}"]) / max(temperature, 1e-4)
        probs, log_probs = masked_log_softmax(logits, m)
        val = float((a2 @ self.params["wv"] + self.params["bv"]).item())

        p0 = probs[0]
        if deterministic or rng is None:
            action = int(np.argmax(np.where(mask, p0, -1.0)))
        else:
            p_safe = np.where(mask, np.maximum(p0, 0.0), 0.0).astype(np.float64)
            s = p_safe.sum()
            if s <= 0.0:
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal))
            else:
                p_safe /= s
                action = int(rng.choice(len(p_safe), p=p_safe))
        return action, float(log_probs[0, action]), val

    def _backprop_from_head(
        self,
        dt: DecisionType,
        cache: Dict[str, np.ndarray],
        dlogits: np.ndarray,
        dvalues: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        a2 = cache["a2"]
        wk = f"w_{dt.value}"
        bk = f"b_{dt.value}"
        dw_h = a2.T @ dlogits
        db_h = np.sum(dlogits, axis=0)
        da2 = dlogits @ self.params[wk].T

        grads: Dict[str, np.ndarray] = {
            wk: dw_h.astype(np.float32),
            bk: db_h.astype(np.float32),
        }

        if dvalues is not None:
            dv_col = dvalues[:, None]
            grads["wv"] = (a2.T @ dv_col).astype(np.float32)
            grads["bv"] = np.sum(dv_col, axis=0).astype(np.float32)
            da2 = da2 + dv_col @ self.params["wv"].T

        dz2 = da2 * relu_grad(cache["z2"])
        grads["w2"] = (cache["a1"].T @ dz2).astype(np.float32)
        grads["b2"] = np.sum(dz2, axis=0).astype(np.float32)

        da1 = dz2 @ self.params["w2"].T
        dz1 = da1 * relu_grad(cache["z1"])
        grads["w1"] = (cache["x"].T @ dz1).astype(np.float32)
        grads["b1"] = np.sum(dz1, axis=0).astype(np.float32)
        return grads

    def bc_loss_and_grads(
        self,
        dt: DecisionType,
        x: np.ndarray,
        actions: np.ndarray,
        mask: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Masked cross-entropy loss & exact gradients for Behavior Cloning."""
        b = x.shape[0]
        probs, log_probs, _, cache = self.forward_head(x, dt, mask)
        w = sample_weights.astype(np.float32) if sample_weights is not None else np.ones(b, dtype=np.float32)
        w_sum = max(float(np.sum(w)), 1e-8)

        chosen_lp = log_probs[np.arange(b), actions]
        loss = -float(np.sum(w * chosen_lp) / w_sum)

        dlogits = probs.copy()
        dlogits[np.arange(b), actions] -= 1.0
        dlogits = np.where(mask, dlogits, 0.0) * (w[:, None] / w_sum)
        grads = self._backprop_from_head(dt, cache, dlogits, dvalues=None)
        return loss, grads

    def ppo_loss_and_grads(
        self,
        dt: DecisionType,
        x: np.ndarray,
        actions: np.ndarray,
        mask: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.015,
        bc_anchor: "MultiHeadPolicyValueMLP | None" = None,
        kl_coef: float = 0.02,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Exact analytical gradients for clipped Masked PPO + value + entropy + optional BC KL anchor."""
        b = x.shape[0]
        probs, log_probs, values, cache = self.forward_head(x, dt, mask)
        idx = np.arange(b)

        cur_lp = log_probs[idx, actions]
        ratio = np.exp(np.clip(cur_lp - old_log_probs, -10.0, 10.0))
        unclipped = ratio * advantages
        clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        clipped = clipped_ratio * advantages
        policy_loss = -float(np.mean(np.minimum(unclipped, clipped)))

        # Derivative of policy loss w.r.t. cur_lp
        # Active when unclipped <= clipped (gradient passes through ratio)
        use_unclipped = (unclipped <= clipped).astype(np.float32)
        dL_dcur_lp = -(use_unclipped * ratio * advantages) / float(b)

        # d(log p_a)/d(logits_j) = 1_{j=a} - p_j (on legal actions)
        dlogits = -probs * dL_dcur_lp[:, None]
        dlogits[idx, actions] += dL_dcur_lp

        # Entropy bonus: L_ent = -entropy_coef * mean(H), H = -sum p log p
        safe_lp = np.where(mask, log_probs, np.float32(0.0))
        entropy = -np.sum(probs * safe_lp, axis=-1)  # [B]
        mean_ent = float(np.mean(entropy))
        if entropy_coef > 0.0:
            # d(-H)/dz_j = p_j * (log p_j + H)
            dL_ent_dz = probs * (safe_lp + entropy[:, None])
            dlogits += (entropy_coef / float(b)) * dL_ent_dz

        # Optional KL anchor toward BC policy: KL(pi || pi_bc) = sum p * (log p - log p_bc)
        mean_kl = 0.0
        if bc_anchor is not None and kl_coef > 0.0:
            _, ref_lp, _, _ = bc_anchor.forward_head(x, dt, mask)
            safe_ref_lp = np.where(mask, ref_lp, np.float32(0.0))
            diff = safe_lp - safe_ref_lp
            kl = np.sum(probs * diff, axis=-1)  # [B]
            mean_kl = float(np.mean(kl))
            # d(KL)/dz_j = p_j * (diff_j - sum_k p_k diff_k) = p_j * (diff_j - kl)
            dlogits += (kl_coef / float(b)) * (probs * (diff - kl[:, None]))

        dlogits = np.where(mask, dlogits, np.float32(0.0))

        # Value loss: 0.5 * value_coef * mean((V - R)^2)
        v_err = values - returns
        value_loss = 0.5 * float(np.mean(v_err ** 2))
        dvalues = (value_coef / float(b)) * v_err

        grads = self._backprop_from_head(dt, cache, dlogits, dvalues=dvalues)
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": mean_ent,
            "kl_bc": mean_kl,
            "approx_kl": float(np.mean(old_log_probs - cur_lp)),
        }
        return metrics, grads

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array([self.player_count, self.h1, self.h2], dtype=np.int64)
        np.savez_compressed(out, meta=meta, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadPolicyValueMLP":
        data = np.load(path)
        player_count, h1, h2 = [int(v) for v in data["meta"]]
        model = cls(player_count=player_count, h1=h1, h2=h2, seed=0)
        for k in list(model.params.keys()):
            model.params[k] = data[k].astype(np.float32).copy()
        return model
