"""Seat-Equivariant Pointer Policy & Value Network for permutation-invariant Werewolf RL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..features import GLOBAL_FEAT_DIM, NUM_ROLES, PLAYER_FEAT_DIM, policy_state_dim
from ..nn import init_linear, masked_log_softmax, relu, relu_grad
from ..protocol import DecisionType, TARGETED_SPEECH_INTENTS, action_space_size

NUM_SPEECH_TARGET_TYPES = len(TARGETED_SPEECH_INTENTS)  # 7
SEAT_ENTITY_DIM = PLAYER_FEAT_DIM + NUM_ROLES  # 24 + 5 = 29
SEAT_INPUT_DIM = GLOBAL_FEAT_DIM + SEAT_ENTITY_DIM * 3  # 19 + 29*3 = 106
GLOBAL_POOL_DIM = GLOBAL_FEAT_DIM + SEAT_ENTITY_DIM * 2  # 19 + 29*2 = 77


def unpack_policy_state(x: np.ndarray, player_count: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack flat [B, 280] policy_state into (seat_inputs [B, N, 106], global_inputs [B, 77])."""
    b = x.shape[0]
    g = x[:, :GLOBAL_FEAT_DIM]  # [B, 19]
    p_end = GLOBAL_FEAT_DIM + player_count * PLAYER_FEAT_DIM
    p = x[:, GLOBAL_FEAT_DIM:p_end].reshape(b, player_count, PLAYER_FEAT_DIM)
    bel = x[:, p_end:].reshape(b, player_count, NUM_ROLES)
    ent = np.concatenate([p, bel], axis=-1)  # [B, N, 29]

    e_mean = ent.mean(axis=1)  # [B, 29]
    e_max = ent.max(axis=1)  # [B, 29]

    g_rep = np.broadcast_to(g[:, None, :], (b, player_count, GLOBAL_FEAT_DIM))
    m_rep = np.broadcast_to(e_mean[:, None, :], (b, player_count, SEAT_ENTITY_DIM))
    x_rep = np.broadcast_to(e_max[:, None, :], (b, player_count, SEAT_ENTITY_DIM))

    seat_in = np.concatenate([g_rep, ent, m_rep, x_rep], axis=-1).astype(np.float32)  # [B, N, 106]
    glob_in = np.concatenate([g, e_mean, e_max], axis=-1).astype(np.float32)  # [B, 77]
    return seat_in, glob_in


class SeatEquivariantPolicyValueNet:
    """Permutation-equivariant pointer policy + global value head.

    Every seat 1..N is scored by the EXACT SAME neural network weights, enabling
    instant generalization across seat shuffles and 9x higher sample efficiency.
    """

    def __init__(
        self,
        player_count: int = 9,
        h_seat: int = 96,
        h_mid: int = 64,
        seed: int = 42,
    ) -> None:
        self.player_count = player_count
        self.in_dim = policy_state_dim(player_count)
        self.h_seat = h_seat
        self.h_mid = h_mid

        rng = np.random.default_rng(seed)
        # Shared seat encoder: 106 -> h_seat -> h_mid
        ws1, bs1 = init_linear(rng, SEAT_INPUT_DIM, h_seat)
        ws2, bs2 = init_linear(rng, h_seat, h_mid)

        # Global encoder (for pass/abstain/heal/stay_low & value): (77 + 2*h_mid) -> h_mid
        g_in_dim = GLOBAL_POOL_DIM + 2 * h_mid
        wg1, bg1 = init_linear(rng, g_in_dim, h_mid)

        # Shared per-seat pointer scoring heads (12 outputs per seat):
        # 0: seer_check, 1: wolf_kill, 2: witch_poison, 3: hunter_shoot, 4: day_vote,
        # 5..11: 7 targeted speech intent types
        w_ptr, b_ptr = init_linear(rng, h_mid, 5 + NUM_SPEECH_TARGET_TYPES, scale=0.2)

        # Global non-target action logits (5 outputs):
        # 0: witch_pass, 1: witch_heal, 2: hunter_pass, 3: vote_abstain, 4: speech_stay_low
        w_gact, b_gact = init_linear(rng, h_mid, 5, scale=0.2)

        # Scalar Value head
        wv, bv = init_linear(rng, h_mid, 1, scale=0.25)

        self.params: Dict[str, np.ndarray] = {
            "ws1": ws1,
            "bs1": bs1,
            "ws2": ws2,
            "bs2": bs2,
            "wg1": wg1,
            "bg1": bg1,
            "w_ptr": w_ptr,
            "b_ptr": b_ptr,
            "w_gact": w_gact,
            "b_gact": b_gact,
            "wv": wv,
            "bv": bv,
        }

    def clone(self) -> "SeatEquivariantPolicyValueNet":
        c = SeatEquivariantPolicyValueNet(self.player_count, self.h_seat, self.h_mid, seed=0)
        c.params = {k: v.copy() for k, v in self.params.items()}
        return c

    def _forward_backbone(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        b = x.shape[0]
        n = self.player_count
        seat_in, glob_in = unpack_policy_state(x, n)  # [B, N, 106], [B, 77]
        s_flat = seat_in.reshape(b * n, SEAT_INPUT_DIM)

        zs1 = s_flat @ self.params["ws1"] + self.params["bs1"]
        as1 = relu(zs1)
        zs2 = as1 @ self.params["ws2"] + self.params["bs2"]
        as2 = relu(zs2)
        h_seats = as2.reshape(b, n, self.h_mid)  # [B, N, H]

        h_mean = h_seats.mean(axis=1)  # [B, H]
        h_max_idx = np.argmax(h_seats, axis=1)  # [B, H]
        h_max = np.take_along_axis(h_seats, h_max_idx[:, None, :], axis=1)[:, 0, :]  # [B, H]

        g_cat = np.concatenate([glob_in, h_mean, h_max], axis=-1)  # [B, 77 + 2H]
        zg1 = g_cat @ self.params["wg1"] + self.params["bg1"]
        ag1 = relu(zg1)  # [B, H]

        ptr_all = (as2 @ self.params["w_ptr"] + self.params["b_ptr"]).reshape(b, n, 5 + NUM_SPEECH_TARGET_TYPES)
        gact_all = ag1 @ self.params["w_gact"] + self.params["b_gact"]  # [B, 5]
        values = (ag1 @ self.params["wv"] + self.params["bv"]).squeeze(-1)  # [B]

        return {
            "b": b,
            "s_flat": s_flat,
            "zs1": zs1,
            "as1": as1,
            "zs2": zs2,
            "as2": as2,
            "h_max_idx": h_max_idx,
            "g_cat": g_cat,
            "zg1": zg1,
            "ag1": ag1,
            "ptr_all": ptr_all,
            "gact_all": gact_all,
            "values": values.astype(np.float32),
        }

    def _assemble_logits(self, dt: DecisionType, cache: Dict[str, np.ndarray]) -> np.ndarray:
        ptr = cache["ptr_all"]  # [B, N, 12]
        ga = cache["gact_all"]  # [B, 5]
        if dt == DecisionType.NIGHT_SEER_CHECK:
            return ptr[:, :, 0]
        if dt == DecisionType.NIGHT_WOLF_KILL:
            return ptr[:, :, 1]
        if dt == DecisionType.NIGHT_WITCH_ACT:
            return np.concatenate([ga[:, 0:2], ptr[:, :, 2]], axis=-1)
        if dt == DecisionType.HUNTER_SHOOT:
            return np.concatenate([ga[:, 2:3], ptr[:, :, 3]], axis=-1)
        if dt == DecisionType.DAY_VOTE:
            return np.concatenate([ga[:, 3:4], ptr[:, :, 4]], axis=-1)
        if dt == DecisionType.DAY_SPEECH:
            # 0 = stay_low; 1..63 = for each of 7 intent types, seats 1..9
            b = ptr.shape[0]
            targeted = np.transpose(ptr[:, :, 5:], (0, 2, 1)).reshape(b, NUM_SPEECH_TARGET_TYPES * self.player_count)
            return np.concatenate([ga[:, 4:5], targeted], axis=-1)
        raise ValueError(f"Unsupported dt: {dt}")

    def forward_head(
        self,
        x: np.ndarray,
        dt: DecisionType,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        cache = self._forward_backbone(x)
        logits = self._assemble_logits(dt, cache)
        probs, log_probs = masked_log_softmax(logits, mask)
        cache["probs"] = probs
        cache["log_probs"] = log_probs
        return probs, log_probs, cache["values"], cache

    def act_single(
        self,
        state: np.ndarray,
        dt: DecisionType,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[int, float, float]:
        cache = self._forward_backbone(state[None, :])
        logits = self._assemble_logits(dt, cache) / max(temperature, 1e-4)
        probs, log_probs = masked_log_softmax(logits, mask[None, :])
        val = float(cache["values"][0])

        p0 = probs[0]
        if deterministic or rng is None:
            action = int(np.argmax(np.where(mask, p0, -1.0)))
        else:
            p_safe = np.where(mask, np.maximum(p0, 0.0), 0.0).astype(np.float64)
            s = p_safe.sum()
            if s <= 0.0:
                action = int(rng.choice(np.flatnonzero(mask)))
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
        b = int(cache["b"])
        n = self.player_count
        d_ptr = np.zeros((b, n, 5 + NUM_SPEECH_TARGET_TYPES), dtype=np.float32)
        d_gact = np.zeros((b, 5), dtype=np.float32)

        if dt == DecisionType.NIGHT_SEER_CHECK:
            d_ptr[:, :, 0] = dlogits
        elif dt == DecisionType.NIGHT_WOLF_KILL:
            d_ptr[:, :, 1] = dlogits
        elif dt == DecisionType.NIGHT_WITCH_ACT:
            d_gact[:, 0:2] = dlogits[:, 0:2]
            d_ptr[:, :, 2] = dlogits[:, 2:]
        elif dt == DecisionType.HUNTER_SHOOT:
            d_gact[:, 2:3] = dlogits[:, 0:1]
            d_ptr[:, :, 3] = dlogits[:, 1:]
        elif dt == DecisionType.DAY_VOTE:
            d_gact[:, 3:4] = dlogits[:, 0:1]
            d_ptr[:, :, 4] = dlogits[:, 1:]
        elif dt == DecisionType.DAY_SPEECH:
            d_gact[:, 4:5] = dlogits[:, 0:1]
            d_targeted = dlogits[:, 1:].reshape(b, NUM_SPEECH_TARGET_TYPES, n)
            d_ptr[:, :, 5:] = np.transpose(d_targeted, (0, 2, 1))

        # Backprop through pointer head
        d_ptr_flat = d_ptr.reshape(b * n, 5 + NUM_SPEECH_TARGET_TYPES)
        as2 = cache["as2"]
        dw_ptr = as2.T @ d_ptr_flat
        db_ptr = np.sum(d_ptr_flat, axis=0)
        d_as2 = d_ptr_flat @ self.params["w_ptr"].T  # [B*N, H]

        # Backprop through global action head & value head
        ag1 = cache["ag1"]
        dw_gact = ag1.T @ d_gact
        db_gact = np.sum(d_gact, axis=0)
        d_ag1 = d_gact @ self.params["w_gact"].T

        grads: Dict[str, np.ndarray] = {
            "w_ptr": dw_ptr.astype(np.float32),
            "b_ptr": db_ptr.astype(np.float32),
            "w_gact": dw_gact.astype(np.float32),
            "b_gact": db_gact.astype(np.float32),
        }

        if dvalues is not None:
            dv_col = dvalues[:, None]
            grads["wv"] = (ag1.T @ dv_col).astype(np.float32)
            grads["bv"] = np.sum(dv_col, axis=0).astype(np.float32)
            d_ag1 = d_ag1 + dv_col @ self.params["wv"].T

        d_zg1 = d_ag1 * relu_grad(cache["zg1"])
        grads["wg1"] = (cache["g_cat"].T @ d_zg1).astype(np.float32)
        grads["bg1"] = np.sum(d_zg1, axis=0).astype(np.float32)
        d_gcat = d_zg1 @ self.params["wg1"].T

        # Flow global pool gradients back into h_seats (mean & max pools)
        d_hmean = d_gcat[:, GLOBAL_POOL_DIM : GLOBAL_POOL_DIM + self.h_mid]  # [B, H]
        d_hmax = d_gcat[:, GLOBAL_POOL_DIM + self.h_mid :]  # [B, H]

        d_hseats = d_as2.reshape(b, n, self.h_mid)
        d_hseats = d_hseats + (d_hmean[:, None, :] / float(n))
        # Scatter max-pool gradient to argmax indices
        b_idx = np.arange(b)[:, None]
        h_idx = np.arange(self.h_mid)[None, :]
        np.add.at(d_hseats, (b_idx, cache["h_max_idx"], h_idx), d_hmax)

        d_as2_total = d_hseats.reshape(b * n, self.h_mid)
        d_zs2 = d_as2_total * relu_grad(cache["zs2"])
        grads["ws2"] = (cache["as1"].T @ d_zs2).astype(np.float32)
        grads["bs2"] = np.sum(d_zs2, axis=0).astype(np.float32)

        d_as1 = d_zs2 @ self.params["ws2"].T
        d_zs1 = d_as1 * relu_grad(cache["zs1"])
        grads["ws1"] = (cache["s_flat"].T @ d_zs1).astype(np.float32)
        grads["bs1"] = np.sum(d_zs1, axis=0).astype(np.float32)
        return grads

    def bc_loss_and_grads(
        self,
        dt: DecisionType,
        x: np.ndarray,
        actions: np.ndarray,
        mask: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
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
        bc_anchor: "SeatEquivariantPolicyValueNet | None" = None,
        kl_coef: float = 0.015,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        b = x.shape[0]
        probs, log_probs, values, cache = self.forward_head(x, dt, mask)
        idx = np.arange(b)

        cur_lp = log_probs[idx, actions]
        ratio = np.exp(np.clip(cur_lp - old_log_probs, -10.0, 10.0))
        unclipped = ratio * advantages
        clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        clipped = clipped_ratio * advantages
        policy_loss = -float(np.mean(np.minimum(unclipped, clipped)))

        use_unclipped = (unclipped <= clipped).astype(np.float32)
        dL_dcur_lp = -(use_unclipped * ratio * advantages) / float(b)

        dlogits = -probs * dL_dcur_lp[:, None]
        dlogits[idx, actions] += dL_dcur_lp

        safe_lp = np.where(mask, log_probs, np.float32(0.0))
        entropy = -np.sum(probs * safe_lp, axis=-1)
        mean_ent = float(np.mean(entropy))
        if entropy_coef > 0.0:
            dlogits += (entropy_coef / float(b)) * (probs * (safe_lp + entropy[:, None]))

        mean_kl = 0.0
        if bc_anchor is not None and kl_coef > 0.0:
            _, ref_lp, _, _ = bc_anchor.forward_head(x, dt, mask)
            safe_ref_lp = np.where(mask, ref_lp, np.float32(0.0))
            diff = safe_lp - safe_ref_lp
            kl = np.sum(probs * diff, axis=-1)
            mean_kl = float(np.mean(kl))
            dlogits += (kl_coef / float(b)) * (probs * (diff - kl[:, None]))

        dlogits = np.where(mask, dlogits, np.float32(0.0))

        v_err = values - returns
        value_loss = 0.5 * float(np.mean(v_err ** 2))
        dvalues = (value_coef / float(b)) * v_err

        grads = self._backprop_from_head(dt, cache, dlogits, dvalues=dvalues)
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": mean_ent,
            "kl_bc": mean_kl,
            "approx_kl": float(np.mean(old_log_probs - cur_lp)),
        }, grads

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array([self.player_count, self.h_seat, self.h_mid], dtype=np.int64)
        np.savez_compressed(out, arch=np.array(["pointer"]), meta=meta, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "SeatEquivariantPolicyValueNet":
        data = np.load(path)
        player_count, h_seat, h_mid = [int(v) for v in data["meta"]]
        model = cls(player_count=player_count, h_seat=h_seat, h_mid=h_mid, seed=0)
        for k in list(model.params.keys()):
            model.params[k] = data[k].astype(np.float32).copy()
        return model
