"""Lightweight, gradient-verified NumPy neural network primitives and Adam optimizer."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically stable masked softmax & log_softmax.

    Args:
        logits: [B, A] float32
        mask: [B, A] bool (at least one True per row)
    Returns:
        probs: [B, A] float32 (0 where mask is False)
        log_probs: [B, A] float32
    """
    neg_inf = np.float32(-1e9)
    masked = np.where(mask, logits, neg_inf)
    max_v = np.max(masked, axis=-1, keepdims=True)
    shifted = masked - max_v
    exp_v = np.where(mask, np.exp(shifted), np.float32(0.0))
    sum_exp = np.maximum(np.sum(exp_v, axis=-1, keepdims=True), np.float32(1e-12))
    probs = exp_v / sum_exp
    log_probs = np.where(mask, shifted - np.log(sum_exp), neg_inf)
    return probs.astype(np.float32), log_probs.astype(np.float32)


def init_linear(rng: np.random.Generator, in_dim: int, out_dim: int, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(2.0 / max(in_dim, 1)) * scale
    w = (rng.standard_normal((in_dim, out_dim)) * std).astype(np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)
    return w, b


class AdamOptimizer:
    """Standard AdamW-style optimizer over a dict of NumPy parameter arrays."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float | None = None) -> float:
        self.t += 1
        cur_lr = self.lr if lr is None else lr

        # Global L2 gradient norm clipping
        sq_norm = 0.0
        for g in grads.values():
            sq_norm += float(np.sum(g.astype(np.float64) ** 2))
        grad_norm = float(np.sqrt(sq_norm))
        clip_coef = 1.0
        if self.max_grad_norm > 0.0 and grad_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-8)

        b1_corr = 1.0 - self.beta1 ** self.t
        b2_corr = 1.0 - self.beta2 ** self.t

        for k, p in params.items():
            if k not in grads:
                continue
            g = grads[k] * clip_coef
            # Decoupled weight decay on weight matrices (not biases)
            if self.weight_decay > 0.0 and p.ndim >= 2:
                p *= (1.0 - cur_lr * self.weight_decay)
            m = self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            v = self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = m / b1_corr
            v_hat = v / b2_corr
            p -= cur_lr * (m_hat / (np.sqrt(v_hat) + self.eps))

        return grad_norm
