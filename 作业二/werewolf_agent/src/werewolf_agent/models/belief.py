"""Seat-shared lightweight MLP Belief Tracker trained on first-person observations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..features import (
    BELIEF_SEAT_FEAT_DIM,
    NUM_ROLES,
    apply_belief_hard_constraints,
    build_belief_seat_features,
)
from ..nn import AdamOptimizer, init_linear, relu, relu_grad
from ..protocol import PlayerObservation


class BeliefMLP:
    """Seat-shared 2-hidden-layer MLP predicting 5-role probabilities per seat."""

    def __init__(
        self,
        in_dim: int = BELIEF_SEAT_FEAT_DIM,
        h1: int = 128,
        h2: int = 64,
        out_dim: int = NUM_ROLES,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        w1, b1 = init_linear(rng, in_dim, h1)
        w2, b2 = init_linear(rng, h1, h2)
        w3, b3 = init_linear(rng, h2, out_dim, scale=0.5)
        self.params: Dict[str, np.ndarray] = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3,
        }

    def forward_logits(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.params["w1"] + self.params["b1"]
        a1 = relu(z1)
        z2 = a1 @ self.params["w2"] + self.params["b2"]
        a2 = relu(z2)
        logits = a2 @ self.params["w3"] + self.params["b3"]
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return logits.astype(np.float32), cache

    def predict_probs_batch(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward_logits(x)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_v = np.exp(shifted)
        return (exp_v / np.sum(exp_v, axis=-1, keepdims=True)).astype(np.float32)

    def predict_obs(self, obs: PlayerObservation, apply_constraints: bool = True) -> np.ndarray:
        """Predict [N, 5] role distribution for a live PlayerObservation."""
        seat_x = build_belief_seat_features(obs)
        raw_probs = self.predict_probs_batch(seat_x)
        if apply_constraints:
            return apply_belief_hard_constraints(raw_probs, obs)
        return raw_probs

    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Cross-entropy loss and exact analytical gradients on [B, in_dim] -> [B] labels."""
        b = x.shape[0]
        logits, cache = self.forward_logits(x)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_v = np.exp(shifted)
        probs = exp_v / np.sum(exp_v, axis=-1, keepdims=True)

        w_sample = (
            class_weights[y].astype(np.float32)
            if class_weights is not None
            else np.ones(b, dtype=np.float32)
        )
        w_sum = max(float(np.sum(w_sample)), 1e-8)

        log_p = np.log(np.clip(probs[np.arange(b), y], 1e-12, 1.0))
        loss = -float(np.sum(w_sample * log_p) / w_sum)

        dlogits = probs.copy()
        dlogits[np.arange(b), y] -= 1.0
        dlogits *= (w_sample[:, None] / w_sum)

        dw3 = cache["a2"].T @ dlogits
        db3 = np.sum(dlogits, axis=0)
        da2 = dlogits @ self.params["w3"].T
        dz2 = da2 * relu_grad(cache["z2"])

        dw2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.params["w2"].T
        dz1 = da1 * relu_grad(cache["z1"])

        dw1 = cache["x"].T @ dz1
        db1 = np.sum(dz1, axis=0)

        grads = {
            "w1": dw1.astype(np.float32),
            "b1": db1.astype(np.float32),
            "w2": dw2.astype(np.float32),
            "b2": db2.astype(np.float32),
            "w3": dw3.astype(np.float32),
            "b3": db3.astype(np.float32),
        }
        return loss, grads

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array([self.in_dim, self.h1, self.h2, self.out_dim], dtype=np.int64)
        np.savez_compressed(out, meta=meta, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefMLP":
        data = np.load(path)
        in_dim, h1, h2, out_dim = [int(v) for v in data["meta"]]
        model = cls(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim)
        for k in list(model.params.keys()):
            model.params[k] = data[k].astype(np.float32).copy()
        return model


def evaluate_belief_predictions(
    probs: np.ndarray,
    true_roles: np.ndarray,
    unknown_mask: np.ndarray,
) -> Dict[str, float]:
    """Evaluate [M, 9, 5] predicted probs on unknown seats."""
    flat_p = probs[unknown_mask]
    flat_y = true_roles[unknown_mask]
    preds = np.argmax(flat_p, axis=-1)
    acc = float(np.mean(preds == flat_y)) if len(flat_y) else 0.0

    # Macro-F1 across 5 roles
    f1s = []
    for c in range(NUM_ROLES):
        tp = int(np.sum((preds == c) & (flat_y == c)))
        fp = int(np.sum((preds == c) & (flat_y != c)))
        fn = int(np.sum((preds != c) & (flat_y == c)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(f1s))

    # Wolf binary F1 (role idx 3 = werewolf)
    wolf_true = flat_y == 3
    wolf_pred = preds == 3
    wtp = int(np.sum(wolf_pred & wolf_true))
    wfp = int(np.sum(wolf_pred & ~wolf_true))
    wfn = int(np.sum(~wolf_pred & wolf_true))
    wolf_f1 = (2.0 * wtp / (2 * wtp + wfp + wfn)) if (2 * wtp + wfp + wfn) > 0 else 0.0

    # Cross-entropy / NLL
    nll = -float(np.mean(np.log(np.clip(flat_p[np.arange(len(flat_y)), flat_y], 1e-12, 1.0)))) if len(flat_y) else 0.0

    return {
        "num_unknown_seats": int(len(flat_y)),
        "unknown_acc": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "wolf_f1": round(wolf_f1, 4),
        "nll": round(nll, 4),
    }


def train_belief_mlp(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> Tuple[BeliefMLP, list[Dict[str, float]]]:
    """Train seat-shared BeliefMLP on unknown seats with validation early selection."""
    rng = np.random.default_rng(seed)
    tr_mask = train_data["belief_unknown_mask"]
    X_tr = train_data["belief_X"][tr_mask].astype(np.float32)
    y_tr = train_data["belief_y"][tr_mask].astype(np.int64)

    # Mild inverse-sqrt class weights to prevent collapsing onto Villager/Werewolf only
    counts = np.bincount(y_tr, minlength=NUM_ROLES).astype(np.float32)
    class_weights = np.power(np.maximum(counts.mean() / np.maximum(counts, 1.0), 0.1), 0.35).astype(np.float32)

    model = BeliefMLP(seed=seed)
    opt = AdamOptimizer(model.params, lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")
    best_params = {k: v.copy() for k, v in model.params.items()}
    history: list[Dict[str, float]] = []

    val_X = val_data["belief_X"].astype(np.float32)
    val_y = val_data["belief_y"].astype(np.int64)
    val_mask = val_data["belief_unknown_mask"]

    n_samples = X_tr.shape[0]
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_samples)
        losses = []
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            loss, grads = model.loss_and_grads(X_tr[idx], y_tr[idx], class_weights=class_weights)
            opt.step(model.params, grads)
            losses.append(loss)

        val_probs = model.predict_probs_batch(val_X.reshape(-1, model.in_dim)).reshape(val_X.shape[0], val_X.shape[1], NUM_ROLES)
        val_metrics = evaluate_belief_predictions(val_probs, val_y, val_mask)
        rec = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_nll": val_metrics["nll"],
            "val_acc": val_metrics["unknown_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_wolf_f1": val_metrics["wolf_f1"],
        }
        history.append(rec)
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_params = {k: v.copy() for k, v in model.params.items()}

    model.params = best_params
    return model, history
