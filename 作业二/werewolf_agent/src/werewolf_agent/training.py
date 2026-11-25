"""Offline Behavior Cloning (BC), Population Self-Play Masked PPO, and Symbolic Evaluation."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .agents import BaseAgent, BeliefRuleAgent, NeuralPolicyAgent, RandomAgent, RuleBasedAgent
from .engine import GameResult, WerewolfEngine
from .features import (
    BELIEF_SEAT_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    NUM_ROLES,
    PLAYER_FEAT_DIM,
    ROLE_TO_IDX,
    default_prior_belief,
    encode_policy_state,
)
from .models.belief import BeliefMLP
from .models.pointer_policy import SeatEquivariantPolicyValueNet
from .models.policy import ALL_DECISION_TYPES, MultiHeadPolicyValueMLP
from .nn import AdamOptimizer
from .protocol import ActionRequest, ActionResponse, Camp, DecisionType, Role


def refresh_policy_states_with_belief(
    policy_X: np.ndarray,
    belief_model: BeliefMLP,
    player_count: int = 9,
) -> np.ndarray:
    """Replace the last 45 belief dims in offline [M, 280] policy_X with BeliefMLP predictions."""
    if policy_X.shape[0] == 0:
        return policy_X.copy()
    m = policy_X.shape[0]
    g = policy_X[:, :GLOBAL_FEAT_DIM]  # [M, 19]
    p = policy_X[:, GLOBAL_FEAT_DIM : GLOBAL_FEAT_DIM + player_count * PLAYER_FEAT_DIM].reshape(
        m, player_count, PLAYER_FEAT_DIM
    )
    p_mean = p.mean(axis=1, keepdims=True)
    p_max = p.max(axis=1, keepdims=True)
    g_rep = np.broadcast_to(g[:, None, :], (m, player_count, GLOBAL_FEAT_DIM))
    mean_rep = np.broadcast_to(p_mean, (m, player_count, PLAYER_FEAT_DIM))
    max_rep = np.broadcast_to(p_max, (m, player_count, PLAYER_FEAT_DIM))
    seat_x = np.concatenate([g_rep, p, mean_rep, max_rep], axis=-1).reshape(m * player_count, BELIEF_SEAT_FEAT_DIM)

    probs = belief_model.predict_probs_batch(seat_x).reshape(m, player_count, NUM_ROLES)

    # Preserve exact deterministic facts already encoded in p[:, :, 0..8] and g[:, 6:11]
    my_role_idx = np.argmax(g[:, 6:11], axis=-1)  # [M]
    for i in range(m):
        mr = int(my_role_idx[i])
        for s in range(player_count):
            if p[i, s, 0] > 0.5:  # is_self
                probs[i, s] = 0.0
                probs[i, s, mr] = 1.0
            elif p[i, s, 4] > 0.5 or p[i, s, 5] > 0.5:  # wolf teammate or seer checked wolf
                probs[i, s] = 0.0
                probs[i, s, 3] = 1.0
            elif p[i, s, 8] > 0.5:  # revealed hunter
                probs[i, s] = 0.0
                probs[i, s, 2] = 1.0
            elif p[i, s, 6] > 0.5:  # seer checked good
                probs[i, s, 3] = 0.0
                probs[i, s, 0] = 0.0
                denom = probs[i, s].sum()
                if denom > 1e-8:
                    probs[i, s] /= denom
        if mr == 3:  # werewolf knows non-teammates are not wolves
            for s in range(player_count):
                if p[i, s, 0] < 0.5 and p[i, s, 4] < 0.5:
                    probs[i, s, 3] = 0.0
                    denom = probs[i, s].sum()
                    if denom > 1e-8:
                        probs[i, s] /= denom

    out = policy_X.copy()
    out[:, GLOBAL_FEAT_DIM + player_count * PLAYER_FEAT_DIM :] = probs.reshape(m, player_count * NUM_ROLES)
    return out.astype(np.float32)


def evaluate_bc_policy(
    policy: MultiHeadPolicyValueMLP,
    data: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute masked CE loss and top-1 accuracy per decision type."""
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    per_dt: Dict[str, Dict[str, float]] = {}

    for dt in ALL_DECISION_TYPES:
        prefix = f"act_{dt.value}"
        X = data[f"{prefix}_X"]
        y = data[f"{prefix}_y"]
        mask = data[f"{prefix}_mask"]
        if X.shape[0] == 0:
            continue
        probs, log_probs, _, _ = policy.forward_head(X, dt, mask)
        preds = np.argmax(np.where(mask, probs, -1.0), axis=-1)
        correct = int(np.sum(preds == y))
        n = int(X.shape[0])
        loss = -float(np.mean(log_probs[np.arange(n), y]))
        per_dt[dt.value] = {
            "count": n,
            "acc": round(correct / n, 4),
            "loss": round(loss, 4),
        }
        total_loss += loss * n
        total_correct += correct
        total_count += n

    return {
        "total_count": total_count,
        "overall_acc": round(total_correct / max(total_count, 1), 4),
        "overall_loss": round(total_loss / max(total_count, 1), 4),
        "per_decision": per_dt,
    }


def train_bc_policy(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    belief_model: Optional[BeliefMLP] = None,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    winner_weight_boost: float = 1.35,
    arch: str = "pointer",
    seed: int = 42,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Train PolicyValue network via offline masked Behavior Cloning."""
    rng = np.random.default_rng(seed)

    # Refresh offline policy states with trained BeliefMLP so BC matches live inference
    tr = dict(train_data)
    va = dict(val_data)
    if belief_model is not None:
        for dt in ALL_DECISION_TYPES:
            key = f"act_{dt.value}_X"
            tr[key] = refresh_policy_states_with_belief(tr[key], belief_model)
            va[key] = refresh_policy_states_with_belief(va[key], belief_model)

    policy: Any = (
        SeatEquivariantPolicyValueNet(seed=seed)
        if arch == "pointer"
        else MultiHeadPolicyValueMLP(seed=seed)
    )
    opt = AdamOptimizer(policy.params, lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_params = {k: v.copy() for k, v in policy.params.items()}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        # Interleave updates across all 6 decision heads
        dt_order = list(ALL_DECISION_TYPES)
        rng.shuffle(dt_order)
        for dt in dt_order:
            prefix = f"act_{dt.value}"
            X = tr[f"{prefix}_X"]
            y = tr[f"{prefix}_y"]
            mask = tr[f"{prefix}_mask"]
            won = tr[f"{prefix}_won"]
            n = X.shape[0]
            if n == 0:
                continue
            # Mildly upweight winning-camp expert decisions
            weights = np.where(won > 0.5, winner_weight_boost, 1.0).astype(np.float32)
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                loss, grads = policy.bc_loss_and_grads(dt, X[idx], y[idx], mask[idx], weights[idx])
                opt.step(policy.params, grads)
                epoch_losses.append(loss)

        val_eval = evaluate_bc_policy(policy, va)
        rec = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(epoch_losses)), 4),
            "val_loss": val_eval["overall_loss"],
            "val_acc": val_eval["overall_acc"],
            "val_vote_acc": val_eval["per_decision"].get("day_vote", {}).get("acc", 0.0),
            "val_speech_acc": val_eval["per_decision"].get("day_speech", {}).get("acc", 0.0),
        }
        history.append(rec)
        if val_eval["overall_loss"] < best_val_loss:
            best_val_loss = val_eval["overall_loss"]
            best_params = {k: v.copy() for k, v in policy.params.items()}

    policy.params = best_params
    return policy, history


@dataclass
class RecordedTransition:
    player_id: int
    role_idx: int
    day: int
    decision_type: DecisionType
    state: np.ndarray
    action_id: int
    mask: np.ndarray
    log_prob: float
    value: float
    step_reward: float = 0.0


class RecordingPolicyAgent(BaseAgent):
    """Training wrapper around MultiHeadPolicyValueMLP that logs transitions for PPO."""

    def __init__(
        self,
        policy_model: MultiHeadPolicyValueMLP,
        belief_model: Optional[BeliefMLP],
        buffer: List[RecordedTransition],
        seed: int = 0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(name="recording_ppo")
        self.policy_model = policy_model
        self.belief_model = belief_model
        self.buffer = buffer
        self._rng = np.random.default_rng(seed)
        self.temperature = temperature

    def act(self, request: ActionRequest) -> ActionResponse:
        obs = request.observation
        belief = (
            self.belief_model.predict_obs(obs, apply_constraints=True)
            if self.belief_model is not None
            else default_prior_belief(obs)
        )
        state = encode_policy_state(obs, belief)
        mask = np.asarray(request.legal_action_mask, dtype=bool)
        action_id, log_prob, value = self.policy_model.act_single(
            state=state,
            dt=request.decision_type,
            mask=mask,
            rng=self._rng,
            deterministic=False,
            temperature=self.temperature,
        )
        self.buffer.append(
            RecordedTransition(
                player_id=obs.player_id,
                role_idx=ROLE_TO_IDX[obs.my_role],
                day=obs.day,
                decision_type=request.decision_type,
                state=state,
                action_id=action_id,
                mask=mask,
                log_prob=log_prob,
                value=value,
            )
        )
        return ActionResponse(action_id=action_id)


def _assign_episode_rewards(
    transitions: List[RecordedTransition],
    result: GameResult,
) -> Tuple[List[RecordedTransition], List[float]]:
    """Compute terminal win/loss + aligned tactical shaping rewards and discounted returns."""
    gamma = 0.97
    roles = result.roles
    by_player: Dict[int, List[RecordedTransition]] = defaultdict(list)

    for tr in transitions:
        pid = tr.player_id
        role = roles[pid]
        dt = tr.decision_type
        act = tr.action_id
        shaping = 0.0

        # Mild, well-aligned tactical shaping to accelerate credit assignment
        if dt == DecisionType.DAY_VOTE and act >= 1:
            target_role = roles[act]
            if role == Role.WEREWOLF:
                # Allow strategic bussing (倒钩); only reward pushing good players slightly
                shaping += 0.05 if target_role != Role.WEREWOLF else 0.0
            else:
                shaping += 0.12 if target_role == Role.WEREWOLF else -0.05
        elif dt == DecisionType.NIGHT_SEER_CHECK:
            target_role = roles[act + 1]
            if target_role == Role.WEREWOLF:
                shaping += 0.15
        elif dt == DecisionType.NIGHT_WOLF_KILL:
            target_role = roles[act + 1]
            if target_role == Role.WEREWOLF:
                shaping -= 0.25
            elif target_role.is_god:
                shaping += 0.18
            else:
                shaping += 0.04
        elif dt == DecisionType.NIGHT_WITCH_ACT:
            if act >= 2:
                target_role = roles[act - 1]
                shaping += 0.35 if target_role == Role.WEREWOLF else -0.35
        elif dt == DecisionType.HUNTER_SHOOT and act >= 1:
            target_role = roles[act]
            shaping += 0.35 if target_role == Role.WEREWOLF else -0.25

        tr.step_reward = shaping
        by_player[pid].append(tr)

    flat_tr: List[RecordedTransition] = []
    flat_ret: List[float] = []
    for pid, seq in by_player.items():
        role = roles[pid]
        if result.winner == Camp.DRAW:
            term = 0.0
        elif (role == Role.WEREWOLF and result.winner == Camp.WEREWOLF) or (
            role != Role.WEREWOLF and result.winner == Camp.GOOD
        ):
            term = 1.0
        else:
            term = -1.0

        seq[-1].step_reward += term
        running = 0.0
        rets = [0.0] * len(seq)
        for i in range(len(seq) - 1, -1, -1):
            running = seq[i].step_reward + gamma * running
            rets[i] = running
        flat_tr.extend(seq)
        flat_ret.extend(rets)
    return flat_tr, flat_ret


def evaluate_matchup(
    good_factory: Callable[[int], BaseAgent],
    wolf_factory: Callable[[int], BaseAgent],
    games: int = 200,
    seed: int = 10000,
) -> Dict[str, float]:
    """Evaluate a Good-team agent family against a Werewolf-team agent family."""
    engine = WerewolfEngine(mode="symbolic")
    good_wins = 0
    wolf_wins = 0
    draws = 0
    good_vote_total = 0
    good_vote_wolf = 0
    witch_poison_total = 0
    witch_poison_wolf = 0
    hunter_shoot_total = 0
    hunter_shoot_wolf = 0
    total_days = 0

    rng = np.random.default_rng(seed)
    base_roles = [
        Role.SEER,
        Role.WITCH,
        Role.HUNTER,
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
    ]

    for g_idx in range(games):
        perm = rng.permutation(9)
        fixed_roles = {pid: base_roles[int(perm[pid - 1])] for pid in range(1, 10)}
        agents: Dict[int, BaseAgent] = {}
        for pid, r in fixed_roles.items():
            s = seed * 17 + g_idx * 13 + pid
            agents[pid] = wolf_factory(s) if r == Role.WEREWOLF else good_factory(s)

        res = engine.run_game(agents, seed=seed + g_idx, game_id=f"eval_{g_idx}", fixed_roles=fixed_roles)
        total_days += res.total_days
        if res.winner == Camp.GOOD:
            good_wins += 1
        elif res.winner == Camp.WEREWOLF:
            wolf_wins += 1
        else:
            draws += 1

        for st in res.trajectory:
            role = res.roles[st.player_id]
            if st.decision_type == DecisionType.DAY_VOTE.value and st.action_id >= 1 and role != Role.WEREWOLF:
                good_vote_total += 1
                if res.roles[st.action_id] == Role.WEREWOLF:
                    good_vote_wolf += 1
            elif st.decision_type == DecisionType.NIGHT_WITCH_ACT.value and st.action_id >= 2:
                witch_poison_total += 1
                if res.roles[st.action_id - 1] == Role.WEREWOLF:
                    witch_poison_wolf += 1
            elif st.decision_type == DecisionType.HUNTER_SHOOT.value and st.action_id >= 1:
                hunter_shoot_total += 1
                if res.roles[st.action_id] == Role.WEREWOLF:
                    hunter_shoot_wolf += 1

    return {
        "games": games,
        "good_win_rate": round(good_wins / games, 4),
        "wolf_win_rate": round(wolf_wins / games, 4),
        "draw_rate": round(draws / games, 4),
        "avg_days": round(total_days / games, 3),
        "good_vote_wolf_acc": round(good_vote_wolf / max(good_vote_total, 1), 4),
        "witch_poison_wolf_acc": round(witch_poison_wolf / max(witch_poison_total, 1), 4),
        "hunter_shoot_wolf_acc": round(hunter_shoot_wolf / max(hunter_shoot_total, 1), 4),
    }


def train_self_play_ppo(
    init_policy: MultiHeadPolicyValueMLP,
    belief_model: Optional[BeliefMLP],
    iterations: int = 100,
    games_per_iter: int = 48,
    ppo_epochs: int = 3,
    batch_size: int = 256,
    lr: float = 4e-4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.015,
    kl_coef: float = 0.015,
    eval_every: int = 10,
    eval_games: int = 150,
    seed: int = 42,
) -> Tuple[MultiHeadPolicyValueMLP, List[Dict[str, Any]]]:
    """Population-based Self-Play Masked PPO with Role-Conditioned Advantage Estimation (RAE)."""
    rng = np.random.default_rng(seed)
    engine = WerewolfEngine(mode="symbolic")

    policy = init_policy.clone()
    bc_anchor = init_policy.clone()
    opt = AdamOptimizer(policy.params, lr=lr, weight_decay=1e-4, max_grad_norm=0.8)

    snapshot_pool: List[MultiHeadPolicyValueMLP] = [init_policy.clone()]
    history: List[Dict[str, Any]] = []

    base_roles = [
        Role.SEER,
        Role.WITCH,
        Role.HUNTER,
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
    ]

    best_score = -1.0
    best_params = {k: v.copy() for k, v in policy.params.items()}

    def run_checkpoint_eval(step_iter: int) -> Dict[str, float]:
        g_vs_rw = evaluate_matchup(
            good_factory=lambda s: NeuralPolicyAgent(policy, belief_model, seed=s, deterministic=False, temperature=0.7),
            wolf_factory=lambda s: RuleBasedAgent(seed=s),
            games=eval_games,
            seed=20000,
        )
        w_vs_rg = evaluate_matchup(
            good_factory=lambda s: RuleBasedAgent(seed=s),
            wolf_factory=lambda s: NeuralPolicyAgent(policy, belief_model, seed=s, deterministic=False, temperature=0.7),
            games=eval_games,
            seed=30000,
        )
        return {
            "eval_good_vs_rule_wolves": g_vs_rw["good_win_rate"],
            "eval_good_vote_wolf_acc": g_vs_rw["good_vote_wolf_acc"],
            "eval_wolf_vs_rule_good": w_vs_rg["wolf_win_rate"],
            "eval_combined_score": round(0.5 * (g_vs_rw["good_win_rate"] + w_vs_rg["wolf_win_rate"]), 4),
        }

    # Initial iteration-0 benchmark before any RL step
    init_eval = run_checkpoint_eval(0)
    history.append({"iter": 0, "games_seen": 0, **init_eval})
    best_score = init_eval["eval_combined_score"]

    t_start = time.perf_counter()

    for it in range(1, iterations + 1):
        # Linear LR decay
        cur_lr = lr * max(0.2, 1.0 - 0.8 * (it / max(iterations, 1)))
        all_tr: List[RecordedTransition] = []
        all_ret: List[float] = []
        iter_role_returns: Dict[int, List[float]] = defaultdict(list)

        for g in range(games_per_iter):
            game_seed = seed * 10000 + it * 500 + g
            perm = rng.permutation(9)
            fixed_roles = {pid: base_roles[int(perm[pid - 1])] for pid in range(1, 10)}

            # Population schedule: 35% self-play, 30% RL-Good vs Rule-Wolves,
            # 20% Rule-Good vs RL-Wolves, 15% RL vs Historical/Random
            u = rng.random()
            game_buf: List[RecordedTransition] = []
            agents: Dict[int, BaseAgent] = {}

            if u < 0.35:
                for pid in range(1, 10):
                    agents[pid] = RecordingPolicyAgent(policy, belief_model, game_buf, seed=game_seed * 11 + pid)
            elif u < 0.65:
                for pid, r in fixed_roles.items():
                    s = game_seed * 11 + pid
                    if r != Role.WEREWOLF:
                        agents[pid] = RecordingPolicyAgent(policy, belief_model, game_buf, seed=s)
                    else:
                        agents[pid] = RuleBasedAgent(seed=s)
            elif u < 0.85:
                for pid, r in fixed_roles.items():
                    s = game_seed * 11 + pid
                    if r == Role.WEREWOLF:
                        agents[pid] = RecordingPolicyAgent(policy, belief_model, game_buf, seed=s)
                    else:
                        agents[pid] = RuleBasedAgent(seed=s)
            else:
                snap = snapshot_pool[int(rng.integers(0, len(snapshot_pool)))]
                rl_camp_is_wolf = bool(rng.random() < 0.5)
                use_random = bool(rng.random() < 0.35)
                for pid, r in fixed_roles.items():
                    s = game_seed * 11 + pid
                    is_rl = (r == Role.WEREWOLF) if rl_camp_is_wolf else (r != Role.WEREWOLF)
                    if is_rl:
                        agents[pid] = RecordingPolicyAgent(policy, belief_model, game_buf, seed=s)
                    elif use_random:
                        agents[pid] = RandomAgent(seed=s)
                    else:
                        agents[pid] = NeuralPolicyAgent(snap, belief_model, seed=s)

            res = engine.run_game(agents, seed=game_seed, game_id=f"ppo_{it}_{g}", fixed_roles=fixed_roles)
            ep_tr, ep_ret = _assign_episode_rewards(game_buf, res)
            all_tr.extend(ep_tr)
            all_ret.extend(ep_ret)
            for tr, ret in zip(ep_tr, ep_ret):
                iter_role_returns[tr.role_idx].append(ret)

        # Role-Conditioned Advantage Estimation (RAE): normalize advantages per role
        raw_adv = np.array([ret - tr.value for tr, ret in zip(all_tr, all_ret)], dtype=np.float32)
        role_ids = np.array([tr.role_idx for tr in all_tr], dtype=np.int64)
        norm_adv = np.zeros_like(raw_adv)
        for r_idx in range(NUM_ROLES):
            rm = role_ids == r_idx
            if np.sum(rm) >= 2:
                vals = raw_adv[rm]
                norm_adv[rm] = (vals - vals.mean()) / (vals.std() + 1e-6)
            elif np.sum(rm) == 1:
                norm_adv[rm] = raw_adv[rm]

        # Group transitions by DecisionType for vectorized PPO updates
        by_dt: Dict[DecisionType, List[int]] = defaultdict(list)
        for idx, tr in enumerate(all_tr):
            by_dt[tr.decision_type].append(idx)

        ppo_stats = defaultdict(list)
        for _ in range(ppo_epochs):
            dts = list(by_dt.keys())
            rng.shuffle(dts)
            for dt in dts:
                idxs = np.array(by_dt[dt], dtype=np.int64)
                if len(idxs) < 4:
                    continue
                rng.shuffle(idxs)
                X_dt = np.stack([all_tr[i].state for i in idxs], axis=0)
                a_dt = np.array([all_tr[i].action_id for i in idxs], dtype=np.int64)
                m_dt = np.stack([all_tr[i].mask for i in idxs], axis=0)
                olp_dt = np.array([all_tr[i].log_prob for i in idxs], dtype=np.float32)
                adv_dt = norm_adv[idxs]
                ret_dt = np.array([all_ret[i] for i in idxs], dtype=np.float32)

                for start in range(0, len(idxs), batch_size):
                    sl = slice(start, start + batch_size)
                    # Stronger BC-KL anchor on speech head to prevent symbolic speech exploits (Sim-to-Real alignment)
                    eff_kl = kl_coef * (5.0 if dt == DecisionType.DAY_SPEECH else 1.0)
                    metrics, grads = policy.ppo_loss_and_grads(
                        dt=dt,
                        x=X_dt[sl],
                        actions=a_dt[sl],
                        mask=m_dt[sl],
                        old_log_probs=olp_dt[sl],
                        advantages=adv_dt[sl],
                        returns=ret_dt[sl],
                        clip_eps=clip_eps,
                        entropy_coef=entropy_coef,
                        bc_anchor=bc_anchor,
                        kl_coef=eff_kl,
                    )
                    opt.step(policy.params, grads, lr=cur_lr)
                    for k, v in metrics.items():
                        ppo_stats[k].append(v)

        if it % 15 == 0:
            snapshot_pool.append(policy.clone())
            if len(snapshot_pool) > 8:
                snapshot_pool.pop(1)  # Always retain initial BC anchor at index 0

        if it % eval_every == 0 or it == iterations:
            ev = run_checkpoint_eval(it)
            rec = {
                "iter": it,
                "games_seen": it * games_per_iter,
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
                "mean_return": round(float(np.mean(all_ret)), 4),
                "policy_loss": round(float(np.mean(ppo_stats["policy_loss"])), 4),
                "value_loss": round(float(np.mean(ppo_stats["value_loss"])), 4),
                "entropy": round(float(np.mean(ppo_stats["entropy"])), 4),
                "kl_bc": round(float(np.mean(ppo_stats["kl_bc"])), 4),
                **ev,
            }
            history.append(rec)
            if ev["eval_combined_score"] >= best_score:
                best_score = ev["eval_combined_score"]
                best_params = {k: v.copy() for k, v in policy.params.items()}

    policy.params = best_params
    return policy, history
