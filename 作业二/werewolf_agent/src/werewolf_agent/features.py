"""Unified NumPy observation feature encoder shared by MaKTO offline data, Belief Tracker, RL, and Frozen LLM."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .protocol import Phase, PlayerObservation, Role, SpeechIntentType

ROLE_ORDER: Tuple[Role, ...] = (
    Role.SEER,
    Role.WITCH,
    Role.HUNTER,
    Role.WEREWOLF,
    Role.VILLAGER,
)
ROLE_TO_IDX: Dict[Role, int] = {r: i for i, r in enumerate(ROLE_ORDER)}
NUM_ROLES = len(ROLE_ORDER)

PHASE_ORDER: Tuple[Phase, ...] = (
    Phase.NIGHT,
    Phase.DAY_SPEECH,
    Phase.DAY_VOTE,
    Phase.HUNTER_RESOLVE,
    Phase.GAME_OVER,
)
PHASE_TO_IDX: Dict[Phase, int] = {p: i for i, p in enumerate(PHASE_ORDER)}

GLOBAL_FEAT_DIM = 19
PLAYER_FEAT_DIM = 24
BELIEF_SEAT_FEAT_DIM = GLOBAL_FEAT_DIM + PLAYER_FEAT_DIM * 3  # 19 + 24*3 = 91


def policy_state_dim(player_count: int = 9) -> int:
    """Dimension of flat policy input: global + per-player + belief probs."""
    return GLOBAL_FEAT_DIM + player_count * (PLAYER_FEAT_DIM + NUM_ROLES)


def encode_observation_parts(obs: PlayerObservation) -> Tuple[np.ndarray, np.ndarray]:
    """Encode PlayerObservation into (global_feat [19], player_feats [N, 24])."""
    n = obs.player_count
    g = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    p = np.zeros((n, PLAYER_FEAT_DIM), dtype=np.float32)

    alive_set = set(obs.alive_players)
    dead_set = set(obs.dead_players)
    vote_cand_set = set(obs.vote_candidates)
    wolf_teammate_set = set(obs.werewolf_teammates)
    seer_wolf_set = {t for _, t, is_w in obs.seer_checks if is_w}
    seer_good_set = {t for _, t, is_w in obs.seer_checks if not is_w}

    # Global features
    g[0] = min(obs.day, 10) / 10.0
    g[1 + PHASE_TO_IDX[obs.phase]] = 1.0
    g[6 + ROLE_TO_IDX[obs.my_role]] = 1.0
    g[11] = len(alive_set) / float(n)
    g[12] = len(dead_set) / float(n)
    g[13] = 1.0 if obs.witch_has_antidote else 0.0
    g[14] = 1.0 if obs.witch_has_poison else 0.0
    g[15] = min(len(seer_wolf_set), 3) / 3.0
    g[16] = min(len(seer_good_set), 6) / 6.0

    # Per-player static & private-view features (cols 0..8)
    for pid in range(1, n + 1):
        i = pid - 1
        p[i, 0] = 1.0 if pid == obs.player_id else 0.0
        p[i, 1] = 1.0 if pid in alive_set else 0.0
        p[i, 2] = 1.0 if pid in dead_set else 0.0
        p[i, 3] = 1.0 if pid in vote_cand_set else 0.0
        p[i, 4] = 1.0 if pid in wolf_teammate_set else 0.0
        p[i, 5] = 1.0 if pid in seer_wolf_set else 0.0
        p[i, 6] = 1.0 if pid in seer_good_set else 0.0
        p[i, 7] = 1.0 if obs.witch_tonight_attacked == pid else 0.0
        p[i, 8] = 1.0 if obs.revealed_roles.get(pid) == Role.HUNTER else 0.0

    # Parse public events sequentially (speeches & votes)
    seer_claimers = set()
    num_speeches = 0
    last_vote_received = np.zeros(n, dtype=np.float32)

    for ev in obs.public_events:
        if ev.event_type == "speech" and ev.actor is not None and 1 <= ev.actor <= n:
            num_speeches += 1
            ai = ev.actor - 1
            intent_d = ev.payload.get("intent", {})
            itype = intent_d.get("intent_type")
            target = intent_d.get("target")

            # Speaker self-claim flags (cols 9..13) — STRICT first-person regex to never confuse quotes
            txt = str(ev.payload.get("text", ""))
            claimed_any_role = False
            if itype in (SpeechIntentType.CLAIM_SEER_GOOD.value, SpeechIntentType.CLAIM_SEER_WOLF.value) or re.search(
                r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)预言家", txt
            ):
                p[ai, 9] = 1.0
                seer_claimers.add(ev.actor)
                claimed_any_role = True
            if itype in (SpeechIntentType.CLAIM_WITCH_SAVE.value, SpeechIntentType.CLAIM_WITCH_POISON.value) or re.search(
                r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)女巫", txt
            ):
                p[ai, 10] = 1.0
                claimed_any_role = True
            if itype == SpeechIntentType.CLAIM_HUNTER_PRESSURE.value or re.search(
                r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)猎人", txt
            ):
                p[ai, 11] = 1.0
                claimed_any_role = True
            if not claimed_any_role:
                if itype in (SpeechIntentType.CLAIM_VILLAGER_ACCUSE.value, SpeechIntentType.CLAIM_VILLAGER_DEFEND.value):
                    p[ai, 12] = 1.0
                elif itype == SpeechIntentType.STAY_LOW.value:
                    p[ai, 13] = 1.0

            # Social signals received by target (cols 14..19) & targeting self (col 23)
            if isinstance(target, int) and 1 <= target <= n:
                ti = target - 1
                if itype == SpeechIntentType.CLAIM_SEER_GOOD.value:
                    p[ti, 14] += 1.0 / 3.0
                elif itype == SpeechIntentType.CLAIM_SEER_WOLF.value:
                    p[ti, 15] += 1.0 / 3.0
                    if target == obs.player_id:
                        p[ai, 23] += 1.0 / 3.0
                elif itype in (SpeechIntentType.CLAIM_VILLAGER_ACCUSE.value, SpeechIntentType.CLAIM_HUNTER_PRESSURE.value):
                    p[ti, 16] += 1.0 / 5.0
                    if target == obs.player_id:
                        p[ai, 23] += 1.0 / 3.0
                elif itype == SpeechIntentType.CLAIM_VILLAGER_DEFEND.value:
                    p[ti, 17] += 1.0 / 5.0
                elif itype == SpeechIntentType.CLAIM_WITCH_SAVE.value:
                    p[ti, 18] += 1.0 / 2.0
                elif itype == SpeechIntentType.CLAIM_WITCH_POISON.value:
                    p[ti, 19] += 1.0 / 2.0
                    if target == obs.player_id:
                        p[ai, 23] += 1.0 / 3.0

        elif ev.event_type == "vote_result":
            votes = ev.payload.get("votes", {})
            round_recv = np.zeros(n, dtype=np.float32)
            for voter_str, tgt in votes.items():
                try:
                    voter = int(voter_str)
                except (TypeError, ValueError):
                    continue
                if isinstance(tgt, int) and 1 <= tgt <= n:
                    p[tgt - 1, 20] += 1.0 / 10.0
                    round_recv[tgt - 1] += 1.0 / float(n)
                    if tgt == obs.player_id and 1 <= voter <= n:
                        p[voter - 1, 22] += 1.0 / 3.0
            last_vote_received = round_recv

    p[:, 21] = last_vote_received
    p[:, 14:24] = np.clip(p[:, 14:24], 0.0, 2.0)

    g[17] = min(len(seer_claimers), 4) / 4.0
    g[18] = min(num_speeches, 40) / 40.0
    return g, p


def build_belief_seat_features(obs: PlayerObservation) -> np.ndarray:
    """Build seat-shared Belief input matrix of shape [N, 91].

    Each row i concatenates [global (19), player_i (24), mean(all players) (24), max(all players) (24)].
    Sharing weights across seats guarantees seat-index equivariance and 9x sample efficiency.
    """
    g, p = encode_observation_parts(obs)
    n = p.shape[0]
    p_mean = p.mean(axis=0)
    p_max = p.max(axis=0)
    out = np.empty((n, BELIEF_SEAT_FEAT_DIM), dtype=np.float32)
    for i in range(n):
        out[i, :GLOBAL_FEAT_DIM] = g
        out[i, GLOBAL_FEAT_DIM : GLOBAL_FEAT_DIM + PLAYER_FEAT_DIM] = p[i]
        out[i, GLOBAL_FEAT_DIM + PLAYER_FEAT_DIM : GLOBAL_FEAT_DIM + 2 * PLAYER_FEAT_DIM] = p_mean
        out[i, GLOBAL_FEAT_DIM + 2 * PLAYER_FEAT_DIM :] = p_max
    return out


def apply_belief_hard_constraints(
    raw_probs: np.ndarray,
    obs: PlayerObservation,
) -> np.ndarray:
    """Enforce exact known role facts from obs onto predicted [N, 5] belief distribution."""
    probs = np.clip(np.asarray(raw_probs, dtype=np.float32), 1e-6, 1.0).copy()
    n = obs.player_count
    assert probs.shape == (n, NUM_ROLES)

    # Normalize initial rows
    probs /= probs.sum(axis=1, keepdims=True)

    # 1. Self role is known with certainty
    self_idx = obs.player_id - 1
    probs[self_idx] = 0.0
    probs[self_idx, ROLE_TO_IDX[obs.my_role]] = 1.0

    # 2. Werewolf knows exact teammates (and all others are non-wolf!)
    if obs.my_role == Role.WEREWOLF:
        wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
        for pid in range(1, n + 1):
            i = pid - 1
            if pid in wolf_set:
                probs[i] = 0.0
                probs[i, ROLE_TO_IDX[Role.WEREWOLF]] = 1.0
            else:
                probs[i, ROLE_TO_IDX[Role.WEREWOLF]] = 0.0
                s = probs[i].sum()
                probs[i] = probs[i] / s if s > 1e-8 else np.array([0.25, 0.25, 0.25, 0.0, 0.25], dtype=np.float32)

    # 3. Seer knows exact check outcomes
    if obs.my_role == Role.SEER:
        for _, tgt, is_wolf in obs.seer_checks:
            i = tgt - 1
            if is_wolf:
                probs[i] = 0.0
                probs[i, ROLE_TO_IDX[Role.WEREWOLF]] = 1.0
            else:
                probs[i, ROLE_TO_IDX[Role.WEREWOLF]] = 0.0
                probs[i, ROLE_TO_IDX[Role.SEER]] = 0.0
                s = probs[i].sum()
                probs[i] = probs[i] / s if s > 1e-8 else np.array([0.0, 0.2, 0.2, 0.0, 0.6], dtype=np.float32)

    # 4. Unique god role exclusion (if I am a unique god, nobody else has my role)
    if obs.my_role in (Role.SEER, Role.WITCH, Role.HUNTER):
        my_ridx = ROLE_TO_IDX[obs.my_role]
        for pid in range(1, n + 1):
            if pid != obs.player_id:
                i = pid - 1
                probs[i, my_ridx] = 0.0
                s = probs[i].sum()
                if s > 1e-8:
                    probs[i] /= s

    # 5. Publicly revealed Hunters
    for rpid, rrole in obs.revealed_roles.items():
        if rrole == Role.HUNTER:
            i = rpid - 1
            probs[i] = 0.0
            probs[i, ROLE_TO_IDX[Role.HUNTER]] = 1.0
            for pid in range(1, n + 1):
                if pid != rpid:
                    j = pid - 1
                    probs[j, ROLE_TO_IDX[Role.HUNTER]] = 0.0
                    s = probs[j].sum()
                    if s > 1e-8:
                        probs[j] /= s

    return probs


def default_prior_belief(obs: PlayerObservation) -> np.ndarray:
    """Uniform board prior (1/9, 1/9, 1/9, 3/9, 3/9) refined by known hard constraints."""
    prior = np.tile(
        np.array([1.0 / 9, 1.0 / 9, 1.0 / 9, 3.0 / 9, 3.0 / 9], dtype=np.float32),
        (obs.player_count, 1),
    )
    return apply_belief_hard_constraints(prior, obs)


def encode_policy_state(
    obs: PlayerObservation,
    belief_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode flat policy state vector of shape [280] for 9-player Werewolf."""
    g, p = encode_observation_parts(obs)
    b = belief_probs if belief_probs is not None else default_prior_belief(obs)
    return np.concatenate([g, p.reshape(-1), b.reshape(-1)], axis=0).astype(np.float32)
