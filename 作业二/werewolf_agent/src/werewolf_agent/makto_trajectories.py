"""Reconstruct strict first-person expert trajectories from MaKTO 9-player Chinese games."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import (
    BELIEF_SEAT_FEAT_DIM,
    NUM_ROLES,
    ROLE_TO_IDX,
    build_belief_seat_features,
    default_prior_belief,
    encode_policy_state,
    policy_state_dim,
)
from .protocol import (
    DecisionType,
    Phase,
    PlayerObservation,
    PublicEvent,
    Role,
    SpeechIntent,
    SpeechIntentType,
    TARGETED_SPEECH_INTENTS,
    action_space_size,
)

MAKTO_ROLE_MAP: Dict[str, Role] = {
    "seer": Role.SEER,
    "witch": Role.WITCH,
    "hunter": Role.HUNTER,
    "werewolf": Role.WEREWOLF,
    "simple_villager": Role.VILLAGER,
    "villager": Role.VILLAGER,
}


def map_speech_to_intent(
    speaker: int,
    speech_text: str,
    summary: Optional[Dict[str, Any]],
    alive_set: set[int],
    player_count: int = 9,
) -> SpeechIntent:
    """Deterministically map MaKTO speech + speech_summary into our 64-dim SpeechIntent."""
    summary = summary or {}
    self_present = [str(x) for x in (summary.get("self_present") or [])]
    call_for_vote = [int(x) for x in (summary.get("call_for_vote") or []) if isinstance(x, int) and 1 <= x <= player_count and x != speaker]
    id_labels_raw = summary.get("identity_label") or {}

    wolf_labeled: List[int] = []
    good_labeled: List[int] = []
    for k, vals in id_labels_raw.items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            continue
        if pid == speaker or not (1 <= pid <= player_count):
            continue
        labels = [str(v) for v in (vals or [])]
        if labels == ["werewolf"]:
            wolf_labeled.append(pid)
        elif any(v in ("seer", "witch", "hunter", "simple_villager", "villager") for v in labels) and "werewolf" not in labels:
            good_labeled.append(pid)

    def first_alive_or_any(cands: Sequence[int]) -> Optional[int]:
        for c in cands:
            if c in alive_set:
                return c
        return cands[0] if cands else None

    mentioned = [int(m) for m in re.findall(r"([1-9])\s*号", speech_text) if int(m) != speaker]

    # 1. Claimed Seer
    if "seer" in self_present:
        # Check explicit 查杀 / 金水 cues in text first, then structured labels
        m_kill = re.search(r"([1-9])\s*号[^，。；\n]{0,8}(查杀|是狼|狼人)", speech_text)
        m_good = re.search(r"([1-9])\s*号[^，。；\n]{0,8}(金水|是好人|好人)", speech_text)
        if m_kill and int(m_kill.group(1)) != speaker:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, int(m_kill.group(1)))
        if wolf_labeled:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, first_alive_or_any(wolf_labeled))
        if call_for_vote:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_WOLF, call_for_vote[0])
        if m_good and int(m_good.group(1)) != speaker:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_GOOD, int(m_good.group(1)))
        if good_labeled:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_GOOD, first_alive_or_any(good_labeled))
        if mentioned:
            return SpeechIntent(SpeechIntentType.CLAIM_SEER_GOOD, mentioned[0])

    # 2. Claimed Witch
    if "witch" in self_present:
        m_save = re.search(r"(救了?|捞了?|银水)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,6}(银水|我救)", speech_text)
        m_poison = re.search(r"(毒了?|泼毒|撒毒|吃毒)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,6}(毒|泼)", speech_text)
        if m_poison:
            t = int(m_poison.group(2) or m_poison.group(3))
            if t != speaker:
                return SpeechIntent(SpeechIntentType.CLAIM_WITCH_POISON, t)
        if m_save:
            t = int(m_save.group(2) or m_save.group(3))
            if t != speaker:
                return SpeechIntent(SpeechIntentType.CLAIM_WITCH_SAVE, t)
        if wolf_labeled:
            return SpeechIntent(SpeechIntentType.CLAIM_WITCH_POISON, first_alive_or_any(wolf_labeled))
        if good_labeled:
            return SpeechIntent(SpeechIntentType.CLAIM_WITCH_SAVE, first_alive_or_any(good_labeled))

    # 3. Claimed Hunter
    if "hunter" in self_present:
        t = first_alive_or_any(call_for_vote) or first_alive_or_any(wolf_labeled) or first_alive_or_any(mentioned)
        if t is not None and t in alive_set:
            return SpeechIntent(SpeechIntentType.CLAIM_HUNTER_PRESSURE, t)

    # 4. Villager / Low-profile accusation or defense (must target alive seat)
    accuse_target = first_alive_or_any(call_for_vote) or first_alive_or_any(wolf_labeled)
    if accuse_target is not None and accuse_target in alive_set:
        return SpeechIntent(SpeechIntentType.CLAIM_VILLAGER_ACCUSE, accuse_target)

    defend_target = first_alive_or_any(good_labeled)
    if defend_target is not None and defend_target in alive_set:
        return SpeechIntent(SpeechIntentType.CLAIM_VILLAGER_DEFEND, defend_target)

    return SpeechIntent(SpeechIntentType.STAY_LOW, None)


def _speech_legal_mask(speaker: int, alive: Dict[int, bool], n: int = 9) -> List[bool]:
    mask = [True]
    for itype in TARGETED_SPEECH_INTENTS:
        for tgt in range(1, n + 1):
            if tgt == speaker:
                mask.append(False)
            elif itype in (
                SpeechIntentType.CLAIM_SEER_GOOD,
                SpeechIntentType.CLAIM_SEER_WOLF,
                SpeechIntentType.CLAIM_WITCH_SAVE,
                SpeechIntentType.CLAIM_WITCH_POISON,
            ):
                mask.append(True)
            else:
                mask.append(alive[tgt])
    return mask


@dataclass
class BeliefSample:
    game_id: str
    day: int
    observer_id: int
    observer_role: int
    seat_features: np.ndarray  # [9, 91]
    true_roles: np.ndarray  # [9] int64 in 0..4
    unknown_mask: np.ndarray  # [9] bool (seats not deterministically known to observer)


@dataclass
class ActionSample:
    game_id: str
    day: int
    decision_type: DecisionType
    player_id: int
    role_idx: int
    policy_state: np.ndarray  # [280]
    action_id: int
    legal_mask: np.ndarray  # [A] bool
    camp_won: float  # 1.0 if player's camp won the game, else 0.0


def reconstruct_game_trajectories(
    events: List[Dict[str, Any]],
    game_id: str,
    player_count: int = 9,
) -> Tuple[List[BeliefSample], List[ActionSample], Dict[str, Any]]:
    """Replay one MaKTO event_zh.json strictly causally to produce first-person samples."""
    n = player_count
    roles: Dict[int, Role] = {}
    winner_camp = None
    for ev in events:
        if ev.get("event") == "roles":
            c = ev["content"]
            roles[int(c["player"])] = MAKTO_ROLE_MAP[str(c["role"])]
        elif ev.get("event") == "end":
            w = str(ev.get("winner", "")).lower()
            if "villager" in w:
                winner_camp = "good"
            elif "werewolf" in w or "werewolves" in w:
                winner_camp = "werewolf"

    if len(roles) != n or set(roles.keys()) != set(range(1, n + 1)):
        raise ValueError(f"{game_id}: invalid role map {roles}")

    true_role_vec = np.array([ROLE_TO_IDX[roles[p]] for p in range(1, n + 1)], dtype=np.int64)
    wolf_set = {p for p, r in roles.items() if r == Role.WEREWOLF}

    alive: Dict[int, bool] = {p: True for p in range(1, n + 1)}
    revealed_roles: Dict[int, Role] = {}
    public_events: List[PublicEvent] = []
    seer_checks: List[Tuple[int, int, bool]] = []
    witch_has_antidote = True
    witch_has_poison = True
    tonight_wolf_target: Optional[int] = None

    day = 0
    phase = Phase.NIGHT
    belief_samples: List[BeliefSample] = []
    action_samples: List[ActionSample] = []
    stats = Counter()

    def camp_won_for(pid: int) -> float:
        if winner_camp is None:
            return 0.5
        is_wolf = roles[pid] == Role.WEREWOLF
        return 1.0 if ((is_wolf and winner_camp == "werewolf") or (not is_wolf and winner_camp == "good")) else 0.0

    def build_obs(
        pid: int,
        cur_phase: Phase,
        witch_attacked: Optional[int] = None,
        vote_cands: Sequence[int] = (),
    ) -> PlayerObservation:
        role = roles[pid]
        teammates = tuple(sorted(wolf_set - {pid})) if role == Role.WEREWOLF else ()
        checks = tuple(seer_checks) if role == Role.SEER else ()
        return PlayerObservation(
            game_id=game_id,
            mode="symbolic",
            day=max(day, 1),
            phase=cur_phase,
            player_id=pid,
            my_role=role,
            player_count=n,
            alive_players=tuple(p for p in range(1, n + 1) if alive[p]),
            dead_players=tuple(p for p in range(1, n + 1) if not alive[p]),
            revealed_roles=dict(revealed_roles),
            public_events=tuple(public_events),
            werewolf_teammates=teammates,
            seer_checks=checks,
            witch_has_antidote=witch_has_antidote if role == Role.WITCH else None,
            witch_has_poison=witch_has_poison if role == Role.WITCH else None,
            witch_tonight_attacked=witch_attacked if (role == Role.WITCH and witch_has_antidote) else None,
            vote_candidates=tuple(vote_cands),
        )

    def add_belief_sample(obs: PlayerObservation) -> None:
        seat_feats = build_belief_seat_features(obs)
        unk = np.ones(n, dtype=bool)
        unk[obs.player_id - 1] = False
        if obs.my_role == Role.WEREWOLF:
            for tm in obs.werewolf_teammates:
                unk[tm - 1] = False
        if obs.my_role == Role.SEER:
            for _, tgt, is_w in obs.seer_checks:
                if is_w:
                    unk[tgt - 1] = False
        for rpid in obs.revealed_roles:
            unk[rpid - 1] = False
        belief_samples.append(
            BeliefSample(
                game_id=game_id,
                day=obs.day,
                observer_id=obs.player_id,
                observer_role=ROLE_TO_IDX[obs.my_role],
                seat_features=seat_feats,
                true_roles=true_role_vec.copy(),
                unknown_mask=unk,
            )
        )

    def add_action_sample(
        obs: PlayerObservation,
        dt: DecisionType,
        action_id: int,
        legal_mask: List[bool],
    ) -> None:
        if not (0 <= action_id < len(legal_mask)) or not legal_mask[action_id]:
            stats[f"skipped_illegal_{dt.value}"] += 1
            return
        state = encode_policy_state(obs, default_prior_belief(obs))
        action_samples.append(
            ActionSample(
                game_id=game_id,
                day=obs.day,
                decision_type=dt,
                player_id=obs.player_id,
                role_idx=ROLE_TO_IDX[obs.my_role],
                policy_state=state,
                action_id=action_id,
                legal_mask=np.asarray(legal_mask, dtype=bool),
                camp_won=camp_won_for(obs.player_id),
            )
        )
        stats[f"kept_{dt.value}"] += 1

    # Index speech_summary by (day_str, player) for pairing with speech
    summary_queue: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev.get("event") == "speech_summary" and isinstance(ev.get("content"), dict):
            c = ev["content"]
            summary_queue[(str(c.get("day", "")), int(c["player"]))].append(c)

    # Track pending witch night heal so heal+poison combine into one NIGHT_WITCH_ACT
    pending_heal: Optional[Tuple[int, Optional[int]]] = None  # (night, healed_player)

    def flush_witch_night(night_num: int, healed_p: Optional[int], poison_p: Optional[int]) -> None:
        nonlocal witch_has_antidote, witch_has_poison
        witch_pids = [p for p in range(1, n + 1) if roles[p] == Role.WITCH and alive[p]]
        if not witch_pids:
            return
        wpid = witch_pids[0]
        can_heal = witch_has_antidote and (tonight_wolf_target is not None)
        poison_mask = [witch_has_poison and alive[p] and p != wpid for p in range(1, n + 1)]
        mask = [True, can_heal] + poison_mask
        obs = build_obs(wpid, Phase.NIGHT, witch_attacked=tonight_wolf_target)
        if healed_p is not None and can_heal:
            act = 1
        elif poison_p is not None and 1 <= poison_p <= n and mask[1 + poison_p]:
            act = 1 + poison_p
        else:
            act = 0
        add_action_sample(obs, DecisionType.NIGHT_WITCH_ACT, act, mask)
        if healed_p is not None:
            witch_has_antidote = False
        if poison_p is not None:
            witch_has_poison = False

    idx = 0
    while idx < len(events):
        ev = events[idx]
        etype = ev.get("event")
        c = ev.get("content")

        if etype == "cycle_round" and isinstance(c, dict):
            if pending_heal is not None:
                flush_witch_night(pending_heal[0], pending_heal[1], None)
                pending_heal = None
            status = c.get("status")
            day = int(c.get("round", day or 1))
            if status == "night":
                phase = Phase.NIGHT
                tonight_wolf_target = None
            elif status == "day":
                phase = Phase.DAY_SPEECH
                ctx = str(c.get("context", ""))
                dead_nums = [] if "平安夜" in ctx else [int(x) for x in re.findall(r"([1-9])\s*号", ctx)]
                for dp in dead_nums:
                    alive[dp] = False
                public_events.append(
                    PublicEvent(
                        event_id=len(public_events) + 1,
                        day=day,
                        phase=Phase.DAY_SPEECH,
                        event_type="night_announcement",
                        actor=None,
                        payload={"dead_players": sorted(dead_nums), "text": ctx},
                    )
                )

        elif etype == "inquired" and isinstance(c, dict):
            seer_pids = [p for p in range(1, n + 1) if roles[p] == Role.SEER and alive[p]]
            tgt = c.get("player")
            if seer_pids and isinstance(tgt, int) and 1 <= tgt <= n:
                spid = seer_pids[0]
                already = {t for _, t, _ in seer_checks}
                mask = [alive[p] and p != spid and p not in already for p in range(1, n + 1)]
                if not any(mask):
                    mask = [alive[p] and p != spid for p in range(1, n + 1)]
                obs = build_obs(spid, Phase.NIGHT)
                add_action_sample(obs, DecisionType.NIGHT_SEER_CHECK, tgt - 1, mask)
                seer_checks.append((int(c.get("night", day)), tgt, bool(c.get("is_werewolf"))))

        elif etype == "werewolf_night_discuss" and isinstance(c, dict):
            dec = c.get("decision_kill") or {}
            for w_str, tgt in dec.items():
                try:
                    wpid = int(w_str)
                except (TypeError, ValueError):
                    continue
                if wpid in wolf_set and alive[wpid] and isinstance(tgt, int) and 1 <= tgt <= n:
                    mask = [alive[p] for p in range(1, n + 1)]
                    obs = build_obs(wpid, Phase.NIGHT)
                    add_action_sample(obs, DecisionType.NIGHT_WOLF_KILL, tgt - 1, mask)

        elif etype == "werewolf_kill" and isinstance(c, dict):
            tgt = c.get("target_player")
            if isinstance(tgt, int) and 1 <= tgt <= n:
                tonight_wolf_target = tgt

        elif etype == "healed" and isinstance(c, dict):
            if pending_heal is not None:
                flush_witch_night(pending_heal[0], pending_heal[1], None)
            pending_heal = (int(c.get("night", day)), c.get("player"))

        elif etype == "poison" and isinstance(c, dict):
            healed_p = pending_heal[1] if pending_heal is not None else None
            pending_heal = None
            flush_witch_night(int(c.get("night", day)), healed_p, c.get("player"))

        elif etype == "shoot" and isinstance(c, dict):
            if pending_heal is not None:
                flush_witch_night(pending_heal[0], pending_heal[1], None)
                pending_heal = None
            shooter = int(c["player"])
            tgt = c.get("shoot_player")
            mask = [True] + [alive[p] and p != shooter for p in range(1, n + 1)]
            obs = build_obs(shooter, Phase.HUNTER_RESOLVE)
            act = int(tgt) if isinstance(tgt, int) and 1 <= tgt <= n else 0
            add_action_sample(obs, DecisionType.HUNTER_SHOOT, act, mask)
            if isinstance(tgt, int) and 1 <= tgt <= n:
                revealed_roles[shooter] = Role.HUNTER
                alive[tgt] = False
                public_events.append(
                    PublicEvent(
                        event_id=len(public_events) + 1,
                        day=day,
                        phase=Phase.HUNTER_RESOLVE,
                        event_type="hunter_shoot",
                        actor=shooter,
                        payload={"shooter": shooter, "target": tgt},
                    )
                )

        elif etype == "speech" and isinstance(c, dict):
            spid = int(c["player"])
            day_str = str(c.get("day", ""))
            text = str(c.get("context", ""))
            q = summary_queue.get((day_str, spid))
            summ = q.pop(0) if q else None
            alive_set = {p for p in range(1, n + 1) if alive[p]}
            intent = map_speech_to_intent(spid, text, summ, alive_set, n)
            action_id = intent.to_action_id(n)
            mask = _speech_legal_mask(spid, alive, n)
            if alive[spid]:
                obs = build_obs(spid, Phase.DAY_SPEECH)
                add_action_sample(obs, DecisionType.DAY_SPEECH, action_id, mask)
            public_events.append(
                PublicEvent(
                    event_id=len(public_events) + 1,
                    day=day,
                    phase=Phase.DAY_SPEECH,
                    event_type="speech",
                    actor=spid,
                    payload={"intent": intent.to_dict(), "action_id": action_id, "text": text},
                )
            )

        elif etype == "voted" and isinstance(c, dict):
            vpid = int(c["player"])
            tgt = c.get("voted_to_player")
            if alive[vpid]:
                cands = [p for p in range(1, n + 1) if alive[p]]
                mask = [True] + [alive[p] for p in range(1, n + 1)]
                obs = build_obs(vpid, Phase.DAY_VOTE, vote_cands=cands)
                act = int(tgt) if isinstance(tgt, int) and 1 <= tgt <= n else 0
                add_belief_sample(obs)
                add_action_sample(obs, DecisionType.DAY_VOTE, act, mask)

        elif etype == "vote_results" and isinstance(c, dict):
            tally = Counter(int(v) for v in c.values() if isinstance(v, int) and 1 <= v <= n)
            exiled = None
            tied: List[int] = []
            if tally:
                mx = max(tally.values())
                top = sorted(p for p, cnt in tally.items() if cnt == mx)
                if len(top) == 1:
                    exiled = top[0]
                    alive[exiled] = False
                else:
                    tied = top
            public_events.append(
                PublicEvent(
                    event_id=len(public_events) + 1,
                    day=day,
                    phase=Phase.DAY_VOTE,
                    event_type="vote_result",
                    actor=None,
                    payload={"votes": {str(k): v for k, v in c.items()}, "exiled": exiled, "tied": tied},
                )
            )

        idx += 1

    if pending_heal is not None:
        flush_witch_night(pending_heal[0], pending_heal[1], None)

    return belief_samples, action_samples, dict(stats)


def load_game_splits(splits_jsonl: Path) -> Dict[str, str]:
    """Load Phase 1 leak-free game splits (train / validation / test)."""
    mapping: Dict[str, str] = {}
    with splits_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["game_id"]] = row["split"]
    return mapping


def save_split_npz(
    out_path: Path,
    belief_samples: List[BeliefSample],
    action_samples: List[ActionSample],
    player_count: int = 9,
) -> Dict[str, Any]:
    """Pack belief & per-decision-type action arrays into one compressed .npz file."""
    arrays: Dict[str, np.ndarray] = {}
    if belief_samples:
        arrays["belief_X"] = np.stack([b.seat_features for b in belief_samples], axis=0)
        arrays["belief_y"] = np.stack([b.true_roles for b in belief_samples], axis=0)
        arrays["belief_unknown_mask"] = np.stack([b.unknown_mask for b in belief_samples], axis=0)
        arrays["belief_observer_role"] = np.array([b.observer_role for b in belief_samples], dtype=np.int64)
        arrays["belief_day"] = np.array([b.day for b in belief_samples], dtype=np.int64)
    else:
        arrays["belief_X"] = np.zeros((0, player_count, BELIEF_SEAT_FEAT_DIM), dtype=np.float32)
        arrays["belief_y"] = np.zeros((0, player_count), dtype=np.int64)
        arrays["belief_unknown_mask"] = np.zeros((0, player_count), dtype=bool)
        arrays["belief_observer_role"] = np.zeros((0,), dtype=np.int64)
        arrays["belief_day"] = np.zeros((0,), dtype=np.int64)

    counts: Dict[str, int] = {"belief_samples": len(belief_samples)}
    sdim = policy_state_dim(player_count)
    for dt in DecisionType:
        sub = [a for a in action_samples if a.decision_type == dt]
        adim = action_space_size(dt, player_count)
        prefix = f"act_{dt.value}"
        if sub:
            arrays[f"{prefix}_X"] = np.stack([a.policy_state for a in sub], axis=0)
            arrays[f"{prefix}_y"] = np.array([a.action_id for a in sub], dtype=np.int64)
            arrays[f"{prefix}_mask"] = np.stack([a.legal_mask for a in sub], axis=0)
            arrays[f"{prefix}_role"] = np.array([a.role_idx for a in sub], dtype=np.int64)
            arrays[f"{prefix}_won"] = np.array([a.camp_won for a in sub], dtype=np.float32)
        else:
            arrays[f"{prefix}_X"] = np.zeros((0, sdim), dtype=np.float32)
            arrays[f"{prefix}_y"] = np.zeros((0,), dtype=np.int64)
            arrays[f"{prefix}_mask"] = np.zeros((0, adim), dtype=bool)
            arrays[f"{prefix}_role"] = np.zeros((0,), dtype=np.int64)
            arrays[f"{prefix}_won"] = np.zeros((0,), dtype=np.float32)
        counts[dt.value] = len(sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return counts
