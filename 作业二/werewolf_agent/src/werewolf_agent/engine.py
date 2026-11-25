"""Dual-mode (Symbolic / NL) nine-player Werewolf engine with strict info isolation."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import GameConfig, load_game_config
from .protocol import (
    ActionRequest,
    ActionResponse,
    Camp,
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


@dataclass
class StepRecord:
    """Private trajectory record for offline training / analysis (never exposed mid-game)."""

    step_index: int
    day: int
    phase: str
    decision_type: str
    player_id: int
    role: str
    action_id: int
    legal_action_ids: List[int]
    resolved_detail: Dict[str, Any]


@dataclass
class GameResult:
    game_id: str
    mode: str
    ruleset_id: str
    seed: int
    winner: Camp
    win_reason: str
    total_days: int
    roles: Dict[int, Role]
    alive_players: List[int]
    death_causes: Dict[int, str]
    public_events: List[PublicEvent]
    trajectory: List[StepRecord]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "mode": self.mode,
            "ruleset_id": self.ruleset_id,
            "seed": self.seed,
            "winner": self.winner.value,
            "win_reason": self.win_reason,
            "total_days": self.total_days,
            "roles": {str(k): v.value for k, v in sorted(self.roles.items())},
            "alive_players": list(self.alive_players),
            "death_causes": {str(k): v for k, v in sorted(self.death_causes.items())},
            "num_public_events": len(self.public_events),
            "num_steps": len(self.trajectory),
        }


class WerewolfEngine:
    """State-machine engine supporting fast symbolic self-play and NL LLM evaluation."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        mode: str = "symbolic",
        speech_parser: Optional[Callable[[str, PlayerObservation, List[int]], int]] = None,
    ) -> None:
        self.config = config or load_game_config()
        self.config.validate()
        if mode not in ("symbolic", "nl"):
            raise ValueError(f"mode must be 'symbolic' or 'nl', got {mode}")
        self.mode = mode
        self.speech_parser = speech_parser

    def run_game(
        self,
        agents: Dict[int, Any],
        seed: int = 0,
        game_id: str = "game_0",
        fixed_roles: Optional[Dict[int, Role]] = None,
    ) -> GameResult:
        rng = random.Random(seed)
        n = self.config.player_count
        expected_ids = set(range(1, n + 1))
        if set(agents.keys()) != expected_ids:
            raise ValueError(f"agents must have keys 1..{n}, got {sorted(agents.keys())}")

        roles = self._assign_roles(rng, fixed_roles)
        alive: Dict[int, bool] = {pid: True for pid in range(1, n + 1)}
        death_causes: Dict[int, str] = {}
        revealed_roles: Dict[int, Role] = {}
        public_events: List[PublicEvent] = []
        trajectory: List[StepRecord] = []

        # Role-private memory managed strictly inside engine
        seer_checks: List[Tuple[int, int, bool]] = []
        witch_has_antidote = True
        witch_has_poison = True
        hunter_has_shot = False

        for pid, agent in agents.items():
            if hasattr(agent, "reset"):
                agent.reset(game_id=game_id, player_id=pid, role=roles[pid])

        def emit_public(day: int, phase: Phase, event_type: str, actor: Optional[int], payload: Dict[str, Any]) -> None:
            public_events.append(
                PublicEvent(
                    event_id=len(public_events) + 1,
                    day=day,
                    phase=phase,
                    event_type=event_type,
                    actor=actor,
                    payload=payload,
                )
            )

        def alive_list() -> List[int]:
            return [p for p in range(1, n + 1) if alive[p]]

        def dead_list() -> List[int]:
            return [p for p in range(1, n + 1) if not alive[p]]

        def check_winner() -> Optional[Tuple[Camp, str]]:
            alive_wolves = [p for p in alive_list() if roles[p] == Role.WEREWOLF]
            alive_villagers = [p for p in alive_list() if roles[p] == Role.VILLAGER]
            alive_gods = [p for p in alive_list() if roles[p].is_god]
            if not alive_wolves:
                return Camp.GOOD, "all_werewolves_eliminated"
            if self.config.end_rule == "side_kill":
                if not alive_villagers:
                    return Camp.WEREWOLF, "all_villagers_eliminated"
                if not alive_gods:
                    return Camp.WEREWOLF, "all_gods_eliminated"
            else:
                if not alive_villagers and not alive_gods:
                    return Camp.WEREWOLF, "all_good_eliminated"
            return None

        def build_obs(
            pid: int,
            day: int,
            phase: Phase,
            witch_tonight_attacked: Optional[int] = None,
            vote_candidates: Sequence[int] = (),
        ) -> PlayerObservation:
            role = roles[pid]
            teammates: Tuple[int, ...] = ()
            if role == Role.WEREWOLF:
                teammates = tuple(
                    p for p in range(1, n + 1) if roles[p] == Role.WEREWOLF and p != pid
                )
            checks_view = tuple(seer_checks) if role == Role.SEER else ()
            w_anti = witch_has_antidote if role == Role.WITCH else None
            w_pois = witch_has_poison if role == Role.WITCH else None
            w_atk = witch_tonight_attacked if (role == Role.WITCH and witch_has_antidote) else None
            return PlayerObservation(
                game_id=game_id,
                mode=self.mode,
                day=day,
                phase=phase,
                player_id=pid,
                my_role=role,
                player_count=n,
                alive_players=tuple(alive_list()),
                dead_players=tuple(dead_list()),
                revealed_roles=dict(revealed_roles),
                public_events=tuple(public_events),
                werewolf_teammates=teammates,
                seer_checks=checks_view,
                witch_has_antidote=w_anti,
                witch_has_poison=w_pois,
                witch_tonight_attacked=w_atk,
                vote_candidates=tuple(vote_candidates),
            )

        def query_agent(
            pid: int,
            day: int,
            phase: Phase,
            decision_type: DecisionType,
            mask: List[bool],
            witch_tonight_attacked: Optional[int] = None,
            vote_candidates: Sequence[int] = (),
        ) -> Tuple[ActionResponse, List[int]]:
            legal_ids = [idx for idx, ok in enumerate(mask) if ok]
            if not legal_ids:
                raise RuntimeError(f"No legal actions for player {pid} in {decision_type}")
            obs = build_obs(
                pid,
                day,
                phase,
                witch_tonight_attacked=witch_tonight_attacked,
                vote_candidates=vote_candidates,
            )
            req = ActionRequest(
                decision_type=decision_type,
                observation=obs,
                legal_action_mask=tuple(mask),
            )
            resp = agents[pid].act(req)
            if not isinstance(resp, ActionResponse):
                resp = ActionResponse(action_id=int(resp))
            action_id = resp.action_id
            if not (0 <= action_id < len(mask)) or not mask[action_id]:
                raise ValueError(
                    f"Illegal action {action_id} by player {pid} ({roles[pid].value}) "
                    f"for {decision_type.value}; legal={legal_ids}"
                )
            return resp, legal_ids

        def record_step(
            day: int,
            phase: Phase,
            decision_type: DecisionType,
            pid: int,
            action_id: int,
            legal_ids: List[int],
            resolved_detail: Dict[str, Any],
        ) -> None:
            trajectory.append(
                StepRecord(
                    step_index=len(trajectory),
                    day=day,
                    phase=phase.value,
                    decision_type=decision_type.value,
                    player_id=pid,
                    role=roles[pid].value,
                    action_id=action_id,
                    legal_action_ids=legal_ids,
                    resolved_detail=resolved_detail,
                )
            )

        def maybe_hunter_shoot(day: int, dead_pid: int, cause: str) -> Optional[Tuple[Camp, str]]:
            nonlocal hunter_has_shot
            if roles[dead_pid] != Role.HUNTER or hunter_has_shot:
                return None
            if cause == "witch_poison" and not self.config.hunter_can_shoot_when_poisoned:
                return None
            # Build legal shoot mask: 0=pass, 1..N=alive target != self
            mask = [True] + [alive[p] and p != dead_pid for p in range(1, n + 1)]
            resp, legal_ids = query_agent(
                dead_pid, day, Phase.HUNTER_RESOLVE, DecisionType.HUNTER_SHOOT, mask
            )
            target = resp.action_id if resp.action_id > 0 else None
            record_step(
                day,
                Phase.HUNTER_RESOLVE,
                DecisionType.HUNTER_SHOOT,
                dead_pid,
                resp.action_id,
                legal_ids,
                {"target": target, "trigger_cause": cause},
            )
            if target is not None:
                hunter_has_shot = True
                if self.config.reveal_hunter_on_shoot:
                    revealed_roles[dead_pid] = Role.HUNTER
                alive[target] = False
                death_causes[target] = "hunter_shoot"
                emit_public(
                    day,
                    Phase.HUNTER_RESOLVE,
                    "hunter_shoot",
                    dead_pid,
                    {"shooter": dead_pid, "target": target},
                )
                return check_winner()
            return None

        winner_info: Optional[Tuple[Camp, str]] = None
        last_day = 0

        for day in range(1, self.config.max_days + 1):
            last_day = day

            # -------------------- NIGHT PHASE --------------------
            # 1. Seer check
            seer_pids = [p for p in alive_list() if roles[p] == Role.SEER]
            if seer_pids:
                seer_pid = seer_pids[0]
                already_checked = {t for _, t, _ in seer_checks}
                mask = [alive[p] and p != seer_pid and p not in already_checked for p in range(1, n + 1)]
                if not any(mask):
                    mask = [alive[p] and p != seer_pid for p in range(1, n + 1)]
                if any(mask):
                    resp, legal_ids = query_agent(
                        seer_pid, day, Phase.NIGHT, DecisionType.NIGHT_SEER_CHECK, mask
                    )
                    target = resp.action_id + 1
                    is_wolf = roles[target] == Role.WEREWOLF
                    seer_checks.append((day, target, is_wolf))
                    record_step(
                        day,
                        Phase.NIGHT,
                        DecisionType.NIGHT_SEER_CHECK,
                        seer_pid,
                        resp.action_id,
                        legal_ids,
                        {"target": target, "is_werewolf": is_wolf},
                    )

            # 2. Werewolf kill vote
            alive_wolves = [p for p in alive_list() if roles[p] == Role.WEREWOLF]
            wolf_votes: Dict[int, int] = {}
            for wpid in alive_wolves:
                mask = [
                    alive[p] and (self.config.allow_werewolf_self_kill or roles[p] != Role.WEREWOLF)
                    for p in range(1, n + 1)
                ]
                if not any(mask):
                    mask = [alive[p] for p in range(1, n + 1)]
                resp, legal_ids = query_agent(
                    wpid, day, Phase.NIGHT, DecisionType.NIGHT_WOLF_KILL, mask
                )
                target = resp.action_id + 1
                wolf_votes[wpid] = target
                record_step(
                    day,
                    Phase.NIGHT,
                    DecisionType.NIGHT_WOLF_KILL,
                    wpid,
                    resp.action_id,
                    legal_ids,
                    {"proposed_target": target},
                )
            wolf_target: Optional[int] = None
            if wolf_votes:
                counts = Counter(wolf_votes.values())
                max_c = max(counts.values())
                top_targets = sorted(t for t, c in counts.items() if c == max_c)
                wolf_target = rng.choice(top_targets)

            # 3. Witch act (0=pass, 1=heal, 2..N+1=poison player 1..N)
            healed = False
            poison_target: Optional[int] = None
            witch_pids = [p for p in alive_list() if roles[p] == Role.WITCH]
            if witch_pids:
                witch_pid = witch_pids[0]
                can_heal = False
                if witch_has_antidote and wolf_target is not None:
                    if wolf_target != witch_pid:
                        can_heal = True
                    elif self.config.witch_self_save_policy == "always":
                        can_heal = True
                    elif self.config.witch_self_save_policy == "night1_only" and day == 1:
                        can_heal = True
                poison_mask = [
                    witch_has_poison and alive[p] and p != witch_pid for p in range(1, n + 1)
                ]
                mask = [True, can_heal] + poison_mask
                resp, legal_ids = query_agent(
                    witch_pid,
                    day,
                    Phase.NIGHT,
                    DecisionType.NIGHT_WITCH_ACT,
                    mask,
                    witch_tonight_attacked=wolf_target if witch_has_antidote else None,
                )
                act = resp.action_id
                detail: Dict[str, Any] = {"witch_action": "pass"}
                if act == 1:
                    healed = True
                    witch_has_antidote = False
                    detail = {"witch_action": "heal", "target": wolf_target}
                elif act >= 2:
                    poison_target = act - 1
                    witch_has_poison = False
                    detail = {"witch_action": "poison", "target": poison_target}
                record_step(
                    day,
                    Phase.NIGHT,
                    DecisionType.NIGHT_WITCH_ACT,
                    witch_pid,
                    act,
                    legal_ids,
                    detail,
                )

            # 4. Night death resolution
            night_deaths: List[Tuple[int, str]] = []
            if wolf_target is not None and not healed and alive[wolf_target]:
                night_deaths.append((wolf_target, "wolf_kill"))
            if poison_target is not None and alive[poison_target]:
                if all(p != poison_target for p, _ in night_deaths):
                    night_deaths.append((poison_target, "witch_poison"))
                else:
                    # Both wolf-killed and poisoned -> poisoned blocks hunter shoot
                    night_deaths = [
                        (p, "witch_poison" if p == poison_target else c) for p, c in night_deaths
                    ]

            for dpid, cause in night_deaths:
                alive[dpid] = False
                death_causes[dpid] = cause

            announced_dead = sorted(dpid for dpid, _ in night_deaths)
            announcement_text = (
                "今天是平安夜"
                if not announced_dead
                else "昨晚死亡玩家：" + "、".join(f"{p}号" for p in announced_dead)
            )
            emit_public(
                day,
                Phase.DAY_SPEECH,
                "night_announcement",
                None,
                {"dead_players": announced_dead, "text": announcement_text},
            )

            winner_info = check_winner()
            if winner_info is not None:
                break

            # Night hunter shoot check
            for dpid, cause in sorted(night_deaths, key=lambda x: x[0]):
                winner_info = maybe_hunter_shoot(day, dpid, cause)
                if winner_info is not None:
                    break
            if winner_info is not None:
                break

            # -------------------- DAY SPEECH PHASE --------------------
            alive_now = alive_list()
            start_seat = ((day - 1) % n) + 1
            speech_order = [p for p in alive_now if p >= start_seat] + [
                p for p in alive_now if p < start_seat
            ]
            for spid in speech_order:
                mask = self._build_speech_mask(spid, alive, n)
                resp, legal_ids = query_agent(
                    spid, day, Phase.DAY_SPEECH, DecisionType.DAY_SPEECH, mask
                )
                action_id = resp.action_id
                utterance = resp.utterance
                if self.mode == "nl" and utterance and self.speech_parser is not None:
                    obs = build_obs(spid, day, Phase.DAY_SPEECH)
                    parsed_id = self.speech_parser(utterance, obs, legal_ids)
                    if parsed_id in legal_ids:
                        action_id = parsed_id
                intent = SpeechIntent.from_action_id(action_id, n)
                if not utterance:
                    utterance = intent.render_template_zh(spid)
                record_step(
                    day,
                    Phase.DAY_SPEECH,
                    DecisionType.DAY_SPEECH,
                    spid,
                    action_id,
                    legal_ids,
                    {"intent": intent.to_dict(), "utterance": utterance},
                )
                emit_public(
                    day,
                    Phase.DAY_SPEECH,
                    "speech",
                    spid,
                    {"intent": intent.to_dict(), "action_id": action_id, "text": utterance},
                )

            # -------------------- DAY VOTE PHASE --------------------
            candidates = alive_list()
            exiled_pid: Optional[int] = None
            for vote_round in range(1 + self.config.max_revotes_per_day):
                votes: Dict[int, Optional[int]] = {}
                voters = alive_list()
                cand_set = set(candidates)
                for vpid in voters:
                    mask = [self.config.allow_abstain_vote] + [
                        (p in cand_set and alive[p]) for p in range(1, n + 1)
                    ]
                    if not any(mask):
                        mask[0] = True
                    resp, legal_ids = query_agent(
                        vpid,
                        day,
                        Phase.DAY_VOTE,
                        DecisionType.DAY_VOTE,
                        mask,
                        vote_candidates=candidates,
                    )
                    target = resp.action_id if resp.action_id > 0 else None
                    votes[vpid] = target
                    record_step(
                        day,
                        Phase.DAY_VOTE,
                        DecisionType.DAY_VOTE,
                        vpid,
                        resp.action_id,
                        legal_ids,
                        {"vote_round": vote_round + 1, "target": target},
                    )

                tally = Counter(t for t in votes.values() if t is not None)
                if not tally:
                    emit_public(
                        day,
                        Phase.DAY_VOTE,
                        "vote_result",
                        None,
                        {
                            "vote_round": vote_round + 1,
                            "votes": {str(k): v for k, v in votes.items()},
                            "exiled": None,
                            "tied": [],
                        },
                    )
                    break

                max_votes = max(tally.values())
                top_cands = sorted(p for p, c in tally.items() if c == max_votes)
                if len(top_cands) == 1:
                    exiled_pid = top_cands[0]
                    emit_public(
                        day,
                        Phase.DAY_VOTE,
                        "vote_result",
                        None,
                        {
                            "vote_round": vote_round + 1,
                            "votes": {str(k): v for k, v in votes.items()},
                            "exiled": exiled_pid,
                            "tied": [],
                        },
                    )
                    break

                # Tie -> revote if rounds remain
                emit_public(
                    day,
                    Phase.DAY_VOTE,
                    "vote_result",
                    None,
                    {
                        "vote_round": vote_round + 1,
                        "votes": {str(k): v for k, v in votes.items()},
                        "exiled": None,
                        "tied": top_cands,
                    },
                )
                candidates = top_cands

            if exiled_pid is not None:
                alive[exiled_pid] = False
                death_causes[exiled_pid] = "vote_exile"
                winner_info = check_winner()
                if winner_info is not None:
                    break
                winner_info = maybe_hunter_shoot(day, exiled_pid, "vote_exile")
                if winner_info is not None:
                    break

        if winner_info is None:
            winner_info = (Camp.DRAW, "max_days_reached")

        winner, reason = winner_info
        emit_public(
            last_day,
            Phase.GAME_OVER,
            "game_end",
            None,
            {"winner": winner.value, "reason": reason},
        )
        return GameResult(
            game_id=game_id,
            mode=self.mode,
            ruleset_id=self.config.ruleset_id,
            seed=seed,
            winner=winner,
            win_reason=reason,
            total_days=last_day,
            roles=roles,
            alive_players=alive_list(),
            death_causes=death_causes,
            public_events=public_events,
            trajectory=trajectory,
        )

    def _assign_roles(
        self, rng: random.Random, fixed_roles: Optional[Dict[int, Role]]
    ) -> Dict[int, Role]:
        n = self.config.player_count
        if fixed_roles is not None:
            if set(fixed_roles.keys()) != set(range(1, n + 1)):
                raise ValueError(f"fixed_roles must specify players 1..{n}")
            counts = Counter(r.value for r in fixed_roles.values())
            if dict(counts) != self.config.role_counts:
                raise ValueError(
                    f"fixed_roles counts {dict(counts)} != config {self.config.role_counts}"
                )
            return dict(fixed_roles)

        pool: List[Role] = []
        for role_name, count in sorted(self.config.role_counts.items()):
            pool.extend([Role(role_name)] * count)
        rng.shuffle(pool)
        return {pid: pool[pid - 1] for pid in range(1, n + 1)}

    @staticmethod
    def _build_speech_mask(speaker: int, alive: Dict[int, bool], n: int) -> List[bool]:
        """Legal speech intents: STAY_LOW always legal; targeted intents require target != speaker."""
        mask = [True]  # 0: STAY_LOW
        for intent_type in TARGETED_SPEECH_INTENTS:
            for target in range(1, n + 1):
                if target == speaker:
                    mask.append(False)
                elif intent_type in (
                    SpeechIntentType.CLAIM_SEER_GOOD,
                    SpeechIntentType.CLAIM_SEER_WOLF,
                    SpeechIntentType.CLAIM_WITCH_SAVE,
                    SpeechIntentType.CLAIM_WITCH_POISON,
                ):
                    # Can report historical checks/saves/poisons on any other seat (alive or dead)
                    mask.append(True)
                else:
                    # Accuse/defend/pressure generally targets alive players
                    mask.append(alive[target])
        assert len(mask) == action_space_size(DecisionType.DAY_SPEECH, n)
        return mask
