"""Discrete action protocol, speech intents, and isolated player observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Role(str, Enum):
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    WEREWOLF = "werewolf"
    VILLAGER = "villager"

    @property
    def camp(self) -> "Camp":
        return Camp.WEREWOLF if self == Role.WEREWOLF else Camp.GOOD

    @property
    def is_god(self) -> bool:
        return self in (Role.SEER, Role.WITCH, Role.HUNTER)

    @property
    def zh_name(self) -> str:
        return {
            Role.SEER: "预言家",
            Role.WITCH: "女巫",
            Role.HUNTER: "猎人",
            Role.WEREWOLF: "狼人",
            Role.VILLAGER: "平民",
        }[self]


class Camp(str, Enum):
    GOOD = "good"
    WEREWOLF = "werewolf"
    DRAW = "draw"


class Phase(str, Enum):
    NIGHT = "night"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"
    HUNTER_RESOLVE = "hunter_resolve"
    GAME_OVER = "game_over"


class DecisionType(str, Enum):
    NIGHT_SEER_CHECK = "night_seer_check"
    NIGHT_WOLF_KILL = "night_wolf_kill"
    NIGHT_WITCH_ACT = "night_witch_act"
    HUNTER_SHOOT = "hunter_shoot"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"


class SpeechIntentType(str, Enum):
    """Macro tactical speech intents shared by Symbolic RL and Frozen LLM modes."""

    STAY_LOW = "stay_low"  # 表水/认民，不点特定目标
    CLAIM_VILLAGER_ACCUSE = "claim_villager_accuse"  # 认民并踩/归票某玩家
    CLAIM_VILLAGER_DEFEND = "claim_villager_defend"  # 认民并保/站边某玩家
    CLAIM_SEER_GOOD = "claim_seer_good"  # 跳/悍跳预言家，给目标发金水
    CLAIM_SEER_WOLF = "claim_seer_wolf"  # 跳/悍跳预言家，给目标发查杀
    CLAIM_WITCH_SAVE = "claim_witch_save"  # 跳/伪装女巫，报救了目标（银水）
    CLAIM_WITCH_POISON = "claim_witch_poison"  # 跳/伪装女巫，宣称毒/威胁毒目标
    CLAIM_HUNTER_PRESSURE = "claim_hunter_pressure"  # 跳/伪装猎人，点踩施压目标


TARGETED_SPEECH_INTENTS: Tuple[SpeechIntentType, ...] = (
    SpeechIntentType.CLAIM_VILLAGER_ACCUSE,
    SpeechIntentType.CLAIM_VILLAGER_DEFEND,
    SpeechIntentType.CLAIM_SEER_GOOD,
    SpeechIntentType.CLAIM_SEER_WOLF,
    SpeechIntentType.CLAIM_WITCH_SAVE,
    SpeechIntentType.CLAIM_WITCH_POISON,
    SpeechIntentType.CLAIM_HUNTER_PRESSURE,
)


@dataclass(frozen=True)
class SpeechIntent:
    intent_type: SpeechIntentType
    target: Optional[int] = None

    def to_action_id(self, player_count: int = 9) -> int:
        if self.intent_type == SpeechIntentType.STAY_LOW:
            return 0
        if self.target is None or not (1 <= self.target <= player_count):
            raise ValueError(f"Targeted intent {self.intent_type} requires target in 1..{player_count}")
        idx = TARGETED_SPEECH_INTENTS.index(self.intent_type)
        return 1 + idx * player_count + (self.target - 1)

    @classmethod
    def from_action_id(cls, action_id: int, player_count: int = 9) -> "SpeechIntent":
        if action_id == 0:
            return cls(SpeechIntentType.STAY_LOW, None)
        max_id = 1 + len(TARGETED_SPEECH_INTENTS) * player_count - 1
        if not (1 <= action_id <= max_id):
            raise ValueError(f"Speech action_id {action_id} out of range [0, {max_id}]")
        offset = action_id - 1
        intent_idx = offset // player_count
        target = (offset % player_count) + 1
        return cls(TARGETED_SPEECH_INTENTS[intent_idx], target)

    @classmethod
    def action_space_size(cls, player_count: int = 9) -> int:
        return 1 + len(TARGETED_SPEECH_INTENTS) * player_count

    def render_template_zh(self, speaker: int) -> str:
        t = self.target
        mapping = {
            SpeechIntentType.STAY_LOW: f"我是{speaker}号平民，目前信息有限，先听大家发言表水。",
            SpeechIntentType.CLAIM_VILLAGER_ACCUSE: f"我是平民，我认为{t}号发言和行为可疑，建议重点关注并投票出{t}号。",
            SpeechIntentType.CLAIM_VILLAGER_DEFEND: f"我是平民，我倾向于相信{t}号是好人，愿意站边保护{t}号。",
            SpeechIntentType.CLAIM_SEER_GOOD: f"我跳预言家，我查验了{t}号，{t}号是金水（好人）。",
            SpeechIntentType.CLAIM_SEER_WOLF: f"我跳预言家，我查验了{t}号，{t}号是查杀（狼人），今天请全票出{t}号。",
            SpeechIntentType.CLAIM_WITCH_SAVE: f"我跳女巫，我此前用解药救过{t}号，{t}号是银水。",
            SpeechIntentType.CLAIM_WITCH_POISON: f"我跳女巫，我怀疑{t}号是狼，准备对{t}号使用毒药或已对其动手。",
            SpeechIntentType.CLAIM_HUNTER_PRESSURE: f"我拍猎人身份，{t}号在我眼里狼面很大，如果我出局会考虑带走{t}号。",
        }
        return mapping[self.intent_type]

    def to_dict(self) -> Dict[str, Any]:
        return {"intent_type": self.intent_type.value, "target": self.target}


def action_space_size(decision_type: DecisionType, player_count: int = 9) -> int:
    """Return fixed discrete action dimension for RL policy heads."""
    if decision_type in (DecisionType.NIGHT_SEER_CHECK, DecisionType.NIGHT_WOLF_KILL):
        return player_count  # 0..N-1 -> player 1..N
    if decision_type == DecisionType.NIGHT_WITCH_ACT:
        return 2 + player_count  # 0=pass, 1=heal, 2..N+1=poison player 1..N
    if decision_type in (DecisionType.HUNTER_SHOOT, DecisionType.DAY_VOTE):
        return 1 + player_count  # 0=pass/abstain, 1..N=target player 1..N
    if decision_type == DecisionType.DAY_SPEECH:
        return SpeechIntent.action_space_size(player_count)
    raise ValueError(f"Unknown decision_type: {decision_type}")


@dataclass(frozen=True)
class PublicEvent:
    event_id: int
    day: int
    phase: Phase
    event_type: str  # night_announcement | speech | vote_result | hunter_shoot | game_end
    actor: Optional[int]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "day": self.day,
            "phase": self.phase.value,
            "event_type": self.event_type,
            "actor": self.actor,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class PlayerObservation:
    """Strict perspective-isolated observation passed to an agent."""

    game_id: str
    mode: str  # "symbolic" | "nl"
    day: int
    phase: Phase
    player_id: int
    my_role: Role
    player_count: int
    alive_players: Tuple[int, ...]
    dead_players: Tuple[int, ...]
    revealed_roles: Dict[int, Role]
    public_events: Tuple[PublicEvent, ...]
    # Strict role-private fields (empty/None for all other roles)
    werewolf_teammates: Tuple[int, ...] = ()
    seer_checks: Tuple[Tuple[int, int, bool], ...] = ()  # (night, target, is_werewolf)
    witch_has_antidote: Optional[bool] = None
    witch_has_poison: Optional[bool] = None
    witch_tonight_attacked: Optional[int] = None
    vote_candidates: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ActionRequest:
    decision_type: DecisionType
    observation: PlayerObservation
    legal_action_mask: Tuple[bool, ...]

    @property
    def legal_action_ids(self) -> List[int]:
        return [idx for idx, ok in enumerate(self.legal_action_mask) if ok]


@dataclass(frozen=True)
class ActionResponse:
    action_id: int
    utterance: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
