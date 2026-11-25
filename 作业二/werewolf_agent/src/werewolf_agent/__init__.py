"""Core package for the decoupled Werewolf role-playing agent."""

from .config import GameConfig, load_game_config
from .engine import GameResult, WerewolfEngine
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
)

__all__ = [
    "ActionRequest",
    "ActionResponse",
    "Camp",
    "DecisionType",
    "GameConfig",
    "GameResult",
    "Phase",
    "PlayerObservation",
    "PublicEvent",
    "Role",
    "SpeechIntent",
    "SpeechIntentType",
    "WerewolfEngine",
    "load_game_config",
]
