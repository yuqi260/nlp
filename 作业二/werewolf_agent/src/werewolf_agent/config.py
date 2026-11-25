"""Configuration loading and validation for Werewolf rulesets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

VALID_ROLES = ("seer", "witch", "hunter", "werewolf", "villager")
VALID_SELF_SAVE_POLICIES = ("never", "night1_only", "always")
VALID_END_RULES = ("side_kill", "kill_all")


@dataclass(frozen=True)
class GameConfig:
    ruleset_id: str
    language: str
    player_count: int
    role_counts: Dict[str, int]
    end_rule: str = "side_kill"
    has_sheriff: bool = False
    allow_werewolf_self_kill: bool = True
    witch_self_save_policy: str = "night1_only"
    witch_allow_both_same_night: bool = False
    hunter_can_shoot_when_poisoned: bool = False
    reveal_hunter_on_shoot: bool = True
    allow_abstain_vote: bool = True
    max_revotes_per_day: int = 1
    max_days: int = 10

    def validate(self) -> None:
        if self.player_count <= 0:
            raise ValueError(f"player_count must be positive, got {self.player_count}")
        unknown = set(self.role_counts) - set(VALID_ROLES)
        if unknown:
            raise ValueError(f"Unknown roles in config: {sorted(unknown)}")
        if sum(self.role_counts.values()) != self.player_count:
            raise ValueError(
                f"Sum of role_counts ({sum(self.role_counts.values())}) "
                f"!= player_count ({self.player_count})"
            )
        if self.role_counts.get("werewolf", 0) <= 0:
            raise ValueError("Config must have at least 1 werewolf")
        if self.end_rule not in VALID_END_RULES:
            raise ValueError(f"Unsupported end_rule: {self.end_rule}")
        if self.witch_self_save_policy not in VALID_SELF_SAVE_POLICIES:
            raise ValueError(
                f"Unsupported witch_self_save_policy: {self.witch_self_save_policy}"
            )
        if self.max_revotes_per_day < 0:
            raise ValueError("max_revotes_per_day must be >= 0")
        if self.max_days < 1:
            raise ValueError("max_days must be >= 1")


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "nine_player_standard.json"


def load_game_config(path: str | Path | None = None) -> GameConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = GameConfig(
        ruleset_id=raw["ruleset_id"],
        language=raw.get("language", "zh"),
        player_count=int(raw["player_count"]),
        role_counts={k: int(v) for k, v in raw["role_counts"].items()},
        end_rule=raw.get("end_rule", "side_kill"),
        has_sheriff=bool(raw.get("has_sheriff", False)),
        allow_werewolf_self_kill=bool(raw.get("allow_werewolf_self_kill", True)),
        witch_self_save_policy=raw.get("witch_self_save_policy", "night1_only"),
        witch_allow_both_same_night=bool(raw.get("witch_allow_both_same_night", False)),
        hunter_can_shoot_when_poisoned=bool(
            raw.get("hunter_can_shoot_when_poisoned", False)
        ),
        reveal_hunter_on_shoot=bool(raw.get("reveal_hunter_on_shoot", True)),
        allow_abstain_vote=bool(raw.get("allow_abstain_vote", True)),
        max_revotes_per_day=int(raw.get("max_revotes_per_day", 1)),
        max_days=int(raw.get("max_days", 10)),
    )
    cfg.validate()
    return cfg
