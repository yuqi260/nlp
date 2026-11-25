"""Abstract base interface for all Werewolf agents (Random, Rule, Belief+RL, Frozen LLM)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..protocol import ActionRequest, ActionResponse, Role


class BaseAgent(ABC):
    def __init__(self, name: str = "agent") -> None:
        self.name = name

    def reset(self, game_id: str, player_id: int, role: Role) -> None:
        """Called once at the start of each game."""

    @abstractmethod
    def act(self, request: ActionRequest) -> ActionResponse:
        """Return a legal ActionResponse given a strictly isolated ActionRequest."""
