"""Uniform random legal baseline agent."""

from __future__ import annotations

import random

from ..protocol import ActionRequest, ActionResponse
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0, name: str = "random") -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def act(self, request: ActionRequest) -> ActionResponse:
        legal = request.legal_action_ids
        return ActionResponse(action_id=self._rng.choice(legal))
