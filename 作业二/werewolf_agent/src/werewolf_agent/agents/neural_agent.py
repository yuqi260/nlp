"""Agents backed by trained Belief Tracker and/or MultiHeadPolicyValueMLP."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..features import ROLE_TO_IDX, default_prior_belief, encode_policy_state
from ..models.belief import BeliefMLP
from ..models.policy import MultiHeadPolicyValueMLP
from ..protocol import ActionRequest, ActionResponse, PlayerObservation, Role
from .base import BaseAgent
from .rule_agent import RuleBasedAgent


class BeliefRuleAgent(RuleBasedAgent):
    """Ablation B1: replaces naive rule suspicion with trained BeliefMLP wolf probabilities."""

    def __init__(self, belief_model: BeliefMLP, seed: int = 0, name: str = "belief_rule") -> None:
        super().__init__(seed=seed, name=name)
        self.belief_model = belief_model

    def _compute_suspicion(self, obs: PlayerObservation) -> dict[int, float]:
        probs = self.belief_model.predict_obs(obs, apply_constraints=True)
        wolf_idx = ROLE_TO_IDX[Role.WEREWOLF]
        # Scale learned wolf probability [0, 1] into suspicion range [-5, +10]
        base = super()._compute_suspicion(obs)
        scores = {}
        for pid in range(1, obs.player_count + 1):
            p_wolf = float(probs[pid - 1, wolf_idx])
            scores[pid] = 10.0 * (p_wolf - 0.33) + 0.35 * base[pid]
        scores[obs.player_id] = -10.0
        return scores


class NeuralPolicyAgent(BaseAgent):
    """Agent using BeliefMLP (optional) + MultiHeadPolicyValueMLP (BC or BC+RL)."""

    def __init__(
        self,
        policy_model: MultiHeadPolicyValueMLP,
        belief_model: Optional[BeliefMLP] = None,
        seed: int = 0,
        deterministic: bool = False,
        temperature: float = 1.0,
        name: str = "neural_policy",
    ) -> None:
        super().__init__(name=name)
        self.policy_model = policy_model
        self.belief_model = belief_model
        self._rng = np.random.default_rng(seed)
        self.deterministic = deterministic
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
            deterministic=self.deterministic,
            temperature=self.temperature,
        )
        return ActionResponse(
            action_id=action_id,
            metadata={"log_prob": log_prob, "value": value},
        )
