"""Heuristic rule-based Werewolf agent used as a strong symbolic opponent & baseline."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

from ..protocol import (
    ActionRequest,
    ActionResponse,
    DecisionType,
    PlayerObservation,
    Role,
    SpeechIntent,
    SpeechIntentType,
)
from .base import BaseAgent


class RuleBasedAgent(BaseAgent):
    """Reasonable heuristic agent with role-specific tactics and simple suspicion tracking."""

    def __init__(self, seed: int = 0, fake_seer_prob: float = 0.35, name: str = "rule") -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self.fake_seer_prob = fake_seer_prob
        self._wolf_claims_seer: bool = False
        self._witch_saved_target: Optional[int] = None

    def reset(self, game_id: str, player_id: int, role: Role) -> None:
        self._wolf_claims_seer = role == Role.WEREWOLF and (self._rng.random() < self.fake_seer_prob)
        self._witch_saved_target = None

    def act(self, request: ActionRequest) -> ActionResponse:
        dt = request.decision_type
        obs = request.observation
        legal = request.legal_action_ids

        if dt == DecisionType.NIGHT_SEER_CHECK:
            action_id = self._choose_seer_check(obs, legal)
        elif dt == DecisionType.NIGHT_WOLF_KILL:
            action_id = self._choose_wolf_kill(obs, legal)
        elif dt == DecisionType.NIGHT_WITCH_ACT:
            action_id = self._choose_witch_act(obs, legal)
        elif dt == DecisionType.HUNTER_SHOOT:
            action_id = self._choose_hunter_shoot(obs, legal)
        elif dt == DecisionType.DAY_SPEECH:
            action_id = self._choose_speech(obs, legal)
        elif dt == DecisionType.DAY_VOTE:
            action_id = self._choose_vote(obs, legal)
        else:
            action_id = self._rng.choice(legal)

        if action_id not in legal:
            action_id = self._rng.choice(legal)
        return ActionResponse(action_id=action_id)

    def _compute_suspicion(self, obs: PlayerObservation) -> Dict[int, float]:
        """Lightweight heuristic suspicion score over all players using only obs."""
        scores: Dict[int, float] = {p: 0.0 for p in range(1, obs.player_count + 1)}
        scores[obs.player_id] = -10.0

        # Revealed hunters are known good
        for pid, role in obs.revealed_roles.items():
            if role != Role.WEREWOLF:
                scores[pid] = -8.0

        # Seer private truth
        for _, target, is_wolf in obs.seer_checks:
            scores[target] = 10.0 if is_wolf else -9.0

        # Witch saved target tends to be good
        if obs.my_role == Role.WITCH and self._witch_saved_target is not None:
            scores[self._witch_saved_target] -= 3.5

        # Werewolf knows teammates
        if obs.my_role == Role.WEREWOLF:
            for tm in obs.werewolf_teammates:
                scores[tm] = -9.0

        # Parse public claims
        seer_claimers: List[int] = []
        seer_wolf_targets: Dict[int, List[int]] = {}
        seer_good_targets: Dict[int, List[int]] = {}
        for ev in obs.public_events:
            if ev.event_type != "speech" or ev.actor is None:
                continue
            intent_d = ev.payload.get("intent", {})
            itype = intent_d.get("intent_type")
            target = intent_d.get("target")
            actor = ev.actor
            if itype in (SpeechIntentType.CLAIM_SEER_GOOD.value, SpeechIntentType.CLAIM_SEER_WOLF.value):
                if actor not in seer_claimers:
                    seer_claimers.append(actor)
                if target is not None:
                    if itype == SpeechIntentType.CLAIM_SEER_WOLF.value:
                        seer_wolf_targets.setdefault(actor, []).append(target)
                    else:
                        seer_good_targets.setdefault(actor, []).append(target)
                    if obs.my_role == Role.SEER and actor != obs.player_id:
                        scores[actor] += 9.0
                    elif itype == SpeechIntentType.CLAIM_SEER_WOLF.value:
                        scores[target] += 2.5
                    else:
                        scores[target] -= 1.5
            elif itype == SpeechIntentType.CLAIM_VILLAGER_ACCUSE.value and target is not None:
                scores[target] += 0.7
                if target == obs.player_id and obs.my_role != Role.WEREWOLF:
                    scores[actor] += 1.0
            elif itype == SpeechIntentType.CLAIM_VILLAGER_DEFEND.value and target is not None:
                scores[target] -= 0.5
            elif itype == SpeechIntentType.CLAIM_WITCH_SAVE.value and target is not None:
                if obs.my_role == Role.WITCH and actor != obs.player_id:
                    scores[actor] += 9.0
                else:
                    scores[target] -= 1.5
                    scores[actor] -= 1.0
            elif itype == SpeechIntentType.CLAIM_WITCH_POISON.value and target is not None:
                if obs.my_role == Role.WITCH and actor != obs.player_id:
                    scores[actor] += 9.0
                else:
                    scores[target] += 1.5
            elif itype == SpeechIntentType.CLAIM_HUNTER_PRESSURE.value and target is not None:
                if obs.my_role == Role.HUNTER and actor != obs.player_id:
                    scores[actor] += 9.0
                else:
                    scores[target] += 1.0

        # Good-side logic when 1 vs >=2 Seer claimers exist
        if obs.my_role != Role.WEREWOLF:
            if len(seer_claimers) == 1:
                only_seer = seer_claimers[0]
                scores[only_seer] -= 4.0
                for wt in seer_wolf_targets.get(only_seer, []):
                    scores[wt] += 4.5
                for gt in seer_good_targets.get(only_seer, []):
                    scores[gt] -= 3.0
            elif len(seer_claimers) >= 2:
                # If a Seer claimer accuses a known-good player (me, revealed hunter, or witch saved),
                # mark that claimer strongly as wolf and trust the other claimer!
                known_good = {obs.player_id} | set(obs.revealed_roles.keys())
                if obs.my_role == Role.WITCH and self._witch_saved_target is not None:
                    known_good.add(self._witch_saved_target)
                discredited: Set[int] = set()
                for sc in seer_claimers:
                    if any(wt in known_good for wt in seer_wolf_targets.get(sc, [])):
                        discredited.add(sc)
                        scores[sc] += 8.0
                trusted_seers = [sc for sc in seer_claimers if sc not in discredited]
                if len(trusted_seers) == 1:
                    ts = trusted_seers[0]
                    scores[ts] -= 3.5
                    for wt in seer_wolf_targets.get(ts, []):
                        scores[wt] += 4.0
                    for gt in seer_good_targets.get(ts, []):
                        scores[gt] -= 2.5
                else:
                    # Focus vote/poison on competing Seer claimers (at least one is a wolf)
                    for sc in seer_claimers:
                        if sc != obs.player_id:
                            scores[sc] += 2.2
        return scores

    def _public_god_claimers(self, obs: PlayerObservation) -> Set[int]:
        gods: Set[int] = set()
        for ev in obs.public_events:
            if ev.event_type == "speech" and ev.actor is not None:
                itype = ev.payload.get("intent", {}).get("intent_type")
                if itype in (
                    SpeechIntentType.CLAIM_SEER_GOOD.value,
                    SpeechIntentType.CLAIM_SEER_WOLF.value,
                    SpeechIntentType.CLAIM_WITCH_SAVE.value,
                    SpeechIntentType.CLAIM_WITCH_POISON.value,
                    SpeechIntentType.CLAIM_HUNTER_PRESSURE.value,
                ):
                    gods.add(ev.actor)
        return gods

    def _choose_seer_check(self, obs: PlayerObservation, legal: List[int]) -> int:
        suspicion = self._compute_suspicion(obs)
        # Prefer checking alive unrevealed players with moderate/high suspicion
        scored = [(suspicion[a + 1] + self._rng.uniform(-0.3, 0.3), a) for a in legal]
        scored.sort(reverse=True)
        return scored[0][1]

    def _choose_wolf_kill(self, obs: PlayerObservation, legal: List[int]) -> int:
        wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
        non_wolf_legal = [a for a in legal if (a + 1) not in wolf_set]
        if not non_wolf_legal:
            return self._rng.choice(legal)
        god_claimers = self._public_god_claimers(obs)
        # Prefer killing non-wolf god claimers (especially Seer/Witch)
        priority = [a for a in non_wolf_legal if (a + 1) in god_claimers]
        if priority:
            return self._rng.choice(priority)
        return self._rng.choice(non_wolf_legal)

    def _choose_witch_act(self, obs: PlayerObservation, legal: List[int]) -> int:
        # Save on Day 1 if antidote available and someone attacked (classic heuristic)
        if 1 in legal and obs.day == 1 and obs.witch_tonight_attacked is not None:
            self._witch_saved_target = obs.witch_tonight_attacked
            return 1
        # From Day 2+, poison highly suspicious alive player if poison available
        if obs.day >= 2:
            suspicion = self._compute_suspicion(obs)
            poison_actions = [a for a in legal if a >= 2]
            if poison_actions:
                best = max(poison_actions, key=lambda a: suspicion[a - 1] + self._rng.uniform(-0.2, 0.2))
                if suspicion[best - 1] >= 2.0:
                    return best
        return 0 if 0 in legal else self._rng.choice(legal)

    def _choose_hunter_shoot(self, obs: PlayerObservation, legal: List[int]) -> int:
        suspicion = self._compute_suspicion(obs)
        shoot_actions = [a for a in legal if a >= 1]
        if not shoot_actions:
            return 0
        best = max(shoot_actions, key=lambda a: suspicion[a] + self._rng.uniform(-0.2, 0.2))
        if suspicion[best] >= 0.5 or len(obs.alive_players) <= 5:
            return best
        return 0 if 0 in legal else best

    def _choose_speech(self, obs: PlayerObservation, legal: List[int]) -> int:
        n = obs.player_count
        legal_set = set(legal)
        suspicion = self._compute_suspicion(obs)

        def pick_intent(itype: SpeechIntentType, target: Optional[int]) -> Optional[int]:
            aid = SpeechIntent(itype, target).to_action_id(n)
            return aid if aid in legal_set else None

        if obs.my_role == Role.SEER and obs.seer_checks:
            # Prefer reporting an alive wolf check; else report latest check
            alive_set = set(obs.alive_players)
            wolf_checks = [(d, t) for d, t, is_w in obs.seer_checks if is_w and t in alive_set]
            if wolf_checks:
                _, t = wolf_checks[-1]
                aid = pick_intent(SpeechIntentType.CLAIM_SEER_WOLF, t)
                if aid is not None:
                    return aid
            _, t, is_w = obs.seer_checks[-1]
            itype = SpeechIntentType.CLAIM_SEER_WOLF if is_w else SpeechIntentType.CLAIM_SEER_GOOD
            aid = pick_intent(itype, t)
            if aid is not None:
                return aid

        if obs.my_role == Role.WEREWOLF:
            wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
            alive_good = [p for p in obs.alive_players if p not in wolf_set]
            teammate_claimed_seer = any(
                ev.event_type == "speech"
                and ev.actor in obs.werewolf_teammates
                and ev.payload.get("intent", {}).get("intent_type")
                in (SpeechIntentType.CLAIM_SEER_GOOD.value, SpeechIntentType.CLAIM_SEER_WOLF.value)
                for ev in obs.public_events
            )
            if self._wolf_claims_seer and not teammate_claimed_seer and alive_good:
                # Fake Seer: accuse a good player or give gold water to a teammate/good
                if self._rng.random() < 0.7:
                    t = self._rng.choice(alive_good)
                    aid = pick_intent(SpeechIntentType.CLAIM_SEER_WOLF, t)
                else:
                    cand = list(obs.werewolf_teammates) or alive_good
                    t = self._rng.choice(cand)
                    aid = pick_intent(SpeechIntentType.CLAIM_SEER_GOOD, t)
                if aid is not None:
                    return aid
            if alive_good:
                t = self._rng.choice(alive_good)
                aid = pick_intent(SpeechIntentType.CLAIM_VILLAGER_ACCUSE, t)
                if aid is not None:
                    return aid
            return 0

        if obs.my_role == Role.WITCH and obs.day >= 2 and self._witch_saved_target is not None:
            # Claim Witch only if saved player is under suspicion or late game
            if suspicion[self._witch_saved_target] >= 1.5:
                aid = pick_intent(SpeechIntentType.CLAIM_WITCH_SAVE, self._witch_saved_target)
                if aid is not None:
                    return aid

        # Good players: accuse most suspicious alive player if suspicion is notable
        alive_others = [p for p in obs.alive_players if p != obs.player_id]
        if alive_others:
            top = max(alive_others, key=lambda p: suspicion[p] + self._rng.uniform(-0.2, 0.2))
            if obs.my_role == Role.HUNTER and obs.day >= 3 and suspicion[top] >= 3.0:
                aid = pick_intent(SpeechIntentType.CLAIM_HUNTER_PRESSURE, top)
                if aid is not None:
                    return aid
            if suspicion[top] >= 0.8:
                aid = pick_intent(SpeechIntentType.CLAIM_VILLAGER_ACCUSE, top)
                if aid is not None:
                    return aid
            most_trusted = min(alive_others, key=lambda p: suspicion[p] + self._rng.uniform(-0.2, 0.2))
            if suspicion[most_trusted] <= -1.0:
                aid = pick_intent(SpeechIntentType.CLAIM_VILLAGER_DEFEND, most_trusted)
                if aid is not None:
                    return aid
        return 0

    def _choose_vote(self, obs: PlayerObservation, legal: List[int]) -> int:
        vote_targets = [a for a in legal if a >= 1 and a != obs.player_id]
        if not vote_targets:
            return 0 if 0 in legal else self._rng.choice(legal)

        if obs.my_role == Role.WEREWOLF:
            wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
            non_wolf_targets = [t for t in vote_targets if t not in wolf_set]
            if non_wolf_targets:
                # Try to push a publicly accused good player or random good
                suspicion = self._compute_suspicion(obs)
                return max(non_wolf_targets, key=lambda t: suspicion[t] + self._rng.uniform(-0.5, 0.5))
            return self._rng.choice(vote_targets)

        suspicion = self._compute_suspicion(obs)
        return max(vote_targets, key=lambda t: suspicion[t] + self._rng.uniform(-0.25, 0.25))
