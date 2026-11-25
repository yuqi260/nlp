"""Frozen-LLM Werewolf agents: PureLLM (B0), BeliefLLM (B1), and Decoupled Belief+Policy+LLM (B2/Ours)."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import numpy as np

from ..fact_tracker import ROLE_PLAYBOOK_ZH, build_cross_round_notes, build_verified_fact_board
from ..features import default_prior_belief, encode_policy_state
from ..models.belief import BeliefMLP
from ..nl_parser import extract_json_object
from ..nl_speech import check_speech_guardrail, format_compact_context, generate_speech_for_intent
from ..protocol import (
    ActionRequest,
    ActionResponse,
    DecisionType,
    PlayerObservation,
    Role,
    SpeechIntent,
)
from .base import BaseAgent

DECISION_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的参赛玩家（3狼3民1预1女1猎，屠边，无警长）。
请严格根据你的真实身份、私有信息与公开局面，从给定的 legal_action_ids 中选择最优动作。
基本常识：女巫首夜通常用解药救人，次夜若发现高度可疑的狼人应果断用毒避免带毒死亡；猎人出局若场上有明确狼面目标尽量开枪带走。
只输出合法 JSON：{"action_id": 整数, "reason": "一句话简短理由"}。action_id 必须在 legal_action_ids 内。"""

PURE_SPEECH_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的参赛玩家（3狼3民1预1女1猎，屠边，无警长）。
请结合你的身份、私有信息和当前局面发表一段 40~110 字的白天公开发言。
硬性要求：
1. 直接输出发言正文，不要输出思考过程、旁白或 JSON。
2. 狼人绝对禁止承认自己是狼或泄露狼队友名单；不要编造游戏已结束等虚假公告。
3. 屠边局基本纪律：预言家可正常起跳报查验；女巫和猎人首日若无紧迫抗推压力或必须公布的关键信息，尽量先隐藏神职以好人口吻发言，避免首日三神全明牌被狼队精准屠神。
4. 发言中明确表达你对局势的判断或怀疑/支持对象。"""


def _describe_legal_actions(req: ActionRequest) -> str:
    dt = req.decision_type
    legal = req.legal_action_ids
    if dt == DecisionType.NIGHT_SEER_CHECK:
        return ", ".join(f"{a}:查验{a + 1}号" for a in legal)
    if dt == DecisionType.NIGHT_WOLF_KILL:
        return ", ".join(f"{a}:刀{a + 1}号" for a in legal)
    if dt == DecisionType.NIGHT_WITCH_ACT:
        parts = []
        for a in legal:
            if a == 0:
                parts.append("0:不用药")
            elif a == 1:
                parts.append(f"1:用解药救{req.observation.witch_tonight_attacked}号")
            else:
                parts.append(f"{a}:用毒药毒{a - 1}号")
        return ", ".join(parts)
    if dt == DecisionType.HUNTER_SHOOT:
        return ", ".join(("0:不开枪" if a == 0 else f"{a}:开枪带走{a}号") for a in legal)
    if dt == DecisionType.DAY_VOTE:
        return ", ".join(("0:弃票" if a == 0 else f"{a}:投给{a}号") for a in legal)
    return str(legal)


class PureLLMAgent(BaseAgent):
    """B0 baseline: frozen LLM decides night actions, speeches, votes, and hunter shoots directly."""

    def __init__(
        self,
        llm_client: Any,
        seed: int = 0,
        name: str = "pure_llm",
    ) -> None:
        super().__init__(name=name)
        self.llm_client = llm_client
        self._rng = random.Random(seed)
        self.guardrail_stats: Dict[str, int] = {"passed": 0, "retried": 0, "fallback": 0}

    def _get_belief_for_prompt(self, obs: PlayerObservation) -> Optional[np.ndarray]:
        return None

    def act(self, request: ActionRequest) -> ActionResponse:
        obs = request.observation
        legal = request.legal_action_ids
        belief = self._get_belief_for_prompt(obs)
        context = format_compact_context(obs, belief_probs=belief)

        if request.decision_type == DecisionType.DAY_SPEECH:
            messages = [
                {"role": "system", "content": PURE_SPEECH_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n轮到你发言，请直接输出公开发言："},
            ]
            for attempt in range(2):
                try:
                    resp = self.llm_client.chat(messages, max_tokens=180, temperature=0.6)
                    text = resp.text.strip().strip('"').strip("“”")
                    rep = check_speech_guardrail(text, obs, expected_intent=None)
                    if rep.passed:
                        self.guardrail_stats["passed"] += 1
                        if attempt > 0:
                            self.guardrail_stats["retried"] += 1
                        return ActionResponse(action_id=0 if 0 in legal else legal[0], utterance=text)
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": f"发言违规（{rep.violations}），请重写 40~100 字合法发言："},
                    ]
                except Exception:
                    break
            self.guardrail_stats["fallback"] += 1
            fallback_aid = 0 if 0 in legal else self._rng.choice(legal)
            fallback_txt = SpeechIntent.from_action_id(fallback_aid, obs.player_count).render_template_zh(obs.player_id)
            return ActionResponse(action_id=fallback_aid, utterance=fallback_txt)

        # Non-speech discrete decision via compact JSON
        options_str = _describe_legal_actions(request)
        user_msg = (
            f"{context}\n\n"
            f"当前决策：{request.decision_type.value}\n"
            f"legal_action_ids: {legal}\n"
            f"动作含义：{options_str}\n"
            f"请只输出 JSON：{{\"action_id\": 整数, \"reason\": \"一句话理由\"}}"
        )
        messages = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            resp = self.llm_client.chat(messages, max_tokens=90, temperature=0.3)
            obj = extract_json_object(resp.text)
            aid = int(obj["action_id"])
            if aid in legal:
                return ActionResponse(action_id=aid, metadata={"reason": str(obj.get("reason", ""))})
        except Exception:
            pass
        return ActionResponse(action_id=self._rng.choice(legal), metadata={"fallback": True})


class BeliefLLMAgent(PureLLMAgent):
    """B1 ablation: frozen LLM augmented with trained BeliefMLP probabilities in context."""

    def __init__(
        self,
        llm_client: Any,
        belief_model: BeliefMLP,
        seed: int = 0,
        name: str = "belief_llm",
    ) -> None:
        super().__init__(llm_client=llm_client, seed=seed, name=name)
        self.belief_model = belief_model

    def _get_belief_for_prompt(self, obs: PlayerObservation) -> Optional[np.ndarray]:
        return self.belief_model.predict_obs(obs, apply_constraints=True)


class DecoupledLLMAgent(BaseAgent):
    """B2 (Belief+BC+LLM) and Ours (Belief+BC+RL+LLM): local neural brain + frozen LLM speaker."""

    def __init__(
        self,
        policy_model: Any,
        belief_model: Optional[BeliefMLP],
        llm_client: Any,
        seed: int = 0,
        deterministic: bool = False,
        temperature: float = 0.7,
        name: str = "decoupled_rl_llm",
    ) -> None:
        super().__init__(name=name)
        self.policy_model = policy_model
        self.belief_model = belief_model
        self.llm_client = llm_client
        self._rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        self.temperature = temperature
        self.guardrail_stats: Dict[str, int] = {"passed": 0, "retried": 0, "fallback": 0}
        self.last_trace: Dict[str, Any] = {}

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

        # Save decision trace for M6 visualization
        self.last_trace = {
            "day": obs.day,
            "decision_type": request.decision_type.value,
            "action_id": action_id,
            "value": round(value, 4),
            "belief_wolf_probs": [round(float(belief[i, 3]), 4) for i in range(obs.player_count)],
        }

        if request.decision_type == DecisionType.DAY_SPEECH and obs.mode == "nl":
            intent = SpeechIntent.from_action_id(action_id, obs.player_count)
            utterance, report, attempts = generate_speech_for_intent(
                llm_client=self.llm_client,
                obs=obs,
                intent=intent,
                belief_probs=belief,
                max_retries=1,
            )
            if report.passed:
                self.guardrail_stats["passed"] += 1
                if attempts > 1:
                    self.guardrail_stats["retried"] += 1
            else:
                self.guardrail_stats["fallback"] += 1
            return ActionResponse(
                action_id=action_id,
                utterance=utterance,
                metadata={
                    "log_prob": log_prob,
                    "value": value,
                    "intent": intent.to_dict(),
                    "guardrail_passed": report.passed,
                    "guardrail_violations": report.violations,
                    "attempts": attempts,
                },
            )

        return ActionResponse(action_id=action_id, metadata={"log_prob": log_prob, "value": value})


CANDIDATE_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的策略助手（3狼3民1预1女1猎，屠边，无警长）。
请结合你的真实身份、私有信息与完整公开对话，从 legal_action_ids 中选出最合理的 1~2 个候选动作（按优先级排序）。
策略提醒：
- 若你是狼人：夜晚优先精准刀死已暴露或疑似神职（预言家/女巫/猎人）；白天若队友已被查杀且必出，可适度倒钩隐藏自己；绝不暴露狼队。
- 若你是好人：优先归票查杀狼或行为矛盾最明显的狼人；女巫首夜尽量救人，次夜毒确定度高的狼人；猎人尽量带走最像狼的玩家。
只输出合法 JSON：{"candidates": [最优action_id, 次优action_id], "reason": "一句话理由"}。所有元素必须在 legal_action_ids 内。"""

DUAL_SPEECH_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的高水平玩家（3狼3民1预1女1猎，屠边，无警长）。
你需要结合完整局势与【系统二战略建议】发表一段 45~115 字、逻辑严密且符合角色利益的白天公开发言。
关键角色纪律：
- 预言家：第1天起应果断起跳报清查验结果（金水/查杀）并带队归票。
- 女巫/猎人：在未受到致命抗推压力或无关键救/毒信息需公布前，尽量先以好人/平民口吻发言隐藏神职，避免过早暴露遭狼队精准屠神！
- 狼人：严禁承认自己是狼或泄露队友；严禁在无必要时乱跳女巫/猎人穿帮；优先以平民表水/逻辑分析伪装好人，或单狼有序悍跳预言家。
硬性要求：直接输出发言正文，不要输出思考过程、旁白或 JSON。"""


class DualSystemLLMAgent(BaseAgent):
    """True System-1 (Frozen LLM) + System-2 (Trained BeliefMLP + Pointer RL Reranker) Agent.

    Implements the Xu et al. (ICML 2024) + Thinker dual-system paradigm:
    - System-1 LLM proposes top-2 text-grounded candidate actions from rich Chinese dialogue.
    - System-2 Trained BeliefMLP + SeatEquivariant Pointer RL scores & selects the optimal
      candidate using learned policy log-probabilities (eliminating LLM strategic biases
      without losing fine-grained natural language comprehension).
    - For DAY_SPEECH, System-2 provides strategic intent & belief guidance while System-1
      generates context-aware, role-disciplined Chinese speech under Guardrail protection.
    """

    def __init__(
        self,
        policy_model: Any,
        belief_model: Optional[BeliefMLP],
        llm_client: Any,
        rl_weight: float = 0.55,
        seed: int = 0,
        name: str = "dual_system_rl_llm",
    ) -> None:
        super().__init__(name=name)
        self.policy_model = policy_model
        self.belief_model = belief_model
        self.llm_client = llm_client
        self.rl_weight = rl_weight
        self._rng = np.random.default_rng(seed)
        self.guardrail_stats: Dict[str, int] = {"passed": 0, "retried": 0, "fallback": 0}
        self.last_trace: Dict[str, Any] = {}

    def _get_rl_distribution(
        self, obs: PlayerObservation, dt: DecisionType, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        belief = (
            self.belief_model.predict_obs(obs, apply_constraints=True)
            if self.belief_model is not None
            else default_prior_belief(obs)
        )
        state = encode_policy_state(obs, belief)
        probs, log_probs, values, _ = self.policy_model.forward_head(state[None, :], dt, mask[None, :])
        return belief, probs[0], log_probs[0], float(values[0])

    def act(self, request: ActionRequest) -> ActionResponse:
        obs = request.observation
        legal = request.legal_action_ids
        mask = np.asarray(request.legal_action_mask, dtype=bool)
        belief, rl_probs, rl_log_probs, rl_val = self._get_rl_distribution(obs, request.decision_type, mask)

        # Top-3 RL choices for trace & System-2 advice
        legal_sorted = sorted(legal, key=lambda a: float(rl_probs[a]), reverse=True)
        top_rl = legal_sorted[0]

        self.last_trace = {
            "day": obs.day,
            "decision_type": request.decision_type.value,
            "rl_top_action": top_rl,
            "rl_top_prob": round(float(rl_probs[top_rl]), 4),
            "value": round(rl_val, 4),
            "belief_wolf_probs": [round(float(belief[i, 3]), 4) for i in range(obs.player_count)],
        }

        if request.decision_type == DecisionType.DAY_SPEECH:
            if obs.mode != "nl":
                return ActionResponse(action_id=top_rl)
            rl_intent = SpeechIntent.from_action_id(top_rl, obs.player_count)
            context = format_compact_context(obs, belief_probs=belief)
            # Provide System-2 strategic direction as structured advice (not brittle forced claim)
            advice = (
                f"系统二 RL 推荐宏观策略：{rl_intent.intent_type.value}"
                + (f"（目标 {rl_intent.target}号）" if rl_intent.target else "")
                + f"，置信度 {float(rl_probs[top_rl]):.0%}。"
            )
            if obs.my_role in (Role.WITCH, Role.HUNTER) and obs.day == 1:
                advice += "【神职隐蔽提醒】第1天若无紧急救人信息或抗推危机，建议先隐藏神职以好人口吻发言防屠神。"
            elif obs.my_role == Role.WEREWOLF:
                advice += "【狼人纪律】保持自然好人视角发言或单狼有序悍跳，切勿无故乱跳女巫/猎人穿帮。"

            messages = [
                {"role": "system", "content": DUAL_SPEECH_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n【系统二战略建议】\n{advice}\n请直接输出你的白天公开发言："},
            ]
            for attempt in range(2):
                try:
                    resp = self.llm_client.chat(messages, max_tokens=180, temperature=0.55)
                    text = resp.text.strip().strip('"').strip("“”")
                    rep = check_speech_guardrail(text, obs, expected_intent=None)
                    if rep.passed:
                        self.guardrail_stats["passed"] += 1
                        if attempt > 0:
                            self.guardrail_stats["retried"] += 1
                        return ActionResponse(action_id=top_rl, utterance=text)
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": f"发言违规（{rep.violations}），请重写合法发言："},
                    ]
                except Exception:
                    break
            self.guardrail_stats["fallback"] += 1
            return ActionResponse(action_id=top_rl, utterance=rl_intent.render_template_zh(obs.player_id))

        # Non-speech discrete decision: System-1 LLM proposes Top-2 candidates -> System-2 RL reranks!
        if len(legal) == 1:
            return ActionResponse(action_id=legal[0])

        context = format_compact_context(obs, belief_probs=belief)
        options_str = _describe_legal_actions(request)
        rl_hints = ", ".join(f"{a}({float(rl_probs[a]):.0%})" for a in legal_sorted[:3])
        user_msg = (
            f"{context}\n\n"
            f"当前决策：{request.decision_type.value}\n"
            f"legal_action_ids: {legal}\n"
            f"动作含义：{options_str}\n"
            f"【系统二 RL 候选参考】Top-3 动作及概率：{rl_hints}\n"
            f"请输出最合理的 1~2 个候选动作 JSON：{{\"candidates\": [最优id, 次优id], \"reason\": \"一句话理由\"}}"
        )
        messages = [
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        cands: List[int] = []
        reason = ""
        try:
            resp = self.llm_client.chat(messages, max_tokens=96, temperature=0.25)
            obj = extract_json_object(resp.text)
            raw_cands = obj.get("candidates")
            if isinstance(raw_cands, list):
                for item in raw_cands:
                    if isinstance(item, int) and item in legal and item not in cands:
                        cands.append(item)
            elif isinstance(obj.get("action_id"), int) and int(obj["action_id"]) in legal:
                cands.append(int(obj["action_id"]))
            reason = str(obj.get("reason", ""))
        except Exception:
            pass

        if not cands:
            return ActionResponse(action_id=top_rl, metadata={"source": "rl_fallback"})

        if len(cands) == 1:
            chosen = cands[0]
        else:
            # System-2 RL reranking over LLM's Top-2 candidates (Xu et al. ICML 2024 fusion)
            # Rank prior: cand[0] gets +0.55 log-bonus over cand[1]; RL adds rl_weight * log P_RL(a)
            best_a = cands[0]
            best_score = -1e9
            for rank_idx, a in enumerate(cands[:2]):
                llm_rank_score = 0.0 if rank_idx == 0 else -0.55
                score = (1.0 - self.rl_weight) * llm_rank_score + self.rl_weight * float(rl_log_probs[a])
                if score > best_score:
                    best_score = score
                    best_a = a
            chosen = best_a

        self.last_trace["llm_candidates"] = cands[:2]
        self.last_trace["chosen_action"] = chosen
        return ActionResponse(action_id=chosen, metadata={"llm_candidates": cands[:2], "reason": reason})



class DualSystemV2Agent(DualSystemLLMAgent):
    """Upgraded Dual-System Agent: Adapted BeliefMLP_NL + System-2 Fact Verifier + Pointer RL Reranker."""

    def __init__(
        self,
        policy_model: Any,
        belief_model: Optional[BeliefMLP],
        llm_client: Any,
        rl_weight: float = 0.50,
        seed: int = 0,
        name: str = "Ours_DualSystem_v2",
    ) -> None:
        super().__init__(
            policy_model=policy_model,
            belief_model=belief_model,
            llm_client=llm_client,
            rl_weight=rl_weight,
            seed=seed,
            name=name,
        )

    def _format_calibrated_belief_hint(self, obs: PlayerObservation, belief: np.ndarray) -> str:
        """Only surface top-2 suspected wolves & high-confidence god cues to avoid noisy prompt clutter."""
        others = [p for p in obs.alive_players if p != obs.player_id]
        if not others:
            return ""
        ranked_wolves = sorted(others, key=lambda p: float(belief[p - 1, 3]), reverse=True)[:3]
        parts = [f"{p}号(狼概率{float(belief[p - 1, 3]):.0%})" for p in ranked_wolves]
        return "【系统二校准信念·最高狼面】" + "、".join(parts)

    def act(self, request: ActionRequest) -> ActionResponse:
        obs = request.observation
        legal = request.legal_action_ids
        mask = np.asarray(request.legal_action_mask, dtype=bool)
        belief, rl_probs, rl_log_probs, rl_val = self._get_rl_distribution(obs, request.decision_type, mask)
        fact_board = build_verified_fact_board(obs)

        # Boost RL log-probs with mathematically proven System-2 logic facts (zero hallucination!)
        adj_log_probs = rl_log_probs.copy()
        if obs.my_role != Role.WEREWOLF:
            if request.decision_type in (DecisionType.DAY_VOTE, DecisionType.HUNTER_SHOOT):
                for pw in fact_board.proven_wolves:
                    if pw in legal:
                        adj_log_probs[pw] += 2.5
                for pg in fact_board.proven_good:
                    if pg in legal:
                        adj_log_probs[pg] -= 2.5
            elif request.decision_type == DecisionType.NIGHT_WITCH_ACT:
                for pw in fact_board.proven_wolves:
                    a = 1 + pw
                    if a in legal:
                        adj_log_probs[a] += 2.8
                for pg in fact_board.proven_good:
                    a = 1 + pg
                    if a in legal:
                        adj_log_probs[a] -= 3.0
        else:
            # Werewolf: strongly boost killing publicly claimed gods at night
            if request.decision_type == DecisionType.NIGHT_WOLF_KILL:
                claimed_gods = set(fact_board.witch_claimers) | set(fact_board.hunter_claimers) | {
                    c.claimer for c in fact_board.seer_claims
                }
                wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
                for gpid in claimed_gods - wolf_set:
                    a = gpid - 1
                    if a in legal:
                        adj_log_probs[a] += 2.0

        legal_sorted = sorted(legal, key=lambda a: float(adj_log_probs[a]), reverse=True)
        top_rl = legal_sorted[0]

        self.last_trace = {
            "day": obs.day,
            "decision_type": request.decision_type.value,
            "rl_top_action": top_rl,
            "value": round(rl_val, 4),
            "proven_wolves": sorted(fact_board.proven_wolves),
            "belief_wolf_probs": [round(float(belief[i, 3]), 4) for i in range(obs.player_count)],
        }

        fact_text = fact_board.format_zh(obs)
        bel_hint = self._format_calibrated_belief_hint(obs, belief)
        base_context = format_compact_context(obs, belief_probs=None, max_recent_speeches=8)
        enriched_context = "\n".join(x for x in (base_context, fact_text, bel_hint) if x)

        if request.decision_type == DecisionType.DAY_SPEECH:
            if obs.mode != "nl":
                return ActionResponse(action_id=top_rl)
            rl_intent = SpeechIntent.from_action_id(top_rl, obs.player_count)
            advice = (
                f"系统二 RL 建议宏观方向：{rl_intent.intent_type.value}"
                + (f"（目标 {rl_intent.target}号）" if rl_intent.target else "")
                + "。请严格依据【系统二核验】中的准确查验记录发言，切勿把金水误读成查杀！"
            )
            if obs.my_role in (Role.WITCH, Role.HUNTER) and obs.day == 1:
                advice += "【防屠神纪律】第1天若无紧迫抗推压力，切勿自爆女巫/猎人身份，请以逻辑清晰的平民口吻发言。"
            elif obs.my_role == Role.SEER and obs.seer_checks:
                advice += "【预言家纪律】清晰报出你的每一夜真实查验结果（金水/查杀）并坚决归票查杀狼。"
            elif obs.my_role == Role.WEREWOLF:
                advice += "【狼人纪律】自然伪装好人找逻辑漏洞或单狼有序悍跳，队友铁暴露时可果断倒钩隐藏自己，绝不自爆。"

            messages = [
                {"role": "system", "content": DUAL_SPEECH_SYSTEM_PROMPT},
                {"role": "user", "content": f"{enriched_context}\n\n【系统二战略建议】\n{advice}\n请直接输出你的白天公开发言："},
            ]
            for attempt in range(2):
                try:
                    resp = self.llm_client.chat(messages, max_tokens=180, temperature=0.5)
                    text = resp.text.strip().strip('"').strip("“”")
                    rep = check_speech_guardrail(text, obs, expected_intent=None)
                    if rep.passed:
                        self.guardrail_stats["passed"] += 1
                        if attempt > 0:
                            self.guardrail_stats["retried"] += 1
                        return ActionResponse(action_id=top_rl, utterance=text)
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": f"发言违规（{rep.violations}），请重写合法发言："},
                    ]
                except Exception:
                    break
            self.guardrail_stats["fallback"] += 1
            return ActionResponse(action_id=top_rl, utterance=rl_intent.render_template_zh(obs.player_id))

        # Non-speech discrete decision
        if len(legal) == 1:
            return ActionResponse(action_id=legal[0])

        # Hard System-2 guard: if Good player has a 100% logically proven alive wolf on Vote/Shoot/Poison,
        # prioritize among proven wolves cleanly without letting LLM hallucinate a wrong target!
        if obs.my_role != Role.WEREWOLF and fact_board.proven_wolves:
            if request.decision_type in (DecisionType.DAY_VOTE, DecisionType.HUNTER_SHOOT):
                proven_legal = [pw for pw in sorted(fact_board.proven_wolves) if pw in legal]
                if proven_legal:
                    best_pw = max(proven_legal, key=lambda a: float(adj_log_probs[a]))
                    return ActionResponse(action_id=best_pw, metadata={"source": "system2_proven_wolf"})
            elif request.decision_type == DecisionType.NIGHT_WITCH_ACT and obs.day >= 2:
                proven_poison = [1 + pw for pw in sorted(fact_board.proven_wolves) if (1 + pw) in legal]
                if proven_poison:
                    best_p = max(proven_poison, key=lambda a: float(adj_log_probs[a]))
                    return ActionResponse(action_id=best_p, metadata={"source": "system2_proven_poison"})

        options_str = _describe_legal_actions(request)
        rl_hints = ", ".join(f"{a}({float(rl_probs[a]):.0%})" for a in legal_sorted[:3])
        user_msg = (
            f"{enriched_context}\n\n"
            f"当前决策：{request.decision_type.value}\n"
            f"legal_action_ids: {legal}\n"
            f"动作含义：{options_str}\n"
            f"【系统二 RL 候选参考】Top-3 动作：{rl_hints}\n"
            f"请输出最合理的 1~2 个候选动作 JSON：{{\"candidates\": [最优id, 次优id], \"reason\": \"一句话理由\"}}"
        )
        messages = [
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        cands: List[int] = []
        reason = ""
        try:
            resp = self.llm_client.chat(messages, max_tokens=96, temperature=0.2)
            obj = extract_json_object(resp.text)
            raw_cands = obj.get("candidates")
            if isinstance(raw_cands, list):
                for item in raw_cands:
                    if isinstance(item, int) and item in legal and item not in cands:
                        cands.append(item)
            elif isinstance(obj.get("action_id"), int) and int(obj["action_id"]) in legal:
                cands.append(int(obj["action_id"]))
            reason = str(obj.get("reason", ""))
        except Exception:
            pass

        if not cands:
            return ActionResponse(action_id=top_rl, metadata={"source": "rl_fallback"})

        # For Werewolf, relax RL vote weight so LLM freely uses natural bussing (倒钩) & god-sniping
        eff_rl_weight = 0.15 if obs.my_role == Role.WEREWOLF else self.rl_weight
        if len(cands) == 1:
            chosen = cands[0]
        else:
            best_a = cands[0]
            best_score = -1e9
            for rank_idx, a in enumerate(cands[:2]):
                llm_rank_score = 0.0 if rank_idx == 0 else -0.50
                score = (1.0 - eff_rl_weight) * llm_rank_score + eff_rl_weight * float(adj_log_probs[a])
                if score > best_score:
                    best_score = score
                    best_a = a
            chosen = best_a

        self.last_trace["llm_candidates"] = cands[:2]
        self.last_trace["chosen_action"] = chosen
        return ActionResponse(action_id=chosen, metadata={"llm_candidates": cands[:2], "reason": reason})


MAKTO_SPEECH_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的顶尖高手（3狼3民1预1女1猎，屠边，无警长）。
模仿人类顶尖玩家的“先内部规划、再公开发言”流程，只输出合法 JSON：
{"plan": "20~50字严格私有的战术盘算（绝不公开）", "display_role": "对外宣称的身份", "vote_target": 归票目标座位整数或null, "speech": "50~120字极具逻辑说服力的中文公开发言正文"}

硬性纪律：
1. speech 中绝对禁止泄露私有规划、绝不能承认自己是狼或泄露狼队友名单、绝不能编造游戏已结束。
2. speech 必须与【系统二核验表】的客观事实严格一致，绝不把金水误读成查杀。
3. 只输出合法 JSON，不要多余解释。"""

MAKTO_DECISION_SYSTEM_PROMPT = """你是中文九人预女猎狼人杀的顶尖高手（3狼3民1预1女1猎，屠边，无警长）。
请先在 thought 字段中做简短的“此时我应该做的”战术权衡（结合角色利益、跨轮次笔记、票型、言行不一与系统二核验），再从 legal_action_ids 中给出最合理的 1~2 个候选动作（按优先级排序）。
关键原则：
- 狼人：夜晚优先精准刀死已暴露或高概率神职（预言家/女巫/猎人）；白天若队友已被查杀且必出局，可果断倒钩投出队友隐藏自己，否则优先推抗推位好人。
- 好人：优先归票/毒杀/开枪带走铁狼、假预言家或言行不一的冲票狼；女巫首夜尽量开解药救银水，次夜发现明确狼面应果断泼毒避免带毒死亡。
只输出合法 JSON：{"thought": "25~55字战术权衡", "candidates": [最优id, 次优id]}。"""


class DualSystemV3Agent(DualSystemV2Agent):
    """MaKTO-Inspired Role-Adaptive Dual-System Agent (0 LLM fine-tuning).

    Integrates 4 improvements inspired by MaKTO (arXiv:2501.14225) & Xu et al. (ICML 2024):
    1. Lossless Cross-Round Structured Notes (incl. say-vs-vote mismatch detection).
    2. Distilled MaKTO Role-Specific Expert Playbook.
    3. Structured Think-before-Respond JSON planning (private plan -> public speech / thought -> candidates).
    4. Role-Adaptive System-1/System-2 fusion: strong System-2 logic+RL discipline for Good roles,
       high NL strategic flexibility + god-sniping for Werewolves.
    """

    def __init__(
        self,
        policy_model: Any,
        belief_model: Optional[BeliefMLP],
        llm_client: Any,
        good_rl_weight: float = 0.50,
        wolf_rl_weight: float = 0.20,
        seed: int = 0,
        name: str = "Ours_DualSystem_v3",
    ) -> None:
        super().__init__(
            policy_model=policy_model,
            belief_model=belief_model,
            llm_client=llm_client,
            rl_weight=good_rl_weight,
            seed=seed,
            name=name,
        )
        self.good_rl_weight = good_rl_weight
        self.wolf_rl_weight = wolf_rl_weight

    def _build_v3_context(self, obs: PlayerObservation, belief: np.ndarray, fact_board: Any) -> str:
        # Compact current-day verbatim speeches + lossless cross-round structured notes
        cur_day_speeches = [
            f"[今日 {ev.actor}号] {str(ev.payload.get('text', '')).replace(chr(10), ' ')[:120]}"
            for ev in obs.public_events
            if ev.event_type == "speech" and ev.day == obs.day
        ]
        parts = [
            f"当前：第{obs.day}天 | 你的座位：{obs.player_id}号 | 你的真实底牌：{obs.my_role.zh_name}",
            f"存活玩家：{list(obs.alive_players)} | 已出局：{list(obs.dead_players)}",
            ROLE_PLAYBOOK_ZH[obs.my_role],
        ]
        if obs.my_role == Role.WEREWOLF:
            parts.append(f"【私有】你的狼队友：{list(obs.werewolf_teammates)}（严禁暴露队友！）")
        elif obs.my_role == Role.SEER:
            chk = "；".join(f"第{d}夜验{t}号={'狼人' if w else '好人'}" for d, t, w in obs.seer_checks)
            parts.append(f"【私有】你的查验记录：{chk or '暂无'}")
        elif obs.my_role == Role.WITCH:
            anti = "有" if obs.witch_has_antidote else "已用"
            pois = "有" if obs.witch_has_poison else "已用"
            atk = f"，今晚被刀：{obs.witch_tonight_attacked}号" if obs.witch_tonight_attacked is not None else ""
            parts.append(f"【私有】女巫药水：解药({anti})、毒药({pois}){atk}")

        notes = build_cross_round_notes(obs)
        if notes:
            parts.append(notes)
        if cur_day_speeches:
            parts.append("【今日原话摘录】\n" + "\n".join(cur_day_speeches))
        fact_txt = fact_board.format_zh(obs)
        if fact_txt:
            parts.append(fact_txt)
        bel_hint = self._format_calibrated_belief_hint(obs, belief)
        if bel_hint:
            parts.append(bel_hint)
        return "\n".join(parts)

    def act(self, request: ActionRequest) -> ActionResponse:
        obs = request.observation
        legal = request.legal_action_ids
        mask = np.asarray(request.legal_action_mask, dtype=bool)
        belief, rl_probs, rl_log_probs, rl_val = self._get_rl_distribution(obs, request.decision_type, mask)
        fact_board = build_verified_fact_board(obs)

        # Role-adaptive System-2 adjustment
        adj_log_probs = rl_log_probs.copy()
        eff_rl_weight = self.wolf_rl_weight if obs.my_role == Role.WEREWOLF else self.good_rl_weight

        if obs.my_role != Role.WEREWOLF:
            if request.decision_type in (DecisionType.DAY_VOTE, DecisionType.HUNTER_SHOOT):
                for pw in fact_board.proven_wolves:
                    if pw in legal:
                        adj_log_probs[pw] += 2.6
                for pg in fact_board.proven_good:
                    if pg in legal:
                        adj_log_probs[pg] -= 2.6
            elif request.decision_type == DecisionType.NIGHT_WITCH_ACT:
                for pw in fact_board.proven_wolves:
                    if (1 + pw) in legal:
                        adj_log_probs[1 + pw] += 2.8
                for pg in fact_board.proven_good:
                    if (1 + pg) in legal:
                        adj_log_probs[1 + pg] -= 3.2
        else:
            if request.decision_type == DecisionType.NIGHT_WOLF_KILL:
                claimed_gods = set(fact_board.witch_claimers) | set(fact_board.hunter_claimers) | {
                    c.claimer for c in fact_board.seer_claims
                }
                wolf_set = set(obs.werewolf_teammates) | {obs.player_id}
                for gpid in claimed_gods - wolf_set:
                    if (gpid - 1) in legal:
                        adj_log_probs[gpid - 1] += 2.4

        legal_sorted = sorted(legal, key=lambda a: float(adj_log_probs[a]), reverse=True)
        top_rl = legal_sorted[0]
        self.last_trace = {
            "day": obs.day,
            "decision_type": request.decision_type.value,
            "rl_top_action": top_rl,
            "value": round(rl_val, 4),
            "proven_wolves": sorted(fact_board.proven_wolves),
            "belief_wolf_probs": [round(float(belief[i, 3]), 4) for i in range(obs.player_count)],
        }

        ctx = self._build_v3_context(obs, belief, fact_board)

        # 1. MaKTO-style Think-before-Speak JSON for DAY_SPEECH
        if request.decision_type == DecisionType.DAY_SPEECH:
            if obs.mode != "nl":
                return ActionResponse(action_id=top_rl)
            rl_intent = SpeechIntent.from_action_id(top_rl, obs.player_count)
            user_msg = (
                f"{ctx}\n\n"
                f"【系统二宏观参考】RL 推荐倾向：{rl_intent.intent_type.value}"
                + (f"（目标 {rl_intent.target}号）" if rl_intent.target else "")
                + "\n请严格按 JSON 格式先思考规划再输出对外公开发言："
            )
            messages = [
                {"role": "system", "content": MAKTO_SPEECH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            for attempt in range(2):
                try:
                    resp = self.llm_client.chat(messages, max_tokens=260, temperature=0.5)
                    obj = extract_json_object(resp.text)
                    speech = str(obj.get("speech", "")).strip().strip('"').strip("“”")
                    rep = check_speech_guardrail(speech, obs, expected_intent=None)
                    if rep.passed:
                        self.guardrail_stats["passed"] += 1
                        if attempt > 0:
                            self.guardrail_stats["retried"] += 1
                        return ActionResponse(
                            action_id=top_rl,
                            utterance=speech,
                            metadata={
                                "private_plan": str(obj.get("plan", "")),
                                "display_role": str(obj.get("display_role", "")),
                                "vote_target": obj.get("vote_target"),
                            },
                        )
                    messages = messages + [
                        {"role": "assistant", "content": resp.text},
                        {"role": "user", "content": f"你的 speech 字段触发校验错误（{rep.violations}），请修正并重新输出合法 JSON："},
                    ]
                except Exception:
                    # If JSON failed on attempt 0, ask concisely on attempt 1
                    messages = messages + [
                        {"role": "user", "content": "请务必只输出合法 JSON，包含 plan、display_role、vote_target 和 speech 四个字段。"}
                    ]
            self.guardrail_stats["fallback"] += 1
            return ActionResponse(action_id=top_rl, utterance=rl_intent.render_template_zh(obs.player_id))

        # 2. Non-speech discrete decision
        if len(legal) == 1:
            return ActionResponse(action_id=legal[0])

        # High-precision System-2 shortcuts (0 extra LLM calls & zero hallucination!)
        # a) Witch Night 1 save (classic MaKTO expert rule)
        if obs.my_role == Role.WITCH and request.decision_type == DecisionType.NIGHT_WITCH_ACT and obs.day == 1 and 1 in legal:
            return ActionResponse(action_id=1, metadata={"source": "system2_witch_n1_save"})

        # b) Good player has a 100% logically proven alive wolf on Vote/Shoot/Poison
        if obs.my_role != Role.WEREWOLF and fact_board.proven_wolves:
            if request.decision_type in (DecisionType.DAY_VOTE, DecisionType.HUNTER_SHOOT):
                proven_legal = [pw for pw in sorted(fact_board.proven_wolves) if pw in legal]
                if proven_legal:
                    best_pw = max(proven_legal, key=lambda a: float(adj_log_probs[a]))
                    return ActionResponse(action_id=best_pw, metadata={"source": "system2_proven_wolf"})
            elif request.decision_type == DecisionType.NIGHT_WITCH_ACT and obs.day >= 2:
                proven_poison = [1 + pw for pw in sorted(fact_board.proven_wolves) if (1 + pw) in legal]
                if proven_poison:
                    best_p = max(proven_poison, key=lambda a: float(adj_log_probs[a]))
                    return ActionResponse(action_id=best_p, metadata={"source": "system2_proven_poison"})

        options_str = _describe_legal_actions(request)
        rl_hints = ", ".join(f"{a}({float(rl_probs[a]):.0%})" for a in legal_sorted[:3])
        user_msg = (
            f"{ctx}\n\n"
            f"当前决策：{request.decision_type.value}\n"
            f"legal_action_ids: {legal}\n"
            f"动作含义：{options_str}\n"
            f"【系统二 RL 候选参考】Top-3：{rl_hints}\n"
            f"请先在 thought 中权衡此时最合理的行动，再输出 JSON：{{\"thought\": \"...\", \"candidates\": [最优id, 次优id]}}"
        )
        messages = [
            {"role": "system", "content": MAKTO_DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        cands: List[int] = []
        thought = ""
        try:
            resp = self.llm_client.chat(messages, max_tokens=150, temperature=0.25)
            obj = extract_json_object(resp.text)
            raw_cands = obj.get("candidates")
            if isinstance(raw_cands, list):
                for item in raw_cands:
                    if isinstance(item, int) and item in legal and item not in cands:
                        cands.append(item)
            elif isinstance(obj.get("action_id"), int) and int(obj["action_id"]) in legal:
                cands.append(int(obj["action_id"]))
            thought = str(obj.get("thought", ""))
        except Exception:
            pass

        if not cands:
            return ActionResponse(action_id=top_rl, metadata={"source": "rl_fallback"})

        if len(cands) == 1:
            chosen = cands[0]
        else:
            best_a = cands[0]
            best_score = -1e9
            for rank_idx, a in enumerate(cands[:2]):
                llm_rank_score = 0.0 if rank_idx == 0 else -0.55
                score = (1.0 - eff_rl_weight) * llm_rank_score + eff_rl_weight * float(adj_log_probs[a])
                if score > best_score:
                    best_score = score
                    best_a = a
            chosen = best_a

        self.last_trace["llm_candidates"] = cands[:2]
        self.last_trace["chosen_action"] = chosen
        return ActionResponse(action_id=chosen, metadata={"llm_candidates": cands[:2], "thought": thought})
