"""Compressed game summary builder, Intent->Speech generator, and Anti-Leak/Hallucination Guardrail."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .features import ROLE_ORDER
from .protocol import PlayerObservation, Role, SpeechIntent, SpeechIntentType

SPEECH_GEN_SYSTEM_PROMPT = """你正在参加一局中文九人预女猎狼人杀（3狼3民1预1女1猎，屠边，无警长）。
你需要严格按照指定的【战术意图】和【当前身份视角】发表一段简短、自然、有说服力的白天公开发言。
硬性要求：
1. 字数控制在 40~110 字之间，直接输出发言正文，不要输出思考过程、旁白、JSON或引号。
2. 绝对禁止泄露不该公开的底牌信息（例如狼人绝不能说出真实狼队友名单或承认自己是狼；好人不要编造不存在的系统公告）。
3. 必须清晰体现指定的【战术意图】与【目标座位号】（如有目标）。"""


def format_compact_context(
    obs: PlayerObservation,
    belief_probs: Optional[Any] = None,
    max_recent_speeches: int = 9,
) -> str:
    """Build a token-efficient structured game state summary (~300-600 tokens)."""
    lines: List[str] = [
        f"当前：第{obs.day}天 | 你的座位：{obs.player_id}号 | 你的真实底牌：{obs.my_role.zh_name}",
        f"存活玩家：{list(obs.alive_players)} | 已出局：{list(obs.dead_players)}",
    ]
    if obs.revealed_roles:
        rev_str = "、".join(f"{p}号({r.zh_name})" for p, r in sorted(obs.revealed_roles.items()))
        lines.append(f"已公开确认身份：{rev_str}")

    # Strict role-private block
    if obs.my_role == Role.WEREWOLF:
        lines.append(f"【私有】你的狼队友：{list(obs.werewolf_teammates)}（严禁在公开发言中暴露队友身份！）")
    elif obs.my_role == Role.SEER:
        if obs.seer_checks:
            chk = "；".join(f"第{d}夜验{t}号={'狼人' if w else '好人'}" for d, t, w in obs.seer_checks)
            lines.append(f"【私有】你的查验记录：{chk}")
        else:
            lines.append("【私有】暂无查验记录")
    elif obs.my_role == Role.WITCH:
        anti = "有" if obs.witch_has_antidote else "已用"
        pois = "有" if obs.witch_has_poison else "已用"
        extra = f"，今晚被刀：{obs.witch_tonight_attacked}号" if obs.witch_tonight_attacked is not None else ""
        lines.append(f"【私有】女巫药水：解药({anti})、毒药({pois}){extra}")

    # Compact public timeline: announcements, shoots, votes, and recent speeches
    announcements: List[str] = []
    votes_hist: List[str] = []
    speeches: List[str] = []
    for ev in obs.public_events:
        if ev.event_type == "night_announcement":
            announcements.append(f"第{ev.day}天：{ev.payload.get('text', '')}")
        elif ev.event_type == "hunter_shoot":
            announcements.append(f"第{ev.day}天：{ev.payload['shooter']}号翻牌猎人带走{ev.payload['target']}号")
        elif ev.event_type == "vote_result":
            ex = ev.payload.get("exiled")
            tied = ev.payload.get("tied")
            if ex is not None:
                votes_hist.append(f"第{ev.day}天放逐{ex}号（票型:{ev.payload.get('votes')}）")
            elif tied:
                votes_hist.append(f"第{ev.day}天平票{tied}")
        elif ev.event_type == "speech":
            txt = str(ev.payload.get("text", "")).replace("\n", " ")[:110]
            speeches.append(f"[D{ev.day} {ev.actor}号] {txt}")

    if announcements:
        lines.append("公开公告：" + "；".join(announcements[-6:]))
    if votes_hist:
        lines.append("历史投票：" + "；".join(votes_hist[-4:]))
    if speeches:
        lines.append("近期发言：\n" + "\n".join(speeches[-max_recent_speeches:]))

    if belief_probs is not None:
        b_items = []
        for pid in obs.alive_players:
            if pid == obs.player_id:
                continue
            row = belief_probs[pid - 1]
            top_idx = int(max(range(len(row)), key=lambda i: float(row[i])))
            p_wolf = float(row[3])
            b_items.append(f"{pid}号(狼概率{p_wolf:.0%},最可能{ROLE_ORDER[top_idx].zh_name})")
        if b_items:
            lines.append("【内部信念参考】" + "；".join(b_items))

    return "\n".join(lines)


def describe_intent_instruction(intent: SpeechIntent) -> str:
    t = intent.target
    mapping = {
        SpeechIntentType.STAY_LOW: "以平民口吻表水、保持低调，先听他人发言，不明确攻击特定座位。",
        SpeechIntentType.CLAIM_VILLAGER_ACCUSE: f"以平民/好人口吻表水，并明确怀疑、点踩或号召今天投票出 {t}号。",
        SpeechIntentType.CLAIM_VILLAGER_DEFEND: f"以平民/好人口吻表水，并明确表示相信、站边或保护 {t}号。",
        SpeechIntentType.CLAIM_SEER_GOOD: f"明确跳预言家身份，宣称自己查验了 {t}号，且 {t}号是金水（好人）。",
        SpeechIntentType.CLAIM_SEER_WOLF: f"明确跳预言家身份，宣称自己查验了 {t}号，且 {t}号是查杀（狼人），号召出 {t}号。",
        SpeechIntentType.CLAIM_WITCH_SAVE: f"明确跳女巫身份，宣称自己之前用解药救过 {t}号（{t}号是银水）。",
        SpeechIntentType.CLAIM_WITCH_POISON: f"明确跳女巫身份，表示怀疑 {t}号是狼人，宣称已毒或准备毒 {t}号。",
        SpeechIntentType.CLAIM_HUNTER_PRESSURE: f"明确拍猎人身份施压，点出 {t}号狼面很大并威慑若出局将考虑带走 {t}号。",
    }
    return mapping[intent.intent_type]


@dataclass
class GuardrailReport:
    passed: bool
    violations: List[str]


def check_speech_guardrail(
    utterance: str,
    obs: PlayerObservation,
    expected_intent: Optional[SpeechIntent] = None,
) -> GuardrailReport:
    """Check generated speech for secret leaks, impossible claims, and intent alignment."""
    violations: List[str] = []
    text = utterance.strip()
    if len(text) < 8:
        violations.append("too_short")
    if len(text) > 260:
        violations.append("too_long")

    # 1. Anti-leak: Werewolf must never confess being a wolf or expose teammates
    if obs.my_role == Role.WEREWOLF:
        if re.search(r"(我是|作为|身为)[^，。；\n]{0,4}狼人|我的狼队友|狼队友是", text):
            violations.append("wolf_self_or_teammate_leak")
        for tm in obs.werewolf_teammates:
            if re.search(rf"{tm}\s*号[^，。；\n]{{0,6}}(是我|是我的)?(狼)?队友", text):
                violations.append(f"wolf_teammate_{tm}_leak")

    # 2. Anti-hallucination: Never claim game already ended during speech
    if re.search(r"(游戏已经结束|好人已经获胜|狼人已经获胜)", text):
        violations.append("premature_game_over_hallucination")

    # 3. Intent alignment (for Decoupled Agent)
    if expected_intent is not None:
        t = expected_intent.target
        if t is not None and f"{t}号" not in text and f"{t} 号" not in text:
            violations.append(f"missing_target_{t}")
        if expected_intent.intent_type in (SpeechIntentType.CLAIM_SEER_GOOD, SpeechIntentType.CLAIM_SEER_WOLF):
            if "预言家" not in text and "查验" not in text and "金水" not in text and "查杀" not in text:
                violations.append("missing_seer_claim")
        elif expected_intent.intent_type in (SpeechIntentType.CLAIM_WITCH_SAVE, SpeechIntentType.CLAIM_WITCH_POISON):
            if "女巫" not in text and "银水" not in text and "解药" not in text and "毒" not in text:
                violations.append("missing_witch_claim")
        elif expected_intent.intent_type == SpeechIntentType.CLAIM_HUNTER_PRESSURE:
            if "猎人" not in text and "开枪" not in text and "带走" not in text:
                violations.append("missing_hunter_claim")

    return GuardrailReport(passed=(len(violations) == 0), violations=violations)


def generate_speech_for_intent(
    llm_client: Any,
    obs: PlayerObservation,
    intent: SpeechIntent,
    belief_probs: Optional[Any] = None,
    max_retries: int = 1,
) -> Tuple[str, GuardrailReport, int]:
    """Use frozen LLM to render a SpeechIntent into natural Chinese with Guardrail verification."""
    context = format_compact_context(obs, belief_probs=belief_probs)
    instruction = describe_intent_instruction(intent)
    base_user = f"{context}\n\n【本轮必须执行的战术意图】\n{instruction}\n请直接输出你的公开发言："

    messages = [
        {"role": "system", "content": SPEECH_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": base_user},
    ]

    last_text = ""
    last_report = GuardrailReport(passed=False, violations=["not_run"])
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            resp = llm_client.chat(messages, max_tokens=180, temperature=0.6)
            text = resp.text.strip().strip('"').strip("“”")
            report = check_speech_guardrail(text, obs, expected_intent=intent)
            last_text, last_report = text, report
            if report.passed:
                return text, report, attempts
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"你的发言触发校验错误：{report.violations}。请修正并重新直接输出 40~100 字的合法发言：",
                },
            ]
        except Exception as exc:
            last_report = GuardrailReport(passed=False, violations=[f"api_error:{exc}"])

    # Safe fallback to structured template so game never crashes or leaks secrets
    fallback = intent.render_template_zh(obs.player_id)
    return fallback, last_report, attempts
