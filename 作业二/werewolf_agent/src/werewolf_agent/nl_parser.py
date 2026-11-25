"""Hybrid NL -> SpeechIntent parser (deterministic regex fast-path + optional LLM fallback)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .protocol import PlayerObservation, SpeechIntent, SpeechIntentType

VALID_INTENT_NAMES = {it.value for it in SpeechIntentType}

PARSER_SYSTEM_PROMPT = """你是狼人杀发言意图解析器。将玩家的一段中文公开发言映射到以下 8 种标准战术意图之一，只输出 JSON：
{"intent_type": "上述英文值", "target": 目标座位整数或null}

可选 intent_type：
- stay_low：仅表水/认民，没有明确点踩或保某一人
- claim_villager_accuse：认民或隐藏身份，明确怀疑/踩/号召投 target
- claim_villager_defend：认民或隐藏身份，明确相信/站边/保护 target
- claim_seer_good：自称预言家并报 target 是金水/好人
- claim_seer_wolf：自称预言家并报 target 是查杀/狼人
- claim_witch_save：自称女巫并报救过 target（银水）
- claim_witch_poison：自称女巫并宣称毒过或要毒 target
- claim_hunter_pressure：自称猎人并点踩/威慑 target

规则：target 必须是 1..9 的整数且不能等于发言者本人；stay_low 时 target 为 null。只输出合法 JSON，不要多余解释。"""


def regex_parse_speech_intent(
    utterance: str,
    speaker: int,
    legal_action_ids: List[int],
    player_count: int = 9,
) -> Optional[Tuple[int, str]]:
    """Fast zero-cost deterministic parser. Returns (action_id, confidence_tag) or None if ambiguous."""
    text = utterance.strip()
    if not text:
        return (0, "regex_empty") if 0 in legal_action_ids else None

    legal_set = set(legal_action_ids)

    def try_return(itype: SpeechIntentType, target: Optional[int], tag: str) -> Optional[Tuple[int, str]]:
        if itype == SpeechIntentType.STAY_LOW:
            return (0, tag) if 0 in legal_set else None
        if target is None or target == speaker or not (1 <= target <= player_count):
            return None
        aid = SpeechIntent(itype, target).to_action_id(player_count)
        return (aid, tag) if aid in legal_set else None

    # 1. Seer claims (STRICT first-person only! Never bridge across "我是好人，2号跳预言家")
    if re.search(
        r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)预言家|(?<![1-9号\s])(?:首夜|昨晚)我?验了?\s*[1-9]\s*号",
        text,
    ):
        m_wolf = re.search(r"([1-9])\s*号[^，。；\n]{0,12}(查杀|是狼|狼人|坏人)", text)
        if m_wolf:
            res = try_return(SpeechIntentType.CLAIM_SEER_WOLF, int(m_wolf.group(1)), "regex_seer_wolf")
            if res is not None:
                return res
        m_wolf2 = re.search(r"(查杀|验出狼)[^1-9]{0,6}([1-9])\s*号", text)
        if m_wolf2:
            res = try_return(SpeechIntentType.CLAIM_SEER_WOLF, int(m_wolf2.group(2)), "regex_seer_wolf")
            if res is not None:
                return res
        m_good = re.search(r"([1-9])\s*号[^，。；\n]{0,12}(金水|是好人|好人|为好人)", text)
        if m_good:
            res = try_return(SpeechIntentType.CLAIM_SEER_GOOD, int(m_good.group(1)), "regex_seer_good")
            if res is not None:
                return res
        m_good2 = re.search(r"(金水)[^1-9]{0,6}([1-9])\s*号", text)
        if m_good2:
            res = try_return(SpeechIntentType.CLAIM_SEER_GOOD, int(m_good2.group(2)), "regex_seer_good")
            if res is not None:
                return res

    # 2. Witch claims (STRICT first-person only!)
    if re.search(
        r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)女巫|我用了?(解药|毒药)|我昨晚(救|毒)了?",
        text,
    ):
        m_pois = re.search(r"(毒了?|泼了?|撒毒|准备毒|要毒)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,8}(毒|泼)", text)
        if m_pois:
            t = int(m_pois.group(2) or m_pois.group(3))
            res = try_return(SpeechIntentType.CLAIM_WITCH_POISON, t, "regex_witch_poison")
            if res is not None:
                return res
        m_save = re.search(r"(救了?|捞了?|银水)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,8}(银水|我救)", text)
        if m_save:
            t = int(m_save.group(2) or m_save.group(3))
            res = try_return(SpeechIntentType.CLAIM_WITCH_SAVE, t, "regex_witch_save")
            if res is not None:
                return res

    # 3. Hunter claims (STRICT first-person only!)
    if re.search(
        r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)猎人",
        text,
    ):
        m_h = re.search(r"(全票出|建议出|带走|开枪|怀疑|踩|投出?|票出)[^1-9]{0,8}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,10}(带走|狼面|可疑|出局)", text)
        if m_h:
            t = int(m_h.group(2) or m_h.group(3))
            res = try_return(SpeechIntentType.CLAIM_HUNTER_PRESSURE, t, "regex_hunter_pressure")
            if res is not None:
                return res

    # 4. Villager accuse / defend (prioritize explicit vote calls like "今天出 X 号" / "全票出 X 号")
    m_call = re.search(r"(全票出|今天先?出|建议先?出|票出|归票|投出?|出掉)[^1-9]{0,6}([1-9])\s*号", text)
    if m_call:
        res = try_return(SpeechIntentType.CLAIM_VILLAGER_ACCUSE, int(m_call.group(2)), "regex_call_vote")
        if res is not None:
            return res
    m_acc = re.search(
        r"(怀疑|投给?|归票|出掉?|票出|点踩|狼面|像狼|可能是狼)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,10}(可疑|狼面|像狼|是狼|建议出|出局|投票)",
        text,
    )
    if m_acc:
        t = int(m_acc.group(2) or m_acc.group(3))
        res = try_return(SpeechIntentType.CLAIM_VILLAGER_ACCUSE, t, "regex_accuse")
        if res is not None:
            return res

    m_def = re.search(
        r"(站边|相信|保一下?|偏向好人|支持)[^1-9]{0,6}([1-9])\s*号|([1-9])\s*号[^，。；\n]{0,10}(好人|可信|真预言家|我保)",
        text,
    )
    if m_def:
        t = int(m_def.group(2) or m_def.group(3))
        res = try_return(SpeechIntentType.CLAIM_VILLAGER_DEFEND, t, "regex_defend")
        if res is not None:
            return res

    # 5. Clear stay-low (no other seat numbers mentioned)
    other_seats = [int(m) for m in re.findall(r"([1-9])\s*号", text) if int(m) != speaker]
    if not other_seats:
        return try_return(SpeechIntentType.STAY_LOW, None, "regex_stay_low")

    return None


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from raw LLM text (handles markdown code fences)."""
    s = raw_text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


class HybridSpeechParser:
    """Parse natural Chinese speeches into 64-dim SpeechIntent IDs with minimal API cost."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        prefer_regex: bool = True,
        player_count: int = 9,
    ) -> None:
        self.llm_client = llm_client
        self.prefer_regex = prefer_regex
        self.player_count = player_count
        self.stats: Dict[str, int] = {"regex_hits": 0, "llm_calls": 0, "llm_fallbacks": 0}

    def __call__(
        self,
        utterance: str,
        obs: PlayerObservation,
        legal_action_ids: List[int],
    ) -> int:
        if self.prefer_regex:
            hit = regex_parse_speech_intent(utterance, obs.player_id, legal_action_ids, self.player_count)
            if hit is not None:
                self.stats["regex_hits"] += 1
                return hit[0]

        if self.llm_client is not None:
            self.stats["llm_calls"] += 1
            messages = [
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "speaker": obs.player_id,
                            "alive_players": list(obs.alive_players),
                            "speech": utterance,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            try:
                resp = self.llm_client.chat(messages, max_tokens=64, temperature=0.0)
                obj = extract_json_object(resp.text)
                itype_str = str(obj.get("intent_type", ""))
                target = obj.get("target")
                if itype_str in VALID_INTENT_NAMES:
                    itype = SpeechIntentType(itype_str)
                    if itype == SpeechIntentType.STAY_LOW:
                        if 0 in legal_action_ids:
                            return 0
                    elif isinstance(target, int) and 1 <= target <= self.player_count and target != obs.player_id:
                        aid = SpeechIntent(itype, target).to_action_id(self.player_count)
                        if aid in legal_action_ids:
                            return aid
            except Exception:
                self.stats["llm_fallbacks"] += 1

        self.stats["llm_fallbacks"] += 1
        return 0 if 0 in legal_action_ids else legal_action_ids[0]
