"""System-2 Symbolic Fact & Contradiction Verifier (eliminates LLM reading hallucinations)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .protocol import PlayerObservation, Role, SpeechIntentType

ROLE_PLAYBOOK_ZH: Dict[Role, str] = {
    Role.SEER: (
        "【预言家专家打法】第1天起果断起跳报清全部查验（金水/查杀）并明确下夜查验意向；"
        "遇到对跳悍跳狼时直指对方逻辑漏洞与护狼共边玩家，坚决号召好人统一票型出查杀狼或悍跳狼。"
    ),
    Role.WITCH: (
        "【女巫专家打法】首夜通常开解药救银水防减员；第1天若无抗推危机先以好人口吻发言防屠神；"
        "第2夜起若确认铁狼、悍跳假预言家或明显冲票狼，务必果断用毒绝不带毒出局；一旦起跳清晰报银水与毒口带队归票。"
    ),
    Role.HUNTER: (
        "【猎人专家打法】前期像平民一样冷静盘逻辑隐藏身份防屠神；受抗推或关键轮次可亮牌威慑狼队；"
        "吃刀或被放逐时结合查杀与票型果断开枪带走最像狼的玩家，切勿盲目闷枪或误伤神职。"
    ),
    Role.VILLAGER: (
        "【平民专家打法】绝不穿神职衣服干扰好人视角；重点比对真假预言家及每位玩家的言行是否一致（是否嘴上踩A却投票B、是否保查杀狼），"
        "坚定站边真预言家并集中票型出狼，不被狼人带节奏。"
    ),
    Role.WEREWOLF: (
        "【狼人专家打法】屠边局核心是精准屠神（预/女/猎）或抗推好人！严禁自爆或暴露队友，切勿无故乱跳女巫/猎人穿帮；"
        "优先以逻辑严密的平民口吻自然融入、顺势引导好人内讧；若队友已被真预言家查杀且必出局，其余狼人应果断倒钩投出该队友换取好人信任；"
        "夜晚务必优先刀死已暴露或高概率神职（预言家→女巫→猎人）。"
    ),
}


def build_cross_round_notes(obs: PlayerObservation) -> str:
    """Build MaKTO-style lossless cross-round structured notes (prevents Day-1 amnesia on Day 3-4)."""
    by_day: Dict[int, List[str]] = {}
    day_speech_targets: Dict[Tuple[int, int], Tuple[str, Optional[int]]] = {}

    for ev in obs.public_events:
        d = ev.day
        by_day.setdefault(d, [])
        if ev.event_type == "night_announcement":
            by_day[d].append(f"公告:{ev.payload.get('text', '')}")
        elif ev.event_type == "hunter_shoot":
            by_day[d].append(f"开枪:{ev.payload['shooter']}号(猎人)带走{ev.payload['target']}号")
        elif ev.event_type == "speech" and ev.actor is not None:
            it = ev.payload.get("intent", {})
            itype = it.get("intent_type", "stay_low")
            tgt = it.get("target")
            txt = str(ev.payload.get("text", ""))
            tags = []
            if re.search(r"(我是|我跳|底牌|拍)[^。；\n]{0,12}(真)?预言家", txt) or itype.startswith("claim_seer"):
                tags.append("跳预")
            if re.search(r"(我是|我跳|底牌|拍)[^。；\n]{0,12}(真)?女巫", txt) or itype.startswith("claim_witch"):
                tags.append("跳女巫")
            if re.search(r"(我是|我跳|底牌|拍)[^。；\n]{0,12}(真)?猎人", txt) or itype.startswith("claim_hunter"):
                tags.append("跳猎")
            act_map = {
                "stay_low": "表水",
                "claim_villager_accuse": f"踩{tgt}号" if tgt else "踩人",
                "claim_villager_defend": f"保{tgt}号" if tgt else "保人",
                "claim_seer_good": f"金水{tgt}号" if tgt else "报金水",
                "claim_seer_wolf": f"查杀{tgt}号" if tgt else "报查杀",
                "claim_witch_save": f"救{tgt}号" if tgt else "报银水",
                "claim_witch_poison": f"毒{tgt}号" if tgt else "报毒",
                "claim_hunter_pressure": f"压{tgt}号" if tgt else "拍枪",
            }
            tag_str = "+".join(tags + [act_map.get(itype, itype)])
            by_day[d].append(f"{ev.actor}号({tag_str})")
            day_speech_targets[(d, ev.actor)] = (itype, tgt if isinstance(tgt, int) else None)
        elif ev.event_type == "vote_result":
            votes = ev.payload.get("votes", {})
            ex = ev.payload.get("exiled")
            tied = ev.payload.get("tied")
            by_tgt: Dict[str, List[int]] = {}
            mismatches: List[str] = []
            for v_str, vt in votes.items():
                try:
                    vp = int(v_str)
                except (TypeError, ValueError):
                    continue
                key = f"{vt}号" if vt is not None else "弃票"
                by_tgt.setdefault(key, []).append(vp)
                # Detect say-vs-vote mismatch (言行不一致)
                prev = day_speech_targets.get((d, vp))
                if prev is not None and vt is not None:
                    p_itype, p_tgt = prev
                    if p_itype in ("claim_villager_accuse", "claim_seer_wolf") and p_tgt is not None and p_tgt != vt:
                        mismatches.append(f"{vp}号(说踩{p_tgt}号却投{vt}号)")
            res_str = f"出{ex}号" if ex is not None else (f"平票{tied}" if tied else "无人出局")
            vote_groups = " ".join(f"{k}←{sorted(vs)}" for k, vs in by_tgt.items())
            line = f"投票[{res_str}]: {vote_groups}"
            if mismatches:
                line += " | 言行不一:" + "、".join(mismatches[:4])
            by_day[d].append(line)

    if not by_day:
        return ""
    out_lines = ["【跨轮次结构化笔记（全历史无损）】"]
    for d in sorted(by_day.keys()):
        out_lines.append(f"- 第{d}天：" + "；".join(by_day[d]))
    return "\n".join(out_lines)


@dataclass
class SeerClaimRecord:
    day: int
    claimer: int
    target: int
    claimed_is_wolf: bool


@dataclass
class VerifiedFactBoard:
    seer_claims: List[SeerClaimRecord] = field(default_factory=list)
    witch_claimers: List[int] = field(default_factory=list)
    hunter_claimers: List[int] = field(default_factory=list)
    proven_wolves: Set[int] = field(default_factory=set)
    proven_good: Set[int] = field(default_factory=set)
    likely_true_seer: Optional[int] = None
    contradiction_notes: List[str] = field(default_factory=list)

    def format_zh(self, obs: PlayerObservation) -> str:
        lines: List[str] = []
        if self.seer_claims:
            sc_strs = [
                f"D{c.day} {c.claimer}号跳预言家报{c.target}号={'查杀(狼)' if c.claimed_is_wolf else '金水(好人)'}"
                for c in self.seer_claims[-8:]
            ]
            lines.append("【系统二核验·预言家查验表】" + "；".join(sc_strs))
        if self.witch_claimers:
            lines.append(f"【系统二核验·曾跳女巫】{self.witch_claimers}")
        if self.hunter_claimers:
            lines.append(f"【系统二核验·曾跳猎人】{self.hunter_claimers}")

        alive_set = set(obs.alive_players)
        alive_pw = sorted(p for p in self.proven_wolves if p in alive_set and p != obs.player_id)
        alive_pg = sorted(p for p in self.proven_good if p in alive_set and p != obs.player_id)
        if alive_pw:
            lines.append(f"【系统二铁证·存活必狼】{alive_pw}号（逻辑铁狼，好人应优先归票/用毒/开枪！）")
        if alive_pg and obs.my_role != Role.WEREWOLF:
            lines.append(f"【系统二铁证·存活好人】{alive_pg}号（切勿错投/错毒）")
        if self.likely_true_seer is not None and obs.my_role != Role.WEREWOLF:
            lines.append(f"【系统二推断·高可信预言家】{self.likely_true_seer}号")
        for note in self.contradiction_notes[:4]:
            lines.append(f"【系统二矛盾提示】{note}")
        return "\n".join(lines)


def build_verified_fact_board(obs: PlayerObservation) -> VerifiedFactBoard:
    """Deterministically verify claims and deduce hard logical contradictions from obs only."""
    board = VerifiedFactBoard()
    me = obs.player_id
    my_role = obs.my_role

    # 1. Seed proven sets from exact role-private & publicly confirmed truth
    if my_role != Role.WEREWOLF:
        board.proven_good.add(me)
    else:
        board.proven_wolves.add(me)
        board.proven_wolves.update(obs.werewolf_teammates)

    for rpid, rrole in obs.revealed_roles.items():
        if rrole != Role.WEREWOLF:
            board.proven_good.add(rpid)

    if my_role == Role.SEER:
        board.likely_true_seer = me
        for _, tgt, is_w in obs.seer_checks:
            if is_w:
                board.proven_wolves.add(tgt)
            else:
                board.proven_good.add(tgt)

    # 2. Extract exact claims from public speeches (both structured intent AND raw text to avoid missed claims)
    seer_claimers_order: List[int] = []
    for ev in obs.public_events:
        if ev.event_type != "speech" or ev.actor is None:
            continue
        actor = ev.actor
        txt = str(ev.payload.get("text", ""))
        itype = ev.payload.get("intent", {}).get("intent_type", "")
        tgt = ev.payload.get("intent", {}).get("target")

        # Seer check extraction (STRICT first-person self-claim only — never trigger on third-person quotes!)
        is_seer_speech = bool(
            re.search(
                r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)预言家",
                txt,
            )
        ) or (
            not txt and itype in (SpeechIntentType.CLAIM_SEER_GOOD.value, SpeechIntentType.CLAIM_SEER_WOLF.value)
        )

        if is_seer_speech:
            if actor not in seer_claimers_order:
                seer_claimers_order.append(actor)
            m_wolf = re.search(r"([1-9])\s*号[^，。；\n]{0,10}(查杀|是狼|狼人)", txt)
            m_good = re.search(r"([1-9])\s*号[^，。；\n]{0,10}(金水|是好人|好人|为好人)", txt)
            if m_wolf and int(m_wolf.group(1)) != actor:
                board.seer_claims.append(SeerClaimRecord(ev.day, actor, int(m_wolf.group(1)), True))
            elif m_good and int(m_good.group(1)) != actor:
                board.seer_claims.append(SeerClaimRecord(ev.day, actor, int(m_good.group(1)), False))
            elif isinstance(tgt, int) and tgt != actor and itype == SpeechIntentType.CLAIM_SEER_WOLF.value:
                board.seer_claims.append(SeerClaimRecord(ev.day, actor, tgt, True))
            elif isinstance(tgt, int) and tgt != actor and itype == SpeechIntentType.CLAIM_SEER_GOOD.value:
                board.seer_claims.append(SeerClaimRecord(ev.day, actor, tgt, False))

        if re.search(
            r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)女巫",
            txt,
        ) or (
            not txt and itype in (SpeechIntentType.CLAIM_WITCH_SAVE.value, SpeechIntentType.CLAIM_WITCH_POISON.value)
        ):
            if actor not in board.witch_claimers:
                board.witch_claimers.append(actor)

        if re.search(
            r"(?:我是(?:\s*[1-9]\s*号)?[，,\s]*(?:真)?|我(?:跳|明牌|拍|自证|底牌|就是)(?:个)?(?:真)?|底牌是(?:真)?)猎人",
            txt,
        ) or (
            not txt and itype == SpeechIntentType.CLAIM_HUNTER_PRESSURE.value
        ):
            if actor not in board.hunter_claimers:
                board.hunter_claimers.append(actor)

    # 3. Logical contradiction deduction (only for Good observers; wolves already know exact wolves!)
    if my_role != Role.WEREWOLF:
        # Counter-claim contradiction if I am a unique god
        if my_role == Role.SEER:
            for sc in seer_claimers_order:
                if sc != me:
                    board.proven_wolves.add(sc)
                    board.contradiction_notes.append(f"我是真预言家，{sc}号对跳预言家必为悍跳狼")
        elif my_role == Role.WITCH:
            for wc in board.witch_claimers:
                if wc != me:
                    board.proven_wolves.add(wc)
                    board.contradiction_notes.append(f"我是真女巫，{wc}号冒充女巫必为狼")
        elif my_role == Role.HUNTER:
            for hc in board.hunter_claimers:
                if hc != me:
                    board.proven_wolves.add(hc)
                    board.contradiction_notes.append(f"我是真猎人，{hc}号冒充猎人必为狼")

        # Any Seer claimant who falsely accuses a proven-good player (me / revealed hunter) is a fake Seer!
        fake_seers: Set[int] = set()
        for rec in board.seer_claims:
            if rec.claimed_is_wolf and rec.target in board.proven_good:
                fake_seers.add(rec.claimer)
                board.proven_wolves.add(rec.claimer)
                board.contradiction_notes.append(
                    f"{rec.claimer}号声称查杀已知好人{rec.target}号，{rec.claimer}号必为假预言家(狼)"
                )
            elif (not rec.claimed_is_wolf) and rec.target in board.proven_wolves:
                fake_seers.add(rec.claimer)
                board.proven_wolves.add(rec.claimer)
                board.contradiction_notes.append(
                    f"{rec.claimer}号给已知狼人{rec.target}号发金水，{rec.claimer}号必为假预言家(狼)"
                )

        # If exactly 1 non-discredited Seer claimant remains, trust their checks!
        remaining_seers = [sc for sc in seer_claimers_order if sc not in fake_seers and sc not in board.proven_wolves]
        if len(remaining_seers) == 1:
            ts = remaining_seers[0]
            board.likely_true_seer = ts
            if len(seer_claimers_order) >= 2 and fake_seers:
                # One fake Seer already proven -> the sole surviving opposing Seer is strongly credible
                board.proven_good.add(ts)
                for rec in board.seer_claims:
                    if rec.claimer == ts:
                        if rec.claimed_is_wolf:
                            board.proven_wolves.add(rec.target)
                            board.contradiction_notes.append(f"高可信预言家{ts}号查杀{rec.target}号，优先出{rec.target}号")
                        else:
                            board.proven_good.add(rec.target)

    return board
