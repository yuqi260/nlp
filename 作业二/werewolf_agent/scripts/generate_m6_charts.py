"""Generate publication-grade academic SVG charts (ICML/NeurIPS serif style, cobalt blue + burnt sienna palette)."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent import DecisionType, SpeechIntent, WerewolfEngine
from werewolf_agent.agents import NeuralPolicyAgent
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet

# Academic publication serif typography matching Word report (Times New Roman + SimSun)
FONT_SERIF = "'Times New Roman', 'SimSun', 'Songti SC', 'STSong', Georgia, serif"
FONT_ARCH = "'Microsoft YaHei', 'PingFang SC', 'Segoe UI', Arial, sans-serif"

# Reference academic color palette (Deep Cobalt Blue + Warm Burnt Sienna / Ochre Earth Ramp + Slate)
C_BLUE_MAIN = "#0B4F9C"      # Hero / Ours (deep cobalt blue)
C_BLUE_MID = "#4A81BF"       # Companion steel blue
C_BLUE_BAND = "#E4EEF8"      # Subtle vertical highlight band
C_BROWN_DARK = "#7F3310"     # Dark umber / chestnut
C_BROWN_MID = "#B85324"      # Burnt sienna / rust
C_BROWN_LIGHT = "#D98A4E"    # Warm ochre / tan
C_TAUPE = "#B5A89D"          # Muted warm stone gray (Random / weak baseline)
C_AXIS = "#2B2B2B"           # Crisp dark charcoal spines & ticks
C_DASH = "#6E737B"           # Subtle reference dashed line


def svg_wrap(w: int, h: int, body: str, font_family: str = FONT_SERIF) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<style>'
        f'text{{font-family:{font_family};fill:#1a1a1a;}}'
        f'.spine{{stroke:{C_AXIS};stroke-width:1.15;fill:none;}}'
        f'.tick{{stroke:{C_AXIS};stroke-width:1.0;}}'
        f'.refline{{stroke:{C_DASH};stroke-width:1.15;stroke-dasharray:5.5 4;}}'
        f'.panel-lbl{{font-size:15px;font-weight:700;}}'
        f'.panel-ttl{{font-size:14.5px;font-weight:400;}}'
        f'.ax-lbl{{font-size:13px;}}'
        f'.tk-lbl{{font-size:12px;}}'
        f'.leg-lbl{{font-size:12px;}}'
        f'.val{{font-size:10.5px;font-weight:400;}}'
        f'.val-b{{font-size:10.5px;font-weight:700;}}'
        f'</style>'
        f"{body}</svg>"
    )


def make_fig0_system_architecture() -> str:
    """Keep the clean two-loop system architecture diagram (unchanged layout, harmonized cobalt/slate palette)."""
    w, h = 960, 500
    parts: List[str] = [
        '<defs>'
        '<marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="{C_BLUE_MAIN}"/></marker>'
        '<marker id="arr-teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#0d9488"/></marker>'
        '<marker id="arr-slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#475569"/></marker>'
        '<marker id="arr-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="{C_BROWN_MID}"/></marker>'
        '</defs>',
        f'<text x="{w/2}" y="28" text-anchor="middle" style="font-size:17px;font-weight:700">语言与决策解耦的双系统狼人杀智能体总体架构</text>',
        f'<text x="{w/2}" y="47" text-anchor="middle" style="font-size:11.5px;fill:#475569">上方：离线符号环境零 API 成本快速训练环（本地 CPU） ｜ 下方：在线真实中文双系统推理环（冻结大模型零微调）</text>',
    ]

    def box(x: float, y: float, bw: float, bh: float, title: str, lines: List[str], fill: str, stroke: str, title_color: str = "#0f172a") -> str:
        s = [
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
            f'<text x="{x+bw/2}" y="{y+21}" text-anchor="middle" style="font-size:12px;font-weight:700;fill:{title_color}">{title}</text>',
        ]
        for idx, ln in enumerate(lines):
            s.append(f'<text x="{x+bw/2}" y="{y+39+idx*15}" text-anchor="middle" style="font-size:10.5px;fill:#334155">{ln}</text>')
        return "\n".join(s)

    # Top Panel
    parts.append(f'<rect x="20" y="62" width="920" height="176" rx="8" fill="#f8fafc" stroke="{C_BLUE_MID}" stroke-width="1.4" stroke-dasharray="6 4"/>')
    parts.append(f'<rect x="32" y="70" width="335" height="22" rx="4" fill="{C_BLUE_BAND}"/>')
    parts.append(f'<text x="42" y="85" style="font-size:11.5px;font-weight:700;fill:{C_BLUE_MAIN}">① 离线训练态：符号环境高速自博弈（本地 CPU，0 API 费）</text>')

    parts.append(box(34, 104, 178, 88, "MaKTO 人类轨迹重构", ["133 局整局防泄漏划分", "1,552 信念快照 / 4,261 步", "91 维席位共享特征编码"], "#f4f8fc", C_BLUE_MID, C_BLUE_MAIN))
    parts.append(box(246, 104, 186, 88, "席位共享 BeliefMLP", ["91 → 128 → 64 → 5", "座位排列等变 + 已知硬约束", "历史 NL 回放跨领域微调"], "#f4f8fc", C_BLUE_MAIN, C_BLUE_MAIN))
    parts.append(box(466, 104, 214, 88, "席位等变指针策略-价值网络", ["106 维实体 → 共享指针打分头", "BC 冷启动（模仿人类分布）", "4,800 局 Masked PPO + RAE + KL"], "#edf3fa", C_BLUE_MAIN, C_BLUE_MAIN))
    parts.append(box(714, 104, 212, 88, "Symbolic Mode 状态机引擎", ["九人预女猎屠边 · 严格视角隔离", "357–637 局/秒 · 种群对手池", "6 类合法动作掩码（含 64 维意图）"], "#f8fafc", "#475569", "#1e293b"))

    parts.append(f'<line x1="212" y1="148" x2="244" y2="148" stroke="{C_BLUE_MAIN}" stroke-width="1.6" marker-end="url(#arr-blue)"/>')
    parts.append(f'<line x1="432" y1="148" x2="464" y2="148" stroke="{C_BLUE_MAIN}" stroke-width="1.6" marker-end="url(#arr-blue)"/>')
    parts.append(f'<line x1="680" y1="136" x2="712" y2="136" stroke="{C_BLUE_MAIN}" stroke-width="1.6" marker-end="url(#arr-blue)"/>')
    parts.append(f'<line x1="712" y1="162" x2="680" y2="162" stroke="{C_BROWN_MID}" stroke-width="1.6" marker-end="url(#arr-amber)"/>')
    parts.append(f'<text x="696" y="130" text-anchor="middle" style="font-size:9.5px;fill:{C_BLUE_MAIN}">动作</text>')
    parts.append(f'<text x="696" y="176" text-anchor="middle" style="font-size:9.5px;fill:{C_BROWN_MID}">回报</text>')

    parts.append(f'<line x1="339" y1="192" x2="339" y2="303" stroke="{C_BLUE_MID}" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#arr-blue)"/>')
    parts.append(f'<line x1="573" y1="192" x2="573" y2="303" stroke="{C_BLUE_MAIN}" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#arr-blue)"/>')

    parts.append('<rect x="195" y="211" width="570" height="20" rx="4" fill="#f8fafc"/>')
    parts.append('<text x="480" y="225" text-anchor="middle" style="font-size:10.5px;fill:#475569">权重离线冻结并导出：BeliefMLP_NL（约 2 万参数）与 SeatEquivariantPolicyValueNet（约 3.2 万参数）</text>')

    parts.append(f'<rect x="275" y="242" width="128" height="18" rx="3" fill="#ffffff" stroke="{C_BLUE_MID}"/>')
    parts.append(f'<text x="339" y="255" text-anchor="middle" style="font-size:9.5px;font-weight:600;fill:{C_BLUE_MAIN}">加载 BeliefMLP_NL</text>')

    parts.append(f'<rect x="502" y="242" width="142" height="18" rx="3" fill="#ffffff" stroke="{C_BLUE_MAIN}"/>')
    parts.append(f'<text x="573" y="255" text-anchor="middle" style="font-size:9.5px;font-weight:600;fill:{C_BLUE_MAIN}">加载 Pointer RL 权重</text>')

    # Bottom Panel
    parts.append(f'<rect x="20" y="270" width="920" height="214" rx="8" fill="#fdfaf6" stroke="{C_BROWN_LIGHT}" stroke-width="1.4" stroke-dasharray="6 4"/>')
    parts.append('<rect x="32" y="278" width="380" height="22" rx="4" fill="#f6e7d8"/>')
    parts.append(f'<text x="42" y="293" style="font-size:11.5px;font-weight:700;fill:{C_BROWN_DARK}">② 在线推理态：中文自然语言双系统融合（冻结 qwen3.5-27b）</text>')

    parts.append(box(34, 310, 178, 86, "NL Mode 公开对话流", ["第一人称观测隔离", "精简上下文（8 条近期发言）", "混合解析：正则快路径+LLM"], "#ffffff", "#64748b", "#334155"))
    parts.append(box(246, 306, 186, 94, "System-2 信念与事实核验", ["BeliefMLP_NL 校准狼概率", "fact_tracker 客观查验账本", "反证逻辑铁狼 / 防误读金水"], "#f4f8fc", C_BLUE_MID, C_BLUE_MAIN))
    parts.append(box(466, 306, 214, 94, "System-1 冻结 LLM + 重排", ["结合原文+事实表提议 Top-2", "Pointer RL 对数概率融合重排", "铁狼强护栏 + 狼人低权放行"], "#edf3fa", C_BLUE_MAIN, C_BLUE_MAIN))
    parts.append(box(714, 310, 212, 86, "Guardrail 护栏与动作执行", ["反泄密（禁爆狼/队友）", "反幻觉（禁编造终局）", "写回状态机驱动下一轮"], "#fdf4ef", C_BROWN_MID, C_BROWN_DARK))

    parts.append('<line x1="212" y1="353" x2="244" y2="353" stroke="#475569" stroke-width="1.6" marker-end="url(#arr-slate)"/>')
    parts.append(f'<line x1="432" y1="353" x2="464" y2="353" stroke="{C_BLUE_MAIN}" stroke-width="1.6" marker-end="url(#arr-blue)"/>')
    parts.append(f'<line x1="680" y1="353" x2="712" y2="353" stroke="{C_BROWN_MID}" stroke-width="1.6" marker-end="url(#arr-amber)"/>')

    parts.append(f'<path d="M 820 396 L 820 456 L 123 456 L 123 398" fill="none" stroke="{C_BROWN_MID}" stroke-width="1.6" marker-end="url(#arr-amber)"/>')
    parts.append(f'<rect x="330" y="445" width="300" height="20" rx="4" fill="#fdfaf6" stroke="{C_BROWN_LIGHT}"/>')
    parts.append(f'<text x="480" y="459" text-anchor="middle" style="font-size:10.5px;font-weight:600;fill:{C_BROWN_DARK}">公开发言 / 夜间技能 / 放逐投票 回写统一状态机</text>')

    return svg_wrap(w, h, "\n".join(parts), font_family=FONT_ARCH)


def make_fig1_ppo_curve(m3_pointer: dict) -> str:
    """Academic line chart matching reference: serif font, outward ticks, no gridlines, frameless legend, cobalt + sienna palette."""
    hist = m3_pointer["ppo_history"]
    w, h = 760, 390
    ml, mr, mt, mb = 64, 58, 44, 56
    pw, ph = w - ml - mr, h - mt - mb
    max_x = max(r["games_seen"] for r in hist) or 4800

    def tx(v: float) -> float:
        return ml + (v / max_x) * pw

    def ty(v: float) -> float:
        return mt + (1.0 - v) * ph

    parts: List[str] = [
        f'<text x="{ml}" y="25"><tspan class="panel-lbl">(a)</tspan><tspan class="panel-ttl"> 种群自博弈 Masked PPO 学习曲线（0–4,800 局，验证种子 20000/30000）</tspan></text>',
    ]

    # Subtle vertical highlight band on best validation checkpoint (4320 games) — matching reference panel (b)/(c)
    x_best = tx(4320)
    parts.append(f'<rect x="{x_best-18:.1f}" y="{mt}" width="36" height="{ph}" fill="{C_BLUE_BAND}"/>')

    # Reference dashed line (Rule baseline = 48.8%)
    y_rule = ty(0.488)
    parts.append(f'<line x1="{ml}" y1="{y_rule:.1f}" x2="{ml+pw}" y2="{y_rule:.1f}" class="refline"/>')
    parts.append(f'<text x="{tx(2550):.1f}" y="{y_rule+14:.1f}" class="val" style="fill:{C_DASH}">规则基线综合胜率 48.8%</text>')

    # L-shaped axes + outward ticks (NO gridlines!)
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" class="spine"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" class="spine"/>')

    for pct in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = ty(pct)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')

    for row in hist:
        x = tx(row["games_seen"])
        parts.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+5}" class="tick"/>')
        lbl_style = 'font-weight:700;' if row["games_seen"] == 4320 else ''
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+20}" text-anchor="middle" class="tk-lbl" style="{lbl_style}">{row["games_seen"]}</text>')

    parts.append(f'<text x="{ml+pw/2}" y="{h-12}" text-anchor="middle" class="ax-lbl">自博弈训练累计对局数（Symbolic Mode, 本地 CPU）</text>')
    parts.append(f'<text x="18" y="{mt+ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 18 {mt+ph/2})">胜率 / 投狼准确率</text>')

    # Series: (key, label, color, stroke_width, hollow_marker, end_dy)
    series = [
        ("eval_wolf_vs_rule_good", "狼人 vs 规则好人 胜率", C_BROWN_DARK, 1.8, False, 4),
        ("eval_combined_score", "综合胜率（双阵营均值）", C_BLUE_MAIN, 2.4, False, -5),
        ("eval_good_vote_wolf_acc", "好人放逐投狼准确率", C_BROWN_LIGHT, 1.8, False, 10),
        ("eval_good_vs_rule_wolves", "好人 vs 规则狼人 胜率", C_BLUE_MID, 1.7, True, 4),
    ]
    for key, label, color, sw, hollow, end_dy in series:
        pts = " ".join(f"{tx(r['games_seen']):.1f},{ty(r[key]):.1f}" for r in hist)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" points="{pts}"/>')
        for r in hist:
            cx, cy = tx(r["games_seen"]), ty(r[key])
            if hollow:
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>')
            else:
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="{color}"/>')
        last = hist[-1]
        parts.append(
            f'<text x="{tx(last["games_seen"])+7:.1f}" y="{ty(last[key])+end_dy:.1f}" class="val-b" style="fill:{color}">{last[key]*100:.1f}%</text>'
        )

    # Frameless legend placed cleanly in open bottom-right whitespace (below green curve? No, above green curve at x=2000..3600 or bottom-right below 40%!)
    # At x=1920..4320, y=0.05..0.24 is completely empty white space! Let's place frameless legend cleanly at bottom-right (x=420, y=252)
    lx = ml + pw - 210
    ly = mt + ph - 92
    for idx, (_, label, color, sw, hollow, _) in enumerate(series):
        yy = ly + idx * 20
        parts.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+26}" y2="{yy}" stroke="{color}" stroke-width="{sw}"/>')
        if hollow:
            parts.append(f'<circle cx="{lx+13}" cy="{yy}" r="3.3" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>')
        else:
            parts.append(f'<circle cx="{lx+13}" cy="{yy}" r="3.3" fill="{color}"/>')
        parts.append(f'<text x="{lx+33}" y="{yy+4}" class="leg-lbl">{label}</text>')

    return svg_wrap(w, h, "\n".join(parts))


def make_fig2_symbolic_ablation(m3_flat: dict, m3_pointer: dict) -> str:
    """Academic grouped bar chart: sharp bars, L-spines, outward ticks, frameless legend, cobalt + sienna palette."""
    w, h = 760, 380
    ml, mr, mt, mb = 62, 24, 54, 62
    pw, ph = w - ml - mr, h - mt - mb

    p_mat = m3_pointer["symbolic_ablation_matrix"]
    f_mat = m3_flat["symbolic_ablation_matrix"]

    groups = [
        ("Random\n随机基线", p_mat["Random"]["as_good_vs_rule_wolves"]["good_win_rate"], p_mat["Random"]["as_wolf_vs_rule_good"]["wolf_win_rate"], p_mat["Random"]["combined_vs_rule"]),
        ("Rule\n启发式规则", p_mat["Rule"]["as_good_vs_rule_wolves"]["good_win_rate"], p_mat["Rule"]["as_wolf_vs_rule_good"]["wolf_win_rate"], p_mat["Rule"]["combined_vs_rule"]),
        ("Flat MLP\nBC+RL", f_mat["Belief+BC+RL"]["as_good_vs_rule_wolves"]["good_win_rate"], f_mat["Belief+BC+RL"]["as_wolf_vs_rule_good"]["wolf_win_rate"], f_mat["Belief+BC+RL"]["combined_vs_rule"]),
        ("Pointer\n仅离线 BC", p_mat["Belief+BC"]["as_good_vs_rule_wolves"]["good_win_rate"], p_mat["Belief+BC"]["as_wolf_vs_rule_good"]["wolf_win_rate"], p_mat["Belief+BC"]["combined_vs_rule"]),
        ("Pointer\nBC+RL (Ours)", p_mat["Belief+BC+RL"]["as_good_vs_rule_wolves"]["good_win_rate"], p_mat["Belief+BC+RL"]["as_wolf_vs_rule_good"]["wolf_win_rate"], p_mat["Belief+BC+RL"]["combined_vs_rule"]),
    ]

    def ty(v: float) -> float:
        return mt + (1.0 - v) * ph

    parts = [
        f'<text x="{ml}" y="25"><tspan class="panel-lbl">(a)</tspan><tspan class="panel-ttl"> 符号环境 300 局架构与方法消融（独立测试种子 50000/60000，对阵规则对手）</tspan></text>',
    ]

    gw = pw / len(groups)
    # Subtle pale-blue highlight band behind Ours (last group)
    cx_last = ml + (len(groups) - 0.5) * gw
    parts.append(f'<rect x="{cx_last - gw*0.44:.1f}" y="{mt}" width="{gw*0.88:.1f}" height="{ph}" fill="{C_BLUE_BAND}"/>')

    # Frameless legend at top-left above plot area
    colors = [C_BROWN_LIGHT, C_BROWN_DARK, C_BLUE_MAIN]
    names = ["好人胜率（对规则狼）", "狼人胜率（对规则好人）", "综合胜率（双阵营均值）"]
    for i, (nm, cl) in enumerate(zip(names, colors)):
        xx = ml + 10 + i * 185
        parts.append(f'<rect x="{xx}" y="35" width="12" height="9" fill="{cl}"/>')
        parts.append(f'<text x="{xx+17}" y="44" class="leg-lbl">{nm}</text>')

    # Spines & outward ticks
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" class="spine"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" class="spine"/>')
    for pct in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = ty(pct)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')
    parts.append(f'<text x="16" y="{mt+ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 16 {mt+ph/2})">胜率</text>')

    bw = 26
    for gi, (gname, v_g, v_w, v_c) in enumerate(groups):
        cx = ml + (gi + 0.5) * gw
        parts.append(f'<line x1="{cx:.1f}" y1="{mt+ph}" x2="{cx:.1f}" y2="{mt+ph+4}" class="tick"/>')
        vals = [v_g, v_w, v_c]
        for bi, val in enumerate(vals):
            bx = cx + (bi - 1.5) * (bw + 4) + 2
            by = ty(val)
            bh = mt + ph - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{colors[bi]}"/>')
            cls = "val-b" if (gi == len(groups) - 1 and bi == 2) else "val"
            parts.append(f'<text x="{bx+bw/2:.1f}" y="{by-5:.1f}" text-anchor="middle" class="{cls}">{val*100:.1f}%</text>')
        for li, line in enumerate(gname.split("\n")):
            st = 'font-weight:700;' if gi == len(groups) - 1 else ''
            parts.append(f'<text x="{cx:.1f}" y="{mt+ph+19+li*15}" text-anchor="middle" class="tk-lbl" style="{st}">{line}</text>')

    return svg_wrap(w, h, "\n".join(parts))


def make_fig3_protocol_drift() -> str:
    """Academic bar chart for 30-game speech intent distribution (Protocol Drift)."""
    belief = BeliefMLP.load(ROOT / "outputs" / "models" / "belief_mlp.npz")
    eng = WerewolfEngine(mode="symbolic")
    intents_order = [
        ("stay_low", "低调表水"),
        ("claim_villager_accuse", "认民点踩"),
        ("claim_villager_defend", "认民站边"),
        ("claim_seer_wolf", "跳预查杀"),
        ("claim_seer_good", "跳预金水"),
        ("claim_witch_save", "跳女巫银水"),
        ("claim_witch_poison", "跳女巫毒"),
        ("claim_hunter_pressure", "拍猎人施压"),
    ]
    dists: Dict[str, Dict[str, float]] = {}
    for label, path in [
        ("BC (人类模仿)", ROOT / "outputs" / "models" / "policy_pointer_bc.npz"),
        ("BC+RL (KL锚定)", ROOT / "outputs" / "models" / "policy_pointer_bc_rl.npz"),
        ("RL_scratch (无KL)", ROOT / "outputs" / "models" / "policy_pointer_rl_scratch.npz"),
    ]:
        pol = SeatEquivariantPolicyValueNet.load(path)
        cnt = Counter()
        for g in range(30):
            agents = {pid: NeuralPolicyAgent(pol, belief, seed=g * 13 + pid, temperature=0.7) for pid in range(1, 10)}
            res = eng.run_game(agents, seed=9000 + g)
            for st in res.trajectory:
                if st.decision_type == DecisionType.DAY_SPEECH.value:
                    cnt[SpeechIntent.from_action_id(st.action_id).intent_type.value] += 1
        tot = max(sum(cnt.values()), 1)
        dists[label] = {k: cnt[k] / tot for k, _ in intents_order}

    w, h = 820, 380
    ml, mr, mt, mb = 60, 22, 54, 52
    pw, ph = w - ml - mr, h - mt - mb
    max_y = 0.45

    def ty(v: float) -> float:
        return mt + (1.0 - min(v, max_y) / max_y) * ph

    parts = [
        f'<text x="{ml}" y="25"><tspan class="panel-lbl">(a)</tspan><tspan class="panel-ttl"> 30 局发言意图分布审计：无 KL 锚定的语言漂移与 BC 锚定保持效果</tspan></text>',
    ]

    series_meta = [
        ("BC（人类模仿）", "BC (人类模仿)", C_BROWN_LIGHT),
        ("BC+RL（KL 锚定，Ours）", "BC+RL (KL锚定)", C_BLUE_MAIN),
        ("RL_scratch（无 KL 锚定）", "RL_scratch (无KL)", C_BROWN_DARK),
    ]
    for i, (disp, _, cl) in enumerate(series_meta):
        xx = ml + 10 + i * 195
        parts.append(f'<rect x="{xx}" y="35" width="12" height="9" fill="{cl}"/>')
        parts.append(f'<text x="{xx+17}" y="44" class="leg-lbl">{disp}</text>')

    # Spines & outward ticks
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" class="spine"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" class="spine"/>')
    for pct in (0.0, 0.1, 0.2, 0.3, 0.4):
        y = ty(pct)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')
    parts.append(f'<text x="16" y="{mt+ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 16 {mt+ph/2})">发言意图占比</text>')

    gw = pw / len(intents_order)
    bw = 20
    for gi, (ikey, zh) in enumerate(intents_order):
        cx = ml + (gi + 0.5) * gw
        parts.append(f'<line x1="{cx:.1f}" y1="{mt+ph}" x2="{cx:.1f}" y2="{mt+ph+4}" class="tick"/>')
        for bi, (_, skey, col) in enumerate(series_meta):
            val = dists[skey][ikey]
            bx = cx + (bi - 1.5) * (bw + 3) + 1.5
            by = ty(val)
            bh = mt + ph - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}"/>')
            if val >= 0.005:
                lbl_txt = f"{val*100:.0f}%" if val >= 0.01 else f"{val*100:.1f}%"
                parts.append(f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" text-anchor="middle" class="val">{lbl_txt}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{mt+ph+20}" text-anchor="middle" class="tk-lbl">{zh}</text>')

    return svg_wrap(w, h, "\n".join(parts))


def make_fig4_m5_nl(m5_80: dict, m5_v2: dict, m5_dual: dict, m5_cross: dict) -> str:
    """Stacked academic two-panel figure (a) Win Rates + (b) Per-game API Cost for all 7 groups (140 games, ¥6.03)."""
    w, h = 840, 480
    ml, mr = 64, 24
    pw = w - ml - mr

    s80 = m5_80["systems"]
    rows = [
        ("B0\nPureLLM", s80["B0_PureLLM"]["good_win_rate"], s80["B0_PureLLM"]["wolf_win_rate"], s80["B0_PureLLM"]["combined_win_rate"], s80["B0_PureLLM"]["avg_cost_per_game_rmb"]),
        ("B1\n+Belief", s80["B1_BeliefLLM"]["good_win_rate"], s80["B1_BeliefLLM"]["wolf_win_rate"], s80["B1_BeliefLLM"]["combined_win_rate"], s80["B1_BeliefLLM"]["avg_cost_per_game_rmb"]),
        ("B2\n+BC", s80["B2_BeliefBC"]["good_win_rate"], s80["B2_BeliefBC"]["wolf_win_rate"], s80["B2_BeliefBC"]["combined_win_rate"], s80["B2_BeliefBC"]["avg_cost_per_game_rmb"]),
        ("Ours_v1\n纯解耦 RL", s80["Ours_BeliefBCRL"]["good_win_rate"], s80["Ours_BeliefBCRL"]["wolf_win_rate"], s80["Ours_BeliefBCRL"]["combined_win_rate"], s80["Ours_BeliefBCRL"]["avg_cost_per_game_rmb"]),
        ("Ours_v2\nSim2Real", m5_v2["systems"]["Ours_BeliefBCRL"]["good_win_rate"], m5_v2["systems"]["Ours_BeliefBCRL"]["wolf_win_rate"], m5_v2["systems"]["Ours_BeliefBCRL"]["combined_win_rate"], m5_v2["systems"]["Ours_BeliefBCRL"]["avg_cost_per_game_rmb"]),
        ("Ours_Dual\n早期双系统", m5_dual["systems"]["Ours_DualSystem"]["good_win_rate"], m5_dual["systems"]["Ours_DualSystem"]["wolf_win_rate"], m5_dual["systems"]["Ours_DualSystem"]["combined_win_rate"], m5_dual["systems"]["Ours_DualSystem"]["avg_cost_per_game_rmb"]),
        ("Ours_Dual\nvs 3.8-Max", m5_cross["systems"]["Ours_DualSystem"]["good_win_rate"], m5_cross["systems"]["Ours_DualSystem"]["wolf_win_rate"], m5_cross["systems"]["Ours_DualSystem"]["combined_win_rate"], m5_cross["systems"]["Ours_DualSystem"]["avg_cost_per_game_rmb"]),
    ]

    parts: List[str] = [
        f'<text x="{ml}" y="24"><tspan class="panel-lbl">(a)</tspan><tspan class="panel-ttl"> 早期 140 局整阵营对抗与跨模型挑战胜率（7 组 × 20 局，未校准环境）</tspan></text>',
    ]

    colors = [C_BROWN_LIGHT, C_BROWN_DARK, C_BLUE_MAIN]
    names = ["好人胜率", "狼人胜率", "综合胜率"]
    for i, (nm, cl) in enumerate(zip(names, colors)):
        xx = ml + 10 + i * 115
        parts.append(f'<rect x="{xx}" y="34" width="12" height="9" fill="{cl}"/>')
        parts.append(f'<text x="{xx+17}" y="43" class="leg-lbl">{nm}</text>')

    # ---------- Panel (a): Grouped Bar Chart ----------
    t_mt, t_ph = 56, 198

    def ty_w(v: float) -> float:
        return t_mt + (1.0 - v) * t_ph

    parts.append(f'<line x1="{ml}" y1="{t_mt}" x2="{ml}" y2="{t_mt+t_ph}" class="spine"/>')
    parts.append(f'<line x1="{ml}" y1="{t_mt+t_ph}" x2="{ml+pw}" y2="{t_mt+t_ph}" class="spine"/>')
    for pct in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = ty_w(pct)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')
    parts.append(f'<text x="16" y="{t_mt+t_ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 16 {t_mt+t_ph/2})">胜率</text>')

    gw = pw / len(rows)
    bw = 21
    for gi, (gname, vg, vw, vc, _) in enumerate(rows):
        cx = ml + (gi + 0.5) * gw
        parts.append(f'<line x1="{cx:.1f}" y1="{t_mt+t_ph}" x2="{cx:.1f}" y2="{t_mt+t_ph+4}" class="tick"/>')
        for bi, val in enumerate((vg, vw, vc)):
            bx = cx + (bi - 1.5) * (bw + 3) + 1.5
            by = ty_w(val)
            bh = t_mt + t_ph - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{colors[bi]}"/>')
            parts.append(f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" text-anchor="middle" class="val">{val*100:.0f}%</text>')

    # ---------- Panel (b): Per-game API Cost Line Chart (matching reference style!) ----------
    b_mt, b_ph = 304, 118
    max_cost = 0.11

    def ty_c(c: float) -> float:
        return b_mt + (1.0 - min(c, max_cost) / max_cost) * b_ph

    parts.append(f'<text x="{ml}" y="{b_mt-14}"><tspan class="panel-lbl">(b)</tspan><tspan class="panel-ttl"> 单局实测 API 成本（元 / 局，140 局合计 ¥6.03）</tspan></text>')
    parts.append(f'<line x1="{ml}" y1="{b_mt}" x2="{ml}" y2="{b_mt+b_ph}" class="spine"/>')
    parts.append(f'<line x1="{ml}" y1="{b_mt+b_ph}" x2="{ml+pw}" y2="{b_mt+b_ph}" class="spine"/>')
    for c_tick in (0.00, 0.03, 0.06, 0.09):
        y = ty_c(c_tick)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">¥{c_tick:.2f}</text>')
    parts.append(f'<text x="16" y="{b_mt+b_ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 16 {b_mt+b_ph/2})">单局费用</text>')

    cost_pts: List[str] = []
    for gi, (gname, _, _, _, cost) in enumerate(rows):
        cx = ml + (gi + 0.5) * gw
        cy = ty_c(cost)
        cost_pts.append(f"{cx:.1f},{cy:.1f}")
        parts.append(f'<line x1="{cx:.1f}" y1="{b_mt+b_ph}" x2="{cx:.1f}" y2="{b_mt+b_ph+5}" class="tick"/>')
        for li, line in enumerate(gname.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{b_mt+b_ph+19+li*15}" text-anchor="middle" class="tk-lbl">{line}</text>')

    parts.append(f'<polyline fill="none" stroke="{C_BLUE_MAIN}" stroke-width="2.1" points="{" ".join(cost_pts)}"/>')
    for gi, (_, _, _, _, cost) in enumerate(rows):
        cx = ml + (gi + 0.5) * gw
        cy = ty_c(cost)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.8" fill="{C_BLUE_MAIN}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy-8:.1f}" text-anchor="middle" class="val-b" style="fill:{C_BLUE_MAIN}">¥{cost:.3f}</text>')

    return svg_wrap(w, h, "\n".join(parts))


def make_fig5_1v8_single_seat(s1v8_sym_src: dict, s1v8_bal: dict, s1v8_v2p: dict, s1v8_v3: dict) -> str:
    """Side-by-side academic (a) / (b) grouped bar chart matching reference aesthetic."""
    w, h = 920, 410
    sym = s1v8_sym_src["symbolic_1v8"]
    nl_b0 = s1v8_bal["systems"]["B0_PureLLM"]
    nl_v2 = s1v8_bal["systems"]["Ours_DualSystem_v2"]
    nl_v2p = s1v8_v2p["systems"]["Ours_DualSystem_v2"]
    nl_v3 = s1v8_v3["systems"]["Ours_DualSystem_v3"]

    def deck9(d: dict) -> float:
        if "deck_weighted_win_rate" in d:
            return float(d["deck_weighted_win_rate"])
        pr = d["per_role"]
        return (pr["seer"]["win_rate"] + pr["witch"]["win_rate"] + pr["hunter"]["win_rate"] + 3 * pr["werewolf"]["win_rate"] + 3 * pr["villager"]["win_rate"]) / 9.0

    parts: List[str] = []

    # ---------- Panel (a): Symbolic 1v8 (Left) ----------
    l_ml, l_mt, l_pw, l_ph = 56, 56, 368, 282

    def ty_l(v: float) -> float:
        return l_mt + (1.0 - v / 0.85) * l_ph

    parts.append(f'<text x="{l_ml}" y="25"><tspan class="panel-lbl">(a)</tspan><tspan class="panel-ttl"> 符号环境 1v8（每系统 1,000 局）</tspan></text>')
    sym_series = [("Random", C_TAUPE), ("Rule", C_BROWN_DARK), ("Belief+BC", C_BROWN_LIGHT), ("Belief+BC+RL", C_BLUE_MAIN)]
    for i, (nm, cl) in enumerate(sym_series):
        xx = l_ml + i * 92
        parts.append(f'<rect x="{xx}" y="35" width="10" height="9" fill="{cl}"/>')
        parts.append(f'<text x="{xx+14}" y="44" style="font-size:11px">{nm}</text>')

    parts.append(f'<line x1="{l_ml}" y1="{l_mt}" x2="{l_ml}" y2="{l_mt+l_ph}" class="spine"/>')
    parts.append(f'<line x1="{l_ml}" y1="{l_mt+l_ph}" x2="{l_ml+l_pw}" y2="{l_mt+l_ph}" class="spine"/>')
    for pct in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = ty_l(pct)
        parts.append(f'<line x1="{l_ml-5}" y1="{y:.1f}" x2="{l_ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{l_ml-8}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')
    parts.append(f'<text x="16" y="{l_mt+l_ph/2}" text-anchor="middle" class="ax-lbl" transform="rotate(-90 16 {l_mt+l_ph/2})">胜率 / 投准率</text>')

    sym_metrics = [
        ("总胜率", lambda d: d["overall_win_rate"]),
        ("好人4角", lambda d: d["good_roles_avg_win_rate"]),
        ("猎人", lambda d: d["per_role_win_rate"]["hunter"]),
        ("女巫", lambda d: d["per_role_win_rate"]["witch"]),
        ("投准率", lambda d: d["target_vote_acc"]),
    ]
    gw_l = l_pw / len(sym_metrics)
    bw_l = 14
    for gi, (mlbl, fn) in enumerate(sym_metrics):
        cx = l_ml + (gi + 0.5) * gw_l
        parts.append(f'<line x1="{cx:.1f}" y1="{l_mt+l_ph}" x2="{cx:.1f}" y2="{l_mt+l_ph+4}" class="tick"/>')
        for bi, (sk, col) in enumerate(sym_series):
            val = fn(sym[sk])
            bx = cx + (bi - 2) * (bw_l + 2) + 1
            by = ty_l(val)
            bh = l_mt + l_ph - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw_l}" height="{bh:.1f}" fill="{col}"/>')
            if bi == 3:
                parts.append(f'<text x="{bx+bw_l/2:.1f}" y="{by-5:.1f}" text-anchor="middle" class="val-b" style="fill:{C_BLUE_MAIN}">{val*100:.1f}%</text>')
        parts.append(f'<text x="{cx:.1f}" y="{l_mt+l_ph+20}" text-anchor="middle" class="tk-lbl">{mlbl}</text>')

    # ---------- Panel (b): Balanced Live NL 1v8 (Right) ----------
    r_ml, r_mt, r_pw, r_ph = 486, 56, 412, 282

    def ty_r(v: float) -> float:
        return r_mt + (1.0 - v) * r_ph

    parts.append(f'<text x="{r_ml}" y="25"><tspan class="panel-lbl">(b)</tspan><tspan class="panel-ttl"> 均衡中文 NL 1v8 配对消融（120 局 vs 8 PureLLM）</tspan></text>')
    nl_series = [
        ("B0", nl_b0, C_BROWN_DARK),
        ("Ours_v2(精简版)", nl_v2, C_BLUE_MAIN),
        ("Ours_v2+(长手册)", nl_v2p, C_BROWN_MID),
        ("Ours_v3(丢原文)", nl_v3, C_BROWN_LIGHT),
    ]
    x_leg_offsets = [0, 48, 166, 290]
    for (nm, _, cl), xo in zip(nl_series, x_leg_offsets):
        xx = r_ml + xo
        parts.append(f'<rect x="{xx}" y="35" width="10" height="9" fill="{cl}"/>')
        parts.append(f'<text x="{xx+14}" y="44" style="font-size:11px">{nm}</text>')

    parts.append(f'<line x1="{r_ml}" y1="{r_mt}" x2="{r_ml}" y2="{r_mt+r_ph}" class="spine"/>')
    parts.append(f'<line x1="{r_ml}" y1="{r_mt+r_ph}" x2="{r_ml+r_pw}" y2="{r_mt+r_ph}" class="spine"/>')
    for pct in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = ty_r(pct)
        parts.append(f'<line x1="{r_ml-5}" y1="{y:.1f}" x2="{r_ml}" y2="{y:.1f}" class="tick"/>')
        parts.append(f'<text x="{r_ml-8}" y="{y+4:.1f}" text-anchor="end" class="tk-lbl">{int(pct*100)}%</text>')

    # Reference dashed line (Xu et al. ICML 2024 Deck9 = 43.3%) — subtle academic gray dashed line like reference!
    y_icml = ty_r(0.433)
    parts.append(f'<line x1="{r_ml}" y1="{y_icml:.1f}" x2="{r_ml+r_pw}" y2="{y_icml:.1f}" class="refline"/>')
    parts.append(f'<text x="{r_ml+6}" y="{r_mt+15}" class="val" style="fill:{C_DASH}">--- 参考量级：Xu et al. (ICML 2024) 43.3%</text>')

    nl_metrics = [
        ("5角总胜率", lambda d: d["overall_win_rate"]),
        ("牌堆胜率\n(Deck9)", deck9),
        ("好人4角", lambda d: d["good_roles_win_rate"]),
        ("狼人", lambda d: d["per_role"]["werewolf"]["win_rate"]),
        ("单兵投准", lambda d: d["target_vote_acc"]),
    ]
    gw_r = r_pw / len(nl_metrics)
    bw_r = 15
    for gi, (mlbl, fn) in enumerate(nl_metrics):
        cx = r_ml + (gi + 0.5) * gw_r
        parts.append(f'<line x1="{cx:.1f}" y1="{r_mt+r_ph}" x2="{cx:.1f}" y2="{r_mt+r_ph+4}" class="tick"/>')
        for bi, (sk, sdata, col) in enumerate(nl_series):
            val = fn(sdata)
            bx = cx + (bi - 2) * (bw_r + 2.5) + 1.25
            by = ty_r(val)
            bh = r_mt + r_ph - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw_r}" height="{bh:.1f}" fill="{col}"/>')

        # Clean Ours_v2 value label with crisp white backdrop mask (no border) so dashed line never cuts through text
        val_v2 = fn(nl_v2)
        val_b0 = fn(nl_b0)
        bx_v2 = cx + (1 - 2) * (bw_r + 2.5) + 1.25
        by_v2 = ty_r(val_v2)
        tx_lbl = bx_v2 + bw_r / 2
        if val_b0 > val_v2 + 0.05:
            tx_lbl += 9.5
        ty_lbl = by_v2 - 6.0
        if gi == 0:
            # 40.0% sits just below the 43.3% dashed line — lift slightly above or mask cleanly
            parts.append(f'<rect x="{tx_lbl-17:.1f}" y="{ty_lbl-10:.1f}" width="34" height="12" fill="#ffffff"/>')
        elif gi == 4:
            # 65.1% sits between 80% gray and 71.7% sienna — mask cleanly above blue bar
            parts.append(f'<rect x="{tx_lbl-17:.1f}" y="{ty_lbl-10:.1f}" width="34" height="12" fill="#ffffff"/>')
        parts.append(f'<text x="{tx_lbl:.1f}" y="{ty_lbl:.1f}" text-anchor="middle" class="val-b" style="fill:{C_BLUE_MAIN}">{val_v2*100:.1f}%</text>')

        for li, ln in enumerate(mlbl.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{r_mt+r_ph+20+li*14}" text-anchor="middle" class="tk-lbl">{ln}</text>')

    return svg_wrap(w, h, "\n".join(parts))


def main() -> None:
    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    export_svg_dir = ROOT.parent / "实验报告配图_SVG"
    export_svg_dir.mkdir(parents=True, exist_ok=True)

    m3_flat = json.loads((ROOT / "outputs" / "metrics" / "m3_results.json").read_text(encoding="utf-8"))
    m3_pointer = json.loads((ROOT / "outputs" / "metrics" / "m3_results_pointer.json").read_text(encoding="utf-8"))
    m5_80 = json.loads((ROOT / "outputs" / "m5_eval" / "ablation_live_qwen35_80g" / "summary.json").read_text(encoding="utf-8"))
    m5_v2 = json.loads((ROOT / "outputs" / "m5_eval" / "ours_only_live_qwen35_ours_v2_20g" / "summary.json").read_text(encoding="utf-8"))
    m5_dual = json.loads((ROOT / "outputs" / "m5_eval" / "dual_only_live_qwen35_dual_20g" / "summary.json").read_text(encoding="utf-8"))
    m5_cross = json.loads((ROOT / "outputs" / "m5_eval" / "cross_dual_live_qwen35_dual_vs_qwen38max_20g" / "summary.json").read_text(encoding="utf-8"))
    s1v8 = json.loads((ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_1v8_90g" / "summary.json").read_text(encoding="utf-8"))
    s1v8_bal = json.loads((ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_balanced_1v8_60g" / "summary.json").read_text(encoding="utf-8"))
    s1v8_v2p = json.loads((ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_balanced_1v8_v2plus_30g" / "summary.json").read_text(encoding="utf-8"))
    s1v8_v3 = json.loads((ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_balanced_1v8_v3_30g" / "summary.json").read_text(encoding="utf-8"))

    outputs = [
        ("fig0_system_architecture.svg", "图1_系统总体架构.svg", make_fig0_system_architecture()),
        ("fig1_ppo_learning_curve.svg", "图2_PPO学习曲线.svg", make_fig1_ppo_curve(m3_pointer)),
        ("fig2_arch_and_symbolic_ablation.svg", "图3_符号环境消融对照.svg", make_fig2_symbolic_ablation(m3_flat, m3_pointer)),
        ("fig3_protocol_drift.svg", "图4_语言漂移发言意图分布.svg", make_fig3_protocol_drift()),
        ("fig5_1v8_single_seat.svg", "图5_1v8单席位主实验.svg", make_fig5_1v8_single_seat(s1v8, s1v8_bal, s1v8_v2p, s1v8_v3)),
        ("fig4_m5_nl_winrate_and_cost.svg", "图6_140局补充实验与成本.svg", make_fig4_m5_nl(m5_80, m5_v2, m5_dual, m5_cross)),
    ]
    for fname, export_name, content in outputs:
        p = fig_dir / fname
        p.write_text(content, encoding="utf-8")
        ep = export_svg_dir / export_name
        ep.write_text(content, encoding="utf-8")
        print(f"Saved {p.name} & {ep.name} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
