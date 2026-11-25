"""Build a self-contained offline interactive HTML demo showcasing real NL replays, Belief heatmaps, RL policy scores, and charts."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from werewolf_agent.features import ROLE_ORDER, encode_policy_state
from werewolf_agent.models import BeliefMLP, SeatEquivariantPolicyValueNet
from werewolf_agent.protocol import DecisionType, Phase, PlayerObservation, PublicEvent, Role, SpeechIntent

ROLE_ZH = {
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "werewolf": "狼人",
    "villager": "平民",
}


def compute_mind_snapshots(
    game_rec: Dict[str, Any],
    belief_model: BeliefMLP,
    rl_policy: SeatEquivariantPolicyValueNet,
) -> List[Dict[str, Any]]:
    """Compute BeliefMLP [9x5] heatmap + Pointer RL Top-5 vote & speech scores at every public event."""
    roles = {int(k): Role(v) for k, v in game_rec["roles"].items()}
    wolf_set = {p for p, r in roles.items() if r == Role.WEREWOLF}

    alive = {p: True for p in range(1, 10)}
    revealed: Dict[int, Role] = {}
    history: List[PublicEvent] = []
    frames: List[Dict[str, Any]] = []

    for raw in game_rec["public_events"]:
        et = raw["event_type"]
        if et == "night_announcement":
            for dp in raw["payload"].get("dead_players", []):
                alive[int(dp)] = False
        elif et == "hunter_shoot":
            revealed[int(raw["payload"]["shooter"])] = Role.HUNTER
            alive[int(raw["payload"]["target"])] = False
        elif et == "vote_result":
            ex = raw["payload"].get("exiled")
            if ex is not None:
                alive[int(ex)] = False

        if et != "game_end":
            history.append(
                PublicEvent(
                    event_id=raw["event_id"],
                    day=raw["day"],
                    phase=Phase(raw["phase"]),
                    event_type=et,
                    actor=raw["actor"],
                    payload=raw["payload"],
                )
            )

        # Compute observer-specific minds for all 9 seats so user can click ANY seat to inspect its brain!
        seat_minds: Dict[str, Any] = {}
        alive_t = tuple(p for p in range(1, 10) if alive[p])
        dead_t = tuple(p for p in range(1, 10) if not alive[p])
        vote_mask = np.array([True] + [alive[p] for p in range(1, 10)], dtype=bool)

        for pid in range(1, 10):
            r = roles[pid]
            obs = PlayerObservation(
                game_id="demo",
                mode="nl",
                day=max(int(raw["day"]), 1),
                phase=Phase.DAY_VOTE,
                player_id=pid,
                my_role=r,
                player_count=9,
                alive_players=alive_t,
                dead_players=dead_t,
                revealed_roles=dict(revealed),
                public_events=tuple(history),
                werewolf_teammates=tuple(sorted(wolf_set - {pid})) if r == Role.WEREWOLF else (),
                vote_candidates=alive_t,
            )
            bel = belief_model.predict_obs(obs, apply_constraints=True)
            p_state = encode_policy_state(obs, bel)
            v_probs, _, v_vals, _ = rl_policy.forward_head(p_state[None, :], DecisionType.DAY_VOTE, vote_mask[None, :])
            vp0 = v_probs[0]
            top_votes = sorted(
                [(i, float(vp0[i])) for i in range(10) if vote_mask[i]],
                key=lambda x: x[1],
                reverse=True,
            )[:4]
            seat_minds[str(pid)] = {
                "value": round(float(v_vals[0]), 3),
                "belief": [[round(float(bel[i, j]), 3) for j in range(5)] for i in range(9)],
                "top_votes": [
                    {"label": ("弃票" if a == 0 else f"投 {a}号"), "prob": round(prob, 3)}
                    for a, prob in top_votes
                ],
            }

        frames.append(
            {
                "event_id": raw["event_id"],
                "day": raw["day"],
                "phase": raw["phase"],
                "event_type": et,
                "actor": raw["actor"],
                "payload": raw["payload"],
                "alive": list(alive_t),
                "revealed": {str(k): v.value for k, v in revealed.items()},
                "seat_minds": seat_minds,
            }
        )
    return frames


def select_curated_games() -> List[tuple[str, Path]]:
    picks: List[tuple[str, Path]] = []
    s1v8_bal = ROOT / "outputs" / "m5_eval" / "single_seat_live_qwen35_balanced_1v8_60g" / "games"
    candidates = [
        ("【主实验·均衡 1v8】Ours_DualSystem_v2 抽中预言家胜局（1 vs 8 PureLLM）", s1v8_bal / "Ours_DualSystem_v2_seer_*.json", True),
        ("【主实验·均衡 1v8】Ours_DualSystem_v2 抽中女巫胜局（1 vs 8 PureLLM）", s1v8_bal / "Ours_DualSystem_v2_witch_*.json", True),
        ("【主实验·均衡 1v8】Ours_DualSystem_v2 抽中平民胜局（1 vs 8 PureLLM）", s1v8_bal / "Ours_DualSystem_v2_villager_*.json", True),
        ("【主实验·均衡 1v8】Ours_DualSystem_v2 抽中狼人胜局（1 vs 8 PureLLM）", s1v8_bal / "Ours_DualSystem_v2_werewolf_*.json", True),
        ("【补充·阵营对抗】Ours_v1 好人队胜局 (qwen3.5-27b)", ROOT / "outputs" / "m5_eval" / "ablation_live_qwen35_80g" / "games" / "Ours_BeliefBCRL_as_good_*.json", True),
        ("【补充·阵营对抗】Ours_DualSystem 狼人队屠神胜局 (qwen3.5-27b)", ROOT / "outputs" / "m5_eval" / "dual_only_live_qwen35_dual_20g" / "games" / "Ours_DualSystem_as_wolf_*.json", True),
        ("【补充·跨模型】Ours_Dual vs qwen3.8-max 狼人队胜局", ROOT / "outputs" / "m5_eval" / "cross_dual_live_qwen35_dual_vs_qwen38max_20g" / "games" / "Ours_DualSystem_as_wolf_*.json", True),
    ]
    for title, pattern, want_win in candidates:
        matched = sorted(Path(p) for p in glob.glob(str(pattern)))
        chosen = None
        for p in matched:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("eval_won") == want_win:
                chosen = p
                break
        if chosen is None and matched:
            chosen = matched[0]
        if chosen is not None:
            picks.append((title, chosen))
    return picks


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>狼人杀双系统智能体（Belief + Pointer RL + 冻结 LLM）交互演示与实验看板</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#f1f5f9; --accent:#38bdf8; --good:#10b981; --wolf:#ef4444; --warn:#f59e0b; --border:#334155; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",sans-serif; background:var(--bg); color:var(--text); }
  header { padding:16px 24px; background:#020617; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }
  header h1 { margin:0; font-size:19px; color:#f8fafc; }
  .tabs { display:flex; gap:8px; }
  .tab-btn { background:var(--card); color:var(--text); border:1px solid var(--border); padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  .tab-btn.active { background:var(--accent); color:#020617; font-weight:700; border-color:var(--accent); }
  .container { max-width:1380px; margin:0 auto; padding:18px; }
  .hidden { display:none !important; }
  .toolbar { display:flex; gap:12px; flex-wrap:wrap; align-items:center; background:var(--card); padding:12px 16px; border-radius:10px; border:1px solid var(--border); margin-bottom:16px; }
  select, button { background:#0f172a; color:var(--text); border:1px solid var(--border); padding:7px 12px; border-radius:7px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .grid-main { display:grid; grid-template-columns: 1.15fr 0.85fr; gap:16px; }
  @media (max-width: 980px) { .grid-main { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; }
  .card h2 { margin:0 0 12px 0; font-size:15px; color:var(--accent); display:flex; justify-content:space-between; align-items:center; }
  .seats { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-bottom:14px; }
  .seat { background:#0f172a; border:2px solid var(--border); border-radius:10px; padding:9px 10px; cursor:pointer; transition:0.15s; }
  .seat:hover { border-color:var(--accent); }
  .seat.selected { border-color:#38bdf8; box-shadow:0 0 0 2px rgba(56,189,248,0.28); }
  .seat.target-seat { background:linear-gradient(135deg, #172554, #0f172a); border-color:#f59e0b; }
  .seat.dead { opacity:0.48; }
  .seat-top { display:flex; justify-content:space-between; align-items:center; font-weight:700; font-size:14px; }
  .badge { font-size:11px; padding:2px 6px; border-radius:5px; font-weight:600; }
  .badge.wolf { background:rgba(239,68,68,0.2); color:#fca5a5; }
  .badge.good { background:rgba(16,185,129,0.2); color:#6ee7b7; }
  .badge.unk { background:rgba(148,163,184,0.2); color:#cbd5e1; }
  .seat-sub { font-size:11px; color:var(--muted); margin-top:4px; }
  .timeline { max-height:460px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; padding-right:4px; }
  .ev { padding:10px 12px; border-radius:8px; background:#0f172a; border:1px solid var(--border); font-size:13px; line-height:1.55; cursor:pointer; }
  .ev.active { border-color:var(--accent); background:#172554; }
  .ev-meta { font-size:11px; color:var(--muted); margin-bottom:4px; display:flex; justify-content:space-between; }
  table.heat { width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }
  table.heat th, table.heat td { border:1px solid var(--border); padding:6px 5px; text-align:center; }
  table.heat th { background:#0f172a; color:var(--muted); }
  .bar-row { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:12px; }
  .bar-lbl { width:76px; text-align:right; color:#cbd5e1; }
  .bar-track { flex:1; height:16px; background:#0f172a; border-radius:4px; overflow:hidden; }
  .bar-fill { height:100%; background:linear-gradient(90deg, #38bdf8, #2563eb); }
  .bar-val { width:44px; font-weight:700; font-size:11px; }
  .charts-grid { display:grid; grid-template-columns:1fr; gap:18px; }
  .chart-box { background:#ffffff; border-radius:12px; padding:10px; overflow-x:auto; }
  .chart-box svg { max-width:100%; height:auto; display:block; margin:0 auto; }
  .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
  @media (max-width:780px){ .kpi-row{grid-template-columns:repeat(2,1fr);} }
  .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
  .kpi .num { font-size:22px; font-weight:700; color:var(--accent); }
  .kpi .desc { font-size:12px; color:var(--muted); margin-top:4px; }
</style>
</head>
<body>
<header>
  <h1>狼人杀角色智能体：Belief + 指针 RL + 冻结 LLM 双系统演示</h1>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('replay', this)">交互式对局复盘 &amp; 智能体心智</button>
    <button class="tab-btn" onclick="switchTab('charts', this)">论文实验图表看板（M3 + M5）</button>
  </div>
</header>

<div class="container">
  <div id="tab-replay">
    <div class="toolbar">
      <label>选择真实对局：<select id="gameSelect" onchange="loadGame(this.value)"></select></label>
      <button onclick="stepFrame(-1)">◀ 上一步</button>
      <button onclick="stepFrame(1)">下一步 ▶</button>
      <span id="frameIndicator" style="font-size:13px;color:var(--muted);"></span>
      <label style="margin-left:auto;font-size:13px;cursor:pointer;">
        <input type="checkbox" id="godViewToggle" checked onchange="renderAll()" /> 上帝视角（显示全部底牌）
      </label>
    </div>

    <div class="grid-main">
      <div class="card">
        <h2><span>九人席位与发言回放（点击任意座位切换第一人称心智视角）</span><span id="gameBadge" class="badge good"></span></h2>
        <div id="seatsGrid" class="seats"></div>
        <div id="timeline" class="timeline"></div>
      </div>

      <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="card">
          <h2><span id="mindTitle">智能体实时内部信念热力图（BeliefMLP）</span><span id="valBadge" class="badge unk"></span></h2>
          <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">
            展示所选座位在当前事件时点对 1–9 号玩家的 5 类身份后验概率预测（已施加本视角硬约束）。
          </div>
          <table class="heat">
            <thead><tr><th>席位</th><th>预言家</th><th>女巫</th><th>猎人</th><th>狼人</th><th>平民</th></tr></thead>
            <tbody id="beliefBody"></tbody>
          </table>
        </div>

        <div class="card">
          <h2><span>System-2 指针 RL 策略候选投票分布</span></h2>
          <div id="voteBars"></div>
          <div id="metaBox" style="margin-top:12px;font-size:12px;color:var(--muted);line-height:1.6;"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-charts" class="hidden">
    <div class="kpi-row">
      <div class="kpi"><div class="num">40.0% / 44.4%</div><div class="desc">均衡 1v8 主实验 5角色/牌堆胜率（PureLLM 26.7%/37.0%，超 ICML 2024）</div></div>
      <div class="kpi"><div class="num">33.3% (4×)</div><div class="desc">1v8 好人 4 角色全线胜率（PureLLM 仅 8.3%）</div></div>
      <div class="kpi"><div class="num">65.7% / 352 局</div><div class="desc">跨领域 BeliefMLP_NL 准确率 / 全部 352 局 0 报错</div></div>
      <div class="kpi"><div class="num">¥14.84</div><div class="desc">全部 352 局真实中文 API 总花费（零模型微调费）</div></div>
    </div>
    <div class="charts-grid">
      <div class="chart-box">__FIG5__</div>
      <div class="chart-box">__FIG1__</div>
      <div class="chart-box">__FIG2__</div>
      <div class="chart-box">__FIG3__</div>
      <div class="chart-box">__FIG4__</div>
    </div>
  </div>
</div>

<script>
const GAMES = __GAMES_JSON__;
const ROLE_ZH = {seer:"预言家", witch:"女巫", hunter:"猎人", werewolf:"狼人", villager:"平民"};
let curGameIdx = 0;
let curFrameIdx = 0;
let selectedSeat = 1;

function switchTab(name, btn) {
  document.getElementById("tab-replay").classList.toggle("hidden", name !== "replay");
  document.getElementById("tab-charts").classList.toggle("hidden", name !== "charts");
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

function init() {
  const sel = document.getElementById("gameSelect");
  GAMES.forEach((g, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${idx+1}. ${g.title}（${g.total_days}天，${g.winner === "good" ? "好人胜" : "狼人胜"}）`;
    sel.appendChild(opt);
  });
  loadGame(0);
}

function loadGame(idx) {
  curGameIdx = Number(idx);
  curFrameIdx = Math.min(10, GAMES[curGameIdx].frames.length - 1);
  selectedSeat = GAMES[curGameIdx].target_seat || 1;
  renderAll();
}

function stepFrame(delta) {
  const g = GAMES[curGameIdx];
  curFrameIdx = Math.max(0, Math.min(g.frames.length - 1, curFrameIdx + delta));
  renderAll();
}

function heatBg(p, isWolfCol) {
  const a = Math.min(0.85, Math.max(0.05, p));
  return isWolfCol ? `rgba(239,68,68,${a.toFixed(2)})` : `rgba(56,189,248,${(a*0.75).toFixed(2)})`;
}

function renderAll() {
  const g = GAMES[curGameIdx];
  const f = g.frames[curFrameIdx];
  const godView = document.getElementById("godViewToggle").checked;

  document.getElementById("frameIndicator").textContent = `事件 ${curFrameIdx+1} / ${g.frames.length}（第${f.day}天 ${f.phase}）`;
  const gb = document.getElementById("gameBadge");
  gb.textContent = (g.winner === "good" ? "终局：好人胜" : "终局：狼人胜") + ` | 费用 ¥${g.cost_rmb}`;
  gb.className = "badge " + (g.winner === "good" ? "good" : "wolf");

  // Seats grid
  const sg = document.getElementById("seatsGrid");
  sg.innerHTML = "";
  for (let p = 1; p <= 9; p++) {
    const ps = String(p);
    const isAlive = f.alive.includes(p);
    const role = g.roles[ps];
    const showRole = godView || p === selectedSeat || f.revealed[ps];
    const isTarget = g.target_seat && p === g.target_seat;
    const div = document.createElement("div");
    div.className = "seat" + (isTarget ? " target-seat" : "") + (isAlive ? "" : " dead") + (p === selectedSeat ? " selected" : "");
    div.onclick = () => { selectedSeat = p; renderAll(); };
    const badgeCls = !showRole ? "unk" : (role === "werewolf" ? "wolf" : "good");
    const badgeTxt = showRole ? ROLE_ZH[role] : "未知";
    div.innerHTML = `
      <div class="seat-top"><span>${isTarget ? "★ " : ""}${p}号 ${isAlive ? "" : "☠"}</span><span class="badge ${badgeCls}">${badgeTxt}</span></div>
      <div class="seat-sub">${g.seat_agents[ps] || ""}${isTarget ? "（被测单席位）" : ""}</div>
    `;
    sg.appendChild(div);
  }

  // Timeline
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  g.frames.forEach((ev, idx) => {
    const div = document.createElement("div");
    div.className = "ev" + (idx === curFrameIdx ? " active" : "");
    div.onclick = () => { curFrameIdx = idx; renderAll(); };
    let body = "";
    if (ev.event_type === "speech") {
      body = `<strong>${ev.actor}号发言</strong>（意图: <code>${ev.payload.intent.intent_type}${ev.payload.intent.target ? "→"+ev.payload.intent.target+"号" : ""}</code>）：<br/>${ev.payload.text}`;
    } else if (ev.event_type === "night_announcement") {
      body = `🌙 <strong>天亮公告</strong>：${ev.payload.text}`;
    } else if (ev.event_type === "vote_result") {
      body = `🗳 <strong>投票结算</strong>：${ev.payload.exiled ? ev.payload.exiled+"号被放逐" : "平票/无人出局"}（票型：${JSON.stringify(ev.payload.votes)}）`;
    } else if (ev.event_type === "hunter_shoot") {
      body = `🔫 <strong>猎人开枪</strong>：${ev.payload.shooter}号翻牌猎人带走 ${ev.payload.target}号`;
    } else {
      body = `🏁 <strong>游戏结束</strong>：${ev.payload.winner}（${ev.payload.reason}）`;
    }
    div.innerHTML = `<div class="ev-meta"><span>#${idx+1} 第${ev.day}天</span><span>${ev.event_type}</span></div>${body}`;
    tl.appendChild(div);
  });

  // Mind panel
  const mind = f.seat_minds[String(selectedSeat)];
  document.getElementById("mindTitle").textContent = `${selectedSeat}号（${ROLE_ZH[g.roles[String(selectedSeat)]]}）实时信念热力图`;
  document.getElementById("valBadge").textContent = `状态估值 V(s) = ${mind.value}`;

  const bb = document.getElementById("beliefBody");
  bb.innerHTML = "";
  for (let i = 0; i < 9; i++) {
    const row = mind.belief[i];
    const tr = document.createElement("tr");
    let cells = `<td><strong>${i+1}号</strong>${(i+1)===selectedSeat ? "(我)" : ""}</td>`;
    for (let j = 0; j < 5; j++) {
      cells += `<td style="background:${heatBg(row[j], j===3)}">${(row[j]*100).toFixed(0)}%</td>`;
    }
    tr.innerHTML = cells;
    bb.appendChild(tr);
  }

  const vb = document.getElementById("voteBars");
  vb.innerHTML = mind.top_votes.map(item => `
    <div class="bar-row">
      <div class="bar-lbl">${item.label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(item.prob*100).toFixed(1)}%"></div></div>
      <div class="bar-val">${(item.prob*100).toFixed(1)}%</div>
    </div>
  `).join("");

  document.getElementById("metaBox").innerHTML =
    `对局配置：主模型 <code>${g.main_model}</code> vs 对手 <code>${g.opp_model}</code> | 评测阵营：<code>${g.side}</code> | 死因：<code>${JSON.stringify(g.death_causes)}</code>`;
}

init();
</script>
</body>
</html>
"""


def main() -> None:
    belief_path = ROOT / "outputs" / "models" / "belief_mlp_nl.npz"
    if not belief_path.is_file():
        belief_path = ROOT / "outputs" / "models" / "belief_mlp.npz"
    belief = BeliefMLP.load(belief_path)
    rl_policy = SeatEquivariantPolicyValueNet.load(ROOT / "outputs" / "models" / "policy_pointer_bc_rl.npz")

    curated = select_curated_games()
    games_payload = []
    for title, path in curated:
        rec = json.loads(path.read_text(encoding="utf-8"))
        frames = compute_mind_snapshots(rec, belief, rl_policy)
        cost = round(
            rec["usage_main"]["estimated_cost_rmb"] + rec["usage_opp"]["estimated_cost_rmb"], 4
        )
        games_payload.append(
            {
                "title": title,
                "system": rec["system"],
                "side": rec["side"],
                "target_seat": rec.get("target_seat"),
                "main_model": rec["main_model"],
                "opp_model": rec["opp_model"],
                "winner": rec["winner"],
                "total_days": rec["total_days"],
                "cost_rmb": cost,
                "roles": rec["roles"],
                "seat_agents": rec["seat_agents"],
                "death_causes": rec["death_causes"],
                "frames": frames,
            }
        )

    fig_dir = ROOT / "outputs" / "figures"
    html = HTML_TEMPLATE
    html = html.replace("__GAMES_JSON__", json.dumps(games_payload, ensure_ascii=False))
    html = html.replace("__FIG5__", (fig_dir / "fig5_1v8_single_seat.svg").read_text(encoding="utf-8"))
    html = html.replace("__FIG1__", (fig_dir / "fig1_ppo_learning_curve.svg").read_text(encoding="utf-8"))
    html = html.replace("__FIG2__", (fig_dir / "fig2_arch_and_symbolic_ablation.svg").read_text(encoding="utf-8"))
    html = html.replace("__FIG3__", (fig_dir / "fig3_protocol_drift.svg").read_text(encoding="utf-8"))
    html = html.replace("__FIG4__", (fig_dir / "fig4_m5_nl_winrate_and_cost.svg").read_text(encoding="utf-8"))

    out_path = ROOT / "outputs" / "demo" / "werewolf_agent_demo.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Built offline interactive demo ({len(games_payload)} curated games): {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
