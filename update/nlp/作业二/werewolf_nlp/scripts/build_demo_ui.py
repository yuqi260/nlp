"""Build a self-contained offline HTML demo UI from saved snapshots and predictions.

Double-click outputs/demo/werewolf_demo.html in any modern browser to inspect:
- Validation (12 games / 20 snapshots) and Test (14 games / 23 snapshots) games
- Day 1 -> Day 2 role inference updates with change highlighting
- Clickable eNNNN event citations linking directly to original public speeches/votes
- Deterministic faithfulness check badges and optional truth reveal toggle
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from werewolf_nlp.behavior import extract_hard_facts
from werewolf_nlp.faithfulness import audit_record_faithfulness, _strip_fence
from run_baseline import read_jsonl

MODEL_RUNS = {
    'validation': [
        ('direct_on', 'Qwen3.5-27B · 直接预测（开思考）', 'outputs/predictions/qwen3_5_27b_validation_on_v1.jsonl'),
        ('hyp_v1_on', 'Qwen3.5-27B · 竞争假设 v1（开思考）', 'outputs/predictions/qwen3_5_27b_validation_hypotheses_thinking_complete_v1.jsonl'),
        ('hyp_v2_on', 'Qwen3.5-27B · 竞争假设 v2（开思考）', 'outputs/predictions/qwen3_5_27b_validation_hypotheses_v2_thinking_v1.jsonl'),
        ('thinking_vote', 'Qwen3.5-27B · 三次思考投票', 'outputs/predictions/qwen3_5_27b_validation_thinking_vote_v1.jsonl'),
        ('direct_off', 'Qwen3.5-27B · 直接预测（关思考）', 'outputs/predictions/qwen3_5_27b_validation_off_v1.jsonl'),
        ('naive_bayes', '字符二元组朴素贝叶斯（传统基线）', 'outputs/predictions/naive_bayes_validation_v2.jsonl'),
    ],
    'test': [
        ('direct_on', 'Qwen3.5-27B · 直接预测（开思考·主基线）', 'outputs/predictions/qwen3_5_27b_test_thinking_direct_complete_v1.jsonl'),
        ('hyp_a_on', 'Qwen3.5-27B · 竞争假设 v1（开思考·次1）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_a_complete_v1.jsonl'),
        ('hyp_b_on', 'Qwen3.5-27B · 竞争假设 v1（开思考·次2）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_b_complete_v1.jsonl'),
        ('thinking_vote', 'Qwen3.5-27B · 三次思考投票', 'outputs/predictions/qwen3_5_27b_test_thinking_vote_v1.jsonl'),
        ('direct_off', 'Qwen3.5-27B · 直接预测（关思考）', 'outputs/predictions/qwen3_5_27b_test_v1.jsonl'),
    ],
}


def parse_hypotheses(raw_output):
    if not isinstance(raw_output, str) or not raw_output.strip():
        return []
    try:
        obj = json.loads(_strip_fence(raw_output))
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    hyps = obj.get('hypotheses')
    return hyps if isinstance(hyps, list) else []


def build_bundle():
    snapshots = read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
    labels = read_jsonl(ROOT / 'data/processed/labels.jsonl')
    splits = {x['game_id']: x['split'] for x in read_jsonl(ROOT / 'data/splits/games.jsonl')}

    truth_by_game = defaultdict(dict)
    for row in labels:
        truth_by_game[row['game_id']][str(row['player_id'])] = row['true_role']

    snapshot_map = {s['snapshot_id']: s for s in snapshots}

    bundle = {'splits': {'test': {'runs': [], 'games': {}}, 'validation': {'runs': [], 'games': {}}}}
    for split_name in ('test', 'validation'):
        selected = [s for s in snapshots if splits.get(s['game_id']) == split_name]
        games = defaultdict(list)
        for s in selected:
            games[s['game_id']].append(s)

        run_preds = {}
        for run_key, run_label, rel_path in MODEL_RUNS[split_name]:
            path = ROOT / rel_path
            if not path.exists():
                continue
            bundle['splits'][split_name]['runs'].append({'key': run_key, 'label': run_label})
            records = read_jsonl(path)
            run_preds[run_key] = {}
            for rec in records:
                sid = rec['snapshot_id']
                if sid not in snapshot_map:
                    continue
                audit = audit_record_faithfulness(rec, snapshot_map[sid])
                run_preds[run_key][sid] = {
                    'status': rec.get('status'),
                    'role_counts_valid': rec.get('role_counts_valid'),
                    'assignments': {str(x['player_id']): x['role'] for x in rec.get('assignments', [])},
                    'hypotheses': parse_hypotheses(rec.get('raw_output')),
                    'faithfulness': {
                        'clean': audit['clean'],
                        'cited_event_count': audit['cited_event_count'],
                        'flags': audit['flags'],
                        'invalid_event_citations': audit['invalid_event_citations'],
                    },
                }

        for gid, gsnaps in sorted(games.items()):
            gsnaps.sort(key=lambda x: x['day'])
            snap_entries = []
            for s in gsnaps:
                sid = s['snapshot_id']
                hard_facts = extract_hard_facts(s)
                preds = {rk: run_preds[rk].get(sid) for rk in run_preds}
                snap_entries.append({
                    'snapshot_id': sid,
                    'day': s['day'],
                    'confirmed_hunters': [h['confirmed_hunter'] for h in hard_facts['confirmed_hunters']],
                    'hard_facts': hard_facts,
                    'events': [
                        {
                            'event_id': e['event_id'].rsplit(':', 1)[-1],
                            'day': e['day'],
                            'type': e['type'],
                            'speaker': e.get('speaker'),
                            'text': e.get('text'),
                            'votes': e.get('votes'),
                            'eliminated_player': e.get('eliminated_player'),
                            'tied_players': e.get('tied_players'),
                            'target': e.get('target'),
                        }
                        for e in s['public_events']
                    ],
                    'predictions': preds,
                })
            bundle['splits'][split_name]['games'][gid] = {
                'game_id': gid,
                'truth': truth_by_game[gid],
                'snapshots': snap_entries,
            }
    return bundle


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>狼人杀身份推断系统 · 逐轮推断与依据溯源演示</title>
<style>
  :root {
    --bg: #f6f7f9;
    --surface: #ffffff;
    --border: #e2e6ee;
    --text: #1c2434;
    --muted: #616e84;
    --accent: #2f5bea;
    --accent-soft: #edf2ff;
    --wolf: #d9383a;
    --wolf-bg: #fff1f0;
    --seer: #6f42c1;
    --seer-bg: #f5f0ff;
    --witch: #0b875b;
    --witch-bg: #ecfdf5;
    --hunter: #c96a00;
    --hunter-bg: #fff7e6;
    --villager: #3b5998;
    --villager-bg: #eef4ff;
    --ok: #15803d;
    --warn: #b45309;
    --err: #b91c1c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px 24px; background: var(--surface); border-bottom: 1px solid var(--border); }
  select, button { font-size: 13px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: #fff; color: var(--text); cursor: pointer; }
  button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .toggle-truth { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); cursor: pointer; }
  main { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; padding: 16px 24px; max-width: 1480px; margin: 0 auto; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .panel h2 { margin: 0 0 12px; font-size: 15px; display: flex; align-items: center; justify-content: space-between; }
  .metrics-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .pill { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: var(--bg); border: 1px solid var(--border); }
  .pill.ok { color: var(--ok); background: #ecfdf5; border-color: #bbf7d0; }
  .pill.warn { color: var(--warn); background: #fffbeb; border-color: #fde68a; }
  .grid9 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .card { border: 1px solid var(--border); border-radius: 8px; padding: 10px; background: #fff; position: relative; transition: border-color .15s; }
  .card.changed { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
  .card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-weight: 600; font-size: 14px; }
  .badge { display: inline-block; font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; }
  .role-werewolf { color: var(--wolf); background: var(--wolf-bg); }
  .role-seer { color: var(--seer); background: var(--seer-bg); }
  .role-witch { color: var(--witch); background: var(--witch-bg); }
  .role-hunter { color: var(--hunter); background: var(--hunter-bg); }
  .role-villager { color: var(--villager); background: var(--villager-bg); }
  .tag { font-size: 11px; padding: 1px 6px; border-radius: 4px; background: var(--bg); color: var(--muted); }
  .tag.delta { background: var(--accent-soft); color: var(--accent); }
  .tag.masked { background: #fff7e6; color: var(--hunter); }
  .truth-row { margin-top: 6px; font-size: 12px; padding-top: 6px; border-top: 1px dashed var(--border); display: flex; justify-content: space-between; }
  .truth-ok { color: var(--ok); font-weight: 600; }
  .truth-bad { color: var(--err); font-weight: 600; }
  .hyp-box { margin-top: 14px; border-top: 1px solid var(--border); padding-top: 12px; }
  .hyp-item { background: var(--bg); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; line-height: 1.55; }
  .hyp-item strong { color: var(--text); }
  .eid-link { display: inline-block; padding: 0 5px; margin: 0 1px; border-radius: 4px; background: var(--accent-soft); color: var(--accent); font-family: monospace; font-size: 12px; cursor: pointer; text-decoration: none; }
  .eid-link:hover { background: var(--accent); color: #fff; }
  .flag-box { margin-top: 10px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 8px 12px; font-size: 12px; color: var(--warn); }
  .events-list { max-height: 74vh; overflow-y: auto; padding-right: 4px; }
  .ev { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; font-size: 13px; line-height: 1.5; transition: background .25s, border-color .25s; }
  .ev.highlight { background: var(--accent-soft); border-color: var(--accent); }
  .ev-meta { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .ev-id { font-family: monospace; font-weight: 600; color: var(--accent); }
  .ev-type-announcement { border-left: 3px solid #64748b; }
  .ev-type-vote_results { border-left: 3px solid var(--accent); }
  .ev-type-shoot { border-left: 3px solid var(--hunter); background: #fffbf2; }
  .ev-type-speech { border-left: 3px solid #cbd5e1; }
</style>
</head>
<body>
<header>
  <div>
    <h1>狼人杀身份推断系统 · 逐轮推断与依据溯源演示</h1>
    <div class="sub">旁观者视角 · 仅展示截止时点前的公开事件 · 点击依据中的事件编号可定位原文</div>
  </div>
</header>

<div class="controls">
  <label>数据集：
    <select id="splitSelect">
      <option value="test">测试集（14局 / 23快照）</option>
      <option value="validation">验证集（12局 / 20快照）</option>
    </select>
  </label>
  <label>对局：
    <select id="gameSelect"></select>
  </label>
  <label>模型方案：
    <select id="runSelect"></select>
  </label>
  <div id="dayButtons" style="display:flex;gap:6px;"></div>
  <label class="toggle-truth">
    <input type="checkbox" id="showTruth" />
    揭晓真实身份（离线评价对比）
  </label>
</div>

<main>
  <section class="panel">
    <h2>
      <span>九人身份推断结果</span>
      <span id="snapLabel" style="font-size:12px;color:var(--muted);font-weight:400;"></span>
    </h2>
    <div class="metrics-bar" id="metricsBar"></div>
    <div class="grid9" id="playerGrid"></div>
    <div class="hyp-box" id="hypBox"></div>
  </section>

  <section class="panel">
    <h2>
      <span>公开事件时间线（可溯源原文）</span>
      <span id="evCount" style="font-size:12px;color:var(--muted);font-weight:400;"></span>
    </h2>
    <div class="events-list" id="eventsList"></div>
  </section>
</main>

<script>
const BUNDLE = __BUNDLE_JSON__;
const ROLE_ZH = { seer: '预言家', witch: '女巫', hunter: '猎人', werewolf: '狼人', villager: '平民' };
const TYPE_ZH = { announcement: '系统公告', speech: '玩家发言', vote_results: '公开票型', shoot: '猎人开枪' };

const state = { split: 'test', gameId: null, runKey: null, dayIdx: 0, showTruth: false };

const splitEl = document.getElementById('splitSelect');
const gameEl = document.getElementById('gameSelect');
const runEl = document.getElementById('runSelect');
const dayBtnsEl = document.getElementById('dayButtons');
const truthEl = document.getElementById('showTruth');

function escapeHtml(str) {
  return String(str || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function linkifyEventIds(text) {
  return escapeHtml(text).replace(/\be\d{4}\b/g, m => `<a class="eid-link" onclick="jumpToEvent('${m}')">${m}</a>`);
}

window.jumpToEvent = function(eid) {
  const el = document.getElementById('ev-' + eid);
  if (!el) return;
  document.querySelectorAll('.ev.highlight').forEach(x => x.classList.remove('highlight'));
  el.classList.add('highlight');
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

function populateSelectors() {
  const sp = BUNDLE.splits[state.split];
  const gids = Object.keys(sp.games);
  gameEl.innerHTML = gids.map(g => `<option value="${g}">${g} (${sp.games[g].snapshots.length}快照)</option>`).join('');
  state.gameId = gids[0];
  runEl.innerHTML = sp.runs.map(r => `<option value="${r.key}">${escapeHtml(r.label)}</option>`).join('');
  state.runKey = sp.runs[0] ? sp.runs[0].key : null;
  state.dayIdx = 0;
  render();
}

function render() {
  const sp = BUNDLE.splits[state.split];
  const game = sp.games[state.gameId];
  if (!game) return;
  if (state.dayIdx >= game.snapshots.length) state.dayIdx = 0;
  const snap = game.snapshots[state.dayIdx];
  const prevSnap = state.dayIdx > 0 ? game.snapshots[state.dayIdx - 1] : null;
  const predObj = snap.predictions[state.runKey];
  const prevPredObj = prevSnap ? prevSnap.predictions[state.runKey] : null;

  dayBtnsEl.innerHTML = game.snapshots.map((s, idx) =>
    `<button class="${idx === state.dayIdx ? 'active' : ''}" onclick="selectDay(${idx})">第 ${s.day} 天</button>`
  ).join('');

  document.getElementById('snapLabel').textContent = snap.snapshot_id;

  // Metrics bar
  const pills = [];
  if (predObj && predObj.status === 'ok') {
    pills.push(`<span class="pill ${predObj.role_counts_valid ? 'ok' : 'warn'}">角色配额：${predObj.role_counts_valid ? '合法' : '不合法'}</span>`);
    const f = predObj.faithfulness;
    if (f) {
      pills.push(`<span class="pill ${f.clean ? 'ok' : 'warn'}">依据自动校验：${f.clean ? '无规则告警' : (f.flags.length + f.invalid_event_citations.length) + ' 项告警'}</span>`);
      if (f.cited_event_count > 0) {
        pills.push(`<span class="pill">引用事件数：${f.cited_event_count}</span>`);
      }
    }
    if (state.showTruth) {
      let ok = 0, total = 0;
      for (let p = 1; p <= 9; p++) {
        if (snap.confirmed_hunters.includes(p)) continue;
        total++;
        if (predObj.assignments[String(p)] === game.truth[String(p)]) ok++;
      }
      pills.push(`<span class="pill ok">未知身份命中：${ok}/${total} (${(ok/total*100).toFixed(1)}%)</span>`);
    }
  } else {
    pills.push('<span class="pill warn">当前方案无该快照预测记录</span>');
  }
  document.getElementById('metricsBar').innerHTML = pills.join('');

  // 9-player cards
  let cardsHtml = '';
  for (let p = 1; p <= 9; p++) {
    const pid = String(p);
    const role = predObj && predObj.assignments ? predObj.assignments[pid] : null;
    const prevRole = prevPredObj && prevPredObj.assignments ? prevPredObj.assignments[pid] : null;
    const changed = Boolean(prevRole && role && prevRole !== role);
    const isMasked = snap.confirmed_hunters.includes(p);
    const trueRole = game.truth[pid];
    const match = role === trueRole;

    cardsHtml += `
      <div class="card ${changed ? 'changed' : ''}">
        <div class="card-head">
          <span>${p} 号玩家</span>
          ${isMasked ? '<span class="tag masked">已公开开枪(掩码)</span>' : (changed ? `<span class="tag delta">较前日变化：${ROLE_ZH[prevRole]}→</span>` : '')}
        </div>
        <div>
          ${role ? `<span class="badge role-${role}">${ROLE_ZH[role]} (${role})</span>` : '<span class="tag">无预测</span>'}
        </div>
        ${state.showTruth ? `
          <div class="truth-row">
            <span>真实：${ROLE_ZH[trueRole]}</span>
            <span class="${match ? 'truth-ok' : 'truth-bad'}">${match ? '✓ 正确' : '✗ 误判'}</span>
          </div>` : ''}
      </div>`;
  }
  document.getElementById('playerGrid').innerHTML = cardsHtml;

  // Hypotheses & faithfulness flags
  let hypHtml = '';
  if (predObj && predObj.hypotheses && predObj.hypotheses.length > 0) {
    hypHtml += '<div style="font-size:13px;font-weight:600;margin-bottom:8px;">竞争假设与原文依据（点击蓝色编号定位原文）</div>';
    for (const h of predObj.hypotheses) {
      hypHtml += `<div class="hyp-item">
        <div><strong>候选预言家：${escapeHtml(h.seer_candidate)} 号</strong></div>
        <div><strong>支持：</strong>${linkifyEventIds(h.support || '无')}</div>
        <div><strong>反对：</strong>${linkifyEventIds(h.opposition || '无')}</div>
      </div>`;
    }
  } else {
    hypHtml += '<div style="font-size:12px;color:var(--muted);">当前所选方案仅输出九人身份分配（如需查看竞争假设与原文依据，请切换至“竞争假设 v1/v2”方案）。</div>';
  }
  if (predObj && predObj.faithfulness && !predObj.faithfulness.clean) {
    const items = [
      ...predObj.faithfulness.invalid_event_citations.map(x => `无效事件引用：${x.event_id}（${x.location}）`),
      ...predObj.faithfulness.flags.map(x => `${x.message}（命中：${x.excerpt}）`),
    ];
    hypHtml += `<div class="flag-box"><strong>自动忠实度告警：</strong><br/>${items.map(escapeHtml).join('<br/>')}</div>`;
  }
  document.getElementById('hypBox').innerHTML = hypHtml;

  // Public events timeline
  document.getElementById('evCount').textContent = `共 ${snap.events.length} 条公开事件`;
  let evHtml = '';
  for (const e of snap.events) {
    let body = '';
    if (e.type === 'speech') {
      body = `<strong>${e.speaker}号发言：</strong>${escapeHtml(e.text)}`;
    } else if (e.type === 'announcement') {
      body = escapeHtml(e.text);
    } else if (e.type === 'shoot') {
      body = `<strong>${e.speaker}号猎人翻牌开枪</strong>，带走 <strong>${e.target}号</strong>（目标身份未公开）`;
    } else if (e.type === 'vote_results') {
      const tally = {};
      const abst = [];
      for (const v of (e.votes || [])) {
        if (v.target == null) abst.push(v.player + '号');
        else { tally[v.target] = tally[v.target] || []; tally[v.target].push(v.player + '号'); }
      }
      const parts = Object.keys(tally).sort((a,b)=>Number(a)-Number(b)).map(t => `投${t}号：[${tally[t].join('、')}]`);
      if (abst.length) parts.push(`弃票：[${abst.join('、')}]`);
      const outcome = e.eliminated_player ? `放逐 ${e.eliminated_player}号` : (e.tied_players && e.tied_players.length ? `平票：${e.tied_players.join('、')}号` : '无放逐');
      body = `<strong>${outcome}</strong>｜${parts.join('；')}`;
    }
    evHtml += `<div class="ev ev-type-${e.type}" id="ev-${e.event_id}">
      <div class="ev-meta"><span class="ev-id">${e.event_id}</span><span>第 ${e.day} 天</span><span>${TYPE_ZH[e.type] || e.type}</span></div>
      <div>${body}</div>
    </div>`;
  }
  document.getElementById('eventsList').innerHTML = evHtml;
}

window.selectDay = function(idx) { state.dayIdx = idx; render(); };
splitEl.addEventListener('change', () => { state.split = splitEl.value; populateSelectors(); });
gameEl.addEventListener('change', () => { state.gameId = gameEl.value; state.dayIdx = 0; render(); });
runEl.addEventListener('change', () => { state.runKey = runEl.value; render(); });
truthEl.addEventListener('change', () => { state.showTruth = truthEl.checked; render(); });

populateSelectors();
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo/werewolf_demo.html')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    bundle = build_bundle()
    payload = json.dumps(bundle, ensure_ascii=False).replace('</', '<\\/')
    html = HTML_TEMPLATE.replace('__BUNDLE_JSON__', payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if args.overwrite else 'x'
    with args.output.open(mode, encoding='utf-8', newline='\n') as f:
        f.write(html)
    print(f'Wrote {args.output} ({len(html):,} chars)')


if __name__ == '__main__':
    main()
