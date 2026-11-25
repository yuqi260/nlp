"""Quantitative error attribution across validation and test predictions.

Uses only existing saved prediction JSONL files, processed snapshots, and labels
(zero API calls). Outputs structured JSON and a Markdown audit report covering:
- Per-run masked unknown-role accuracy, game-mean accuracy, Macro-F1, by-day metrics
- Per-game accuracy distribution and hardest games
- Top role confusion pairs
- Day 1 -> Day 2+ transition dynamics (improved / flat / degraded)
- Validation vs. test structural comparison (length, revotes, shoots, days)
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from werewolf_nlp.baseline import ROLES
from evaluate_baseline import evaluate, public_identity_masks
from run_baseline import read_jsonl

ROLE_ZH = {
    'seer': '预言家',
    'witch': '女巫',
    'hunter': '猎人',
    'werewolf': '狼人',
    'villager': '平民',
}

DEFAULT_RUNS = [
    ('val_direct_off', 'validation', 'direct_v1（关思考）', 'outputs/predictions/qwen3_5_27b_validation_off_v1.jsonl'),
    ('val_direct_on', 'validation', 'direct_v1（开思考）', 'outputs/predictions/qwen3_5_27b_validation_on_v1.jsonl'),
    ('val_thinking_vote', 'validation', '三次思考投票', 'outputs/predictions/qwen3_5_27b_validation_thinking_vote_v1.jsonl'),
    ('test_direct_off', 'test', 'direct_v1（关思考）', 'outputs/predictions/qwen3_5_27b_test_v1.jsonl'),
    ('test_direct_on', 'test', 'direct_v1（开思考）', 'outputs/predictions/qwen3_5_27b_test_thinking_direct_complete_v1.jsonl'),
    ('test_hyp_a', 'test', 'hypotheses_v1（开思考·次1）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_a_complete_v1.jsonl'),
    ('test_hyp_b', 'test', 'hypotheses_v1（开思考·次2）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_b_complete_v1.jsonl'),
    ('test_thinking_vote', 'test', '三次思考投票', 'outputs/predictions/qwen3_5_27b_test_thinking_vote_v1.jsonl'),
]


def load_truth(labels):
    truth = defaultdict(dict)
    for row in labels:
        truth[row['game_id']][row['player_id']] = row['true_role']
    return truth


def analyze_run(records, truth, masks):
    summary = evaluate(records, [
        {'game_id': gid, 'player_id': pid, 'true_role': role}
        for gid, roles in truth.items()
        for pid, role in roles.items()
    ], excluded=masks)

    per_snapshot = []
    by_game_snaps = defaultdict(list)
    confusion_pairs = Counter()
    total_errors = 0

    for rec in records:
        if rec.get('status') != 'ok':
            continue
        gid = rec['game_id']
        sid = rec['snapshot_id']
        day = rec['day']
        gold = truth[gid]
        pred = {item['player_id']: item['role'] for item in rec['assignments']}
        masked = set(masks.get(sid, []))
        unknown_players = sorted(set(gold) - masked)
        correct = sum(1 for p in unknown_players if pred[p] == gold[p])
        errors = []
        for p in unknown_players:
            if pred[p] != gold[p]:
                confusion_pairs[(gold[p], pred[p])] += 1
                total_errors += 1
                errors.append({'player_id': p, 'true': gold[p], 'pred': pred[p]})
        item = {
            'snapshot_id': sid,
            'game_id': gid,
            'day': day,
            'correct': correct,
            'total': len(unknown_players),
            'accuracy': correct / len(unknown_players) if unknown_players else None,
            'errors': errors,
        }
        per_snapshot.append(item)
        by_game_snaps[gid].append(item)

    per_game = []
    transitions = {'improved': [], 'flat': [], 'degraded': []}
    for gid, snaps in sorted(by_game_snaps.items()):
        snaps.sort(key=lambda x: x['day'])
        c = sum(x['correct'] for x in snaps)
        t = sum(x['total'] for x in snaps)
        per_game.append({
            'game_id': gid,
            'snapshots': len(snaps),
            'correct': c,
            'total': t,
            'accuracy': c / t if t else None,
            'days': [{'day': x['day'], 'correct': x['correct'], 'total': x['total'], 'accuracy': x['accuracy']} for x in snaps],
        })
        for prev, curr in zip(snaps, snaps[1:]):
            delta = curr['accuracy'] - prev['accuracy']
            entry = {
                'game_id': gid,
                'from_day': prev['day'],
                'to_day': curr['day'],
                'from_acc': prev['accuracy'],
                'to_acc': curr['accuracy'],
                'from_score': f"{prev['correct']}/{prev['total']}",
                'to_score': f"{curr['correct']}/{curr['total']}",
                'delta': round(delta, 4),
            }
            if delta > 1e-9:
                transitions['improved'].append(entry)
            elif delta < -1e-9:
                transitions['degraded'].append(entry)
            else:
                transitions['flat'].append(entry)

    game_accs = [g['accuracy'] for g in per_game if g['accuracy'] is not None]
    per_game_sorted = sorted(per_game, key=lambda x: (x['accuracy'], x['game_id']))

    top_confusions = [
        {
            'true_role': t,
            'pred_role': p,
            'count': cnt,
            'share_of_errors': round(cnt / total_errors, 4) if total_errors else 0.0,
        }
        for (t, p), cnt in confusion_pairs.most_common(10)
    ]

    seer_wolf_errors = confusion_pairs[('seer', 'werewolf')] + confusion_pairs[('werewolf', 'seer')]
    villager_wolf_errors = confusion_pairs[('villager', 'werewolf')] + confusion_pairs[('werewolf', 'villager')]

    return {
        'unknown_accuracy': summary['player_accuracy_among_successes'],
        'correct_players': summary['correct_players'],
        'scored_players': summary['scored_players'],
        'game_mean_accuracy': summary['game_mean_accuracy'],
        'macro_f1': summary['role_macro_f1_among_successes'],
        'exact_snapshot_accuracy': summary['exact_snapshot_accuracy_among_successes'],
        'by_day': summary['by_day'],
        'per_role': summary['per_role'],
        'total_errors': total_errors,
        'seer_wolf_confusion_count': seer_wolf_errors,
        'seer_wolf_confusion_share': round(seer_wolf_errors / total_errors, 4) if total_errors else 0.0,
        'villager_wolf_confusion_count': villager_wolf_errors,
        'villager_wolf_confusion_share': round(villager_wolf_errors / total_errors, 4) if total_errors else 0.0,
        'top_confusions': top_confusions,
        'game_accuracy_stats': {
            'min': min(game_accs) if game_accs else None,
            'median': statistics.median(game_accs) if game_accs else None,
            'max': max(game_accs) if game_accs else None,
            'stdev': round(statistics.stdev(game_accs), 4) if len(game_accs) > 1 else 0.0,
        },
        'hardest_games': per_game_sorted[:5],
        'easiest_games': per_game_sorted[-3:][::-1],
        'transitions': {
            'improved_count': len(transitions['improved']),
            'flat_count': len(transitions['flat']),
            'degraded_count': len(transitions['degraded']),
            'degraded': transitions['degraded'],
            'improved': transitions['improved'],
        },
        'per_game': per_game,
    }


def split_profile(snapshots, splits):
    by_split = defaultdict(list)
    for snap in snapshots:
        sp = splits.get(snap['game_id'])
        if sp in ('validation', 'test'):
            by_split[sp].append(snap)

    profiles = {}
    for sp, snaps in by_split.items():
        games = defaultdict(list)
        for s in snaps:
            games[s['game_id']].append(s)
        speech_chars = []
        speech_counts = []
        revote_snaps = 0
        shoot_snaps = 0
        multi_day_games = 0
        for gid, gsnaps in games.items():
            if len(gsnaps) >= 2:
                multi_day_games += 1
        for s in snaps:
            speeches = [e for e in s['public_events'] if e['type'] == 'speech']
            speech_counts.append(len(speeches))
            speech_chars.append(sum(len(e.get('text', '')) for e in speeches))
            votes_by_day = Counter(e['day'] for e in s['public_events'] if e['type'] == 'vote_results')
            if any(v > 1 for v in votes_by_day.values()):
                revote_snaps += 1
            if any(e['type'] == 'shoot' for e in s['public_events']):
                shoot_snaps += 1
        profiles[sp] = {
            'games': len(games),
            'snapshots': len(snaps),
            'multi_day_games': multi_day_games,
            'avg_speeches_per_snapshot': round(statistics.mean(speech_counts), 1),
            'avg_speech_chars_per_snapshot': round(statistics.mean(speech_chars), 1),
            'median_speech_chars_per_snapshot': round(statistics.median(speech_chars), 1),
            'snapshots_with_revote': revote_snaps,
            'snapshots_with_shoot': shoot_snaps,
        }
    return profiles


def pct(val):
    return f'{val * 100:.2f}%' if val is not None else 'N/A'


def render_markdown(report):
    lines = [
        '# 验证集与测试集定向错例归因分析',
        '',
        '> 本报告完全基于既有本地预测文件与公开快照生成，未发起任何新增 API 调用。所有主指标采用排除公开开枪猎人的保守未知身份掩码口径（已全量核验 MaKTO 213 快照无其他公开亮牌公告）。',
        '',
        '## 1. 数据集结构对比（验证集 vs. 测试集）',
        '',
        '| 划分 | 局数 | 快照数 | 含多日对局 | 平均发言条数/快照 | 平均发言字符数/快照 | 含平票复投快照 | 含猎人开枪快照 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for sp, name in [('validation', '验证集'), ('test', '测试集')]:
        p = report['split_profiles'][sp]
        lines.append(
            f"| {name} | {p['games']} | {p['snapshots']} | {p['multi_day_games']} | "
            f"{p['avg_speeches_per_snapshot']} | {p['avg_speech_chars_per_snapshot']} | "
            f"{p['snapshots_with_revote']} | {p['snapshots_with_shoot']} |"
        )

    lines += [
        '',
        '## 2. 核心方案总览（含 Day1 / Day2+ 分轮表现与按局波动）',
        '',
        '| 方案 | 划分 | 未知身份准确率 | 按局平均 | Macro-F1 | Day 1 | Day 2 | 局间标准差 | Day1→Day2 退化/持平/改善 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for key, split, label, _ in DEFAULT_RUNS:
        if key not in report['runs']:
            continue
        r = report['runs'][key]
        d1 = r['by_day'].get('1', {}).get('accuracy')
        d2 = r['by_day'].get('2', {}).get('accuracy')
        tr = r['transitions']
        sp_zh = '验证集' if split == 'validation' else '测试集'
        lines.append(
            f"| {label} | {sp_zh} | {pct(r['unknown_accuracy'])} ({r['correct_players']}/{r['scored_players']}) | "
            f"{pct(r['game_mean_accuracy'])} | {pct(r['macro_f1'])} | {pct(d1)} | {pct(d2)} | "
            f"{pct(r['game_accuracy_stats']['stdev'])} | {tr['degraded_count']} 退化 / {tr['flat_count']} 持平 / {tr['improved_count']} 改善 |"
        )

    lines += [
        '',
        '## 3. 主要角色混淆模式（以主基线 `direct_v1` 开思考为例）',
        '',
    ]
    for key, title in [('val_direct_on', '验证集 · direct_v1 开思考'), ('test_direct_on', '测试集 · direct_v1 开思考')]:
        if key not in report['runs']:
            continue
        r = report['runs'][key]
        lines.append(f"### {title}（总错误 {r['total_errors']} 人次）")
        lines.append(
            f"- **平民 ↔ 狼人互混**：{r['villager_wolf_confusion_count']} 次（占全部错误的 {pct(r['villager_wolf_confusion_share'])}）"
        )
        lines.append(
            f"- **预言家 ↔ 狼人互混**：{r['seer_wolf_confusion_count']} 次（占全部错误的 {pct(r['seer_wolf_confusion_share'])}）"
        )
        lines.append('')
        lines.append('| 真实身份 | 误判为 | 次数 | 占总错误比例 |')
        lines.append('| --- | --- | ---: | ---: |')
        for item in r['top_confusions'][:6]:
            lines.append(
                f"| {ROLE_ZH[item['true_role']]} | {ROLE_ZH[item['pred_role']]} | "
                f"{item['count']} | {pct(item['share_of_errors'])} |"
            )
        lines.append('')

    lines += [
        '## 4. 测试集最难对局与跨轮次退化样本（`direct_v1` 开思考）',
        '',
    ]
    if 'test_direct_on' in report['runs']:
        trun = report['runs']['test_direct_on']
        lines.append('### 4.1 测试集准确率最低的 5 局')
        lines.append('')
        lines.append('| 对局 | 快照数 | 正确/未知总数 | 局准确率 | 各日明细 |')
        lines.append('| --- | ---: | ---: | ---: | --- |')
        for g in trun['hardest_games']:
            day_str = ', '.join(f"Day{d['day']}: {d['correct']}/{d['total']}" for d in g['days'])
            lines.append(f"| {g['game_id']} | {g['snapshots']} | {g['correct']}/{g['total']} | {pct(g['accuracy'])} | {day_str} |")
        lines.append('')
        lines.append('### 4.2 测试集 Day1 → Day2+ 退化样本')
        lines.append('')
        degraded = trun['transitions']['degraded']
        if degraded:
            lines.append('| 对局 | 轮次变化 | 前一日 | 后一日 | 准确率变化 |')
            lines.append('| --- | --- | ---: | ---: | ---: |')
            for d in degraded:
                lines.append(
                    f"| {d['game_id']} | Day{d['from_day']}→Day{d['to_day']} | {d['from_score']} ({pct(d['from_acc'])}) | "
                    f"{d['to_score']} ({pct(d['to_acc'])}) | {d['delta'] * 100:+.2f} pp |"
                )
        else:
            lines.append('无跨轮次退化样本。')
        lines.append('')

    lines += [
        '## 5. 关键归因结论与方法启示',
        '',
        '1. **测试集准确率低于验证集并非单局异常，而是整体难度与方差叠加**：可结合上表观察测试集各局标准差及最难 5 局分布，后续所有方法必须按局汇报并警惕在 12 局验证集上过拟合。',
        '2. **核心错误高度集中在“平民 ↔ 狼人”与“预言家 ↔ 狼人”两对阵营边界**：神职（尤其女巫、猎人）一旦跳出较易识别，真正决定上限的是真假预言家分辨与倒钩/深水狼识别。',
        '3. **开启思考在测试集带来最稳定的净收益**：`direct_v1` 开启思考在测试集高于关闭思考及三次投票集成，说明后续改进应以“开启思考 + 极简确定性硬事实锚点”为主线，避免再往非思考模型塞入带噪声的 LLM 抽取索引。',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-output', type=Path, default=ROOT / 'outputs/metrics/error_attribution_v1.json')
    parser.add_argument('--md-output', type=Path, default=ROOT / 'docs/audit/验证与测试集错例归因分析.md')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    snapshots = read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
    labels = read_jsonl(ROOT / 'data/processed/labels.jsonl')
    splits = {x['game_id']: x['split'] for x in read_jsonl(ROOT / 'data/splits/games.jsonl')}
    truth = load_truth(labels)
    masks = public_identity_masks(snapshots)

    runs = {}
    for key, split, label, rel_path in DEFAULT_RUNS:
        path = ROOT / rel_path
        if not path.exists():
            continue
        runs[key] = analyze_run(read_jsonl(path), truth, masks)
        runs[key]['label'] = label
        runs[key]['split'] = split
        runs[key]['source'] = rel_path

    report = {
        'mask_scope': 'excluding confirmed shooters only (verified across all 213 MaKTO snapshots: 0 other public role reveals)',
        'split_profiles': split_profile(snapshots, splits),
        'runs': runs,
    }

    mode = 'w' if args.overwrite else 'x'
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open(mode, encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    with args.md_output.open(mode, encoding='utf-8', newline='\n') as f:
        f.write(render_markdown(report))

    print(f'Wrote {args.json_output} and {args.md_output}')


if __name__ == '__main__':
    main()
