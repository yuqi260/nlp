"""Compare paired validation runs and index errors to public evidence."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_baseline import evaluate, public_identity_masks
from run_baseline import ROOT, read_jsonl

ROLE_NAMES = dict(seer='预言家', witch='女巫', hunter='猎人', werewolf='狼人', villager='平民')


def load_run(path, snapshots):
    records = read_jsonl(path)
    indexed = {row['snapshot_id']: row for row in records}
    if len(indexed) != len(records) or indexed.keys() != snapshots.keys():
        raise ValueError(f'{path.name} must contain each validation snapshot exactly once')
    for key, row in indexed.items():
        snapshot = snapshots[key]
        if (row['split'], row['game_id'], row['day']) != ('validation', snapshot['game_id'], snapshot['day']):
            raise ValueError(f'mismatched validation record {key}')
        if row['status'] not in ('ok', 'failed'):
            raise ValueError(f'expected real API result for {key}')
    return indexed


def wrong_roles(prediction, gold, players):
    return '、'.join(
        f'{p}号 {ROLE_NAMES[gold[p]]}→{ROLE_NAMES[prediction[p]]}'
        for p in players if gold[p] != prediction[p]
    ) or '无'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validation = {r['game_id'] for r in read_jsonl(ROOT / 'data/splits/games.jsonl') if r['split'] == 'validation'}
    snapshots = {
        s['snapshot_id']: s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
        if s['game_id'] in validation
    }
    labels = read_jsonl(ROOT / 'data/processed/labels.jsonl')
    masks = public_identity_masks(snapshots.values())
    truth = defaultdict(dict)
    for row in labels:
        truth[row['game_id']][row['player_id']] = row['true_role']
    runs = [load_run(path, snapshots) for path in (args.baseline, args.candidate)]
    report = [
        '# 验证集逐局错例与配对比较', '',
        f'- 基线组文件 `{args.baseline.name}`',
        f'- 对照组文件 `{args.candidate.name}`',
        f'- 共 {len(validation)} 局、{len(snapshots)} 个验证快照。排除截止点前实际开枪的玩家。',
        '- 错例按真实身份→预测身份展示。事件编号供回看公开记录，具体错因需人工判断。', '',
        '| 组别 | 成功快照 | 正确身份 | 按局平均 | Macro-F1 | 最常见混淆（前三） |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for name, run in zip(('基线', '对照'), runs):
        result = evaluate(list(run.values()), labels, masks)
        accuracy = result['game_mean_accuracy']
        f1 = result['role_macro_f1_among_successes']
        accuracy_text = '—' if accuracy is None else f'{accuracy:.2%}'
        f1_text = '—' if f1 is None else f'{f1:.2%}'
        confusions = Counter({
            f'{ROLE_NAMES[actual]}→{ROLE_NAMES[pred]}': count
            for actual, row in result['confusion'].items()
            for pred, count in row.items() if pred != actual and count
        })
        common = '、'.join(f'{label} {count}' for label, count in confusions.most_common(3)) or '—'
        report.append(
            f"| {name} | {result['successful_snapshots']}/{len(snapshots)} | "
            f"{result['correct_players']}/{result['scored_players']} | {accuracy_text} | {f1_text} | {common} |"
        )
    report.extend([
        '', '## 逐轮错例', '',
        '| 对局 | 第几天 | 排除玩家 | 基线正确 | 对照正确 | 差值 | 基线错例 | 对照错例 | 错例玩家发言及公开结算事件 |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |',
    ])
    deltas = defaultdict(int)
    round_drops = [[], []]
    previous = {}
    paired = 0
    for key, snapshot in sorted(snapshots.items(), key=lambda item: (item[1]['game_id'], item[1]['day'])):
        game, day = snapshot['game_id'], snapshot['day']
        gold = truth[game]
        players = sorted(set(gold) - set(masks[key]))
        counts, errors, wrong_players = [], [], set()
        for i, run in enumerate(runs):
            row = run[key]
            if row['status'] == 'failed':
                counts.append(None)
                errors.append('调用失败')
                previous.pop((game, i), None)
                continue
            pred = {p['player_id']: p['role'] for p in row['assignments']}
            counts.append(sum(pred[p] == gold[p] for p in players))
            errors.append(wrong_roles(pred, gold, players))
            wrong_players.update(p for p in players if pred[p] != gold[p])
            if (game, i) in previous:
                prev_day, prev_pred, prev_players = previous[game, i]
                shared = set(players) & prev_players
                delta = sum(int(pred[p] == gold[p]) - int(prev_pred[p] == gold[p]) for p in shared)
                if delta < 0:
                    round_drops[i].append(f'`{game}` 第 {prev_day}→{day} 天（{delta}）')
            previous[game, i] = day, pred, set(players)
        delta_text = '—'
        if None not in counts:
            delta = counts[1] - counts[0]
            paired += 1
            deltas[game] += delta
            delta_text = f'{delta:+d}'
        refs = [
            e['event_id'].rsplit(':', 1)[-1] for e in snapshot['public_events']
            if (e['type'] == 'speech' and e['speaker'] in wrong_players)
            or e['type'] in ('announcement', 'vote_results', 'shoot')
        ]
        refs_text = '、'.join(f'`{ref}`' for ref in refs) or '—'
        counts_text = ['—' if count is None else f'{count}/{len(players)}' for count in counts]
        masked = '、'.join(map(str, masks[key])) or '无'
        report.append(
            f'| `{game}` | {day} | {masked} | {counts_text[0]} | {counts_text[1]} | '
            f'{delta_text} | {errors[0]} | {errors[1]} | {refs_text} |'
        )
    improved = sum(delta > 0 for delta in deltas.values())
    declined = sum(delta < 0 for delta in deltas.values())
    report.extend([
        '', '## 逐局变化', '',
        f'- 双方都成功的快照共 {paired} 个。对照组净增加 {sum(deltas.values()):+d} 个正确身份。',
        f'- 有配对结果的对局中，{improved} 局改善，{declined} 局下降，{len(deltas) - improved - declined} 局持平。',
    ])
    for name, drops in zip(('基线', '对照'), round_drops):
        drop_text = '、'.join(drops) or '无'
        report.append(f'- {name}组新增公开信息后正确人数下降的轮次（只计算前后均未公开确认的玩家）为 {drop_text}。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as file:
        file.write('\n'.join(report) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
