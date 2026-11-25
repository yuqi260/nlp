"""Run deterministic faithfulness checks across rationale-bearing prediction files."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from werewolf_nlp.faithfulness import audit_record_faithfulness
from run_baseline import read_jsonl

DEFAULT_FILES = [
    ('val_hyp_v1_thinking', '验证集 · hypotheses_v1（开思考·首次）', 'outputs/predictions/qwen3_5_27b_validation_hypotheses_thinking_complete_v1.jsonl'),
    ('val_hyp_v2_thinking', '验证集 · hypotheses_v2（开思考）', 'outputs/predictions/qwen3_5_27b_validation_hypotheses_v2_thinking_v1.jsonl'),
    ('val_fact_anchor_v1_thinking', '验证集 · fact_anchor_v1（开思考）', 'outputs/predictions/qwen3_5_27b_validation_fact_anchor_on_v1.jsonl'),
    ('test_hyp_a_thinking', '测试集 · hypotheses_v1（开思考·次1）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_a_complete_v1.jsonl'),
    ('test_hyp_b_thinking', '测试集 · hypotheses_v1（开思考·次2）', 'outputs/predictions/qwen3_5_27b_test_thinking_hyp_b_complete_v1.jsonl'),
    ('train_teacher_pilot', '训练集试点 · hypotheses_v2（关思考）', 'outputs/predictions/qwen3_5_27b_train_hypotheses_teacher_pilot_v1.jsonl'),
]


def audit_file(records, snapshot_map):
    items = []
    flag_counts = Counter()
    with_citations = 0
    total_citations = 0
    invalid_citations = 0
    hunter_ok = 0
    clean_count = 0

    for rec in records:
        snap = snapshot_map[rec['snapshot_id']]
        res = audit_record_faithfulness(rec, snap)
        items.append(res)
        if res['cited_event_count'] > 0:
            with_citations += 1
        total_citations += res['cited_event_count']
        invalid_citations += len(res['invalid_event_citations'])
        if res['confirmed_hunter_consistent']:
            hunter_ok += 1
        if res['clean']:
            clean_count += 1
        for flag in res['flags']:
            flag_counts[flag['code']] += 1

    return {
        'snapshots': len(items),
        'clean_snapshots': clean_count,
        'flagged_snapshots': len(items) - clean_count,
        'snapshots_with_event_citations': with_citations,
        'total_event_citations': total_citations,
        'invalid_event_citations': invalid_citations,
        'confirmed_hunter_consistent_snapshots': hunter_ok,
        'flag_type_counts': dict(flag_counts),
        'flagged_details': [x for x in items if not x['clean']],
    }


def render_markdown(report):
    lines = [
        '# 模型依据忠实度确定性自动校验报告',
        '',
        '> 说明：本校验脚本检查四项可确定性验证的忠实度规则——(1) 引用事件编号 `eNNNN` 是否存在于当前快照；(2) 已公开开枪猎人是否与预测身份一致；(3) “全票”表述是否与实际 `vote_results` 冲突；(4) 是否出现“未公布胜负却宣称获胜”“把投票出局/开枪目标当成已证实身份”等越界断言。通过自动校验不代表语义完全正确，仍需结合人工抽检。',
        '',
        '## 1. 各组预测自动校验汇总',
        '',
        '| 预测组 | 快照数 | 无告警快照 | 告警快照 | 含 `eNNNN` 引用 | 总引用次数 | 越界/无效编号 | 开枪猎人一致 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for key, label, _ in DEFAULT_FILES:
        if key not in report['runs']:
            continue
        r = report['runs'][key]
        lines.append(
            f"| {label} | {r['snapshots']} | {r['clean_snapshots']} | {r['flagged_snapshots']} | "
            f"{r['snapshots_with_event_citations']}/{r['snapshots']} | {r['total_event_citations']} | "
            f"{r['invalid_event_citations']} | {r['confirmed_hunter_consistent_snapshots']}/{r['snapshots']} |"
        )

    lines += ['', '## 2. 检出的具体告警明细', '']
    any_flag = False
    for key, label, _ in DEFAULT_FILES:
        if key not in report['runs']:
            continue
        details = report['runs'][key]['flagged_details']
        if not details:
            continue
        any_flag = True
        lines.append(f'### {label}（{len(details)} 个快照存在告警）')
        lines.append('')
        lines.append('| 快照 ID | 告警类型 | 位置 | 命中片段 | 说明 |')
        lines.append('| --- | --- | --- | --- | --- |')
        for item in details:
            for inv in item['invalid_event_citations']:
                lines.append(
                    f"| {item['snapshot_id']} | invalid_event_id | {inv['location']} | `{inv['event_id']}` | 引用了当前快照不存在的事件编号 |"
                )
            for flag in item['flags']:
                excerpt = flag['excerpt'].replace('|', '\\|')
                lines.append(
                    f"| {item['snapshot_id']} | {flag['code']} | {flag['location']} | `{excerpt}` | {flag['message']} |"
                )
        lines.append('')
    if not any_flag:
        lines.append('本次校验的所有预测组均未检出规则告警。')
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-output', type=Path, default=ROOT / 'outputs/metrics/faithfulness_audit_v1.json')
    parser.add_argument('--md-output', type=Path, default=ROOT / 'docs/audit/依据忠实度自动校验报告.md')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    snapshots = read_jsonl(ROOT / 'data/processed/snapshots.jsonl')
    snapshot_map = {s['snapshot_id']: s for s in snapshots}

    runs = {}
    for key, label, rel_path in DEFAULT_FILES:
        path = ROOT / rel_path
        if not path.exists():
            continue
        res = audit_file(read_jsonl(path), snapshot_map)
        res['label'] = label
        res['source'] = rel_path
        runs[key] = res

    report = {'runs': runs}
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
