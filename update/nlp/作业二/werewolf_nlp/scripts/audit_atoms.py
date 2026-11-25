"""Summarize grounding by atom without assuming semantic correctness."""
import json
from collections import Counter
from pathlib import Path
from extract_atoms import check
from run_baseline import ROOT, read_jsonl


def main():
    source = ROOT / 'outputs/predictions/validation_atomic_four_types_v1.jsonl'
    snapshots = {s['snapshot_id']: s for s in read_jsonl(ROOT / 'data/processed/snapshots.jsonl')}
    summary = Counter()
    by_kind = {}
    for row in read_jsonl(source):
        summary['snapshots'] += 1
        raw = row.get('raw_output', '').strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        try:
            atoms = json.loads(raw)['atoms']
            if not isinstance(atoms, list):
                raise ValueError('invalid list')
        except (ValueError, KeyError, TypeError):
            summary['unparseable_snapshots'] += 1
            continue
        summary['fully_valid_snapshots'] += row['status'] == 'ok'
        for atom in atoms:
            kind = atom.get('kind', 'unknown') if isinstance(atom, dict) else 'invalid'
            counts = by_kind.setdefault(kind, Counter())
            counts['total'] += 1
            try:
                check(json.dumps({'atoms': [atom]}, ensure_ascii=False), snapshots[row['snapshot_id']])
                counts['grounded'] += 1
            except (ValueError, KeyError, TypeError, AttributeError):
                counts['invalid'] += 1
    result = {'note': 'Field and source grounding only, not semantic correctness or recall.', 'summary': dict(summary), 'by_kind': by_kind}
    output = ROOT / 'outputs/metrics/validation_atomic_four_types_v1_audit.json'
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
