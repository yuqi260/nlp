"""Run a bounded direct baseline; credentials are read from environment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.local_config import load_local_config
load_local_config(ROOT / '.env')
from werewolf_nlp.baseline import PROMPTS, PROMPT_VERSION, assign_scores, call_api, make_messages, parse_response, validate_claims
from werewolf_nlp.world_solver import solve


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--offset', type=int, default=0, help='skip this many snapshots in the selected split')
    p.add_argument('--model', required=True, help='exact provider model identifier')
    p.add_argument('--prompt-version', choices=PROMPTS, default=PROMPT_VERSION)
    p.add_argument('--atom-input', type=Path, help='candidate extraction JSONL required for atom_index_v1')
    p.add_argument('--base-url', default=os.environ.get('WEREWOLF_API_BASE_URL'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--retries', type=int, default=1)
    p.add_argument('--thinking', choices=['on', 'off', 'provider-default'], default='provider-default', help='provider must support enable_thinking')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.limit < 1 or args.offset < 0 or args.max_tokens < 1 or not 0 <= args.retries <= 3:
        p.error('limit and max-tokens must be positive, offset nonnegative, retries 0–3')
    if (args.prompt_version in ('atom_index_v1', 'atom_links_v1')) != bool(args.atom_input):
        p.error('atom prompts require --atom-input and other prompts do not accept it')
    atom_records = {}
    if args.atom_input:
        rows = read_jsonl(args.atom_input)
        atom_records = {r['snapshot_id']: r for r in rows}
        if len(atom_records) != len(rows):
            p.error('duplicate atom snapshots')
    key = os.environ.get('WEREWOLF_API_KEY')
    if not args.dry_run:
        if not args.base_url or not key:
            p.error('set WEREWOLF_API_BASE_URL and WEREWOLF_API_KEY before calling')
        parsed = urlparse(args.base_url)
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')):
            p.error('API URL must use HTTPS except for localhost')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            p.error('API URL must not contain credentials, query or fragment')
    if args.output.exists():
        p.error('output already exists; use a new file')
    splits = {x['game_id']: x['split'] for x in read_jsonl(ROOT / 'data/splits/games.jsonl')}
    selected = [x for x in read_jsonl(ROOT / 'data/processed/snapshots.jsonl') if splits.get(x['game_id']) == args.split]
    snapshots = selected[args.offset:args.offset + args.limit]
    if not snapshots:
        p.error('no matching snapshots')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        for snapshot in snapshots:
            messages = make_messages(snapshot, args.prompt_version)
            atom_stats = None
            if args.atom_input:
                from extract_atoms import check
                source = atom_records.get(snapshot['snapshot_id'])
                if not source or not source.get('raw_output'):
                    raise ValueError('missing extraction response')
                raw_atoms = source['raw_output'].strip()
                if raw_atoms.startswith('```') and raw_atoms.endswith('```'):
                    raw_atoms = raw_atoms.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                atoms = json.loads(raw_atoms)['atoms']
                if not isinstance(atoms, list):
                    raise ValueError('invalid extraction atoms')
                grounded = []
                for atom in atoms:
                    try:
                        grounded.extend(check(json.dumps({'atoms': [atom]}, ensure_ascii=False), snapshot))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        continue
                grounded = [{key: atom[key] for key in ('event_id', 'speaker', 'kind', 'target', 'value', 'quote')} for atom in grounded]
                payload = json.loads(messages[1]['content'])
                payload['unverified_extracted_candidates'] = grounded
                if args.prompt_version == 'atom_links_v1':
                    from werewolf_nlp.behavior import relate_stances_to_votes
                    payload['unverified_stance_vote_links'] = relate_stances_to_votes(snapshot, grounded)
                messages[1]['content'] = json.dumps(payload, ensure_ascii=False)
                atom_stats = {'source': str(args.atom_input), 'total': len(atoms), 'grounded': len(grounded)}
            record = {'snapshot_id': snapshot['snapshot_id'], 'game_id': snapshot['game_id'], 'day': snapshot['day'], 'split': args.split, 'model_id': args.model, 'prompt_version': args.prompt_version, 'temperature': args.temperature, 'max_tokens': args.max_tokens, 'thinking': args.thinking, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'dry_run' if args.dry_run else 'failed', 'attempts': 0, 'attempt_log': []}
            if atom_stats is not None:
                record['atom_stats'] = atom_stats
            if args.dry_run:
                record['input_characters'] = sum(len(x['content']) for x in messages)
            else:
                started = time.monotonic()
                for attempt in range(args.retries + 1):
                    record['attempts'] += 1
                    try:
                        thinking = None if args.thinking == 'provider-default' else args.thinking == 'on'
                        raw, usage, actual_model, finish = call_api(args.base_url, key, args.model, messages, args.temperature, args.max_tokens, args.timeout, thinking)
                        record['attempt_log'].append({'status': 'response', 'usage': usage, 'finish_reason': finish})
                        record.update(raw_output=raw, usage=usage, returned_model=actual_model, finish_reason=finish)
                        if args.prompt_version == 'scores_v1':
                            assignment, scores, score_total, ties = assign_scores(raw, snapshot)
                            record.update(scores=scores, score_total=score_total, optimal_assignments=ties)
                            quota_ok = True
                        elif args.prompt_version == 'scores_float_v1':
                            content = raw.strip()
                            if content.startswith('```') and content.endswith('```'):
                                content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                            rows = json.loads(content)['scores']
                            if not isinstance(rows, list) or len(rows) != 9 or any(type(row.get('player_id')) is not int for row in rows):
                                raise ValueError('invalid score rows')
                            scores = {row['player_id']: {role: row[role] for role in ('seer', 'witch', 'hunter', 'werewolf', 'villager')} for row in rows}
                            if len(scores) != 9 or any(type(v) not in (int, float) or not 0 <= v <= 100 for row in scores.values() for v in row.values()):
                                raise ValueError('scores must be numeric from 0 to 100')
                            result = solve(scores, snapshot)
                            assignment, quota_ok = result['assignment'], True
                            record.update(scores=scores, score_total=result['score'], optimal_assignments=result['optimal_assignments'], feasible_assignments=result['feasible_assignments'])
                        else:
                            assignment, quota_ok = parse_response(raw)
                        record.update(status='ok', assignments=[{'player_id': k, 'role': v} for k,v in sorted(assignment.items())], role_counts_valid=quota_ok)
                        if args.prompt_version == 'claims_v1':
                            try:
                                record['claims'] = validate_claims(raw, snapshot)
                                record['claims_valid'] = True
                            except (ValueError, TypeError) as exc:
                                record['claims_valid'] = False
                                record['claims_error'] = str(exc)
                        break
                    except Exception as exc:
                        record['error'] = type(exc).__name__ + (': ' + str(exc) if type(exc) in (ValueError, RuntimeError) else '')
                        if not record['attempt_log'] or len(record['attempt_log']) < record['attempts']:
                            record['attempt_log'].append({'status': 'error', 'error_type': type(exc).__name__})
                record['elapsed_seconds'] = round(time.monotonic()-started, 3)
                if record['status'] == 'ok':
                    record.pop('error', None)
            out.write(json.dumps(record, ensure_ascii=False) + '\n'); out.flush()
            print(snapshot['snapshot_id'], record['status'])


if __name__ == '__main__':
    main()
