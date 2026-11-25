"""Deterministic faithfulness checks for model rationales and hypotheses.

Flags verifiable grounding or factual-discipline issues without claiming full
semantic verification:
1. Missing or out-of-snapshot event ID citations (eNNNN or <game_id>:eNNNN).
2. Confirmed public hunter contradicted by predicted assignments.
3. Unanimous-vote ("全票") claims when actual vote_results had dissent/abstention.
4. Overclaim patterns: asserting unannounced win/loss outcomes, or treating
   vote elimination / shoot target as confirmed identity.
"""
import json
import re

EVENT_ID_RE = re.compile(r'\be\d{4}\b')

OVERCLAIM_PATTERNS = [
    (
        'unannounced_game_outcome',
        re.compile(r'(好人(阵营)?(已经)?获胜|狼人(阵营)?(已经)?获胜|游戏(已|已经)结束|锁定胜局|好人赢了|狼人赢了)'),
        '快照未公布胜负，但依据声称游戏已结束或某阵营获胜',
    ),
    (
        'elimination_treated_as_confirmed_role',
        re.compile(r'(被(投票|放逐|公投|投出|出局)[^。；，]{0,12}(证实|确认|实锤|证明)[^。；，]{0,8}(身份|是狼|为狼|好人|预言家|女巫|猎人|平民)|已证实(是|为)(狼人|好人|预言家|女巫|平民))'),
        '将投票出局或未公开事件表述为已证实身份',
    ),
    (
        'shoot_target_treated_as_confirmed_role',
        re.compile(r'(开枪带走|枪杀|带走)[^。；，]{0,12}(确认|证实|实锤|证明)[^。；，]{0,8}(身份|好人|狼人|平民|神职)'),
        '将猎人开枪目标表述为已证实身份',
    ),
]

UNANIMOUS_VOTE_RE = re.compile(r'全票(出局|放逐|投出|投|打飞|归票|通过)')


def _strip_fence(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return text


def short_event_id(event_id):
    return str(event_id).rsplit(':', 1)[-1]


def extract_rationale_texts(record):
    """Collect all rationale strings (hypotheses + per-player reasons) from a record."""
    texts = []
    raw = record.get('raw_output')
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(_strip_fence(raw))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for hyp in parsed.get('hypotheses') or []:
                if isinstance(hyp, dict):
                    for key in ('support', 'opposition'):
                        val = hyp.get(key)
                        if isinstance(val, str) and val.strip():
                            texts.append((f"hypothesis[{hyp.get('seer_candidate')}].{key}", val))
            for item in parsed.get('assignments') or []:
                if isinstance(item, dict):
                    val = item.get('reason')
                    if isinstance(val, str) and val.strip():
                        texts.append((f"assignment[{item.get('player_id')}].reason", val))
                    eids = item.get('evidence_ids')
                    if isinstance(eids, list) and eids:
                        texts.append((f"assignment[{item.get('player_id')}].evidence_ids", ' '.join(str(x) for x in eids)))
    return texts


def _has_any_unanimous_elimination(snapshot):
    for event in snapshot['public_events']:
        if event['type'] != 'vote_results':
            continue
        eliminated = event.get('eliminated_player')
        votes = event.get('votes') or []
        if eliminated is None or not votes:
            continue
        if all(v.get('target') == eliminated for v in votes):
            return True
    return False


def audit_record_faithfulness(record, snapshot):
    """Run deterministic faithfulness checks for one prediction record against its snapshot."""
    valid_short_ids = set()
    for e in snapshot['public_events']:
        if e.get('game_id') == snapshot['game_id'] and e.get('order', float('inf')) <= snapshot['cutoff_order']:
            valid_short_ids.add(short_event_id(e['event_id']))
    shooters = sorted({e['speaker'] for e in snapshot['public_events'] if e['type'] == 'shoot'})

    texts = extract_rationale_texts(record)
    cited_ids = []
    invalid_ids = []
    flags = []

    for location, text in texts:
        found = EVENT_ID_RE.findall(text)
        cited_ids.extend(found)
        for eid in found:
            if eid not in valid_short_ids:
                invalid_ids.append({'location': location, 'event_id': eid})
        for code, pattern, message in OVERCLAIM_PATTERNS:
            match = pattern.search(text)
            if match:
                flags.append({
                    'code': code,
                    'location': location,
                    'excerpt': match.group(0),
                    'message': message,
                })
        if UNANIMOUS_VOTE_RE.search(text) and not _has_any_unanimous_elimination(snapshot):
            match = UNANIMOUS_VOTE_RE.search(text)
            flags.append({
                'code': 'false_unanimous_vote_claim',
                'location': location,
                'excerpt': match.group(0),
                'message': '声称全票放逐/出局，但快照中无全票一致的放逐票型',
            })

    hunter_consistent = True
    if shooters and record.get('status') == 'ok' and record.get('assignments'):
        pred = {item['player_id']: item['role'] for item in record['assignments']}
        for shooter in shooters:
            if pred.get(shooter) != 'hunter':
                hunter_consistent = False
                flags.append({
                    'code': 'confirmed_hunter_mismatch',
                    'location': f'assignment[{shooter}]',
                    'excerpt': f'predicted={pred.get(shooter)}',
                    'message': f'{shooter}号已公开开枪，但预测身份不是 hunter',
                })

    return {
        'snapshot_id': record['snapshot_id'],
        'game_id': record['game_id'],
        'day': record['day'],
        'has_rationale': bool(texts),
        'cited_event_count': len(cited_ids),
        'unique_cited_events': sorted(set(cited_ids)),
        'invalid_event_citations': invalid_ids,
        'confirmed_hunter_consistent': hunter_consistent,
        'flag_count': len(flags) + len(invalid_ids),
        'flags': flags,
        'clean': (len(flags) == 0 and len(invalid_ids) == 0),
    }
