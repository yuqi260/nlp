"""Experimental joint assignment with explicitly supplied inspection claims.

Claims must be manually verified against the snapshot before use. Their truth
is conditional on the speaker being the seer, not a public fact.
"""
import math
from itertools import combinations, permutations
from .baseline import ROLES


def solve(scores, snapshot, checks=()):
    if set(scores) != set(range(1, 10)):
        raise ValueError('scores must cover players 1 through 9')
    for row in scores.values():
        if set(row) != set(ROLES) or any(type(x) not in (int, float) or not math.isfinite(x) for x in row.values()):
            raise ValueError('each player needs five finite numeric scores')
    events = {e['event_id']: e for e in snapshot['public_events'] if e.get('type') == 'speech'}
    for check in checks:
        speaker, target, result = check.get('speaker'), check.get('target'), check.get('result')
        event = events.get(check.get('event_id'))
        quote = check.get('quote')
        if (type(speaker) is not int or speaker not in range(1, 10) or
                type(target) is not int or target not in range(1, 10) or target == speaker or
                result not in ('wolf', 'nonwolf') or not event or event.get('speaker') != speaker or
                not isinstance(quote, str) or not quote.strip() or quote not in event.get('text', '')):
            raise ValueError('invalid or ungrounded inspection claim')
    shooters = {e['speaker'] for e in snapshot['public_events'] if e['type'] == 'shoot'}
    if len(shooters) > 1 or any(type(p) is not int or p not in range(1, 10) for p in shooters):
        raise ValueError('invalid public shooters')
    best, best_score, ties, feasible = None, None, 0, 0
    for seer, witch, hunter in permutations(range(1, 10), 3):
        if shooters and hunter not in shooters:
            continue
        remaining = [p for p in range(1, 10) if p not in (seer, witch, hunter)]
        for wolves in combinations(remaining, 3):
            if any(c['speaker'] == seer and ((c['target'] in wolves) != (c['result'] == 'wolf')) for c in checks):
                continue
            roles = {p: 'villager' for p in remaining}
            roles.update({p: 'werewolf' for p in wolves})
            roles.update({seer: 'seer', witch: 'witch', hunter: 'hunter'})
            total = math.fsum(scores[p][roles[p]] for p in range(1, 10))
            feasible += 1
            if best_score is None or total > best_score and not math.isclose(total, best_score, rel_tol=1e-12, abs_tol=1e-12):
                best, best_score, ties = roles, total, 1
            elif math.isclose(total, best_score, rel_tol=1e-12, abs_tol=1e-12):
                ties += 1
    if best is None:
        raise ValueError('no feasible assignment')
    return {'assignment': best, 'score': best_score, 'optimal_assignments': ties, 'feasible_assignments': feasible}
