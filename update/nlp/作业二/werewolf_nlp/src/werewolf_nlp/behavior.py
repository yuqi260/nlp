"""Extract traceable public actions without interpreting player intent."""
from collections import defaultdict


def extract_behavior(snapshot):
    votes, shots, announcements = [], [], []
    previous = {}
    for event in snapshot['public_events']:
        if event.get('game_id') != snapshot['game_id'] or event.get('order', float('inf')) > snapshot['cutoff_order']:
            raise ValueError('event outside snapshot')
        kind = event['type']
        if kind == 'vote_results':
            day = event['day']
            current = {v['player']: v['target'] for v in event['votes']}
            old = previous.get(day, {})
            changes = [{'player': p, 'from': old[p], 'to': target} for p, target in current.items() if p in old and old[p] != target]
            votes.append({'event_id': event['event_id'], 'day': day, 'votes': event['votes'],
                          'changes_from_previous_vote': changes, 'tied_players': event.get('tied_players', []),
                          'eliminated_player': event.get('eliminated_player')})
            previous[day] = current
        elif kind == 'shoot':
            shots.append({'event_id': event['event_id'], 'day': event['day'], 'speaker': event['speaker'], 'target': event['target']})
        elif kind == 'announcement':
            announcements.append({'event_id': event['event_id'], 'day': event['day'], 'text': event['text']})
    return {'votes': votes, 'shots': shots, 'announcements': announcements}


def extract_hard_facts(snapshot):
    """Build a compact, deterministic public-fact anchor with zero LLM noise.

    Includes only objective public announcements, compact vote tallies/outcomes,
    and confirmed hunter shoot events up to snapshot['cutoff_order'].
    """
    behavior = extract_behavior(snapshot)
    vote_outcomes = []
    for item in behavior['votes']:
        tally = defaultdict(list)
        abstainers = []
        for vote in item['votes']:
            target = vote.get('target')
            if target is None:
                abstainers.append(vote['player'])
            else:
                tally[str(target)].append(vote['player'])
        vote_outcomes.append({
            'event_id': item['event_id'],
            'day': item['day'],
            'eliminated_player': item['eliminated_player'],
            'tied_players': item['tied_players'],
            'tally_by_target': {k: sorted(v) for k, v in sorted(tally.items(), key=lambda kv: int(kv[0]))},
            'abstainers': sorted(abstainers),
        })
    confirmed_hunters = [
        {
            'event_id': shot['event_id'],
            'day': shot['day'],
            'confirmed_hunter': shot['speaker'],
            'shot_target_role_unknown': shot['target'],
        }
        for shot in behavior['shots']
    ]
    return {
        'announcements': behavior['announcements'],
        'vote_outcomes': vote_outcomes,
        'confirmed_hunters': confirmed_hunters,
    }


def relate_stances_to_votes(snapshot, atoms):
    """Link a prior extracted stance to an actual vote for the same target.

    Candidate stances may be wrong, tentative, or outdated; links are not facts
    about intent or alignment. Never compare a later speech to an earlier vote.
    """
    events = {e['event_id']: e for e in snapshot['public_events']}
    links = []
    for vote in extract_behavior(snapshot)['votes']:
        vote_event = events[vote['event_id']]
        for item in vote['votes']:
            for atom in atoms:
                event = events.get(atom.get('event_id'))
                if (atom.get('kind') == 'stance' and event and event['type'] == 'speech' and
                        event['speaker'] == atom.get('speaker') == item['player'] and
                        atom.get('target') == item['target'] and item['target'] is not None and
                        atom.get('value') in ('support', 'oppose') and isinstance(atom.get('quote'), str) and
                        bool(atom['quote'].strip()) and atom['quote'] in event.get('text', '') and
                        event['order'] < vote_event['order']):
                    links.append({'vote_event_id': vote['event_id'], 'speech_event_id': event['event_id'],
                                  'player': item['player'], 'target': item['target'],
                                  'stance': atom['value'], 'quote': atom['quote']})
    return links
