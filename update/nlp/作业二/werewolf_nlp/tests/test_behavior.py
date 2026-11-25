import unittest
from werewolf_nlp.behavior import extract_behavior, relate_stances_to_votes


class BehaviorTest(unittest.TestCase):
    def test_revote_changes_and_cutoff(self):
        events = [
            {'game_id': 'g', 'order': 1, 'day': 1, 'event_id': 'a', 'type': 'vote_results', 'votes': [{'player': 1, 'target': 2}]},
            {'game_id': 'g', 'order': 2, 'day': 1, 'event_id': 'b', 'type': 'vote_results', 'votes': [{'player': 1, 'target': 3}]},
            {'game_id': 'g', 'order': 3, 'day': 2, 'event_id': 'c', 'type': 'vote_results', 'votes': [{'player': 1, 'target': 4}]},
        ]
        snapshot = {'game_id': 'g', 'cutoff_order': 3, 'public_events': events}
        result = extract_behavior(snapshot)
        self.assertEqual(result['votes'][1]['changes_from_previous_vote'], [{'player': 1, 'from': 2, 'to': 3}])
        self.assertEqual(result['votes'][2]['changes_from_previous_vote'], [])
        snapshot['cutoff_order'] = 2
        with self.assertRaises(ValueError):
            extract_behavior(snapshot)

    def test_only_prior_matching_stance_is_linked(self):
        snapshot = {'game_id': 'g', 'cutoff_order': 3, 'public_events': [
            {'game_id': 'g', 'event_id': 's1', 'order': 1, 'day': 1, 'type': 'speech', 'speaker': 1, 'text': '支持2号'},
            {'game_id': 'g', 'event_id': 'v1', 'order': 2, 'day': 1, 'type': 'vote_results', 'votes': [{'player': 1, 'target': 2}]},
            {'game_id': 'g', 'event_id': 's2', 'order': 3, 'day': 1, 'type': 'speech', 'speaker': 1, 'text': '反对2号'},
        ]}
        atoms = [{'event_id': 's1', 'kind': 'stance', 'speaker': 1, 'target': 2, 'value': 'support', 'quote': '支持2号'},
                 {'event_id': 's2', 'kind': 'stance', 'speaker': 1, 'target': 2, 'value': 'oppose', 'quote': '反对2号'}]
        links = relate_stances_to_votes(snapshot, atoms)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['speech_event_id'], 's1')
