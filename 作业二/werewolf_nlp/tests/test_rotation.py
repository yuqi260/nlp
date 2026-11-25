import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_rotation_candidates import transform


class RotationTest(unittest.TestCase):
    def test_round_trip_and_ambiguous_number(self):
        snapshot = {'game_id': 'g', 'cutoff_order': 2, 'public_events': [
            {'game_id': 'g', 'event_id': 'e1', 'day': 1, 'order': 1, 'phase': 'day', 'type': 'speech', 'speaker': 9, 'text': '我是9号，支持1号'},
            {'game_id': 'g', 'event_id': 'e2', 'day': 1, 'order': 2, 'phase': 'day', 'type': 'vote_results', 'speaker': None, 'votes': [{'player': 9, 'target': 1}], 'tied_players': [1, 2], 'eliminated_player': None},
        ]}
        rotated = transform(snapshot, 2)
        self.assertEqual(rotated['public_events'][0]['text'], '我是2号，支持3号')
        self.assertEqual(rotated['public_events'][1]['votes'][0], {'player': 2, 'target': 3})
        self.assertEqual(transform(rotated, 7), snapshot)
        snapshot['public_events'][0]['text'] = '我支持1和2号'
        with self.assertRaises(ValueError):
            transform(snapshot, 2)
