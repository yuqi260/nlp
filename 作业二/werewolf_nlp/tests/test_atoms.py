import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extract_atoms import check


class AtomTest(unittest.TestCase):
    def test_grounding_and_self_save(self):
        snapshot = {'public_events': [{'event_id': 'e1', 'type': 'speech', 'speaker': 2, 'text': '我救了2号'}]}
        atom = {'event_id': 'e1', 'speaker': 2, 'kind': 'witch_info', 'target': 2, 'value': 'save', 'quote': '我救了 2 号'}
        self.assertEqual(check(json.dumps({'atoms': [atom]}), snapshot)[0]['quote'], '我救了2号')
        atom['quote'] = '我救了3号'
        with self.assertRaises(ValueError):
            check(json.dumps({'atoms': [atom]}), snapshot)

    def test_flag_negated_and_unsubstantiated_poison(self):
        snapshot = {'public_events': [{'event_id': 'e1', 'type': 'speech', 'speaker': 1, 'text': '我没有救9号'}]}
        atom = {'event_id': 'e1', 'speaker': 1, 'kind': 'witch_info', 'target': 9, 'value': 'poison', 'quote': '我没有救9号'}
        flags = check(json.dumps({'atoms': [atom]}), snapshot)[0]['review_flags']
        self.assertIn('possible_negation', flags)
        self.assertIn('poison_not_explicit_in_quote', flags)
