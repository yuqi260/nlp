import unittest
from werewolf_nlp.baseline import ROLES
from werewolf_nlp.world_solver import solve


class WorldSolverTest(unittest.TestCase):
    def test_conditional_checks_and_ties(self):
        scores = {p: {r: 0.0 for r in ROLES} for p in range(1, 10)}
        snapshot = {'public_events': [{'event_id': 'e1', 'type': 'speech', 'speaker': 1, 'text': '我查验2号是狼'}]}
        claim = {'event_id': 'e1', 'speaker': 1, 'target': 2, 'result': 'wolf', 'quote': '查验2号是狼'}
        result = solve(scores, snapshot, [claim])
        self.assertEqual(result['optimal_assignments'], result['feasible_assignments'])
        self.assertLess(result['feasible_assignments'], 10080)
        scores[1]['seer'] = 100.1
        result = solve(scores, snapshot, [claim])
        self.assertEqual(result['assignment'][1], 'seer')
        self.assertEqual(result['assignment'][2], 'werewolf')
        scores[3]['seer'] = 200.2
        scores[2]['villager'] = 50.0
        result = solve(scores, snapshot, [claim])
        self.assertEqual(result['assignment'][3], 'seer')
        self.assertEqual(result['assignment'][2], 'villager')

    def test_rejects_bad_input(self):
        scores = {p: {r: 0 for r in ROLES} for p in range(1, 10)}
        with self.assertRaises(ValueError):
            solve(scores, {'public_events': []}, [{'speaker': 1, 'target': 2, 'result': 'wolf', 'event_id': 'missing', 'quote': 'x'}])
        scores[1]['seer'] = float('nan')
        with self.assertRaises(ValueError):
            solve(scores, {'public_events': []})


if __name__ == '__main__':
    unittest.main()
