import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from werewolf_nlp.preprocess import convert


def fixture():
    roles = ['seer', 'witch', 'hunter'] + ['werewolf'] * 3 + ['simple_villager'] * 3
    return [{'event': 'roles', 'content': {'player': i, 'role': r, 'uid': 123}} for i, r in enumerate(roles, 1)]


class PublicSnapshotsTest(unittest.TestCase):
    def test_excludes_private_and_future_and_keeps_post_vote_shot(self):
        events = fixture() + [
            {'event': 'cycle_round', 'content': {'round': 1, 'status': 'night'}},
            {'event': 'inquired', 'content': {'is_werewolf': True, 'player': 4}},
            {'event': 'cycle_round', 'content': {'round': 1, 'status': 'day', 'context': '死亡玩家：3号玩家'}},
            {'event': 'speech', 'content': {'player': 1, 'day': '1-1', 'context': '我是预言家'}},
            {'event': 'voted', 'content': {'player': 1, 'voted_to_player': 4, 'reason': 'private secret'}},
            {'event': 'vote_results', 'content': {'1': 4, '2': 4, '4': 1}},
            {'event': 'shoot', 'content': {'player': 3, 'shoot_player': 5, 'reason': 'private secret'}},
            {'event': 'cycle_round', 'content': {'round': 2, 'status': 'night'}},
            {'event': 'end', 'winner': 'werewolves', 'content': None},
        ]
        public, labels, snapshots, warnings = convert(events, 'g')
        self.assertEqual(len(labels), 9)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual([x['type'] for x in snapshots[0]['public_events']], ['announcement', 'speech', 'vote_results', 'shoot'])
        self.assertNotIn('private secret', str(snapshots))
        self.assertNotIn('werewolves', str(snapshots))
        self.assertNotIn('true_role', str(snapshots))
        self.assertLess(snapshots[0]['cutoff_order'], len(events)-2)
        self.assertEqual(warnings, [])

    def test_tie_revote_and_unresolved_tie(self):
        prefix = fixture() + [{'event': 'cycle_round', 'content': {'round': 1, 'status': 'day'}}, {'event': 'vote_results', 'content': {'1': 2, '2': 1}}]
        end = [{'event': 'cycle_round', 'content': {'round': 2, 'status': 'night'}}]
        _, _, snaps, warnings = convert(prefix + end, 'g')
        self.assertEqual(snaps, [])
        self.assertEqual(warnings[0]['reason'], 'unresolved_tie')
        _, _, snaps, _ = convert(prefix + [{'event': 'vote_results', 'content': {'1': 2, '2': 2}}] + end, 'g')
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]['public_events'][-1]['eliminated_player'], 2)

    def test_invalid_labels_rejected(self):
        events = fixture(); events[0]['content']['role'] = 'werewolf'
        with self.assertRaises(ValueError):
            convert(events, 'g')


if __name__ == '__main__':
    unittest.main()
