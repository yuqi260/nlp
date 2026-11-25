import copy
import unittest
from werewolf_nlp.wolfgame_adapter import convert, public_identity_masks
from werewolf_nlp.baseline import make_messages


def event(seq, day, phase, kind, text, meta=None, actor=None):
    return {'sequence': seq, 'day': day, 'phase': phase, 'kind': kind, 'content': text, 'actor_id': actor, 'metadata': meta or {}}


class WolfgameAdapterTest(unittest.TestCase):
    def setUp(self):
        names = list('甲乙丙丁戊己庚辛壬')
        roles = ['seer', 'witch', 'hunter'] + ['werewolf'] * 3 + ['villager'] * 3
        self.state = {'players': [{'name': n, 'role': r, 'player_id': 'secret'+str(i), 'seat_ref': str(9-i)+'号'} for i, (n, r) in enumerate(zip(names, roles))]}
        roster = '本局共有 9 名玩家：' + '、'.join(names) + '。'
        self.events = [event(1, 0, 'setup', 'gm_announcement', roster, {'event_type': 'game_started'}),
            event(2, 1, 'day_speech', 'player_speech', '甲：我是预言家', actor='p1'),
            event(3, 1, 'day_vote', 'gm_announcement', '投票明细：甲投给乙；乙投给甲。', {'event_type': 'day_vote_result', 'votes': [{'voter_id':'p1','target_id':'p2'}, {'voter_id':'p2','target_id':'p1'}], 'leaders':['p2'], 'eliminated':'p2'}),
            event(4, 1, 'hunter_action', 'gm_announcement', '甲 翻开猎人身份，可以选择开枪', {'event_type':'hunter_revealed','hunter_id':'p1','death_cause':'SECRET'}),
            event(5, 2, 'night', 'gm_announcement', '第二夜开始', {'event_type':'night_started'}),
            event(6, 2, 'finished', 'gm_announcement', '狼人获胜', {'event_type':'game_finished'}),
            event(7, 2, 'finished', 'role_reveal', '所有真实身份', {})]

    def test_public_cutoff_and_private_state_independence(self):
        snapshots, labels = convert(self.events, self.state, 'g')
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['cutoff_order'], 4)
        vote = next(e for e in snapshots[0]['public_events'] if e['type'] == 'vote_results')
        self.assertEqual(vote['eliminated_player'], 2)
        self.assertEqual(vote['text'], '投票明细：甲投给乙；乙投给甲。')
        self.assertNotIn('SECRET', str(snapshots))
        self.assertNotIn('狼人获胜', str(snapshots))
        self.assertNotIn('所有真实身份', str(snapshots))
        self.assertEqual(snapshots[0]['public_events'][-1]['type'], 'identity_reveal')
        self.assertEqual(public_identity_masks(snapshots)['g:day1'], [1])
        other = copy.deepcopy(self.state)
        other['players'][0]['role'], other['players'][1]['role'] = other['players'][1]['role'], other['players'][0]['role']
        other['players'][0]['alive'] = 'changed secret'
        snapshots2, labels2 = convert(self.events, other, 'g')
        self.assertEqual(snapshots, snapshots2)
        self.assertNotEqual(labels, labels2)
        messages = make_messages(snapshots[0], 'wolfgame_direct_v1')
        self.assertIn('player_directory', messages[1]['content'])
        self.assertNotIn('secret', str(messages))

    def test_reject_unresolved_vote(self):
        self.events[2]['metadata'].update(leaders=['p1','p2'], eliminated=None)
        snapshots, _ = convert(self.events, self.state, 'g')
        self.assertEqual(snapshots, [])
