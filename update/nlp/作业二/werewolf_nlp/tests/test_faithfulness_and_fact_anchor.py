import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.baseline import make_messages, parse_response
from werewolf_nlp.behavior import extract_hard_facts
from werewolf_nlp.faithfulness import audit_record_faithfulness


def _sample_snapshot():
    return {
        'snapshot_id': 'g1_day1',
        'game_id': 'g1',
        'day': 1,
        'cutoff_order': 5,
        'public_events': [
            {'event_id': 'e0001', 'game_id': 'g1', 'day': 1, 'order': 1, 'type': 'announcement', 'text': '今天是平安夜'},
            {'event_id': 'e0002', 'game_id': 'g1', 'day': 1, 'order': 2, 'type': 'speech', 'speaker': 1, 'text': '我是预言家，查杀3号'},
            {'event_id': 'e0003', 'game_id': 'g1', 'day': 1, 'order': 3, 'type': 'speech', 'speaker': 3, 'text': '我才是真预言家'},
            {
                'event_id': 'e0004', 'game_id': 'g1', 'day': 1, 'order': 4, 'type': 'vote_results',
                'votes': [
                    {'player': 1, 'target': 3},
                    {'player': 2, 'target': 3},
                    {'player': 3, 'target': 1},
                    {'player': 4, 'target': None},
                ],
                'eliminated_player': 3,
                'tied_players': [],
            },
            {'event_id': 'e0005', 'game_id': 'g1', 'day': 1, 'order': 5, 'type': 'shoot', 'speaker': 3, 'target': 8},
        ],
    }


def _valid_assignments(hunter_id=3):
    roles = {1: 'seer', 2: 'witch', 3: 'hunter', 4: 'werewolf', 5: 'werewolf', 6: 'werewolf', 7: 'villager', 8: 'villager', 9: 'villager'}
    if hunter_id != 3:
        roles[3], roles[hunter_id] = roles[hunter_id], 'hunter'
    return [{'player_id': p, 'role': r} for p, r in sorted(roles.items())]


class FactAnchorAndFaithfulnessTest(unittest.TestCase):
    def test_extract_hard_facts_and_prompt_payload(self):
        snap = _sample_snapshot()
        facts = extract_hard_facts(snap)
        self.assertEqual(len(facts['announcements']), 1)
        self.assertEqual(facts['vote_outcomes'][0]['tally_by_target'], {'1': [3], '3': [1, 2]})
        self.assertEqual(facts['vote_outcomes'][0]['abstainers'], [4])
        self.assertEqual(facts['confirmed_hunters'][0]['confirmed_hunter'], 3)
        self.assertEqual(facts['confirmed_hunters'][0]['shot_target_role_unknown'], 8)

        messages = make_messages(snap, 'fact_anchor_v1')
        self.assertEqual(len(messages), 2)
        payload = json.loads(messages[1]['content'])
        self.assertIn('hard_facts', payload)
        self.assertEqual(len(payload['events']), 5)

    def test_fact_anchor_output_compatible_with_parse_response(self):
        raw = json.dumps({
            'hypotheses': [{'seer_candidate': 1, 'support': 'e0002 自称预言家', 'opposition': 'e0003 对跳'}],
            'assignments': [
                {'player_id': item['player_id'], 'role': item['role'], 'evidence_ids': ['e0002'], 'reason': '测试依据'}
                for item in _valid_assignments()
            ],
        }, ensure_ascii=False)
        parsed, quota_ok = parse_response(raw)
        self.assertTrue(quota_ok)
        self.assertEqual(parsed[1], 'seer')

    def test_faithfulness_flags_overclaims_invalid_ids_and_hunter_mismatch(self):
        snap = _sample_snapshot()
        raw = json.dumps({
            'hypotheses': [{
                'seer_candidate': 1,
                'support': '3号全票出局证实是狼人，好人已经获胜，见 e0004 与 e9999',
                'opposition': '3号开枪带走8号确认8号是好人',
            }],
            'assignments': _valid_assignments(hunter_id=7),
        }, ensure_ascii=False)
        rec = {
            'snapshot_id': 'g1_day1',
            'game_id': 'g1',
            'day': 1,
            'status': 'ok',
            'raw_output': raw,
            'assignments': _valid_assignments(hunter_id=7),
        }
        res = audit_record_faithfulness(rec, snap)
        self.assertFalse(res['clean'])
        self.assertEqual(len(res['invalid_event_citations']), 1)
        self.assertEqual(res['invalid_event_citations'][0]['event_id'], 'e9999')
        codes = {f['code'] for f in res['flags']}
        self.assertIn('unannounced_game_outcome', codes)
        self.assertIn('false_unanimous_vote_claim', codes)
        self.assertIn('shoot_target_treated_as_confirmed_role', codes)
        self.assertIn('confirmed_hunter_mismatch', codes)
        self.assertFalse(res['confirmed_hunter_consistent'])

    def test_faithfulness_clean_record_passes(self):
        snap = _sample_snapshot()
        raw = json.dumps({
            'hypotheses': [{
                'seer_candidate': 1,
                'support': 'e0002 中1号自称预言家并查杀3号，e0004 中1、2号投3号',
                'opposition': 'e0005 显示3号实际开枪，确认3号为猎人，与1号查杀矛盾',
            }],
            'assignments': _valid_assignments(hunter_id=3),
        }, ensure_ascii=False)
        rec = {
            'snapshot_id': 'g1_day1',
            'game_id': 'g1',
            'day': 1,
            'status': 'ok',
            'raw_output': raw,
            'assignments': _valid_assignments(hunter_id=3),
        }
        res = audit_record_faithfulness(rec, snap)
        self.assertTrue(res['clean'])
        self.assertTrue(res['confirmed_hunter_consistent'])
        self.assertEqual(res['unique_cited_events'], ['e0002', 'e0004', 'e0005'])


if __name__ == '__main__':
    unittest.main()
