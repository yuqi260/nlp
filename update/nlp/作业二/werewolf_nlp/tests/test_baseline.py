import json
import sys
import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from werewolf_nlp.baseline import assign_scores, make_messages, parse_response, call_api, validate_claims
from evaluate_baseline import evaluate


class BaselineTest(unittest.TestCase):
    def test_public_mask_does_not_count_known_correct_player(self):
        roles = ['seer', 'witch', 'hunter'] + ['werewolf']*3 + ['villager']*3
        labels = [{'game_id': 'g', 'player_id': i+1, 'true_role': role} for i, role in enumerate(roles)]
        assignments = [{'player_id': i+1, 'role': role} for i, role in enumerate(roles)]
        assignments[0]['role'] = 'werewolf'
        r = [{'snapshot_id': 's', 'game_id': 'g', 'day': 1, 'status': 'ok', 'assignments': assignments}]
        result = evaluate(r, labels, {'s': [3]})
        self.assertEqual(result['scored_players'], 8)
        self.assertEqual(result['player_accuracy_among_successes'], 7/8)
        empty = evaluate(r, labels, {'s': list(range(1, 10))})
        self.assertIsNone(empty['exact_snapshot_accuracy_among_successes'])

    def test_chat_api_request_and_response(self):
        response = {'model': 'qwen3-8b', 'usage': {'total_tokens': 10}, 'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
        with patch('werewolf_nlp.baseline.request.urlopen', return_value=BytesIO(json.dumps(response).encode())) as mocked:
            content, usage, model, finish = call_api('http://localhost/v1', 'test', 'qwen3-8b', [], .6, 20, 2, False)
        self.assertEqual((content, usage['total_tokens'], model, finish), ('{}', 10, 'qwen3-8b', 'stop'))
        req = mocked.call_args.args[0]
        self.assertEqual(req.full_url, 'http://localhost/v1/chat/completions')
        self.assertFalse(json.loads(req.data)['enable_thinking'])

    def test_input_whitelist_and_cutoff(self):
        snap = {'game_id':'g', 'day':1, 'cutoff_order':2, 'public_events':[{'game_id':'g', 'order':2, 'event_id':'e2', 'type':'speech', 'text':'hello', 'true_role':'secret'}]}
        self.assertNotIn('secret', str(make_messages(snap)))
        snap['public_events'][0]['order'] = 3
        with self.assertRaises(ValueError): make_messages(snap)

    def test_validation_and_metrics(self):
        roles = ['seer', 'witch', 'hunter'] + ['werewolf']*3 + ['villager']*3
        assignments = [{'player_id': i+1, 'role': role} for i, role in enumerate(roles)]
        pred, quota = parse_response(json.dumps({'assignments': assignments}))
        self.assertTrue(quota)
        labels = [{'game_id':'g', 'player_id':i, 'true_role':role} for i,role in pred.items()]
        results = evaluate([{'snapshot_id':'s', 'game_id':'g', 'day':1, 'status':'ok', 'assignments':assignments, 'role_counts_valid':True}, {'snapshot_id':'s2', 'game_id':'g', 'day':2, 'status':'failed'}], labels)
        self.assertEqual(results['player_accuracy_among_successes'], 1)
        self.assertEqual(results['role_macro_f1_among_successes'], 1)
        self.assertEqual(results['success_rate'], .5)
        assignments[-1]['player_id'] = 1
        with self.assertRaises(ValueError): parse_response(json.dumps({'assignments': assignments}))

    def test_rules_prompt_preserves_public_payload(self):
        event = {'game_id': 'g', 'order': 1, 'type': 'speech', 'text': '我是预言家', 'true_role': 'private'}
        snapshot = {'game_id': 'g', 'day': 1, 'cutoff_order': 1, 'public_events': [event]}
        direct = make_messages(snapshot)
        rules = make_messages(snapshot, 'rules_v1')
        self.assertEqual(direct[1], rules[1])
        self.assertTrue(rules[0]['content'].startswith(direct[0]['content']))
        self.assertIn('tied_players', rules[0]['content'])
        self.assertNotIn('private', str(rules))

    def test_speech_index_only_references_visible_events(self):
        snapshot = {'game_id': 'g', 'day': 1, 'cutoff_order': 2, 'public_events': [
            {'game_id': 'g', 'event_id': 'e1', 'day': 1, 'order': 1, 'type': 'speech', 'speaker': 2, 'text': 'hello'},
            {'game_id': 'g', 'event_id': 'e2', 'day': 1, 'order': 2, 'type': 'announcement', 'text': 'day'},
        ]}
        rules = json.loads(make_messages(snapshot, 'rules_v1')[1]['content'])
        indexed = json.loads(make_messages(snapshot, 'evidence_index_v1')[1]['content'])
        self.assertEqual(rules['events'], indexed['events'])
        self.assertEqual(indexed['speech_index_by_day_and_player'], {'1': {'2': ['e1']}})

    def test_claims_require_matching_public_speaker_and_quote(self):
        snapshot = {'public_events': [{'event_id': 'e1', 'type': 'speech', 'speaker': 2, 'text': '我是预言家，查验3号是狼人'}]}
        claim = {'event_id': 'e1', 'speaker': 2, 'kind': 'inspection_claim', 'quote': '查验3号是狼人'}
        self.assertEqual(validate_claims(json.dumps({'claims': [claim]}), snapshot), [claim])
        claim['quote'] = '查验4号是狼人'
        with self.assertRaises(ValueError):
            validate_claims(json.dumps({'claims': [claim]}), snapshot)
        claim['quote'] = '我是预言家'
        claim['speaker'] = 3
        with self.assertRaises(ValueError):
            validate_claims(json.dumps({'claims': [claim]}), snapshot)

    def test_score_assignment_enforces_quota_and_public_hunter(self):
        rows = [{'player_id': p, **{role: 0 for role in ('seer', 'witch', 'hunter', 'werewolf', 'villager')}} for p in range(1, 10)]
        snapshot = {'public_events': [{'type': 'shoot', 'speaker': 9}]}
        assignment, _, total, ties = assign_scores(json.dumps({'scores': rows}), snapshot)
        self.assertEqual(assignment[9], 'hunter')
        self.assertEqual(total, 0)
        self.assertEqual(ties, 1120)
        self.assertEqual(sorted(assignment.values()), sorted(['seer', 'witch', 'hunter'] + ['werewolf'] * 3 + ['villager'] * 3))
        rows[0]['seer'] = True
        with self.assertRaises(ValueError):
            assign_scores(json.dumps({'scores': rows}), snapshot)


if __name__ == '__main__': unittest.main()
