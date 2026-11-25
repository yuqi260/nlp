import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_naive_bayes import features


class NaiveBayesTest(unittest.TestCase):
    def test_features_use_only_player_speech_and_normalize_seat(self):
        snapshot = {'public_events': [
            {'type': 'speech', 'speaker': 1, 'text': '1号认为2号是狼'},
            {'type': 'speech', 'speaker': 2, 'text': '秘密文本'},
            {'type': 'announcement', 'speaker': 1, 'text': '不应读取'},
        ]}
        f = features(snapshot, 1)
        self.assertIn('自己', f)
        self.assertIn('他人', f)
        self.assertNotIn('秘密', f)
        self.assertNotIn('不应', f)
