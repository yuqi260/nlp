import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from werewolf_nlp.local_config import load_local_config


class LocalConfigTest(unittest.TestCase):
    def test_load_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('# comment\nWEREWOLF_API_KEY="test-key"\nWEREWOLF_API_BASE_URL=https://example.test/v1\n', encoding='utf8')
            load_local_config(path)
            self.assertEqual(os.environ['WEREWOLF_API_KEY'], 'test-key')
            os.environ['WEREWOLF_API_KEY'] = 'override'
            load_local_config(path)
            self.assertEqual(os.environ['WEREWOLF_API_KEY'], 'override')
            path.write_text('UNEXPECTED=value', encoding='utf8')
            with self.assertRaises(ValueError):
                load_local_config(path)
