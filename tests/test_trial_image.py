"""Exercise the real trial continuation path, without an engine or fabricated labels."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zhixun.data import ROOT, digest


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        'trial_entrypoint', ROOT / 'submission/trial_entrypoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrialImageTests(unittest.TestCase):
    def test_existing_cli_is_forwarded_without_starting_default_training(self):
        entry = load_entrypoint()
        with patch('run.main', return_value=0) as command, patch.object(entry.sys, 'argv', []):
            self.assertEqual(entry.main(['predict', '--checkpoint', 'example.npz']), 0)
            command.assert_called_once_with()
            self.assertEqual(entry.sys.argv[1:], ['predict', '--checkpoint', 'example.npz'])

    def test_invalid_limits_do_not_create_training_output(self):
        entry = load_entrypoint()
        with self.assertRaises(ValueError):
            entry.run_trial('A', 11, 1, ROOT / 'must-not-be-created')

    def test_real_rounds_chain_selected_checkpoint_and_keep_original_inputs(self):
        entry = load_entrypoint()
        original = ROOT / 'data/A/checkpoint.npz'
        initial = digest(original)
        with tempfile.TemporaryDirectory(dir=ROOT / '指导/临时') as temp:
            with patch('zhixun.train.Recorder') as recorder:
                recorder.return_value.reference.return_value = {'mode': 'unit-test-local'}
                result = entry.run_trial('A', 2, 1, Path(temp))
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['official_runtime_protocol_verified'])
            rounds = result['rounds']
            self.assertEqual(len(rounds), 2)
            second_report = Path(rounds[1]['checkpoint']).parent / 'run.json'
            stored = json.loads(second_report.read_text(encoding='utf-8'))
            self.assertEqual(stored['parent_checkpoint_sha256'],
                             digest(Path(rounds[0]['checkpoint'])))
            self.assertEqual(digest(original), initial)


if __name__ == '__main__':
    unittest.main()
