"""Synthetic arrays here are isolated unit fixtures, never experiment/training data."""
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from zhixun.agent import LlmPolicy
from zhixun.data import ROOT, load_dataset
from zhixun.harness_evaluation import validate_response
from zhixun.metrics import classification_metrics
from zhixun.predict import predict_batch
from zhixun.prior_correction import (
    CoralAdapter,
    PriorCorrectedModel,
    bbse_prior,
    coral_probabilities,
    reweight,
)
from zhixun.reward import agent_reward, rank_guarded
from zhixun.vendor.model import Model


class PriorCorrectionTests(unittest.TestCase):
    def test_class_metrics_absent_class_and_zero_recall_are_explicit(self):
        p = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [1., 0, 0]])
        metrics, details = classification_metrics(p, np.array([0, 0, 1, 1]), ['a', 'b', 'c'])
        self.assertEqual(metrics['accuracy'], .75)
        self.assertEqual(metrics['balanced_accuracy'], .75)
        self.assertAlmostEqual(metrics['g_mean'], np.sqrt(.5))
        self.assertEqual(metrics['cohen_kappa'], .5)
        self.assertEqual(details['metric_conventions']['present_classes'], ['a', 'b'])
        metrics, _ = classification_metrics(np.array([[1., 0], [1., 0]]),
                                            np.array([0, 1]), ['a', 'b'])
        self.assertEqual(metrics['g_mean'], 0)
        with self.assertRaises(ValueError):
            classification_metrics(np.array([[2., 0]]), np.array([0]), ['a', 'b'])

    def test_past_labels_only_sequential_release_and_initial_prior_immutable(self):
        class ModelDouble:
            meta = {'classes': ['a', 'b']}

            def predict(self, x):
                return np.tile([.6, .4], (len(x), 1))

        pc = PriorCorrectedModel(ModelDouble(), window=1)
        pc.fit_prior(np.array([0, 1]))
        before = pc.predict(np.zeros((1, 2)))
        self.assertEqual(before.argmax(), 0)
        pc.release(np.array([1, 1, 1]), batch=2)
        self.assertEqual(pc.predict(np.zeros((1, 2))).argmax(), 1)
        self.assertTrue(np.array_equal(before, [[.6, .4]]))
        for batch in [2, 4]:
            with self.assertRaises(ValueError):
                pc.release(np.array([0, 1]), batch=batch)
        with self.assertRaises(ValueError):
            pc.fit_prior(np.array([1, 1]))

    def test_soft_bbse_recovers_known_mixture_without_target_labels(self):
        source = np.eye(3)[np.repeat(np.arange(3), 10)]
        labels = np.repeat(np.arange(3), 10)
        target = np.eye(3)[np.array([0] * 10 + [1] * 30 + [2] * 60)]
        prior, audit = bbse_prior(source, labels, target, ridge=1e-8)
        np.testing.assert_allclose(prior, [.1, .3, .6], atol=1e-6)
        self.assertFalse(audit['target_labels_used'])
        with self.assertRaises(ValueError):
            bbse_prior(source[:-10], labels[:-10], target)

    def test_inference_wrappers_preserve_model_weights_moments_and_normalization(self):
        model = Model.load(ROOT / 'data/A/checkpoint.npz')
        source = load_dataset(ROOT / 'data/A/train.jsonl', model, 'A')
        before = copy.deepcopy(model)
        p = model.predict(source.x)
        np.testing.assert_allclose(reweight(p, np.ones(6), np.ones(6)), p)
        np.testing.assert_allclose(coral_probabilities(model, source.x, source.x, strength=0), p,
                                   atol=1e-14)
        q = coral_probabilities(model, source.x, source.x[:50], strength=.5)
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(q.sum(axis=1), 1)
        for key in ['params', 'first', 'second']:
            for name in getattr(model, key):
                np.testing.assert_array_equal(getattr(model, key)[name], getattr(before, key)[name])
        np.testing.assert_array_equal(model.mean, before.mean)
        np.testing.assert_array_equal(model.scale, before.scale)
        self.assertEqual(model.meta, before.meta)

    def test_floor_ranking_rejects_cheaper_action_that_breaks_accuracy_floor(self):
        def row(accuracy, epochs):
            return {'action': {'epochs': epochs}, 'epochs': epochs,
                    'next_metrics': {'accuracy': accuracy}}
        outcomes = [row(.899, 0), row(.91, 5), row(.915, 10)]
        ranked = rank_guarded(outcomes, {'epochs': 5})
        self.assertEqual(ranked[0]['action']['epochs'], 5)
        self.assertFalse(ranked[-1]['floor_feasible'])
        self.assertLess(agent_reward(.899, 0, flat_baseline=.91), 0)
        with self.assertRaises(ValueError):
            agent_reward(float('nan'), 0, flat_baseline=.9)

    def test_new_policy_evaluation_checks_catalog_and_budget(self):
        hold = {'action': 'hold', 'epochs': 0, 'learning_rate': 0., 'target_version': None}
        train = {'action': 'retrain', 'epochs': 5, 'learning_rate': .001, 'target_version': None}
        context = {'action_catalog': [hold, train],
                   'observation': {'terminal': False, 'remaining_epochs': 2}}
        self.assertEqual(validate_response(json.dumps(hold), context), hold)
        with self.assertRaises(ValueError):
            validate_response(json.dumps(train), context)

    def test_portable_coral_batch_predictions_and_label_rejection(self):
        model = Model.load(ROOT / 'data/A/checkpoint.npz')
        data = load_dataset(ROOT / 'data/A/train.jsonl', model, 'A')
        adapter = CoralAdapter.from_source(model, data.x)
        folder = ROOT / '指导/临时'
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path = Path(temp) / 'adapter.npz'
            adapter.save(path)
            restored = CoralAdapter.load(path)
            np.testing.assert_allclose(adapter.predict(model, data.x),
                                       restored.predict(model, data.x), atol=1e-14)
            np.testing.assert_allclose(adapter.predict(model, data.x),
                                       coral_probabilities(model, data.x, data.x, strength=.25),
                                       atol=1e-14)
            original = json.loads((ROOT / 'data/A/train.jsonl').read_text(
                encoding='utf-8').splitlines()[0])
            with self.assertRaisesRegex(ValueError, 'only inputs'):
                predict_batch(ROOT / 'data/A/checkpoint.npz', path,
                              io.StringIO(json.dumps(original)), io.StringIO())
            stream = io.StringIO('\n'.join(json.dumps({'inputs': {'features': row.tolist()}})
                                           for row in data.x[:4]))
            output = io.StringIO()
            predict_batch(ROOT / 'data/A/checkpoint.npz', path, stream, output)
            self.assertEqual(len(output.getvalue().splitlines()), 4)

    def test_real_derived_corpus_floor_and_temporal_lineage(self):
        provenance_path = ROOT / 'data/harness-guarded-v4-expanded/provenance.json'
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        states = json.loads((ROOT / 'artifacts' / provenance['source_experiment'] /
                             'states.json').read_text(encoding='utf-8'))
        self.assertFalse(provenance['official_llm_corpus'])
        self.assertEqual(len(states), len({s['state_id'] for s in states}))
        for state in states:
            ranked = rank_guarded(state['outcomes'], state['flat_reference_action'])
            self.assertEqual(state['chosen'], ranked[0]['action'])
            self.assertTrue(ranked[0]['floor_feasible'])
            if state['split'] == 'train':
                self.assertLessEqual(state['observation']['batch'], 6)
                self.assertTrue(all(s['lookahead_batch'] <= 7 for s in state['source_lineage']))
            context = json.loads(state['prompt'].split('\n', 1)[1])
            self.assertEqual(set(context), {'action_catalog', 'observation'})
            self.assertNotIn('outcomes', context['observation'])

    def test_remote_timeout_cancels_future_and_records_explicit_fallback(self):
        config = json.loads((ROOT / 'configs/replay_guarded.json').read_text(encoding='utf-8'))
        row = json.loads((ROOT / 'data/harness-guarded-v4-expanded/train.jsonl').read_text(
            encoding='utf-8').splitlines()[0])
        observation = json.loads(row['prompt'].split('\n', 1)[1])['observation']

        class Future:
            cancelled = False

            def result(self, *, timeout):
                raise TimeoutError('unit-test-only remote wait')

            def cancel(self):
                self.cancelled = True

        future, reserved = Future(), []
        client = types.SimpleNamespace(sample=lambda **kw: future)
        tokenizer = types.SimpleNamespace(encode=lambda *args, **kw: [1, 2, 3])
        sdk = types.SimpleNamespace(ModelInput=types.SimpleNamespace(from_ints=lambda **kw: kw),
                                    SamplingParams=lambda **kw: kw)
        policy = LlmPolicy(config, model='unit-only', client=client, tokenizer=tokenizer,
                           before_request=lambda: reserved.append(True))
        with patch.dict('sys.modules', {'pytrio': sdk}):
            _, audit = policy.decide(observation)
        self.assertTrue(future.cancelled)
        self.assertEqual(len(reserved), 1)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(audit['source'], 'fallback_rule')
        self.assertIn('TimeoutError', audit['fallback_reason'])


if __name__ == '__main__':
    unittest.main()
