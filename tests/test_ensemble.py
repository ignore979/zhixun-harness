"""Temporary models and synthetic labels below are unit fixtures only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_loop import TestRecorder

from zhixun.data import ROOT, load_dataset
from zhixun.domain_views import (
    BlockCoralAdapter,
    domain_transform,
    load_domain_adapter,
    view_probabilities,
)
from zhixun.ensemble import EnsemblePredictor, combine_predictions, temporal_versions
from zhixun.ensemble_study import select_development
from zhixun.fair_replay import run_fair
from zhixun.prior_correction import coral_probabilities
from zhixun.registry import ModelRegistry
from zhixun.vendor.model import Model
from zhixun.view_selector import ViewSelector


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '指导/临时').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '指导/临时')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.base = Model.load(ROOT / 'data/A/checkpoint.npz')
        self.data = load_dataset(ROOT / 'data/A/train.jsonl', self.base, 'A')
        self.registry = ModelRegistry(self.folder, {}, ROOT / 'data/A/checkpoint.npz')
        future = copy.deepcopy(self.base)
        future.params['b2'][0] += 100
        path = self.folder / 'future.npz'
        future.save(path)
        self.registry.register(1, path, parent=0, adopted=True)
        self.registry.register(2, path, parent=0, adopted=False)
        self.registry.state['current'] = 1

    def test_complete_registry_cannot_leak_future_adopted_versions(self):
        predictor = EnsemblePredictor(self.registry, k=5, allowed_versions={0}, current_version=0)
        output = predictor.predict(self.data.x, 'classification', batch=2)
        np.testing.assert_allclose(output, self.base.predict(self.data.x), atol=1e-14)
        self.assertEqual(predictor.last_audit['versions'], [0])
        with self.assertRaises(ValueError):
            EnsemblePredictor(self.registry, allowed_versions={0}).predict(self.data.x)

    def test_k1_matches_incumbent_and_rejected_version_is_excluded(self):
        predictor = EnsemblePredictor(self.registry, k=1)
        output = predictor.predict(self.data.x, batch=2)
        expected = Model.load(self.folder / 'future.npz').predict(self.data.x)
        np.testing.assert_allclose(output, expected, atol=1e-14)
        predictor = EnsemblePredictor(self.registry, k=None)
        predictor.predict(self.data.x, batch=2)
        self.assertEqual(predictor.last_audit['versions'], [0, 1])

    def test_temporal_cutoff_uses_committed_events_instead_of_version_number(self):
        events = [{'batch_index': 0, 'parent_version': 0, 'selected_version': 6},
                  {'batch_index': 1, 'parent_version': 6, 'selected_version': 6},
                  {'batch_index': 2, 'parent_version': 6, 'selected_version': 15}]
        self.assertEqual(temporal_versions(events, 0), {0})
        self.assertEqual(temporal_versions(events, 2), {0, 6})
        with self.assertRaises(ValueError):
            temporal_versions([{**events[0], 'parent_version': 15}], 0)

    def test_reliability_updates_only_after_release_and_cannot_repeat(self):
        predictor = EnsemblePredictor(self.registry, k=3, mode='class_reliability')
        x = self.data.x[:10]
        predictor.predict(x, batch=2)
        before = predictor.weights(predictor._adopted(), 6).copy()
        self.assertEqual(predictor.history, {})
        with self.assertRaises(ValueError):
            predictor.predict(x, batch=3)
        predictor.release(np.arange(10) % 6, batch=2)
        self.assertFalse(np.array_equal(before, predictor.weights(predictor._adopted(), 6)))
        with self.assertRaises(ValueError):
            predictor.release(np.arange(10) % 6, batch=2)
        self.assertTrue(all(r['batch'] == 2 for rows in predictor.history.values() for r in rows))

    def test_class_weights_and_regression_are_finite_normalized(self):
        values = np.array([[[.9, .1]], [[.1, .9]]])
        output = combine_predictions(values, [[9., 1.], [1., 9.]], 'classification')
        np.testing.assert_allclose(output, [[.5, .5]])
        out = combine_predictions([[2., 4.], [6., 8.]], [1., 3.], 'regression')
        np.testing.assert_allclose(out, [5., 7.])
        with self.assertRaises(ValueError):
            combine_predictions(values, [0, 0], 'classification')

    def test_development_choice_ignores_better_audit_answer(self):
        def row(accuracy, audit, recall=.4):
            metrics = {'accuracy': accuracy, 'mean_macro_f1': .6, 'mean_g_mean': .4,
                       'pooled_per_class_recall': {'Acetaldehyde': recall}}
            return {'spec': {'strength': .25}, 'summary': {'development': metrics,
                     'audit': {**metrics, 'accuracy': audit}}}
        baseline = row(.7, .7)
        best_development, best_audit = row(.8, .1), row(.75, .99)
        selected = select_development([best_development, best_audit], baseline)
        self.assertIs(selected, best_development)
        self.assertIs(select_development([row(.9, .9, recall=.1)], baseline), baseline)

    def test_bad_settings_and_changed_frozen_normalization_are_rejected(self):
        for settings in [{'k': 0}, {'decay': float('nan')}, {'mode': 'unknown'}]:
            with self.assertRaises(ValueError):
                EnsemblePredictor(self.registry, **settings)
        changed = copy.deepcopy(self.base)
        changed.mean[0] += 1
        path = self.folder / 'changed.npz'
        changed.save(path)
        self.registry.register(3, path, parent=0, adopted=True)
        self.registry.state['current'] = 3
        with self.assertRaises(ValueError):
            EnsemblePredictor(self.registry, k=None).predict(self.data.x)

    def test_split_alignment_matches_previous_coral_and_does_not_change_model(self):
        before = copy.deepcopy(self.base)
        target = self.data.x[:100]
        predicted = view_probabilities(self.base, self.data, target,
                                        {'kind': 'coral', 'strength': .25})
        np.testing.assert_allclose(predicted, coral_probabilities(
            self.base, self.data.x, target, strength=.25), atol=1e-14)
        for key in ['params', 'first', 'second']:
            for name in getattr(self.base, key):
                np.testing.assert_array_equal(getattr(before, key)[name],
                                              getattr(self.base, key)[name])
        np.testing.assert_array_equal(self.base.mean, before.mean)

    def test_class_shift_rejects_current_labels_and_starts_at_identity(self):
        spec = {'kind': 'past_class_shift', 'strength': .5, 'window': 2}
        z = domain_transform(self.base, self.data, self.data.x[:10], spec, current_batch=2)
        np.testing.assert_array_equal(z, self.base.normalized(self.data.x[:10]))
        with self.assertRaises(ValueError):
            domain_transform(self.base, self.data, self.data.x[:10], spec,
                             past=[(2, self.data)], current_batch=2)

    def test_block_and_diagonal_views_are_finite_and_validate_index_partition(self):
        for kind in ['block_coral', 'diagonal', 'split_coral', 'multi_view']:
            p = view_probabilities(self.base, self.data, self.data.x[:100],
                                   {'kind': kind, 'strength': .25})
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError):
            domain_transform(self.base, self.data, self.data.x[:100],
                             {'kind': 'block_coral', 'block_size': 7})

    def test_delayed_view_selector_has_no_access_to_current_or_future_labels(self):
        selector = ViewSelector(['a', 'b'], baseline=0, experts=2, window=1)
        p = np.array([[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]])
        np.testing.assert_allclose(selector.predict(p, batch=2), p[0])
        with self.assertRaises(ValueError):
            selector.predict(p, batch=3)
        selector.release(np.array([1, 0]), batch=2)
        np.testing.assert_allclose(selector.predict(p, batch=3), p[1])
        self.assertEqual(selector.history[0]['batch'], 2)
        with self.assertRaises(ValueError):
            selector.release(np.array([0, 1]), batch=4)

    def test_live_ensemble_resume_preserves_feedback_without_retraining(self):
        with patch('zhixun.fair_replay.Recorder', TestRecorder):
            result = run_fair('flat', root=ROOT, output=self.folder,
                              prediction_config=ROOT / 'configs/predict_ensemble_k3_coral.json')
            session = ROOT / result['session']
            with patch.object(Model, 'fit', side_effect=AssertionError('no repeated training')):
                resumed = run_fair('flat', root=ROOT, output=self.folder, resume=session,
                    prediction_config=ROOT / 'configs/predict_ensemble_k3_coral.json')
        self.assertEqual(result['summary'], resumed['summary'])
        self.assertEqual(result['events'][0]['ensemble']['versions'], [0])
        self.assertEqual(result['summary']['epochs_charged'], 45)

    def test_portable_block_adapter_matches_study_and_survives_serialization(self):
        adapter = BlockCoralAdapter.from_source(self.base, self.data.x)
        expected = view_probabilities(self.base, self.data, self.data.x[:100],
                                     {'kind': 'block_coral', 'strength': .25, 'block_size': 8})
        path = self.folder / 'adapter.npz'
        adapter.save(path)
        restored = load_domain_adapter(path)
        np.testing.assert_allclose(restored.predict(self.base, self.data.x[:100]), expected,
                                   atol=1e-14)


if __name__ == '__main__':
    unittest.main()
