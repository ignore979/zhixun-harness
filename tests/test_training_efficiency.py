"""Unit fixtures only; no artificial rows become experiment training/evaluation data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_loop import TestRecorder

from zhixun.data import ROOT, Dataset, load_dataset
from zhixun.efficiency import run_capacity
from zhixun.llm_policy import parse_decision
from zhixun.loop import EvaluationBatch
from zhixun.model_extension import ClassWeightedModel, class_weights, expand_hidden
from zhixun.policy import UpdatePlan
from zhixun.pool import TrainingPool
from zhixun.registry import ModelRegistry
from zhixun.replay import execute_replay, load_replay_config
from zhixun.training_schedule import lr_at, schedule_epochs
from zhixun.vendor.model import Model


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "指导/临时/efficiency-tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.x = np.random.default_rng(7).normal(size=(30, 3))
        self.y = np.array([0] * 25 + [1] * 5)
        model = Model("classification", ["a", "b", "c"], ["major", "minor"],
                      self.x, self.y, hidden=4)
        self.path = self.folder / "initial.npz"
        model.save(self.path)

    def test_weights_are_train_only_mean_one_and_handle_absent_classes(self):
        weights = class_weights(self.y, 3, 1.5)
        self.assertAlmostEqual(float(weights[self.y].mean()), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertTrue(np.isfinite(weights).all())
        with self.assertRaises(ValueError):
            class_weights(self.y, 3, float("nan"))

    def test_zero_power_matches_original_training_exactly(self):
        original = Model.load(self.path)
        weighted = ClassWeightedModel.load(self.path)
        original.fit(self.x, self.y, 2)
        weighted.fit(self.x, self.y, 2, cw_power=0)
        for group in ["params", "first", "second"]:
            for key in getattr(original, group):
                np.testing.assert_array_equal(getattr(original, group)[key],
                                              getattr(weighted, group)[key])

    def test_weighted_resume_is_identical_and_keeps_normalization(self):
        first, second = ClassWeightedModel.load(self.path), ClassWeightedModel.load(self.path)
        inherited = first.mean.copy()
        first.fit(self.x, self.y, 3, cw_power=1.5)
        second.fit(self.x, self.y, 1, cw_power=1.5)
        second.save(self.folder / "weighted.npz")
        second = ClassWeightedModel.load(self.folder / "weighted.npz")
        second.fit(self.x, self.y, 2, cw_power=1.5)
        np.testing.assert_array_equal(first.predict(self.x), second.predict(self.x))
        np.testing.assert_array_equal(first.mean, inherited)
        self.assertEqual(first.meta["steps"], second.meta["steps"])

    def test_weighted_gradient_changes_parameters_relative_to_unweighted(self):
        first, second = ClassWeightedModel.load(self.path), Model.load(self.path)
        first.fit(self.x, self.y, 1, cw_power=1.5)
        second.fit(self.x, self.y, 1)
        self.assertFalse(np.array_equal(first.params["b2"], second.params["b2"]))

    def test_b_width_migration_preserves_function_moments_and_parent(self):
        model = Model.load(ROOT / "data/B/checkpoint.npz")
        train = load_dataset(ROOT / "data/B/train.jsonl", model, "B")
        predicted = model.predict(train.x[:100])
        with patch.object(Model, "__init__", side_effect=AssertionError("no reinitialization")):
            expanded = expand_hidden(model, 96)
        self.assertEqual(expanded.meta["parameter_count"], 3457)
        np.testing.assert_allclose(expanded.predict(train.x[:100]), predicted,
                                   atol=1e-10, rtol=1e-12)
        for name in ["first", "second"]:
            source, target = getattr(model, name), getattr(expanded, name)
            np.testing.assert_array_equal(source["w1"], target["w1"][:, :24])
            np.testing.assert_array_equal(source["w2"], target["w2"][:24])
            self.assertTrue(np.all(target["w1"][:, 24:] == 0))
        self.assertEqual(model.meta["hidden"], 24)
        self.assertEqual(expanded.meta["steps"], model.meta["steps"])
        np.testing.assert_array_equal(expanded.mean, model.mean)
        expanded.params["w1"][0, 0] += 1
        np.testing.assert_array_equal(model.predict(train.x[:100]), predicted)

    def test_classification_width_migration_preserves_all_probabilities(self):
        model = Model.load(self.path)
        expanded = expand_hidden(model, 12)
        np.testing.assert_allclose(expanded.predict(self.x), model.predict(self.x), atol=1e-14)
        expanded.save(self.folder / "expanded.npz")
        np.testing.assert_array_equal(Model.load(self.folder / "expanded.npz").predict(self.x),
                                      expanded.predict(self.x))

    def test_invalid_width_migration_is_rejected(self):
        model = Model.load(self.path)
        for width in [4, 2, 129, True]:
            with self.assertRaises(ValueError):
                expand_hidden(model, width)

    def test_history_sampling_is_bounded_reproducible_and_keeps_each_chunk(self):
        def data(name, count, shift):
            return Dataset(tuple(f"{name}-{i}" for i in range(count)),
                           np.arange(count * 3).reshape(count, 3).astype(float) + shift,
                           np.arange(count) % 2, frozenset({name}))

        pool = TrainingPool.from_dataset(data("old", 3, 0))
        pool.append(data("new", 30, 10000), batch_id="new")
        first, info = pool.sample(cap_rows=10, seed=42)
        second, again = pool.sample(cap_rows=10, seed=42)
        self.assertEqual(len(first.ids), 10)
        self.assertEqual(first.ids, second.ids)
        self.assertEqual(info, again)
        self.assertEqual(info["per_chunk_rows"], [3, 7])
        self.assertEqual(len(set(first.ids)), 10)
        self.assertTrue(set(first.ids) <= set(pool.dataset().ids))
        with self.assertRaises(ValueError):
            pool.sample(cap_rows=1)

    def test_low_psi_does_not_veto_real_performance_drop(self):
        self.assertGreaterEqual(schedule_epochs(0, 0.2, 5000), 5)
        self.assertLessEqual(schedule_epochs(100, 1.0, 100000), 20)

    def test_learning_rate_boundary_is_fixed_in_advance(self):
        rates = [lr_at(i, 10, 0.001) for i in range(10)]
        self.assertEqual(rates, [0.001] * 6 + [0.0005] * 4)

    def test_policy_proposals_accept_twenty_epochs_but_reject_twenty_one(self):
        UpdatePlan("retrain", epochs=20, learning_rate=0.001)
        plan = {"action": "retrain", "epochs": 20,
                "learning_rate": 0.001, "target_version": None}
        parse_decision(json.dumps(plan))
        with self.assertRaises(ValueError):
            parse_decision(json.dumps({**plan, "epochs": 21}))

    def test_public_b_training_uses_real_train_and_preserves_optimizer(self):
        original = Model.fit
        source = Model.load(ROOT / "data/B/checkpoint.npz")
        data = load_dataset(ROOT / "data/B/train.jsonl", source, "B")

        def checked(model, x, y, epochs):
            self.assertEqual(model.meta["hidden"], 96)
            np.testing.assert_array_equal(x, data.x)
            np.testing.assert_array_equal(y, data.y)
            return original(model, x, y, epochs)

        with (patch("zhixun.efficiency.Recorder", TestRecorder),
              patch.object(Model, "fit", checked)):
            report = run_capacity(hidden=96, epochs=2, output=self.folder)
        self.assertEqual(report["train_rows"], 7032)
        self.assertEqual(report["validation_rows"], 1679)
        self.assertEqual(report["final_steps"] - report["initial_steps"],
                         2 * int(np.ceil(7032 / source.meta["batch_size"])))

    def test_weighted_sampled_replay_never_trains_on_current_batch(self):
        initial = ROOT / "data/A/checkpoint.npz"
        model = Model.load(initial)
        train = load_dataset(ROOT / "data/A/train.jsonl", model, "A")
        current = [load_dataset(ROOT / "data/A/validation.jsonl", model, "A"),
                   load_dataset(ROOT / "data/replay-gas/batch3.jsonl", model, "A")]
        batches = [EvaluationBatch(i, f"test-real-source-{i}", d, "unit-test-only")
                   for i, d in enumerate(current)]
        config = load_replay_config(ROOT / "configs/replay_weighted.json")
        config["training_upgrade"]["pool_cap_rows"] = 10
        registry = ModelRegistry(self.folder, {"unit_test_only": True}, initial)
        fit = ClassWeightedModel.fit
        calls = []

        def checked(candidate, x, y, epochs, **kwargs):
            batch = len(calls) // config["epochs"]
            past = [train] if batch == 0 else [train, current[0]]
            eligible = {row.tobytes(): int(label) for d in past
                        for row, label in zip(d.x, d.y, strict=True)}
            self.assertEqual(len(x), 10)
            for row, label in zip(x, y, strict=True):
                self.assertIn(row.tobytes(), eligible)
                self.assertEqual(int(label), eligible[row.tobytes()])
            calls.append(len(x))
            return fit(candidate, x, y, epochs, **kwargs)

        with patch.object(ClassWeightedModel, "fit", checked):
            execute_replay(registry, train, batches, initial, "always", config, TestRecorder())
        self.assertEqual(len(calls), 10)


if __name__ == "__main__":
    unittest.main()
