"""Artificial fixtures below are isolated engineering tests, never training evidence."""
import dataclasses
import json
import tempfile
import unittest
import zipfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_loop import TestRecorder

from zhixun.cli import doctor
from zhixun.data import ROOT, Dataset
from zhixun.loop import EvaluationBatch
from zhixun.pool import TrainingPool
from zhixun.prepare_gas import convert, verify_replay
from zhixun.registry import ModelRegistry
from zhixun.replay import decide, execute_replay, load_replay_config
from zhixun.vendor.model import Model


class ReplayTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "指导/临时/replay-tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        x = np.column_stack([np.arange(40), np.arange(40) % 3]).astype(float)
        y = np.zeros(40, dtype=int)
        self.train = Dataset(tuple(f"t-{i}" for i in range(40)), x, y, frozenset({"train"}))
        model = Model("classification", ["x", "z"], ["zero", "one"], x, y, hidden=4)
        model.meta["profile"] = "A"
        self.initial = self.folder / "initial.npz"
        model.save(self.initial)
        self.registry = ModelRegistry(self.folder, {"unit_test_only": True}, self.initial)
        self.config = load_replay_config(ROOT / "configs/replay.json")

    def batches(self):
        return [EvaluationBatch(i, f"fixture-{i}", Dataset(
            tuple(f"b{i}-{j}" for j in range(40)), self.train.x + 100 * (i + 1),
            self.train.y, frozenset({f"group-{i}"})), "test-fixture-only") for i in range(2)]

    def test_pool_rejects_duplicate_groups_ids_and_unimplemented_partial_labels(self):
        pool = TrainingPool.from_dataset(self.train)
        first = self.batches()[0]
        pool.append(first.data, batch_id=first.identity)
        with self.assertRaises(ValueError):
            pool.append(first.data, batch_id=first.identity)
        with self.assertRaises(ValueError):
            pool.append(self.batches()[1].data, batch_id="second", label_frac=0.5)
        self.assertEqual(pool.summary()["rows"], 80)
        self.assertEqual(len(pool.provenance), 2)

    def test_pool_training_excludes_current_and_future_batches(self):
        original = Model.fit
        calls = []

        def checked(model, x, y, epochs):
            batch_index = len(calls) // self.config["epochs"]
            expected = self.train.x if batch_index == 0 else np.concatenate([
                self.train.x, self.batches()[0].data.x
            ])
            np.testing.assert_array_equal(x, expected)
            calls.append(len(x))
            return original(model, x, y, epochs)

        with patch.object(Model, "fit", checked):
            events = execute_replay(self.registry, self.train, self.batches(), self.initial,
                                    "always", self.config, TestRecorder())
        self.assertEqual(calls, [40] * 5 + [80] * 5)
        self.assertEqual(events[0]["pool_before"]["rows"], 40)
        self.assertEqual(events[1]["pool_after"]["rows"], 120)
        selected = Model.load(self.registry.checkpoint(self.registry.state["current"]))
        np.testing.assert_array_equal(selected.mean, self.train.x.mean(axis=0))

    def test_resume_restores_pool_without_duplicating_released_labels(self):
        execute_replay(self.registry, self.train, self.batches(), self.initial,
                       "static", self.config, TestRecorder())
        registry = ModelRegistry(self.folder, {"unit_test_only": True})
        with patch.object(Model, "fit", side_effect=AssertionError("committed")):
            execute_replay(registry, self.train, self.batches(), self.initial,
                           "static", self.config, TestRecorder())
        self.assertEqual(registry.state["pool"]["rows"], 120)
        self.assertEqual(len(registry.state["pool"]["provenance"]), 3)

    def test_interrupted_update_does_not_release_current_labels(self):
        with patch.object(Model, "fit", side_effect=RuntimeError("test interruption")):
            with self.assertRaises(RuntimeError):
                execute_replay(self.registry, self.train, self.batches(), self.initial,
                               "always", self.config, TestRecorder())
        stored = ModelRegistry(self.folder, {"unit_test_only": True})
        self.assertEqual(stored.state["cursor"], 0)
        self.assertNotIn("pool", stored.state)
        self.assertEqual(stored.state["updates_charged"], 1)

    def test_modified_pool_provenance_is_rejected_on_resume(self):
        execute_replay(self.registry, self.train, self.batches(), self.initial,
                       "static", self.config, TestRecorder())
        self.registry.state["pool"]["rows"] += 1
        with self.assertRaisesRegex(ValueError, "provenance"):
            execute_replay(self.registry, self.train, self.batches(), self.initial,
                           "static", self.config, TestRecorder())

    def test_performance_policy_does_not_train_on_psi_alone(self):
        plan = decide("performance", {"macro_f1": 0.7}, 0.71, None, self.config, [])
        self.assertEqual(plan["action"], "hold")
        plan = decide("performance", {"macro_f1": 0.5}, 0.71, None, self.config, [])
        self.assertEqual(plan["action"], "retrain")

    def test_performance_reference_uses_macro_not_accuracy(self):
        plan = decide("performance", {"macro_f1": 0.1, "accuracy": 0.99},
                      0.7, None, self.config, [])
        self.assertEqual(plan["action"], "retrain")

    def test_converter_exactly_reproduces_received_rows_and_metadata(self):
        manifest = verify_replay()
        with zipfile.ZipFile(ROOT / "data/replay-gas" / manifest["source_file"]) as source:
            for batch, name in [(1, "train.jsonl"), (2, "validation.jsonl")]:
                actual = convert(source.read(f"Dataset/batch{batch}.dat").decode(), batch)
                expected = [json.loads(line) for line in (
                    ROOT / "data/A" / name
                ).read_text(encoding="utf-8").splitlines()]
                self.assertEqual(actual, expected)

    def test_converter_rejects_missing_duplicate_or_nonfinite_features(self):
        raw = "1 " + " ".join(f"{i}:1" for i in range(1, 129))
        self.assertEqual(len(convert(raw, 3)[0]["inputs"]["features"]), 128)
        for bad in [raw.replace("128:1", "127:1"), raw.replace("128:1", "128:nan"), "1 1:2"]:
            with self.assertRaises(ValueError):
                convert(bad, 3)

    def test_prequential_predictions_are_from_incumbent_before_training(self):
        batches = self.batches()
        expected = Model.load(self.initial).predict(batches[0].data.x)
        execute_replay(self.registry, self.train, batches, self.initial,
                       "always", self.config, TestRecorder())
        actual = [json.loads(line)["probabilities"] for line in (
            self.folder / "prequential-batch-0.jsonl"
        ).read_text().splitlines()]
        np.testing.assert_array_equal(actual, expected)

    def test_batch_overlap_with_released_history_is_rejected(self):
        first, second = self.batches()
        second = dataclasses.replace(second, data=dataclasses.replace(second.data,
                                                                      x=first.data.x.copy()))
        with self.assertRaisesRegex(ValueError, "inputs overlap"):
            execute_replay(self.registry, self.train, [first, second], self.initial,
                           "static", self.config, TestRecorder())

    def test_doctor_rejects_dependency_version_drift(self):
        with patch("zhixun.cli.version", side_effect=lambda name:
                   "unexpected-version" if name == "numpy" else version(name)):
            with self.assertRaisesRegex(ValueError, "Dependency drift: numpy"):
                doctor()


if __name__ == "__main__":
    unittest.main()
