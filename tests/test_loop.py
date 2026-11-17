"""Synthetic fixtures are unit tests only; never contest data or experiment evidence."""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from zhixun.attribution import attribute_drift, attribute_performance
from zhixun.data import ROOT, Dataset
from zhixun.drift import detect_drift, fit_reference, psi_from_reference
from zhixun.loop import EvaluationBatch, execute_batches, load_config, run_stream, verify_sequence
from zhixun.policy import Budget, RulePolicy
from zhixun.registry import ModelRegistry, process_lock
from zhixun.vendor.model import Model


class TestRecorder:
    __test__ = False

    def __init__(self, folder=None, *args, **kwargs):
        self.folder = folder
        self.rows = []

    def log(self, step, metrics):
        self.rows.append({"step": step, "metrics": metrics})
        if self.folder:
            with (self.folder / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.rows[-1]) + "\n")

    def finish(self, **kwargs):
        pass

    def text(self, *args):
        pass

    def reference(self):
        return {"unit_test_only": True}


class LoopTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / "指导/临时/loop-tests"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        x = np.column_stack([np.arange(40), np.arange(40) % 3]).astype(float)
        y = np.zeros(40, dtype=int)
        self.train = Dataset(tuple(f"train-{i}" for i in range(40)), x, y,
                             frozenset({"training-fixture"}))
        self.model = Model("classification", ["x", "z"], ["zero", "one"], x, y, hidden=4)
        self.model.meta["profile"] = "A"
        self.initial = self.folder / "initial.npz"
        self.model.save(self.initial)
        self.config, self.budget = load_config(ROOT / "configs/stream.json")
        self.registry = ModelRegistry(self.folder, {"unit_test_only": True}, self.initial)

    def batch(self, shift=1e-8, index=0):
        data = Dataset(tuple(f"eval-{index}-{i}" for i in range(40)),
                       self.train.x + shift, self.train.y,
                       frozenset({f"evaluation-fixture-{index}"}))
        return EvaluationBatch(index, f"fixture-{index}", data, "unit-test-not-official-data")

    def execute(self, batches, budget=None):
        return execute_batches(self.registry, self.train, batches, self.initial, "A",
                               self.config, budget or self.budget, TestRecorder())

    def test_reference_same_distribution_is_zero_even_with_ties_and_constants(self):
        x = np.column_stack([np.zeros(40), np.arange(40) % 3])
        ref = fit_reference(x, self.model)
        np.testing.assert_array_equal(psi_from_reference(ref, x), [0, 0])
        self.assertGreater(ref.expected_counts[1].max(), ref.expected_counts[1].min())

    def test_constant_feature_shift_detected_in_both_directions(self):
        x = np.zeros((40, 2))
        ref = fit_reference(x, self.model)
        for sign in [-1, 1]:
            self.assertTrue(np.all(psi_from_reference(ref, x + sign) > 1))

    def test_invalid_or_empty_drift_input_is_rejected(self):
        ref = fit_reference(self.train.x, self.model)
        for x in [np.empty((0, 2)), np.zeros((3, 1)), np.full((3, 2), np.nan)]:
            with self.assertRaises(ValueError):
                psi_from_reference(ref, x)

    def test_attribution_is_ranked_and_exposes_zero_recall(self):
        signal = detect_drift(self.model, fit_reference(self.train.x, self.model),
                              self.train.x + 100)
        diagnosis = attribute_drift(signal, ["x", "z"])
        values = [row["psi"] for row in diagnosis["top_features"]]
        self.assertEqual(values, sorted(values, reverse=True))
        before = {"gas": {"f1": 0.5, "recall": 1, "support": 4}}
        after = {"gas": {"f1": 0.0, "recall": 0, "support": 4}}
        result = attribute_performance(before, after)
        self.assertEqual(result["zero_recall_classes"], ["gas"])
        self.assertEqual(result["per_class_f1_delta"]["gas"], -0.5)

    def test_no_drift_skips_all_optimizer_work(self):
        with patch.object(Model, "fit", side_effect=AssertionError("must skip")):
            events = self.execute([self.batch()])
        self.assertEqual(events[0]["plan"]["rationale"], "no_drift_skip")
        self.assertEqual(self.registry.state["updates_charged"], 0)
        self.assertEqual(events[0]["training_seconds"], 0)

    def test_stream_fit_only_receives_original_training_arrays(self):
        original = Model.fit
        calls = []

        def checked(model, x, y, epochs):
            np.testing.assert_array_equal(x, self.train.x)
            np.testing.assert_array_equal(y, self.train.y)
            calls.append(epochs)
            return original(model, x, y, epochs)

        with patch.object(Model, "fit", checked):
            events = self.execute([self.batch(100)])
        self.assertEqual(len(calls), 5)
        self.assertEqual(events[0]["final_steps"] - events[0]["initial_steps"], 5)
        self.assertEqual(self.registry.state["cursor"], 1)

    def test_resume_does_not_repeat_committed_training(self):
        self.execute([self.batch(100)])
        self.registry = ModelRegistry(self.folder, {"unit_test_only": True})
        with patch.object(Model, "fit", side_effect=AssertionError("already committed")):
            self.execute([self.batch(100)])
        self.assertEqual(len(self.registry.state["events"]), 1)

    def test_two_batches_advance_in_order_without_inventing_real_data(self):
        events = self.execute([self.batch(), self.batch(index=1)])
        self.assertEqual([r["batch_index"] for r in events], [0, 1])
        self.assertEqual(self.registry.state["cursor"], 2)

    def test_order_duplicate_samples_and_group_reuse_are_rejected(self):
        first, second = self.batch(), self.batch(index=1)
        for batches in [[second, first], [first, first],
                        [first, dataclasses.replace(second, data=dataclasses.replace(
                            second.data, groups=first.data.groups))]]:
            with self.assertRaises(ValueError):
                verify_sequence(batches, self.train)

    def test_zero_update_budget_holds_instead_of_training(self):
        with patch.object(Model, "fit", side_effect=AssertionError("budget exhausted")):
            result = self.execute([self.batch(100)], Budget(max_updates=0))
        self.assertEqual(result[0]["plan"]["rationale"], "training_budget_exhausted")

    def test_timeout_retains_reservation_and_never_adopts_candidate(self):
        with (patch.object(Model, "fit", side_effect=AssertionError("must time out first")),
              patch("zhixun.loop.monotonic", side_effect=[0.0, 1.0])):
            with self.assertRaises(TimeoutError):
                self.execute([self.batch(100)], Budget(max_seconds_per_round=1e-9))
        self.assertEqual(self.registry.state["cursor"], 0)
        self.assertEqual(self.registry.state["current"], 0)
        self.assertEqual(self.registry.state["updates_charged"], 1)

    def test_parent_tamper_during_training_never_commits_candidate(self):
        original = Model.fit

        def tamper(model, x, y, epochs):
            result = original(model, x, y, epochs)
            with (self.folder / "version-0.npz").open("ab") as stream:
                stream.write(b"unit-test-tamper")
            return result

        with patch.object(Model, "fit", tamper):
            with self.assertRaisesRegex(ValueError, "integrity"):
                self.execute([self.batch(100)])
        self.assertEqual(self.registry.state["cursor"], 0)
        self.assertEqual(self.registry.state["current"], 0)

    def test_interruption_retains_reservation_and_no_free_training_retry(self):
        with patch.object(Model, "fit", side_effect=RuntimeError("unit-test interruption")):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.execute([self.batch(100)], Budget(max_updates=1))
        self.registry = ModelRegistry(self.folder, {"unit_test_only": True})
        self.assertEqual(self.registry.state["training_seconds_charged"], 300)
        self.assertEqual(self.registry.state["cursor"], 0)
        with patch.object(Model, "fit", side_effect=AssertionError("no free retry")):
            result = self.execute([self.batch(100)], Budget(max_updates=1))
        self.assertEqual(result[0]["status"], "interrupted_attempt")
        self.assertEqual(result[1]["plan"]["action"], "hold")

    def test_policy_responds_to_prior_rejection_and_never_blindly_rolls_back(self):
        signal = detect_drift(self.model, fit_reference(self.train.x, self.model),
                              self.train.x + 100)
        first = RulePolicy().decide(signal, {}, self.budget, [])
        second = RulePolicy().decide(signal, {}, self.budget, [
            {"plan": dataclasses.asdict(first), "accepted": False}
        ])
        self.assertEqual(second.learning_rate, first.learning_rate / 2)
        self.assertEqual(second.action, "retrain")

    def test_rollback_can_reach_version_zero_and_refuses_rejected_versions(self):
        for version in [1, 2]:
            path = self.folder / f"extra-{version}.npz"
            self.model.save(path)
            self.registry.register(version, path, parent=version - 1, adopted=version == 1)
        self.registry.state["current"] = 1
        self.registry.rollback(0)
        self.assertEqual(self.registry.state["current"], 0)
        with self.assertRaisesRegex(ValueError, "previously adopted"):
            self.registry.rollback(2)

    def test_automatic_rollback_compares_old_and_current_on_same_batch(self):
        good = Model.load(self.initial)
        good.params["w1"].fill(0)
        good.params["b1"].fill(0)
        good.params["w2"].fill(0)
        good.params["b2"][:] = [4, 0]
        good.save(self.folder / "good.npz")
        # A separate registry initial version is deliberately good in this unit fixture.
        session = self.folder / "rollback-fixture"
        session.mkdir()
        self.registry = ModelRegistry(session, {"unit_test_only": True}, self.folder / "good.npz")
        bad = Model.load(self.folder / "good.npz")
        bad.params["b2"][:] = [0, 0.04]
        bad.save(session / "bad.npz")
        self.registry.register(7, session / "bad.npz", parent=0, adopted=True)
        self.registry.state["current"] = 7
        with patch.object(Model, "fit", side_effect=AssertionError("rollback needs no training")):
            result = execute_batches(self.registry, self.train, [self.batch(100)],
                                     self.folder / "good.npz", "A", self.config,
                                     self.budget, TestRecorder())
        self.assertEqual(result[0]["plan"]["action"], "rollback")
        self.assertEqual(self.registry.state["current"], 0)
        self.assertTrue(result[0]["rollback_comparisons"])

    def test_registry_detects_checkpoint_tampering_and_changed_resume_signature(self):
        with self.assertRaisesRegex(ValueError, "changed"):
            ModelRegistry(self.folder, {"unit_test_only": False})
        with self.registry.checkpoint(0).open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.registry.checkpoint(0)

    def test_concurrent_session_access_is_rejected(self):
        with process_lock(self.folder):
            with self.assertRaises(RuntimeError):
                with process_lock(self.folder):
                    pass

    def test_configuration_rejects_invalid_limits_unknown_keys_and_paid_calls(self):
        for change in [{"extra": 1}, {"psi_threshold": float("nan")},
                       {"budget": {**self.config["budget"], "max_llm_calls": 1}}]:
            path = self.folder / "invalid.json"
            path.write_text(json.dumps({**self.config, **change}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_real_public_file_remains_one_complete_evaluation_batch(self):
        with patch("zhixun.loop.Recorder", TestRecorder):
            report = run_stream("A", output=self.folder)
        self.assertEqual(report["completed_batches"], 1)
        self.assertEqual(report["events"][0]["validation_rows"], 1244)
        self.assertEqual(report["resources"]["llm_calls"], 0)

    def test_launcher_supports_redirected_windows_sdk_unicode_logs(self):
        script = (
            "import importlib.util; "
            "s=importlib.util.spec_from_file_location('entry','run.py'); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.main(); print(chr(0x1f4be))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, "doctor"], cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "gbk"}, capture_output=True, check=True,
        )
        self.assertIn(chr(0x1f4be), result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
