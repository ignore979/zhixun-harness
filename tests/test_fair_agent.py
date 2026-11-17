"""Unit fixtures only; oracle training data are built exclusively from real replay traces."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_loop import TestRecorder

from zhixun.agent import LlmPolicy
from zhixun.data import ROOT, Dataset
from zhixun.diagnosis import confusion_errors, feature_saturation
from zhixun.fair_core import (
    build_prompt,
    load_fair_config,
    rule_action,
    train_action,
    validate_action,
)
from zhixun.fair_replay import run_fair
from zhixun.model_extension import ClassWeightedModel
from zhixun.pool import TrainingPool
from zhixun.vendor.model import Model


class FairTests(unittest.TestCase):
    def setUp(self):
        self.config = load_fair_config(ROOT)

    def observation(self, batch=2, remaining=45):
        return {"batch": batch, "remaining_epochs": remaining, "terminal": batch == 10,
                "remaining_decisions": 10 - batch, "mean_psi": 1.0,
                "performance_drop": 0.1, "pool_rows": 1000}

    def test_clip_diagnostic_distinguishes_cells_rows_and_unchanged_boundary(self):
        x = np.array([[10., 11.], [0., -12.]])
        result = feature_saturation(x, np.zeros(2), np.ones(2))
        self.assertEqual(result["clipped_cell_fraction"], 0.5)
        self.assertEqual(result["rows_with_any_clip_fraction"], 1)
        self.assertEqual(result["boundary_cell_fraction"], 0.25)
        with self.assertRaises(ValueError):
            feature_saturation(x, np.zeros(2), np.zeros(2))

    def test_confusion_reports_true_class_direction(self):
        result = confusion_errors([[3, 7], [2, 8]], ["a", "b"])
        self.assertEqual(result[0], {"actual": "a", "predicted": "b", "count": 7,
                                     "fraction_of_actual_class": 0.7})

    def test_flat_control_uses_45_epochs_without_terminal_training(self):
        remaining, consumed = 45, 0
        for batch in range(2, 11):
            observation = self.observation(batch, remaining)
            action = rule_action(observation, self.config, flat=True)
            validate_action(action, observation, self.config)
            consumed += action["epochs"]
            remaining -= action["epochs"]
        self.assertEqual(consumed, 45)
        self.assertEqual(action["action"], "hold")

    def test_soft_allocation_never_exceeds_cap_or_catalog(self):
        remaining = 45
        for batch in range(2, 11):
            observation = self.observation(batch, remaining)
            action = rule_action(observation, self.config)
            validate_action(action, observation, self.config)
            remaining -= action["epochs"]
        self.assertGreaterEqual(remaining, 0)

    def test_terminal_overbudget_and_out_of_catalog_actions_are_rejected(self):
        action = self.config["action_catalog"][-1]
        for observation in [self.observation(10), self.observation(2, 2)]:
            with self.assertRaises(ValueError):
                validate_action(action, observation, self.config)
        with self.assertRaises(ValueError):
            validate_action({**action, "epochs": 9}, self.observation(), self.config)

    def test_future_fields_cannot_be_in_agent_prompt(self):
        for key in ["next_metrics", "candidate_results", "chosen", "completion", "future_labels"]:
            with self.assertRaises(ValueError):
                build_prompt({**self.observation(), key: "not allowed"}, self.config)

    def test_shared_trainer_only_fits_released_history_and_keeps_parent(self):
        x = np.random.default_rng(3).normal(size=(20, 3))
        y = np.array([0, 1] * 10)
        model = Model("classification", ["x", "y", "z"], ["a", "b"], x, y, hidden=4)
        pool = TrainingPool.from_dataset(Dataset(tuple(f"train-{i}" for i in range(20)),
                                                x, y, frozenset({"past"})))
        current = Dataset(tuple(f"current-{i}" for i in range(20)), x + 4, y,
                          frozenset({"current"}))
        snapshot = copy.deepcopy(model.params)
        original_fit = ClassWeightedModel.fit

        def checked(candidate, features, labels, epochs, **kwargs):
            np.testing.assert_array_equal(features, x)
            np.testing.assert_array_equal(labels, y)
            return original_fit(candidate, features, labels, epochs, **kwargs)

        with patch.object(ClassWeightedModel, "fit", checked):
            selected, result = train_action(model, pool, current, self.config["action_catalog"][1],
                                            self.config)
        self.assertEqual(result["optimizer_steps"], 2)
        for key in snapshot:
            np.testing.assert_array_equal(model.params[key], snapshot[key])
        np.testing.assert_array_equal(selected.mean, model.mean)
        with self.assertRaises(TypeError):
            train_action(model, pool, current, self.config["action_catalog"][1],
                         self.config, next_batch=current)

    def test_hold_does_not_train(self):
        x, y = np.zeros((2, 1)), np.array([0, 1])
        model = Model("classification", ["x"], ["a", "b"], x, y, hidden=2)
        pool = TrainingPool.from_dataset(Dataset(("a", "b"), x, y, frozenset({"past"})))
        current = Dataset(("c", "d"), x + 1, y, frozenset({"current"}))
        with patch.object(ClassWeightedModel, "fit", side_effect=AssertionError("must not train")):
            _, outcome = train_action(model, pool, current, self.config["action_catalog"][0],
                                      self.config)
        self.assertEqual(outcome["optimizer_steps"], 0)

    def test_invalid_live_response_falls_back_without_executing_code(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1, 2]

            def decode(self, tokens, **kwargs):
                return '{"action":"retrain","epochs":999,"learning_rate":1,"command":"bad"}'

        response = types.SimpleNamespace(sequences=[types.SimpleNamespace(text="bad", tokens=[3])],
                                         input_tokens=2, output_tokens=1)
        client = types.SimpleNamespace(sample=lambda **kw: types.SimpleNamespace(
            result=lambda: response))
        reservations = []
        policy = LlmPolicy(self.config, model="unit-test-only", client=client,
                           tokenizer=Tokenizer(),
                           before_request=lambda: reservations.append(1))
        action, audit = policy.decide(self.observation())
        self.assertEqual(len(reservations), 1)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(audit["source"], "fallback_rule")
        validate_action(action, self.observation(), self.config)

    def test_real_loop_executes_policy_actions_and_resumes_without_retraining(self):
        class Policy:
            def __init__(self, config, **kwargs):
                self.config = config
                self.before = kwargs["before_request"]

            def decide(self, observation):
                if not observation["terminal"]:
                    self.before()
                return rule_action(observation, self.config, flat=True), {
                    "source": "terminal_guard" if observation["terminal"] else "llm",
                    "proposal_valid": True, "input_tokens": 2, "output_tokens": 1,
                    "fallback_reason": None,
                }

        temporary = ROOT / "指导/临时/fair-tests"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as temp:
            with patch("zhixun.fair_replay.Recorder", TestRecorder):
                report = run_fair("agent_base", output=Path(temp), policy_factory=Policy)
            self.assertEqual(report["summary"]["llm_calls"], 8)
            self.assertEqual(report["summary"]["epochs_charged"], 45)
            self.assertEqual(report["events"][-1]["action"]["action"], "hold")
            session = Path(temp) / report["experiment_id"]
            with (patch("zhixun.fair_replay.Recorder", TestRecorder),
                  patch.object(ClassWeightedModel, "fit", side_effect=AssertionError("committed"))):
                resumed = run_fair("agent_base", output=Path(temp), resume=session,
                                    policy_factory=Policy)
            self.assertEqual(resumed["summary"], report["summary"])

    def test_oracle_corpus_has_no_evaluation_lookahead_in_training_rows(self):
        provenance = json.loads((ROOT / "data/harness-oracle-v3/provenance.json").read_text())
        folder = ROOT / "artifacts" / provenance["source_experiment"]
        states = json.loads((folder / "states.json").read_text(encoding="utf-8"))
        self.assertEqual(len({s["state_id"] for s in states}), len(states))
        for state in states:
            maximum = max(p["lookahead_batch"] for p in state["source_lineage"])
            if state["split"] == "train":
                self.assertLessEqual(maximum, 7)
            else:
                self.assertGreater(maximum, 7)
            ranked = sorted(state["outcomes"], key=lambda row: (-row["value"], row["epochs"],
                                                              -row["next_metrics"]["macro_f1"]))
            self.assertEqual(state["chosen"], ranked[0]["action"])
            self.assertNotIn("next_metrics", json.loads(state["prompt"].split("\n", 1)[1])[
                "observation"])


if __name__ == "__main__":
    unittest.main()
