"""Real supplied CPU data; model/host replies here are explicitly unit-test doubles."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from zhixun.contest_runtime import parse_plan, predict, train
from zhixun.data import ROOT, digest
from zhixun.vendor.model import Model


class RoundDouble:
    def __init__(self, task):
        self.dataset_key, self.round_index = task, 2
        self.model = self.query
        checkpoint = ROOT / "data" / task / "checkpoint.npz"
        model = Model.load(checkpoint)
        self.task = {"task": model.meta["task"]}
        self.parent = SimpleNamespace(
            descriptor={"checkpoint_paths": ["checkpoint.npz"], "code_paths": []},
            artifact_id="unit-parent",
            checkpoint_sha256=digest(checkpoint),
            path=lambda _: checkpoint,
        )
        self.data = SimpleNamespace(
            training_rows=lambda: self.rows("train"),
            validation_rows=lambda: self.rows("validation"),
        )
        self.max_output_tokens, self.wall_seconds = 512, 180
        self.history_ids, self.call_ids, self.messages = [], [], []
        self.record = self.saved = None

    def rows(self, split):
        return [
            json.loads(line)
            for line in (ROOT / "data" / self.dataset_key / (split + ".jsonl"))
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    def history(self):
        self.history_ids = ["unit-history-previous"]
        return [{"id": "unit-history-previous", "kind": "unit-host-double"}]

    def query(self, messages, **options):
        self.messages.append(messages)
        call = "unit-call-" + str(len(self.call_ids))
        self.call_ids.append(call)
        text = (
            {"action": "retrain", "epochs": 1, "learning_rate": 0.0005}
            if len(self.call_ids) == 1
            else {"select": "candidate", "reason": "Unit host double."}
        )
        return SimpleNamespace(call_id=call, text=json.dumps(text))

    def submit_candidate(self, directory, **options):
        self.candidate = Path(directory)
        return {"artifact_id": "unit-candidate", "summary": {"source": "unit-host-double"}}

    def state_workspace(self, path):
        path.mkdir()
        return path

    def save_state(self, path):
        self.saved = list(path.glob("round-*.json"))

    def finish(self, **record):
        self.record = record


class OfficialRuntimeTests(unittest.TestCase):
    def test_executable_plan_and_boolean_parameters_are_rejected(self):
        for value in [
            'TRAINING_PLAN = __import__("os").getcwd()',
            '{"action":"retrain","epochs":true,"learning_rate":0.001}',
            '{"action":"retrain","epochs":11,"learning_rate":0.001}',
        ]:
            with self.assertRaises((ValueError, SyntaxError)):
                parse_plan(value)

    def test_real_abc_training_receipts_history_and_parent_are_preserved(self):
        for task in "ABC":
            with (
                self.subTest(task=task),
                tempfile.TemporaryDirectory(dir=ROOT / "指导/临时") as temp,
            ):
                rnd = RoundDouble(task)
                checkpoint = rnd.parent.path("checkpoint.npz")
                before_hash = digest(checkpoint)
                parent = Model.load(checkpoint)
                with (
                    patch.dict(os.environ, {"CONTEST_WORK": temp}),
                    patch(
                        "zhixun.tracking.Recorder",
                        side_effect=ImportError("unit offline logger double"),
                    ),
                ):
                    train(rnd)
                self.assertEqual(rnd.record["model_call_ids"], ["unit-call-0", "unit-call-1"])
                self.assertEqual(rnd.record["history_ids"], ["unit-history-previous"])
                self.assertEqual(digest(checkpoint), before_hash)
                candidate = Model.load(rnd.candidate / "checkpoint.npz")
                self.assertEqual(candidate.meta["epochs"], parent.meta["epochs"] + 1)
                self.assertTrue(np.array_equal(parent.mean, candidate.mean))
                self.assertTrue(np.array_equal(parent.scale, candidate.scale))
                self.assertTrue(rnd.saved)
                audit = json.loads(rnd.saved[0].read_text(encoding="utf-8"))
                self.assertEqual(len(audit["model_receipts"]), 2)
                self.assertFalse(audit["swanlab_offline_recorded"])

    def test_prediction_uses_selected_model_without_labels(self):
        for task in "ABC":
            source = ROOT / "data" / task
            row = json.loads(
                (source / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            model = Model.load(source / "checkpoint.npz")
            sample = SimpleNamespace(task=model.meta["task"], inputs=row["inputs"])
            with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(source / "checkpoint.npz")}):
                result = predict(sample, source)
            if sample.task == "classification":
                self.assertEqual(len(result), len(model.meta["classes"]))
            else:
                self.assertGreaterEqual(result["value"], 0)


if __name__ == "__main__":
    unittest.main()
