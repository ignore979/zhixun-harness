import dataclasses
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from zhixun.data import ROOT, digest, load_dataset, verify_inputs, verify_parent, verify_split
from zhixun.metrics import promotion
from zhixun.packaging import submission_files
from zhixun.predict import predict_stream
from zhixun.pytrio_training import load_training_data, tokens_and_weights
from zhixun.train import RoundPlan, run_round
from zhixun.vendor.model import Model


class JournalOnly:
    """Unit-test replacement; never represented as a real SwanLab cloud run."""

    def __init__(self, folder, config, **kwargs):
        self.folder = folder

    def log(self, step, metrics):
        with (self.folder / "test-journal.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "metrics": metrics}) + "\n")

    def finish(self, **kwargs):
        pass

    def reference(self):
        return {"unit_test_only": True, "cloud_record_verified": False}


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = ROOT / "指导/临时/tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)
        cls.model = Model.load(ROOT / "data/A/checkpoint.npz")
        cls.train = load_dataset(ROOT / "data/A/train.jsonl", cls.model, "A")
        cls.validation = load_dataset(ROOT / "data/A/validation.jsonl", cls.model, "A")

    def test_all_original_download_hashes_match(self):
        for profile in "ABC":
            self.assertEqual(len(verify_inputs(profile)["files"]), 4)

    def test_task_and_group_separation(self):
        verify_parent(self.model, "A", self.train)
        verify_split(self.train, self.validation)
        self.assertEqual((len(self.train.ids), len(self.validation.ids)), (445, 1244))
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            verify_split(self.train, dataclasses.replace(
                self.validation, groups=self.train.groups
            ))

    def test_tampered_download_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "data/A", root / "data/A")
            shutil.copy2(ROOT / "data/manifest.json", root / "data/manifest.json")
            with (root / "data/A/train.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_inputs("A", root)

    def test_resume_keeps_adam_and_matches_uninterrupted_training(self):
        first = Model.load(ROOT / "data/A/checkpoint.npz")
        second = Model.load(ROOT / "data/A/checkpoint.npz")
        first.fit(self.train.x, self.train.y, 3)
        second.fit(self.train.x, self.train.y, 1)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp) / "checkpoint.npz"
            second.save(path)
            second = Model.load(path)
            second.fit(self.train.x, self.train.y, 2)
        for group in ["params", "first", "second"]:
            for name in getattr(first, group):
                np.testing.assert_array_equal(
                    getattr(first, group)[name], getattr(second, group)[name]
                )
        self.assertEqual(first.meta["steps"], second.meta["steps"])
        np.testing.assert_array_equal(second.mean, self.model.mean)

    def test_predictions_match_supplied_model(self):
        row = json.loads((ROOT / "data/A/validation.jsonl").read_text().splitlines()[0])
        destination = io.StringIO()
        predict_stream(ROOT / "data/A/checkpoint.npz",
                       io.StringIO(json.dumps({"inputs": row["inputs"]})), destination)
        result = json.loads(destination.getvalue())["outputs"]
        expected = self.model.predict(self.validation.x[:1])[0]
        np.testing.assert_array_equal([r["score"] for r in result], expected)

    def test_prediction_refuses_label_and_metadata_input(self):
        with self.assertRaisesRegex(ValueError, "never labels"):
            predict_stream(ROOT / "data/A/checkpoint.npz",
                           io.StringIO('{"inputs":{},"answer":"Ethanol"}'), io.StringIO())

    def test_gate_rejects_metric_tradeoffs(self):
        self.assertFalse(promotion({"macro_f1": 0.7, "accuracy": 0.88},
                                   {"macro_f1": 0.75, "accuracy": 0.87}, "A")[0])
        self.assertFalse(promotion({"rmse": 20.0, "mae": 15.0},
                                   {"rmse": 19.0, "mae": 16.0}, "B")[0])

    def test_training_only_receives_public_train_features_and_labels(self):
        original_fit = Model.fit
        calls = []

        def checked_fit(model, features, labels, epochs):
            np.testing.assert_array_equal(features, self.train.x)
            np.testing.assert_array_equal(labels, self.train.y)
            calls.append(epochs)
            return original_fit(model, features, labels, epochs)

        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            with (
                patch("zhixun.train.Recorder", JournalOnly),
                patch.object(Model, "fit", checked_fit),
            ):
                report = run_round(RoundPlan("A", epochs=2), output=Path(temp))
            self.assertEqual(calls, [1, 1])
            self.assertEqual(report["final_steps"] - report["initial_steps"], 8)
            folder = Path(temp) / report["experiment_id"]
            selected = folder / "checkpoint.npz"
            expected = (
                folder / "candidate.npz" if report["accepted"] else ROOT / "data/A/checkpoint.npz"
            )
            self.assertEqual(digest(selected), digest(expected))

    def test_round_budget_stops_before_training(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            with patch("zhixun.train.Recorder", JournalOnly), patch.object(Model, "fit") as fit:
                with self.assertRaises(TimeoutError):
                    run_round(RoundPlan("A", max_seconds=1e-9), output=Path(temp))
                fit.assert_not_called()
            self.assertEqual(len(list(Path(temp).glob("*/failure.json"))), 1)

    def test_failed_export_never_creates_a_completed_report(self):
        original_open = Path.open

        def checked_open(path, *args, **kwargs):
            if path.name == "metrics.csv":
                raise OSError("unit-test export failure")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            with (
                patch("zhixun.train.Recorder", JournalOnly),
                patch.object(Path, "open", checked_open),
            ):
                with self.assertRaisesRegex(OSError, "export failure"):
                    run_round(RoundPlan("A", epochs=1), output=Path(temp))
            self.assertFalse(list(Path(temp).glob("*/run.json")))
            self.assertEqual(len(list(Path(temp).glob("*/failure.json"))), 1)

    def test_submission_contains_no_guidance_environment_or_credentials(self):
        files = submission_files(ROOT)
        self.assertTrue(files)
        self.assertTrue(all("指导" not in p.relative_to(ROOT).parts for p in files))

    def test_cpu_data_cannot_silently_become_llm_training_data(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            provenance = Path(temp) / "provenance.json"
            provenance.write_text(json.dumps({"sha256": "incorrect"}))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_training_data(ROOT / "data/A/train.jsonl", provenance)

    def test_sft_uses_shifted_completion_only_loss(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                return [ord(c) for c in text]

        inputs, targets, weights = tokens_and_weights(
            {"prompt": "ab", "completion": "cd"}, Tokenizer()
        )
        self.assertEqual(inputs, [97, 98, 99])
        self.assertEqual(targets, [98, 99, 100])
        self.assertEqual(weights, [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
