"""All remote training doubles in this module are unit-test only."""
import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_loop import TestRecorder

from zhixun.data import ROOT
from zhixun.harness_evaluation import evaluate_harness
from zhixun.llm_policy import parse_decision
from zhixun.pytrio_training import tokens_and_weights, train_sft


class HarnessTests(unittest.TestCase):
    def test_temporal_split_and_derived_label_disclosure(self):
        folder = ROOT / "data/harness-sft"
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["official_llm_corpus"])
        self.assertEqual((provenance["training_rows"], provenance["evaluation_rows"]), (18, 9))
        train = {r["batch"] for r in provenance["row_lineage"] if r["split"] == "train"}
        evaluation = {r["batch"] for r in provenance["row_lineage"]
                      if r["split"] == "evaluation"}
        self.assertFalse(train & evaluation)
        for split in ["train", "evaluation"]:
            for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                context = json.loads(row["prompt"].split("\n", 1)[1])
                self.assertNotIn("candidate_metrics", context)
                self.assertNotIn("selected_same_batch", context)
                self.assertNotIn("future_batches", context)
                parse_decision(row["completion"],
                               rollback_versions=context["allowed_rollback_versions"])

    def test_policy_json_rejects_code_extra_fields_nan_and_unknown_rollbacks(self):
        good = {"action": "hold", "epochs": 0, "learning_rate": 0, "target_version": None}
        self.assertEqual(parse_decision(json.dumps(good)), good)
        bad = ["print('run')", json.dumps({**good, "command": "delete"}),
               json.dumps({**good, "epochs": True}),
               json.dumps({**good, "learning_rate": float("nan")}),
               json.dumps({**good, "action": "rollback", "target_version": 999}),
               '{"action":"hold","action":"retrain"}']
        for raw in bad:
            with self.assertRaises(ValueError):
                parse_decision(raw)

    def test_chat_template_and_eos_are_used_with_completion_only_weights(self):
        class Tokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

            def encode(self, text, **kwargs):
                return [4, 5]

        inputs, targets, weights = tokens_and_weights(
            {"prompt": "x", "completion": "y"}, Tokenizer()
        )
        self.assertEqual(inputs, [1, 2, 3, 4, 5])
        self.assertEqual(targets, [2, 3, 4, 5, 99])
        self.assertEqual(weights, [0, 0, 1, 1, 1])

    def test_normal_sft_schedule_is_three_epochs_not_old_ten_step_cap(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

        class Future:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class Training:
            updates = 0
            sizes = []

            def get_tokenizer(self):
                return Tokenizer()

            def forward(self, data, loss):
                return Future(types.SimpleNamespace(metrics={"loss_sum": 10.0}))

            def forward_backward(self, data, loss):
                self.sizes.append(len(data))
                return self.forward(data, loss)

            def optim_step(self, params):
                self.updates += 1
                return Future(None)

            def save_state(self, **kwargs):
                return Future(types.SimpleNamespace(path="unit-test-only://state"))

            def save_weights_for_sampler(self, **kwargs):
                return Future(types.SimpleNamespace(path="unit-test-only://sampler"))

        training = Training()
        service = types.SimpleNamespace(get_supported_models=lambda: ["unit-test-model"],
                                        create_lora_training_client=lambda **kw: training)
        sdk = types.SimpleNamespace(ServiceClient=lambda: service,
                                    ModelInput=types.SimpleNamespace(from_ints=lambda **kw: kw),
                                    Datum=lambda **kw: types.SimpleNamespace(**kw),
                                    AdamParams=lambda **kw: kw)
        folder = ROOT / "指导/临时/harness-tests"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            with (patch.dict("sys.modules", {"pytrio": sdk}),
                  patch("zhixun.pytrio_training.ROOT", Path(temp)),
                  patch("zhixun.pytrio_training.Recorder", TestRecorder)):
                report = train_sft(ROOT / "data/harness-sft/train.jsonl",
                                   ROOT / "data/harness-sft/provenance.json", "unit-test-model",
                                   None, 0.0001, project="test", workspace=None,
                                   evaluation_path=ROOT / "data/harness-sft/evaluation.jsonl")
        self.assertEqual(training.updates, 18)
        self.assertEqual(training.sizes, [3] * 18)
        self.assertEqual(report["completed_optimizer_steps"], 18)

    def test_evaluation_records_invalid_proposals_without_executing_them(self):
        folder = ROOT / "指导/临时/harness-tests"
        folder.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in (ROOT / "data/harness-sft/evaluation.jsonl").read_text(
            encoding="utf-8").splitlines()]

        class Sampling:
            def __init__(self, trained):
                self.trained, self.index = trained, 0

            def get_tokenizer(self):
                return types.SimpleNamespace(encode=lambda text, **kw: [1, 2, 3],
                                             decode=lambda tokens, **kw: tokens[0])

            def sample(self, **kwargs):
                text = rows[self.index]["completion"] if self.trained else "run arbitrary code"
                self.index += 1
                response = types.SimpleNamespace(
                    sequences=[types.SimpleNamespace(text=text + "<EOS>", tokens=[text])],
                    input_tokens=3, output_tokens=4)
                return types.SimpleNamespace(result=lambda: response)

        service = types.SimpleNamespace(create_sampling_client=lambda **kw: Sampling(
            kw["model_path"] is not None))
        sdk = types.SimpleNamespace(ServiceClient=lambda: service,
                                    ModelInput=types.SimpleNamespace(from_ints=lambda **kw: kw),
                                    SamplingParams=lambda **kw: kw)
        provenance = json.loads((ROOT / "data/harness-sft/provenance.json").read_text())
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root = Path(temp)
            data = root / "data/harness-sft"
            data.mkdir(parents=True)
            shutil.copy2(ROOT / "data/harness-sft/evaluation.jsonl", data / "evaluation.jsonl")
            run = root / "unit-training"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({
                "status": "completed", "completed_optimizer_steps": 18,
                "experiment_id": "unit-only", "base_model": "unit-model",
                "sampler_checkpoint_path": "unit-test-only://sampler",
                "evaluation_sha256": provenance["evaluation_sha256"],
            }))
            with (patch("zhixun.harness_evaluation.ROOT", root),
                  patch("zhixun.harness_evaluation.Recorder", TestRecorder),
                  patch.dict("sys.modules", {"pytrio": sdk})):
                report = evaluate_harness(run)
        self.assertEqual(report["summary"]["base"]["valid_json_rate"], 0)
        self.assertEqual(report["summary"]["trained"]["recorded_decision_agreement"], 1)
        self.assertFalse(report["outputs_executed"])


if __name__ == "__main__":
    unittest.main()
