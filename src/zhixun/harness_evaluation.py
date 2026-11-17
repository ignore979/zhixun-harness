"""Temporal policy imitation evaluation. Outputs are validated, never executed."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from zhixun.data import ROOT, digest, write_json
from zhixun.llm_policy import parse_decision
from zhixun.pytrio_training import prompt_tokens
from zhixun.tracking import Recorder


def validate_response(decoded, context):
    if "action_catalog" in context:
        from zhixun.fair_core import validate_action

        return validate_action(parse_decision(decoded), context["observation"],
                               {"action_catalog": context["action_catalog"]})
    return parse_decision(decoded, rollback_versions=context.get("allowed_rollback_versions", []))


def evaluate_harness(run: Path, *, project="zhixun-school", workspace=None) -> dict:
    trained = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if trained["status"] != "completed" or trained["completed_optimizer_steps"] <= 0:
        raise ValueError("Evaluation requires actually completed PyTRIO training")
    data_path = ROOT / trained.get("evaluation_dataset_path", "data/harness-sft/evaluation.jsonl")
    if digest(data_path) != trained["evaluation_sha256"]:
        raise ValueError("Evaluation data differs from the registered temporal holdout")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    folder = ROOT / "artifacts" / (
        f"run-evaluate-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    registration = {
        "experiment_id": folder.name, "task": "harness_temporal_imitation_evaluation",
        "training_experiment": trained["experiment_id"], "base_model": trained["base_model"],
        "sampler_checkpoint_path": trained["sampler_checkpoint_path"],
        "evaluation_sha256": digest(data_path), "evaluation_rows": len(rows),
        "temperature": 0, "max_tokens": 256,
        "evaluation_dataset_path": str(data_path.relative_to(ROOT)),
        "scope": "recorded_rule_imitation_not_downstream_industrial_accuracy",
    }
    write_json(folder / "registration.json", registration)
    os.environ["MODELSCOPE_CACHE"] = str(ROOT / "指导/模型缓存/modelscope")
    os.environ["HF_HOME"] = str(ROOT / "指导/模型缓存/huggingface")
    import pytrio

    service = pytrio.ServiceClient()
    recorder = None
    outputs, summary = [], {}
    try:
        recorder = Recorder(folder, registration, mode="online", project=project,
                            workspace=workspace)
        for phase, path in [("base", None), ("trained", trained["sampler_checkpoint_path"])]:
            client = service.create_sampling_client(base_model=trained["base_model"],
                                                     model_path=path)
            tokenizer = client.get_tokenizer()
            valid, agreement = 0, 0
            for index, row in enumerate(rows):
                response = client.sample(
                    prompt=pytrio.ModelInput.from_ints(
                        tokens=prompt_tokens(row["prompt"], tokenizer)),
                    num_samples=1,
                    sampling_params=pytrio.SamplingParams(max_tokens=256, temperature=0, seed=42),
                ).result()
                sequence = response.sequences[0]
                raw = sequence.text or tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                decoded = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                context = json.loads(row["prompt"].split("\n", 1)[1])
                error, proposal = None, None
                try:
                    proposal = validate_response(decoded, context)
                    valid += 1
                    agreement += proposal == json.loads(row["completion"])
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                outputs.append({"phase": phase, "evaluation_row": index + 1,
                                "raw_response": raw, "proposal": proposal, "error": error,
                                "decoded_response": decoded, "tokens": sequence.tokens,
                                "target": json.loads(row["completion"]),
                                "input_tokens": response.input_tokens,
                                "output_tokens": response.output_tokens,
                                "executed": False})
                write_json(folder / "responses.json", outputs)
                write_json(folder / "progress.json", {"phase": phase, "completed_rows": index + 1})
            summary[phase] = {"valid_json_rate": valid / len(rows),
                              "recorded_decision_agreement": agreement / len(rows)}
            recorder.log(0 if phase == "base" else 1,
                         {f"evaluation/{k}": v for k, v in summary[phase].items()})
        recorder.finish()
        report = {**registration, "status": "completed", "summary": summary,
                  "inference_calls": len(outputs), "tracking": recorder.reference(),
                  "outputs_executed": False, "live_harness_adoption": False}
        write_json(folder / "run.json", report)
        return report
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "message": str(exc), "responses_saved": len(outputs)})
        if recorder:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise


def rescore_harness(run: Path, *, project="zhixun-school", workspace=None) -> dict:
    """Correct transport decoding of already generated samples; no new inference calls."""
    original = json.loads((run / "run.json").read_text(encoding="utf-8"))
    responses = json.loads((run / "responses.json").read_text(encoding="utf-8"))
    data_path = ROOT / original.get("evaluation_dataset_path", "data/harness-sft/evaluation.jsonl")
    if digest(data_path) != original["evaluation_sha256"]:
        raise ValueError("Original evaluation dataset changed")
    targets = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    identities = {(r["phase"], r["evaluation_row"]) for r in responses}
    if len(responses) != 2 * len(targets) or identities != {
        (phase, index) for phase in ["base", "trained"] for index in range(1, len(targets) + 1)
    }:
        raise ValueError("Original responses are incomplete or duplicated")
    os.environ["MODELSCOPE_CACHE"] = str(ROOT / "指导/模型缓存/modelscope")
    os.environ["HF_HOME"] = str(ROOT / "指导/模型缓存/huggingface")
    from modelscope import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(original["base_model"], local_files_only=True)
    folder = ROOT / "artifacts" / (
        f"run-rescore-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    registration = {
        "experiment_id": folder.name, "task": "transport_decode_rescore_same_saved_outputs",
        "source_experiment": original["experiment_id"],
        "source_responses_sha256": digest(run / "responses.json"),
        "evaluation_sha256": digest(data_path), "base_model": original["base_model"],
        "transport_decoding": "tokenizer_encode_then_decode_skip_special_tokens",
        "new_remote_inference_calls": 0,
    }
    write_json(folder / "registration.json", registration)
    recorder = Recorder(folder, registration, mode="online", project=project, workspace=workspace)
    scored, summary = [], {}
    try:
        for phase in ["base", "trained"]:
            valid, agreement = 0, 0
            for row in [r for r in responses if r["phase"] == phase]:
                target = targets[row["evaluation_row"] - 1]
                if row["target"] != json.loads(target["completion"]):
                    raise ValueError("Original response target does not match the held-out row")
                # Only tokenizer-declared special tokens are removed. No JSON/prose extraction.
                tokens = tokenizer.encode(row["raw_response"], add_special_tokens=False)
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                context = json.loads(target["prompt"].split("\n", 1)[1])
                error, proposal = None, None
                try:
                    proposal = validate_response(decoded, context)
                    valid += 1
                    agreement += proposal == row["target"]
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                scored.append({**row, "decoded_response": decoded, "proposal": proposal,
                               "error": error, "rescored": True})
            summary[phase] = {"valid_json_rate": valid / len(targets),
                              "recorded_decision_agreement": agreement / len(targets)}
            recorder.log(0 if phase == "base" else 1,
                         {f"evaluation/{k}": v for k, v in summary[phase].items()})
        write_json(folder / "responses.json", scored)
        recorder.finish()
        report = {**registration, "status": "completed", "summary": summary,
                  "tracking": recorder.reference(), "outputs_executed": False,
                  "original_report_preserved": True}
        write_json(folder / "run.json", report)
        return report
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "message": str(exc)})
        try:
            recorder.finish(failed=True)
        except Exception:
            pass
        raise
