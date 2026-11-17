from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from zhixun.data import ROOT, digest, write_json
from zhixun.tracking import Recorder


def prompt_tokens(prompt: str, tokenizer) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("Expected one encoded chat prompt")
            encoded = encoded[0]
        return [int(token) for token in encoded]
    return tokenizer.encode(prompt, add_special_tokens=True)


def list_models() -> dict:
    import pytrio

    models = pytrio.ServiceClient().get_supported_models()
    return {"supported_models": models, "contest_model_approval": "must_be_confirmed"}


def load_training_data(data_path: Path, provenance_path: Path) -> tuple[list[dict], dict]:
    if not data_path.is_file() or data_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("LLM training requires an explicit data file up to 2 MiB")
    declared = json.loads(provenance_path.read_text(encoding="utf-8"))
    if declared.get("sha256") != digest(data_path):
        raise ValueError("LLM dataset does not match its declared SHA-256")
    if (
        declared.get("authorized_for_llm_training") is not True
        or not str(declared.get("source_url", "")).startswith("https://")
        or not str(declared.get("reviewed_by", "")).strip()
    ):
        raise ValueError(
            "LLM data requires an explicit source, reviewer and authorization declaration"
        )
    rows = []
    with data_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict) or set(row) != {"prompt", "completion"}
                or not all(isinstance(v, str) and v.strip() for v in row.values())
            ):
                raise ValueError(
                    "Internal LLM training format requires prompt and completion strings"
                )
            rows.append(row)
    if not 1 <= len(rows) <= 256:
        raise ValueError("First-stage SFT is bounded to 1..256 reviewed examples")
    return rows, declared


def tokens_and_weights(row: dict, tokenizer) -> tuple[list[int], list[int], list[int]]:
    prompt = prompt_tokens(row["prompt"], tokenizer)
    completion = tokenizer.encode(row["completion"], add_special_tokens=False)
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and completion and completion[-1] != eos:
        completion.append(eos)
    tokens = prompt + completion
    weights = [0] * len(prompt) + [1] * len(completion)
    if not completion or not 2 <= len(tokens) <= 8192:
        raise ValueError("SFT examples need completion tokens and at most 8192 total tokens")
    return tokens[:-1], tokens[1:], weights[1:]


def train_sft(
    data_path: Path, provenance_path: Path, model: str, steps: int | None, learning_rate: float,
    *, project: str, workspace: str | None, epochs: int = 3, batch_size: int = 3,
    evaluation_path: Path | None = None,
) -> dict:
    if (type(epochs) is not int or not 1 <= epochs <= 100
            or type(batch_size) is not int or not 1 <= batch_size <= 64
            or (steps is not None and (type(steps) is not int or not 1 <= steps <= 1000))
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.001):
        raise ValueError("Invalid SFT schedule or learning rate")
    rows, provenance = load_training_data(data_path, provenance_path)
    evaluation = []
    if evaluation_path:
        if digest(evaluation_path) != provenance.get("evaluation_sha256"):
            raise ValueError("SFT evaluation does not match its provenance")
        evaluation = [json.loads(line) for line in evaluation_path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
        if not evaluation or len(evaluation) > 256 or any(row in rows for row in evaluation):
            raise ValueError("SFT evaluation is empty, oversized or overlaps training")
    planned_steps = steps or epochs * math.ceil(len(rows) / batch_size)
    cache = ROOT / "指导/模型缓存"
    os.environ["MODELSCOPE_CACHE"] = str(cache / "modelscope")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    import pytrio

    service = pytrio.ServiceClient()
    if model not in service.get_supported_models():
        raise ValueError("Requested model is not available to this account")
    folder = ROOT / "artifacts" / (
        f"pytrio-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    registration = {
        "experiment_id": folder.name, "created_at": datetime.now(UTC).isoformat(),
        "task": "harness_llm_sft", "base_model": model, "steps": planned_steps,
        "schedule_epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "lora_rank": 32, "seed": 42,
        "training_rows": len(rows), "training_sha256": digest(data_path),
        "data_provenance_declared_by_submitter": provenance,
        "independent_model_evaluation": "not_performed",
        "evaluation_rows": len(evaluation),
        "evaluation_sha256": digest(evaluation_path) if evaluation_path else None,
        "evaluation_dataset_path": (evaluation_path.resolve().relative_to(ROOT).as_posix()
                                    if evaluation_path.resolve().is_relative_to(ROOT)
                                    else str(evaluation_path.resolve()))
        if evaluation_path else None,
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p)
                           for p in sorted((ROOT / "src").rglob("*.py"))},
        "monetary_limit": None,
    }
    write_json(folder / "registration.json", registration)
    recorder = None
    completed_steps = 0
    try:
        recorder = Recorder(folder, registration, mode="online", project=project,
                            workspace=workspace)
        training = service.create_lora_training_client(base_model=model, rank=32, seed=42)
        write_json(folder / "progress.json", {"phase": "training_client_created",
                                               "completed_optimizer_steps": 0})
        tokenizer = training.get_tokenizer()

        def datum(row):
            inputs, targets, weights = tokens_and_weights(row, tokenizer)
            return pytrio.Datum(
                model_input=pytrio.ModelInput.from_ints(tokens=inputs),
                loss_fn_inputs={"target_tokens": np.asarray(targets, dtype=np.int32),
                                "weights": np.asarray(weights, dtype=np.float32)},
            ), sum(weights)

        prepared = [datum(row) for row in rows]
        validation = [datum(row) for row in evaluation]

        def validation_loss():
            total, count = 0.0, 0
            for offset in range(0, len(validation), batch_size):
                part = validation[offset:offset + batch_size]
                result = training.forward([d for d, _ in part], "cross_entropy").result()
                total += float(result.metrics["loss_sum"])
                count += sum(n for _, n in part)
            return total / count

        before_loss = validation_loss() if validation else None
        if before_loss is not None:
            recorder.log(0, {"pytrio/evaluation_loss_per_token": before_loss})
        generator = np.random.default_rng(42)
        schedule = []
        while len(schedule) < planned_steps:
            order = generator.permutation(len(rows)).tolist()
            schedule.extend([order[i:i + batch_size] for i in range(0, len(rows), batch_size)])
        for step, indices in enumerate(schedule[:planned_steps], 1):
            part = [prepared[i] for i in indices]
            supervised_tokens = sum(n for _, n in part)
            result = training.forward_backward([d for d, _ in part], "cross_entropy").result()
            training.optim_step(pytrio.AdamParams(learning_rate=learning_rate)).result()
            completed_steps = step
            recorder.log(step, {
                "pytrio/loss_per_token": float(result.metrics["loss_sum"]) / supervised_tokens,
                "pytrio/supervised_tokens": float(supervised_tokens),
                "pytrio/optimizer_steps": float(step),
            })
            write_json(folder / "progress.json", {"phase": "training",
                                                   "completed_optimizer_steps": step,
                                                   "planned_optimizer_steps": planned_steps})
        after_loss = validation_loss() if validation else None
        if after_loss is not None:
            recorder.log(planned_steps + 1, {"pytrio/evaluation_loss_per_token": after_loss})
        state = training.save_state(name=folder.name + "-train").result()
        sampler = training.save_weights_for_sampler(name=folder.name + "-sampler").result()
        recorder.finish()
        report = {
            **registration, "status": "completed", "train_checkpoint_path": state.path,
            "sampler_checkpoint_path": sampler.path, "tracking": recorder.reference(),
            "official_contest_model_approval": "not_independently_verified",
            "completed_optimizer_steps": completed_steps,
            "temporal_evaluation_loss_before": before_loss,
            "temporal_evaluation_loss_after": after_loss,
            "inference_policy_integrated": False,
        }
        write_json(folder / "run.json", report)
        return report
    except BaseException as exc:
        write_json(folder / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "message": str(exc),
            "completed_optimizer_steps": completed_steps,
        })
        if recorder is not None:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise
