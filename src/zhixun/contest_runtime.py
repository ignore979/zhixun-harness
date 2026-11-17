"""Official workspace-v2 adapter. All model access belongs to the host gateway."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from zhixun.data import Dataset, digest, write_json
from zhixun.metrics import promotion, score
from zhixun.model_extension import ClassWeightedModel
from zhixun.vendor.data import flatten, json_value
from zhixun.vendor.model import Model

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
_PREDICT_MODELS = {}


def parse_plan(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    if text.startswith("TRAINING_PLAN"):
        tree = ast.parse(text)
        if (
            len(tree.body) != 1
            or not isinstance(tree.body[0], ast.Assign)
            or len(tree.body[0].targets) != 1
            or not isinstance(tree.body[0].targets[0], ast.Name)
            or tree.body[0].targets[0].id != "TRAINING_PLAN"
        ):
            raise ValueError("Only a literal TRAINING_PLAN assignment is allowed")
        value = ast.literal_eval(tree.body[0].value)
    else:
        value = json_value(text)
    if (
        not isinstance(value, dict)
        or set(value) - {"action", "epochs", "learning_rate", "target_version", "cw_power"}
        or not {"action", "epochs", "learning_rate"} <= set(value)
    ):
        raise ValueError("Unexpected training plan fields")
    action, epochs, rate = value["action"], value["epochs"], value["learning_rate"]
    weight = value.get("cw_power", 0.5)
    if (
        type(epochs) is not int
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0 <= weight <= 1.5
        or value.get("target_version") is not None
    ):
        raise ValueError("Invalid optimizer parameters")
    if action == "retrain":
        if not 1 <= epochs <= 10 or not 0 < rate <= 0.001:
            raise ValueError("Training plan exceeds epoch/rate limits")
    elif action != "hold" or epochs != 0 or rate != 0:
        raise ValueError("Invalid hold plan")
    return dict(action=action, epochs=epochs, learning_rate=float(rate), cw_power=float(weight))


def rows_dataset(rows, model):
    identifiers, x, y, seen = [], [], [], set()
    for row in rows:
        if len(identifiers) >= 1000000:
            raise ValueError("Public projection exceeds the local row budget")
        identity = row["id"]
        if not identity or identity in seen or "answer" not in row:
            raise ValueError("Repeated IDs or missing public labels")
        paths, values = flatten(row["inputs"])
        if (len(identifiers) + 1) * len(values) > 8000000:
            raise ValueError("Projection exceeds the bounded feature-memory budget")
        if paths != model.meta["paths"]:
            raise ValueError("Projection differs from the parent feature schema")
        answer = row["answer"]
        if model.meta["task"] == "classification":
            if not isinstance(answer, str) or answer not in model.meta["classes"]:
                raise ValueError("Public label differs from parent classes")
            answer = model.meta["classes"].index(answer)
        elif (
            isinstance(answer, bool)
            or not isinstance(answer, (int, float))
            or not math.isfinite(answer)
            or answer < 0
        ):
            raise ValueError("Regression labels must be finite and nonnegative")
        seen.add(identity)
        identifiers.append(identity)
        x.append(values)
        y.append(answer)
    if not identifiers:
        raise ValueError("Empty public projection")
    dtype = np.int64 if model.meta["task"] == "classification" else np.float64
    return Dataset(
        tuple(identifiers), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=dtype), frozenset()
    )


def execute_plan(parent, data, plan, deadline):
    candidate = copy.deepcopy(parent)
    candidate.__class__ = ClassWeightedModel
    losses = []
    initial_steps, initial_epochs = candidate.meta["steps"], candidate.meta["epochs"]
    for _ in range(plan["epochs"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training budget exhausted")
        candidate.meta["learning_rate"] = plan["learning_rate"]
        losses.extend(candidate.fit(data.x, data.y, 1, cw_power=plan["cw_power"]))
    if plan["epochs"] and time.monotonic() >= deadline:
        raise TimeoutError("Training exceeded the reserved deadline")
    if (
        not np.array_equal(parent.mean, candidate.mean)
        or not np.array_equal(parent.scale, candidate.scale)
        or parent.meta["target_mean"] != candidate.meta["target_mean"]
        or parent.meta["target_scale"] != candidate.meta["target_scale"]
        or candidate.meta["epochs"] != initial_epochs + plan["epochs"]
        or candidate.meta["steps"]
        != initial_steps + plan["epochs"] * math.ceil(len(data.ids) / candidate.meta["batch_size"])
    ):
        raise ValueError("Parent normalization/optimizer continuation invariant failed")
    candidate.validate()
    return candidate, losses


def model_query(rnd, payload, system, audit):
    receipt = rnd.model(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ],
        max_tokens=min(512, rnd.max_output_tokens),
        temperature=0,
        enable_thinking=False,
    )
    audit.append({"call_id": receipt.call_id, "response": receipt.text[:16000]})
    return receipt.text


def train(rnd):
    started = time.monotonic()
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    run = work / f"zhixun-{rnd.dataset_key}-r{rnd.round_index}"
    run.mkdir(parents=True, exist_ok=False)
    history = rnd.history()
    for relative in rnd.parent.descriptor["code_paths"]:
        rnd.parent.path(relative)  # SDK digest check before citing parent code.
    checkpoints = [p for p in rnd.parent.descriptor["checkpoint_paths"] if p.endswith(".npz")]
    if len(checkpoints) != 1:
        raise ValueError("Exactly one declared CPU NPZ checkpoint is required")
    parent_path = rnd.parent.path(checkpoints[0])
    parent = Model.load(parent_path)
    if parent.meta["task"] != rnd.task["task"]:
        raise ValueError("Frozen task differs from parent checkpoint")
    if parent.meta.get("profile", rnd.dataset_key) != rnd.dataset_key:
        raise ValueError("Checkpoint belongs to another dataset")
    training = rows_dataset(rnd.data.training_rows(), parent)
    validation = rows_dataset(rnd.data.validation_rows(), parent)
    if set(training.ids) & set(validation.ids):
        raise ValueError("Training and public validation IDs overlap")
    before, _, _ = score(parent, validation)
    calls = []
    events = [
        {"tool": "read_history", "history_ids": list(rnd.history_ids)},
        {"tool": "read_code", "function": "execute_plan"},
    ]
    system = (
        "You control CPU continuation via the write_training_plan tool. Read the provided "
        'history and code. Return ONLY TRAINING_PLAN = {"action":"retrain", "epochs":5, '
        '"learning_rate":0.001, "cw_power":0.5}, or the same dictionary as JSON. '
        "Allowed actions: retrain (1..10 epochs, 0<learning_rate<=0.001) and hold "
        "(epochs=0, learning_rate=0). cw_power is 0..1.5, classification only. "
        "Use actual past outcomes; validation labels are never training inputs. "
        "A maximizes macro_f1 without losing accuracy; B/C reduce rmse and mae. "
        "No imports, executable expressions, tools outside this plan, or prose."
    )
    payload = {
        "dataset": rnd.dataset_key,
        "task": rnd.task,
        "round": rnd.round_index,
        "train_rows": len(training.ids),
        "validation_rows": len(validation.ids),
        "local_parent_validation": before,
        "parent_checkpoint_sha256": digest(parent_path),
        "history": history[-3:],
        "training_tool_code": inspect.getsource(execute_plan),
    }
    raw = model_query(rnd, payload, system, calls)
    try:
        plan = parse_plan(raw)
    except (ValueError, SyntaxError, TypeError) as exc:
        events.append({"proposal_error": str(exc)})
        raw = model_query(rnd, {"invalid_reply": raw[:4000], "error": str(exc)}, system, calls)
        try:
            plan = parse_plan(raw)
        except (ValueError, SyntaxError, TypeError) as error:
            events.append({"repair_error": str(error), "guard": "retain_parent"})
            plan = dict(action="hold", epochs=0, learning_rate=0.0, cw_power=0.5)
    candidate_dir = run / "candidate"
    candidate_dir.mkdir()
    plan_path = candidate_dir / "training_plan.py"
    plan_path.write_text("TRAINING_PLAN = " + repr(plan) + "\n", encoding="utf-8")
    # The bounded write-plan tool emits actual code, which is read and validated again.
    executed_plan = parse_plan(plan_path.read_text(encoding="utf-8"))
    events.append({"tool": "write_training_plan", "sha256": digest(plan_path)})
    allowance = min(180, max(0, (rnd.wall_seconds or 300) - (time.monotonic() - started) - 15))
    try:
        candidate, losses = execute_plan(
            parent, training, executed_plan, time.monotonic() + allowance
        )
    except TimeoutError:
        candidate, losses = copy.deepcopy(parent), []
        events.append({"training_timeout": True, "candidate_reset_to_parent": True})
    after, _, _ = score(candidate, validation)
    candidate.save(candidate_dir / "checkpoint.npz")
    audit = {
        "protocol": "contest.harness.workspace.v2",
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "history_ids": list(rnd.history_ids),
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "actual_parent_file_sha256": digest(parent_path),
        "plan": executed_plan,
        "training_plan_sha256": digest(plan_path),
        "plan_code_renderer": "bounded_write_plan_tool",
        "local_public_before": before,
        "local_public_after": after,
        "train_losses": losses,
        "model_receipts": calls,
        "events": events,
        "tracking": "local_json_plus_optional_swanlab",
    }
    events.append({"tool": "execute_training_plan", "completed_epochs": len(losses)})
    # SwanLab writes only in the writable workspace; its archive is retained in native state.
    log_zip = None
    try:
        from zhixun.tracking import Recorder

        track = run / "tracking"
        track.mkdir()
        recorder = Recorder(
            track,
            {k: v for k, v in audit.items() if k != "model_receipts"},
            mode="offline",
            project="zhixun-official-sdk",
            workspace=None,
        )
        recorder.log(0, {f"parent/{k}": v for k, v in before.items()})
        recorder.log(1, {f"candidate/{k}": v for k, v in after.items()})
        recorder.finish()
        log_zip = run / "swanlab-offline.zip"
        with zipfile.ZipFile(log_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in track.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(track).as_posix())
        audit["swanlab_offline_recorded"] = True
    except Exception as exc:
        audit.update(swanlab_offline_recorded=False, tracking_error=f"{type(exc).__name__}: {exc}")
    write_json(candidate_dir / "experiment.json", audit)
    hypothesis = "Use bounded parent continuation and training-only class weighting, then validate."
    change = (
        f"Parent Adam state retained; requested {plan['epochs']} epochs at {plan['learning_rate']}."
    )
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change,
        configuration={
            "plan": executed_plan,
            "local_public_before": before,
            "local_public_after": after,
            "training_plan_sha256": digest(plan_path),
        },
        seed=parent.meta["seed"],
        checkpoint_paths=["checkpoint.npz"],
        code_paths=["training_plan.py", "experiment.json"],
        predict_argv=PREDICT_ARGV,
    )
    choice_text = model_query(
        rnd,
        {
            "parent_artifact_id": rnd.parent.artifact_id,
            "candidate_artifact_id": submitted["artifact_id"],
            "host_candidate_receipt": submitted,
            "local_public_before": before,
            "local_public_after": after,
        },
        "Choose using the real host receipt. Return ONLY JSON "
        '{"select":"candidate" or "parent","reason":"actual reason"}. '
        "Do not claim hidden scores. Reject quality regressions.",
        calls,
    )
    try:
        choice = json_value(choice_text.strip().removeprefix("```json").removesuffix("```").strip())
        select = choice.get("select")
        if select not in ("parent", "candidate"):
            select = {"hold": "parent", "retrain": "candidate"}.get(choice.get("action"))
        reason = str(choice.get("reason", "Model returned a finite hold/retrain selection."))[:2000]
    except (ValueError, AttributeError):
        select, reason = "parent", "Invalid selection response; conservative parent retention."
    eligible, gate = promotion(before, after, rnd.dataset_key)
    if select != "candidate" or not eligible:
        selected = rnd.parent.artifact_id
        reason += " Parent retained; local quality guard: " + gate
    else:
        selected = submitted["artifact_id"]
    audit.update(
        host_candidate_receipt=submitted,
        model_receipts=calls,
        selected_artifact_id=selected,
        selection_reason=reason,
    )
    state = rnd.state_workspace(run / "state")
    write_json(state / f"round-{rnd.round_index:02d}.json", audit)
    if log_zip is not None and log_zip.is_file() and log_zip.stat().st_size <= 1024 * 1024:
        shutil.copy2(log_zip, state / f"round-{rnd.round_index:02d}-swanlab.zip")
    rnd.save_state(state)
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis,
        change_summary=change,
        reason=reason,
        configuration={"plan": executed_plan, "guard_passed": eligible},
        seed=parent.meta["seed"],
        history_ids=list(rnd.history_ids),
        model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    root = Path(model_dir).resolve()
    path = (
        Path(os.environ["CONTEST_ARTIFACT_PATH"])
        if os.environ.get("CONTEST_ARTIFACT_PATH")
        else root / "checkpoint.npz"
    )
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Prediction checkpoint must belong to the selected artifact")
    key = str(path.resolve())
    if key not in _PREDICT_MODELS:
        _PREDICT_MODELS[key] = Model.load(path)
    model = _PREDICT_MODELS[key]
    if sample.task != model.meta["task"]:
        raise ValueError("Prediction task differs from selected checkpoint")
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("Prediction feature schema differs from selected checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if sample.task == "regression":
        return {"value": float(output)}
    maximum = output.max()
    if np.count_nonzero(output == maximum) > 1:
        output[int(np.argmax(output))] = np.nextafter(maximum, 1.0)
    return [
        {"label": label, "score": float(value)}
        for label, value in zip(model.meta["classes"], output, strict=True)
    ]
