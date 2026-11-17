"""End-to-end agent replay: no oracle/lookahead is available on the runtime path."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from zhixun.agent import LlmPolicy
from zhixun.data import ROOT, digest, verify_inputs, write_json
from zhixun.drift import fit_reference
from zhixun.ensemble import EnsemblePredictor
from zhixun.fair_core import (
    load_fair_config,
    load_fair_data,
    observe,
    rule_action,
    train_action,
    validate_action,
)
from zhixun.metrics import classification_metrics, score
from zhixun.pool import TrainingPool
from zhixun.prior_correction import CoralAdapter
from zhixun.registry import ModelRegistry, process_lock
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def run_fair(strategy, *, training_run: Path | None = None, resume: Path | None = None,
             mode="offline", project="zhixun-school", root=ROOT, output=None, policy_factory=None,
             config_path=None, prediction_config=None):
    if strategy not in {"flat", "soft", "agent_base", "agent_trained"}:
        raise ValueError("Unknown fair strategy")
    started_wall = perf_counter()
    config = load_fair_config(root, config_path)
    evaluation_scope = json.loads((root / "configs/fair_evaluation_scope.json").read_text(
        encoding="utf-8"))
    initial, base, train, batches = load_fair_data(root)
    prediction_spec = (json.loads(prediction_config.read_text(encoding="utf-8"))
                       if prediction_config else None)
    adapter = None
    if prediction_spec:
        if set(prediction_spec) != {"ensemble", "coral"}:
            raise ValueError("Prediction recipe requires ensemble and coral sections")
        settings = prediction_spec["coral"]
        if settings["strength"]:
            adapter = CoralAdapter.from_source(base, train.x, **settings)
    model, sampler = "Qwen/Qwen3.8-27B", None
    if strategy == "agent_trained":
        if training_run is None:
            raise ValueError("Trained agent requires its real completed training run")
        trained = json.loads((training_run / "run.json").read_text(encoding="utf-8"))
        if trained["status"] != "completed" or trained["completed_optimizer_steps"] <= 0:
            raise ValueError("Trained agent weights were not successfully produced")
        model, sampler = trained["base_model"], trained["sampler_checkpoint_path"]
    signature = {"strategy": strategy, "config": config, "base_model": model,
                 "evaluation_scope": evaluation_scope,
                 "sampler_checkpoint_path": sampler, "initial_sha256": digest(initial),
                 "batches": [{"id": b.identity, "sha256": b.source_sha256} for b in batches],
                 "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                                    for p in sorted((root / "src").rglob("*.py"))}}
    if prediction_spec:
        signature["prediction_spec"] = prediction_spec
    folder = (output or root / "artifacts") / (
        f"run-fair-{strategy}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}")
    folder.mkdir(parents=True, exist_ok=False)
    session = resume.resolve() if resume else folder
    if resume and not (session / "stream-state.json").is_file():
        raise ValueError("Resume requires the original fair session directory")
    registration = {"experiment_id": folder.name, "format": "zhixun.live-agent-fair.v3",
                    "created_at": datetime.now(UTC).isoformat(), **signature,
                    "session": session.relative_to(root).as_posix(),
                    "resumed": resume is not None,
                    "primary_scope": "prediction_before_current_label_based_action",
                    "independence": "repeated_public_development_not_hidden_test"}
    write_json(folder / "registration.json", registration)
    recorder = None
    try:
        with process_lock(session):
            registry = ModelRegistry(session, signature, initial)
            registry.recover_interruption()
            state = registry.state
            state.setdefault("epochs_charged", 0)
            state.setdefault("llm_calls", 0)
            if prediction_spec and state["cursor"] and "ensemble_history" not in state:
                raise ValueError("Resume requires committed prequential expert reliability")
            if (not 0 <= state["cursor"] <= len(batches)
                    or not 0 <= state["epochs_charged"] <= config["total_epoch_budget"]
                    or not 0 <= state["llm_calls"] <= config["max_llm_calls"]):
                raise ValueError("Saved fair session exceeds its resource or cursor bounds")
            pool = TrainingPool.from_dataset(train)
            for b in batches[:state["cursor"]]:
                pool.append(b.data, batch_id=b.identity)
            if "pool" in state and state["pool"] != pool.summary():
                raise ValueError("Recovered fair pool changed")
            recorder = Recorder(folder, registration, mode=mode, project=project, workspace=None)

            def guard_inputs():
                verify_inputs("A", root)
                for index, batch in enumerate(batches):
                    path = root / ("data/A/validation.jsonl" if index == 0
                                    else f"data/replay-gas/batch{index + 2}.jsonl")
                    if digest(path) != batch.source_sha256:
                        raise ValueError("Fair replay input changed during execution")

            def reserve_call():
                if state["llm_calls"] >= config["max_llm_calls"]:
                    raise ValueError("Persistent LLM call budget exhausted")
                state["llm_calls"] += 1
                registry.save()

            policy = None
            if strategy.startswith("agent"):
                policy = (policy_factory or LlmPolicy)(
                    config, model=model, model_path=sampler, calls=state["llm_calls"],
                    before_request=reserve_call)
            ref = fit_reference(train.x, base)
            for b in batches[state["cursor"]:]:
                parent_version = state["current"]
                parent = Model.load(registry.checkpoint(parent_version))
                prediction_started = perf_counter()
                ensemble = None
                if prediction_spec:
                    ensemble = EnsemblePredictor(
                        registry, **prediction_spec["ensemble"],
                        history=state.get("ensemble_history", {}))
                    transform = adapter.transform(base, b.data.x) if adapter else None
                    probabilities = ensemble.predict(b.data.x, "classification",
                                                      transform=transform, batch=b.index + 2)
                else:
                    probabilities = parent.predict(b.data.x)
                prediction_seconds = perf_counter() - prediction_started
                prediction_path = session / f"prequential-batch-{b.index}.jsonl"
                with prediction_path.open("w", encoding="utf-8") as stream:
                    for identity, values in zip(b.data.ids, probabilities, strict=True):
                        stream.write(json.dumps({"id": identity, "probabilities": values.tolist()})
                                     + "\n")
                metrics, details = classification_metrics(probabilities, b.data.y,
                                                           parent.meta["classes"])
                state.setdefault("reference_macro_f1", metrics["macro_f1"])
                history = [{"action": e["action"]["action"], "epochs": e["action"]["epochs"],
                            "accepted": e["outcome"]["accepted"],
                            "selected_metrics": {k: e["selected_metrics"][k]
                                                 for k in ["accuracy", "macro_f1"]}}
                           for e in state["events"] if e.get("status") == "completed"]
                observation = observe(parent, b.data, pool, b.index + 2,
                                      config["total_epoch_budget"] - state["epochs_charged"],
                                      state["reference_macro_f1"], history, ref, config)
                if prediction_spec:
                    observation.update(current_metrics=metrics,
                                       performance_drop=state["reference_macro_f1"] - metrics[
                                           "macro_f1"],
                                       per_class_recall={c: details[c]["recall"]
                                                         for c in parent.meta["classes"]})
                if policy:
                    action, audit = policy.decide(observation)
                else:
                    action = rule_action(observation, config, flat=strategy == "flat")
                    audit = {"source": strategy, "input_tokens": 0, "output_tokens": 0,
                             "proposal_valid": True, "fallback_reason": None}
                validate_action(action, observation, config)
                if state["training_seconds_charged"] >= config["max_training_seconds"]:
                    action = config["action_catalog"][0].copy()
                    audit["resource_guard"] = "training_seconds_exhausted"
                attempt = state["updates_charged"] + 1
                path = session / f"attempt-{attempt}-candidate.npz"
                version = max(v["version"] for v in state["versions"]) + attempt
                if action["action"] == "retrain":
                    state["epochs_charged"] += action["epochs"]
                    state["updates_charged"] += 1
                    state["training_seconds_charged"] += config["max_seconds_per_round"]
                    state["inflight"] = {"batch": b.index, "epochs_reserved": action["epochs"]}
                    registry.save()

                def log_epoch(epoch, loss, rate, seconds, batch_index=b.index):
                    recorder.log(batch_index * 100 + epoch + 1, {"train/loss": loss,
                                 "train/learning_rate": rate, "train/seconds": seconds})

                selected, outcome = train_action(
                    parent, pool, b.data, action, config, on_epoch=log_epoch,
                    on_candidate=lambda candidate, target=path: candidate.save(target))
                guard_inputs()
                if action["action"] == "retrain":
                    registry.register(version, path, parent=parent_version,
                                      adopted=outcome["accepted"])
                    if outcome["accepted"]:
                        state["current"] = version
                    state["training_seconds_charged"] += outcome["training_seconds"] - config[
                        "max_seconds_per_round"]
                    state["inflight"] = None
                selected_metrics, _, _ = score(selected, b.data)
                event = {"status": "completed", "batch_index": b.index, "batch": b.index + 2,
                         "parent_version": parent_version, "selected_version": state["current"],
                         "prequential": metrics, "rows": len(b.data.ids),
                         "prequential_details": details, "prediction_seconds": prediction_seconds,
                         "observation": observation, "action": action, "agent": audit,
                         "outcome": outcome, "selected_metrics": selected_metrics,
                         "epochs_charged_total": state["epochs_charged"]}
                if ensemble:
                    event["ensemble"] = ensemble.last_audit
                    event["promotion_gate_scope"] = "raw_candidate_vs_raw_parent_same_batch"
                    ensemble.release(b.data.y, batch=b.index + 2)
                    state["ensemble_history"] = ensemble.history
                pool.append(b.data, batch_id=b.identity)
                state["pool"] = pool.summary()
                state["cursor"] = b.index + 1
                state["events"].append(event)
                verify_inputs("A", root)
                registry.checkpoint(parent_version)
                recorder.log(b.index * 100 + 90, {
                    "batch/number": b.index + 2, "prequential/accuracy": metrics["accuracy"],
                    **{f"prequential/{k}": v for k, v in metrics.items()},
                    "decision/epochs": action["epochs"],
                    "decision/accepted": float(outcome["accepted"]),
                    "agent/fallback": float(audit["source"] == "fallback_rule"),
                    "agent/input_tokens": audit["input_tokens"],
                    "agent/output_tokens": audit["output_tokens"],
                    "budget/epochs_charged": state["epochs_charged"],
                    "budget/llm_calls": state["llm_calls"],
                    "budget/optimizer_steps": outcome["optimizer_steps"],
                    "prediction/seconds": prediction_seconds,
                    "prediction/forward_passes": len(event.get("ensemble", {}).get(
                        "versions", [parent_version])),
                    **{f"recall/{c}": details[c]["recall"] for c in parent.meta["classes"]},
                })
                recorder.text(b.index * 100 + 90, "agent/decision", event)
                registry.save()
            completed = [e for e in state["events"] if e.get("status") == "completed"]
            total_rows = sum(e["rows"] for e in completed)
            summary = {
                "prequential_accuracy": sum(e["prequential"]["accuracy"] * e["rows"]
                                             for e in completed) / total_rows,
                "mean_batch_macro_f1": float(np.mean([
                    e["prequential"]["macro_f1"] for e in completed])),
                "epochs_charged": state["epochs_charged"], "llm_calls": state["llm_calls"],
                "optimizer_steps": sum(e["outcome"]["optimizer_steps"] for e in completed),
                "training_seconds": state["training_seconds_charged"],
                "fallback_decisions": sum(e["agent"]["source"] == "fallback_rule"
                                          for e in completed),
                "input_tokens": sum(e["agent"]["input_tokens"] for e in completed),
                "output_tokens": sum(e["agent"]["output_tokens"] for e in completed),
                "mean_batch_g_mean": float(np.mean([e["prequential"]["g_mean"]
                                                     for e in completed])),
            }
            tail = [e for e in completed if e["batch"] in evaluation_scope[
                "primary_policy_evaluation_batches"]]
            summary["policy_tail_accuracy"] = sum(e["prequential"]["accuracy"] * e["rows"]
                                                   for e in tail) / sum(e["rows"] for e in tail)
            summary["policy_tail_mean_macro_f1"] = float(np.mean([
                e["prequential"]["macro_f1"] for e in tail]))
            summary["policy_tail_mean_g_mean"] = float(np.mean([
                e["prequential"]["g_mean"] for e in tail]))
            recorder.log(1000, {f"summary/{k}": v for k, v in summary.items()})
            execution_seconds = perf_counter() - started_wall
            recorder.log(1001, {"timing/execution_seconds_before_final_sync": execution_seconds})
            shutil.copy2(registry.checkpoint(state["current"]), folder / "selected-checkpoint.npz")
            recorder.finish()
            report = {**registration, "status": "completed", "events": state["events"],
                      "summary": summary, "tracking": recorder.reference(),
                      "execution_seconds_before_final_sync": execution_seconds,
                      "actual_llm_connected": strategy.startswith("agent")
                      and state["llm_calls"] > 0}
            write_json(folder / "run.json", report)
            return report
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "message": str(exc)})
        if recorder:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise
