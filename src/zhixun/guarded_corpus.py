"""Relabel verified finite-action results with a same-state accuracy constraint."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from zhixun.data import ROOT, digest, write_json
from zhixun.drift import fit_reference
from zhixun.fair_core import (
    build_prompt,
    load_fair_config,
    load_fair_data,
    observe,
    rule_action,
    train_action,
)
from zhixun.metrics import score
from zhixun.pool import TrainingPool
from zhixun.prepare_gas import SOURCE_URL
from zhixun.reward import rank_guarded
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def prepare_guarded_corpus(source, *, root=ROOT, mode="offline", project="zhixun-school",
                           extra_runs=(), destination_name="harness-guarded-v4"):
    config = load_fair_config(root, root / "configs/replay_guarded.json")
    states_path = source / "states.json"
    source_report = json.loads((source / "run.json").read_text(encoding="utf-8"))
    states = json.loads(states_path.read_text(encoding="utf-8"))
    if (source_report["status"] != "completed" or digest(states_path) != source_report[
            "corpus"]["states_sha256"]):
        raise ValueError("Source verified outcomes changed")
    if destination_name not in {"harness-guarded-v4", "harness-guarded-v4-expanded"}:
        raise ValueError("Unsupported versioned corpus destination")
    destination = root / "data" / destination_name
    if destination.exists():
        raise ValueError("Versioned guarded corpus already exists")
    _, base, train, batches = load_fair_data(root)
    folder = root / "artifacts" / (f"run-guarded-corpus-{datetime.now(UTC):%Y%m%dT%H%M%S}-"
                                    f"{uuid4().hex[:8]}")
    folder.mkdir(parents=True)
    registration = {"experiment_id": folder.name, "created_at": datetime.now(UTC).isoformat(),
                    "task": "accuracy_floor_relabel_actual_verified_actions",
                    "source_run": source.name, "source_states_sha256": digest(states_path),
                    "config": config, "extra_source_runs": [p.name for p in extra_runs]}
    write_json(folder / "registration.json", registration)
    recorder = Recorder(folder, registration, mode=mode, project=project, workspace=None)
    relabeled = []
    try:
        new_candidates = 0
        ref = fit_reference(train.x, base)
        reference = score(base, batches[0].data)[0]["macro_f1"]
        for extra in extra_runs:
            report = json.loads((extra / "run.json").read_text(encoding="utf-8"))
            if report["status"] != "completed":
                raise ValueError("Extra source run is not complete")
            registry = json.loads((extra / "stream-state.json").read_text(encoding="utf-8"))
            versions = {v["version"]: v for v in registry["versions"]}
            pool, spent, history = TrainingPool.from_dataset(train), 0, []
            for event in report["events"]:
                if event["status"] != "completed":
                    continue
                batch = event["batch_index"] + 2
                if batch > 8:
                    break
                version = versions[event["parent_version"]]
                checkpoint = extra / version["path"]
                if digest(checkpoint) != version["sha256"]:
                    raise ValueError("Extra source checkpoint changed")
                parent = Model.load(checkpoint)
                observation = observe(parent, batches[batch - 2].data, pool, batch,
                                      45 - spent, reference, history, ref, config)
                prompt = build_prompt(observation, config)
                outcomes = []
                for action in config["action_catalog"]:
                    if action["epochs"] > observation["remaining_epochs"]:
                        continue
                    selected, outcome = train_action(parent, pool, batches[batch - 2].data,
                                                      action, config)
                    metrics = score(selected, batches[batch - 1].data)[0]
                    outcomes.append({"action": action, "current_gate": outcome,
                                     "next_metrics": metrics, "epochs": action["epochs"]})
                    new_candidates += 1
                old_best = max(outcomes, key=lambda r: r["next_metrics"]["accuracy"] - .001 * r[
                    "epochs"])["action"]
                states.append({"state_id": hashlib.sha256(prompt.encode()).hexdigest(),
                               "observation": observation,
                               "split": "train" if batch <= 6 else "evaluation",
                               "outcomes": outcomes, "chosen": old_best,
                               "source_lineage": [{"run": extra.name,
                                   "report_sha256": digest(extra / "run.json"),
                                   "parent_checkpoint_sha256": digest(checkpoint),
                                   "decision_batch": batch, "lookahead_batch": batch + 1}]})
                pool.append(batches[batch - 2].data, batch_id=batches[batch - 2].identity)
                action = event.get("action", event.get("plan"))
                spent += action["epochs"]
                history.append({"action": action["action"], "epochs": action["epochs"],
                                "accepted": event.get("outcome", {}).get("accepted",
                                    event.get("accepted")),
                                "selected_metrics": {k: event.get("selected_metrics",
                                    event.get("selected_same_batch"))[k]
                                    for k in ["accuracy", "macro_f1"]}})
        seen = {}
        for index, state in enumerate(states):
            observation = state["observation"].copy()
            batch = observation["batch"]
            if state["split"] != ("train" if 2 <= batch <= 6 else "evaluation") or batch > 8:
                raise ValueError("Source split violates temporal boundary")
            lineage = state["source_lineage"][0]
            source_folder = root / "artifacts" / lineage["run"]
            registry = json.loads((source_folder / "stream-state.json").read_text(encoding="utf-8"))
            version = next(v for v in registry["versions"] if v["sha256"] == lineage[
                "parent_checkpoint_sha256"])
            checkpoint = source_folder / version["path"]
            if digest(checkpoint) != version["sha256"]:
                raise ValueError("Teacher parent checkpoint changed")
            observation["current_metrics"] = score(Model.load(checkpoint),
                                                    batches[batch - 2].data)[0]
            reference_action = rule_action(observation, config, flat=True)
            ranked = rank_guarded(state["outcomes"], reference_action,
                                  floor_ratio=config["reward_floor_ratio"],
                                  lam=config["offline_cost_weight_per_epoch"])
            prompt = build_prompt(observation, config)
            chosen = ranked[0]["action"]
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key in seen:
                seen[key]["source_lineage"].extend(state["source_lineage"])
                continue
            row = {"source_state_id": state["state_id"],
                   "state_id": key,
                   "split": state["split"], "prompt": prompt, "observation": observation,
                   "chosen": chosen, "old_chosen": state["chosen"],
                   "chosen_over": ranked[-1]["action"], "outcomes": ranked,
                   "source_lineage": state["source_lineage"],
                   "flat_reference_action": reference_action}
            relabeled.append(row)
            seen[key] = row
            recorder.log(index, {"corpus/changed_label": float(chosen != state["chosen"]),
                                "chosen/epochs": chosen["epochs"],
                                "chosen/reward": ranked[0]["reward"],
                                "chosen/floor_feasible": float(ranked[0]["floor_feasible"])})
        if len({r["state_id"] for r in relabeled}) != len(relabeled):
            raise ValueError("Duplicate new policy observations")
        destination.mkdir()
        for split in ["train", "evaluation"]:
            rows = [{"prompt": s["prompt"], "completion": json.dumps(s["chosen"], sort_keys=True,
                     separators=(",", ":"))} for s in relabeled if s["split"] == split]
            (destination / f"{split}.jsonl").write_text("".join(
                json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (destination / "preferences.jsonl").write_text("".join(json.dumps({
            "state_id": s["state_id"], "split": s["split"], "chosen": s["chosen"],
            "rejected": s["chosen_over"], "source": "verified_same_state_accuracy_floor"
        }) + "\n" for s in relabeled), encoding="utf-8")
        write_json(folder / "states.json", relabeled)
        provenance = {"format": "zhixun.accuracy_guarded_verified_policy.v4",
                      "sha256": digest(destination / "train.jsonl"),
                      "evaluation_sha256": digest(destination / "evaluation.jsonl"),
                      "source_url": SOURCE_URL, "authorized_for_llm_training": True,
                      "reviewed_by": "user_targeted_plan_and_temporal_integrity_checks",
                      "official_llm_corpus": False, "source_experiment": folder.name,
                      "source_outcomes_sha256": digest(states_path),
                      "new_candidate_actions_evaluated": new_candidates,
                      "extra_source_reports": {p.name: digest(p / "run.json") for p in extra_runs},
                      "states_sha256": digest(folder / "states.json"),
                      "training_rows": sum(s["split"] == "train" for s in relabeled),
                      "evaluation_rows": sum(s["split"] == "evaluation" for s in relabeled),
                      "training_label_lookahead_max_batch": 7,
                      "changed_training_labels": sum(s["chosen"] != s["old_chosen"]
                                                     for s in relabeled if s["split"] == "train"),
                      "floor_reference": "actual_same_state_flat_action_next_batch_accuracy",
                      "policy_guarantee": "offline_candidate_constraint_only_no_future_guarantee",
                      "independence": "repeated_public_development_not_blind_test"}
        write_json(destination / "provenance.json", provenance)
        recorder.finish()
        result = {**registration, "status": "completed", "corpus": provenance,
                  "tracking": recorder.reference()}
        write_json(folder / "run.json", result)
        return result
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "message": str(exc)})
        recorder.finish(failed=True)
        raise
