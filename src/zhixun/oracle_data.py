"""Offline finite-action lookahead labels, confined to the declared training/evaluation prefixes."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from zhixun.data import ROOT, digest, write_json
from zhixun.drift import fit_reference
from zhixun.fair_core import build_prompt, load_fair_config, load_fair_data, observe, train_action
from zhixun.metrics import score
from zhixun.pool import TrainingPool
from zhixun.prepare_gas import SOURCE_URL
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def prepare_oracle(*, mode="offline", project="zhixun-school", root=ROOT):
    config = load_fair_config(root)
    initial, base, train, batches = load_fair_data(root)
    ref = fit_reference(train.x, base)
    reference = score(base, batches[0].data)[0]["macro_f1"]
    folder = root / "artifacts" / (
        f"run-oracle-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}")
    folder.mkdir(parents=True, exist_ok=False)
    registration = {"experiment_id": folder.name, "task": "offline_finite_action_verification",
                    "config": config, "initial_sha256": digest(initial),
                    "training_label_scope": "decisions_2_to_6_lookahead_at_most_batch7",
                    "evaluation_scope": "decisions_7_8_next_batches_8_9_already_public_development",
                    "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                                       for p in sorted((root / "src").rglob("*.py"))}}
    write_json(folder / "registration.json", registration)
    recorder = Recorder(folder, registration, mode=mode, project=project, workspace=None)
    states, seen, counter = [], {}, 0
    try:
        for source in config["teacher_source_runs"]:
            source_folder = root / "artifacts" / source
            report = json.loads((source_folder / "run.json").read_text(encoding="utf-8"))
            registry = json.loads((source_folder / "stream-state.json").read_text(encoding="utf-8"))
            versions = {v["version"]: v for v in registry["versions"]}
            pool, spent, history = TrainingPool.from_dataset(train), 0, []
            for event in report["events"]:
                if event["status"] != "completed":
                    continue
                index, batch = event["batch_index"], event["batch_index"] + 2
                if batch > 8:
                    break
                if spent <= config["total_epoch_budget"]:
                    version = versions[event["parent_version"]]
                    path = source_folder / version["path"]
                    if digest(path) != version["sha256"]:
                        raise ValueError("Teacher source checkpoint changed")
                    parent = Model.load(path)
                    observation = observe(parent, batches[index].data, pool, batch,
                                          config["total_epoch_budget"] - spent, reference,
                                          history, ref, config)
                    prompt = build_prompt(observation, config)
                    key = hashlib.sha256(prompt.encode()).hexdigest()
                    lineage = {"run": source, "report_sha256": digest(source_folder / "run.json"),
                               "parent_checkpoint_sha256": digest(path), "decision_batch": batch,
                               "lookahead_batch": batch + 1}
                    if key in seen:
                        seen[key]["source_lineage"].append(lineage)
                    else:
                        split = "train" if batch <= 6 else "evaluation"
                        if split == "train" and batch + 1 > config[
                            "oracle_training_next_batch_max"
                        ]:
                            raise ValueError("Training teacher crossed the prefix boundary")
                        outcomes = []
                        for action in config["action_catalog"]:
                            if action["epochs"] > observation["remaining_epochs"]:
                                continue
                            selected, outcome = train_action(parent, pool, batches[index].data,
                                                              action, config)
                            metrics, _, _ = score(selected, batches[index + 1].data)
                            value = metrics["accuracy"] - config[
                                "offline_cost_weight_per_epoch"] * action["epochs"]
                            row = {"action": action, "current_gate": outcome,
                                   "next_metrics": metrics, "epochs": action["epochs"],
                                   "value": value}
                            outcomes.append(row)
                            recorder.log(counter, {"oracle/value": value,
                                                   "oracle/next_accuracy": metrics["accuracy"],
                                                   "oracle/epochs": action["epochs"],
                                                   "oracle/training_seconds": outcome[
                                                       "training_seconds"]})
                            counter += 1
                        ranked = sorted(outcomes, key=lambda a: (-a["value"], a["epochs"],
                                                                -a["next_metrics"]["macro_f1"]))
                        entry = {"state_id": key, "split": split, "prompt": prompt,
                                 "observation": observation, "source_lineage": [lineage],
                                 "outcomes": outcomes, "chosen": ranked[0]["action"],
                                 "chosen_over": ranked[1]["action"] if len(ranked) > 1 else None,
                                 "value_gain": ranked[0]["value"] - ranked[1]["value"]
                                 if len(ranked) > 1 else None,
                                 "source": "offline_enumeration_best_within_declared_catalog"}
                        states.append(entry)
                        seen[key] = entry
                        write_json(folder / "states.json", states)
                pool.append(batches[index].data, batch_id=batches[index].identity)
                spent += event["plan"]["epochs"]
                history.append({"action": event["plan"]["action"],
                                "epochs": event["plan"]["epochs"],
                                "accepted": event["accepted"],
                                "selected_metrics": event["selected_same_batch"]})
        destination = root / "data/harness-oracle-v3"
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "provenance.json").exists():
            raise ValueError("Oracle corpus already exists; version it instead of overwriting")
        for split in ["train", "evaluation"]:
            rows = [{"prompt": s["prompt"], "completion": json.dumps(s["chosen"], sort_keys=True,
                     separators=(",", ":"))} for s in states if s["split"] == split]
            if not rows:
                raise ValueError("Oracle split is empty")
            (destination / f"{split}.jsonl").write_text("".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        (destination / "preferences.jsonl").write_text("".join(json.dumps({
            "state_id": s["state_id"], "split": s["split"], "chosen": s["chosen"],
            "chosen_over": s["chosen_over"], "value_gain": s["value_gain"], "source": s["source"]
        }) + "\n" for s in states), encoding="utf-8")
        provenance = {"format": "zhixun.outcome_verified_policy.v3",
                      "sha256": digest(destination / "train.jsonl"), "source_url": SOURCE_URL,
                      "authorized_for_llm_training": True,
                      "reviewed_by": "user_v3_plan_and_automated_temporal_integrity_checks",
                      "official_llm_corpus": False, "source_experiment": folder.name,
                      "states_sha256": digest(folder / "states.json"),
                      "evaluation_sha256": digest(destination / "evaluation.jsonl"),
                      "training_rows": sum(s["split"] == "train" for s in states),
                      "evaluation_rows": sum(s["split"] == "evaluation" for s in states),
                      "training_decision_batches": list(range(2, 7)),
                      "training_label_lookahead_max_batch": 7,
                      "evaluation_decision_batches": [7, 8],
                      "evaluated_candidates": counter,
                      "independence": "not_independent_of_prior_public_development",
                      "optimality": "best_of_finite_candidates_for_observed_source_state_only"}
        write_json(destination / "provenance.json", provenance)
        recorder.log(counter + 1, {"corpus/training_rows": provenance["training_rows"],
                                   "corpus/evaluation_rows": provenance["evaluation_rows"],
                                   "oracle/candidates_evaluated": counter})
        recorder.finish()
        result = {**registration, "status": "completed", "corpus": provenance,
                  "tracking": recorder.reference()}
        write_json(folder / "run.json", result)
        return result
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "message": str(exc)})
        try:
            recorder.finish(failed=True)
        except Exception:
            pass
        raise
