"""User-authorized policy imitation data from ACTUAL replay traces, not official action labels."""
from __future__ import annotations

import json
from pathlib import Path

from zhixun.data import ROOT, digest, write_json
from zhixun.llm_policy import parse_decision
from zhixun.prepare_gas import SOURCE_URL
from zhixun.replay import STRATEGIES


def prepare_harness(runs: list[Path], *, root: Path = ROOT) -> dict:
    reports = [json.loads((p / "run.json").read_text(encoding="utf-8")) for p in runs]
    if len(reports) != 3 or {r["strategy"] for r in reports} != STRATEGIES:
        raise ValueError("Provide the actual completed static/always/performance replay runs")
    reference = reports[0]
    for report in reports:
        if any(e.get("plan", {}).get("epochs", 0) > 5 for e in report["events"]):
            raise ValueError("Upgraded actions need a versioned 20-epoch corpus; keep legacy data")
        if (report["status"] != "completed" or report["completed_batches"] != 9
                or report["config"] != reference["config"]
                or report["dataset"] != reference["dataset"]
                or report["batches"] != reference["batches"]):
            raise ValueError("Replay sources must have identical data and protocol")
    folder = root / "data/harness-sft"
    folder.mkdir(parents=True, exist_ok=True)
    rows, lineage = {"train": [], "evaluation": []}, []
    for report in reports:
        history = []
        adopted = {0}
        for event in report["events"]:
            if event["status"] != "completed":
                continue
            allowed = sorted(adopted - {event["parent_version"]})
            context = {
                "objective": ("Imitate the configured recorded controller, "
                              "not hindsight optimality."),
                "strategy": report["strategy"], "batch_id": event["batch_id"],
                "performance_drop_threshold": report["config"]["performance_drop_threshold"],
                "planned_epochs": report["config"]["epochs"],
                "planned_learning_rate": report["config"]["learning_rate"],
                "observed_metrics_before_action": event["prequential"],
                "reference_macro_f1": event["reference_macro_f1"],
                "performance_drop": event["performance_drop"],
                "mean_psi": event["signal"]["mean_psi"],
                "confidence_drop": event["signal"]["confidence_drop"],
                "budget_remaining": event["budget_before"],
                "training_pool_rows": event["pool_before"]["rows"],
                "allowed_rollback_versions": allowed, "past_decisions": history[-3:],
            }
            prompt = (
                "You control an industrial training harness. Return only one JSON object with "
                "action (hold/retrain/rollback), epochs (integer 0..5), learning_rate "
                "(0..0.001), target_version (null except an allowed rollback target). "
                "For hold/rollback use epochs=0 and learning_rate=0. "
                "The controller must obey the configured strategy and available budget.\n"
                + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            decision = {k: event["plan"][k] for k in ["action", "epochs", "learning_rate"]}
            decision["target_version"] = (event["selected_version"]
                                          if decision["action"] == "rollback" else None)
            completion = json.dumps(decision, sort_keys=True, separators=(",", ":"))
            parse_decision(completion, rollback_versions=allowed)
            split = "train" if event["batch_index"] < 6 else "evaluation"
            rows[split].append({"prompt": prompt, "completion": completion})
            lineage.append({"split": split, "row": len(rows[split]),
                            "source_run": report["experiment_id"], "batch": event["batch_id"],
                            "label_origin": "executed_program_decision_not_official_label"})
            history.append({"action": event["plan"]["action"],
                            "learning_rate": event["plan"]["learning_rate"],
                            "accepted": event["accepted"]})
            if event["accepted"]:
                adopted.add(event["selected_version"])
    for split, values in rows.items():
        path = folder / f"{split}.jsonl"
        content = "".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n"
                          for v in values)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite a different SFT corpus; version it separately")
        path.write_text(content, encoding="utf-8")
    provenance = {
        "format": "zhixun.derived-harness-imitation.v1", "sha256": digest(folder / "train.jsonl"),
        "source_url": SOURCE_URL, "authorized_for_llm_training": True,
        "reviewed_by": "user_authorized_trace_derivation_and_programmatic_integrity_checks",
        "user_authorization": "2026-11-15_explicit_Harness_replay_demonstration_selection",
        "official_llm_corpus": False,
        "label_origin": "actual_recorded_controller_actions; no claim of optimality",
        "training_batches": list(range(2, 8)), "evaluation_batches": [8, 9, 10],
        "evaluation_sha256": digest(folder / "evaluation.jsonl"),
        "training_rows": len(rows["train"]), "evaluation_rows": len(rows["evaluation"]),
        "sources": [{"experiment_id": r["experiment_id"], "run_sha256": digest(p / "run.json")}
                    for p, r in zip(runs, reports, strict=True)],
        "row_lineage": lineage,
        "prompt_excludes": ["current_candidate_metrics", "current_selection", "future_batches"],
        "evaluation_limit": "temporal_holdout_same_UCI_stream_not_independent_task_generalization",
    }
    write_json(folder / "provenance.json", provenance)
    return provenance
