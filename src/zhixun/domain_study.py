"""Fixed follow-up ablations after the negative historical-version ensemble experiment."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import numpy as np

from zhixun.adaptation_study import summarize
from zhixun.data import ROOT, digest, write_json
from zhixun.domain_views import view_probabilities
from zhixun.ensemble import temporal_versions
from zhixun.ensemble_study import safe_candidate, select_development
from zhixun.fair_core import load_fair_data
from zhixun.metrics import classification_metrics
from zhixun.registry import ModelRegistry
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def run_domain_study(*, root=ROOT, mode="offline", project="zhixun-school"):
    config = json.loads((root / "configs/domain_followup.json").read_text(encoding="utf-8"))
    _, base, train, batches = load_fair_data(root)
    study = root / "artifacts" / f"study-domain-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    study.mkdir()
    write_json(study / "registration.json", {"config": config,
                                             "created_at": datetime.now(UTC).isoformat()})
    results = []
    for source in config["source_runs"]:
        src = root / "artifacts" / source
        report = json.loads((src / "run.json").read_text(encoding="utf-8"))
        state = json.loads((src / "stream-state.json").read_text(encoding="utf-8"))
        if report["status"] != "completed":
            raise ValueError("Source not completed")
        registry = ModelRegistry(src, state["signature"])
        source_events = [e for e in report["events"] if e["status"] == "completed"]
        folder = root / "artifacts" / (f"run-domain-{datetime.now(UTC):%Y%m%dT%H%M%S}-"
                                       f"{uuid4().hex[:8]}")
        folder.mkdir()
        registration = {"experiment_id": folder.name, "created_at": datetime.now(UTC).isoformat(),
                        "task": "separated_domain_view_followup", "source_run": source,
                        "source_report_sha256": digest(src / "run.json"), "config": config,
                        "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                                           for p in sorted((root / "src").rglob("*.py"))},
                        "additional_training_epochs": 0, "current_labels_used": False}
        write_json(folder / "registration.json", registration)
        recorder, cases = None, []
        try:
            recorder = Recorder(folder, registration, mode=mode, project=project, workspace=None)
            for case_index, spec in enumerate(config["methods"]):
                case_folder = folder / f"case-{case_index:03d}"
                case_folder.mkdir()
                events = []
                for index, batch in enumerate(batches):
                    temporal_versions(source_events, index)
                    parent = Model.load(registry.checkpoint(source_events[index]["parent_version"]))
                    if (not np.array_equal(parent.mean, base.mean)
                            or not np.array_equal(parent.scale, base.scale)):
                        raise ValueError("Frozen normalization differs")
                    began = perf_counter()
                    prediction = view_probabilities(parent, train, batch.data.x, spec,
                        past=[(j + 2, previous.data) for j, previous in enumerate(batches[:index])],
                        current_batch=index + 2)
                    seconds = perf_counter() - began
                    path = case_folder / f"batch-{index + 2:02d}.npz"
                    np.savez_compressed(path, probabilities=prediction,
                                        ids=np.array(batch.data.ids))
                    metrics, details = classification_metrics(prediction, batch.data.y,
                                                               base.meta["classes"])
                    row = {"batch": index + 2, "rows": len(batch.data.ids), "metrics": metrics,
                           "details": details, "prediction_sha256": digest(path),
                           "prediction_seconds": seconds,
                           "parent_version": source_events[index]["parent_version"],
                           "allowed_past_label_batches": list(range(2, index + 2))}
                    events.append(row)
                    recorder.log(case_index * 100 + index, {
                        **{f"prequential/{k}": v for k, v in metrics.items()},
                        **{f"recall/{c}": details[c]["recall"] for c in base.meta["classes"]},
                        "batch/number": index + 2, "case/number": case_index,
                        "prediction/seconds": seconds})
                summaries = {"full": summarize(events, base.meta["classes"], list(range(2, 11))),
                             "development": summarize(events, base.meta["classes"],
                                                      config["development_batches"]),
                             "audit": summarize(events, base.meta["classes"],
                                                config["audit_batches"])}
                row = {"case": case_index, "spec": spec, "summary": summaries, "events": events,
                       "experiment_id": folder.name,
                       "case_directory": str(case_folder.relative_to(root))}
                cases.append(row)
                write_json(case_folder / "result.json", row)
                recorder.log(case_index * 100 + 90, {f"summary/{scope}/{k}": v
                    for scope, summary in summaries.items() for k, v in summary.items()
                    if isinstance(v, (int, float))})
            baseline = next(r for r in cases if r["spec"]["name"] == "coral025")
            selected = select_development(cases, baseline)
            accepted = safe_candidate(selected["summary"]["audit"], baseline["summary"]["audit"])
            recorder.log(10000, {"selection/accepted_audit_gate": float(accepted)})
            for index, (candidate, control) in enumerate(zip(selected["events"], baseline["events"],
                                                            strict=True)):
                recorder.log(11000 + index, {
                    **{f"selected/{k}": v for k, v in candidate["metrics"].items()},
                    **{f"reference/{k}": v for k, v in control["metrics"].items()},
                    "selected/acetaldehyde_recall": candidate["details"]["Acetaldehyde"]["recall"],
                    "reference/acetaldehyde_recall": control["details"]["Acetaldehyde"]["recall"]})
            recorder.finish()
            result = {**registration, "status": "completed", "cases": cases, "selected": selected,
                      "baseline": baseline, "audit_accepted": accepted,
                      "tracking": recorder.reference()}
            write_json(folder / "run.json", result)
            results.append({"source": source, "experiment_id": folder.name, "selected": selected,
                            "baseline": baseline, "audit_accepted": accepted})
        except BaseException as exc:
            write_json(folder / "failure.json", {"status": "failed", "message": str(exc)})
            if recorder:
                recorder.finish(failed=True)
            raise
    final = {"status": "completed", "config": config, "sources": results,
             "study_directory": str(study.relative_to(root))}
    write_json(study / "study.json", final)
    return final
