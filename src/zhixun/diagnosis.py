"""Read-only statistical diagnosis. Associations do not establish a causal bottleneck."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zhixun.data import ROOT, digest, load_dataset, verify_inputs, write_json
from zhixun.metrics import score
from zhixun.vendor.model import Model


def feature_saturation(x, mean, scale, *, feature_names=None) -> dict:
    if (x.ndim != 2 or not len(x) or mean.shape != (x.shape[1],)
            or scale.shape != mean.shape or not np.isfinite(x).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid saturation diagnostic matrix or scaling")
    z = (x - mean) / scale
    clipped = np.abs(z) > 10
    per_feature = clipped.mean(axis=0)
    order = np.argsort(-per_feature, kind="stable")[:10]
    names = feature_names or [str(i) for i in range(x.shape[1])]
    return {"clipped_cell_fraction": float(clipped.mean()),
            "rows_with_any_clip_fraction": float(clipped.any(axis=1).mean()),
            "features_over_20_percent": int((per_feature > 0.2).sum()),
            "boundary_cell_fraction": float((np.abs(z) == 10).mean()),
            "top_saturated": [{"feature": names[i], "fraction": float(per_feature[i])}
                              for i in order]}


def confusion_errors(matrix, classes) -> list[dict]:
    matrix = np.asarray(matrix)
    return [{"actual": classes[i], "predicted": classes[j], "count": int(matrix[i, j]),
             "fraction_of_actual_class": float(matrix[i, j] / matrix[i].sum())}
            for i, j in sorted(((i, j) for i in range(len(classes)) for j in range(len(classes))
                                if i != j and matrix[i, j]), key=lambda pair: -matrix[pair])]


def diagnose_run(run: Path, *, root: Path = ROOT) -> dict:
    verify_inputs("A", root)
    report = json.loads((run / "run.json").read_text(encoding="utf-8"))
    state = json.loads((run / "stream-state.json").read_text(encoding="utf-8"))
    versions = {row["version"]: row for row in state["versions"]}
    original = Model.load(root / "data/A/checkpoint.npz")
    train = load_dataset(root / "data/A/train.jsonl", original, "A")
    classes = original.meta["classes"]
    acetaldehyde = classes.index("Acetaldehyde")
    batches = []
    for event in report["events"]:
        if event["status"] != "completed":
            continue
        batch = event["batch_index"] + 2
        path = root / ("data/A/validation.jsonl" if batch == 2
                        else f"data/replay-gas/batch{batch}.jsonl")
        row = versions[event["parent_version"]]
        checkpoint = run / row["path"]
        if digest(checkpoint) != row["sha256"]:
            raise ValueError("Diagnostic source checkpoint changed")
        model = Model.load(checkpoint)
        data = load_dataset(path, model, "A")
        metrics, details, probabilities = score(model, data)
        saturation = feature_saturation(data.x, model.mean, model.scale,
                                        feature_names=model.meta["paths"])
        hidden, _ = model.forward(model.normalized(data.x))
        predictions = probabilities.argmax(axis=1)
        wrong = (data.y == acetaldehyde) & (predictions != acetaldehyde)
        attribution = []
        for target in range(len(classes)):
            mistaken = wrong & (predictions == target)
            real_target = data.y == target
            if not mistaken.any() or not real_target.any():
                continue
            # Diagnostics only: these statistics are never fitted into the predictor.
            mistaken_center = model.normalized(data.x[mistaken]).mean(axis=0)
            target_center = model.normalized(data.x[real_target]).mean(axis=0)
            initial_acetal = model.normalized(train.x[train.y == acetaldehyde]).mean(axis=0)
            attribution.append({
                "mistaken_as": classes[target], "samples": int(mistaken.sum()),
                "normalized_centroid_distance_to_same_batch_target": float(np.linalg.norm(
                    mistaken_center - target_center)),
                "normalized_centroid_distance_to_initial_acetaldehyde": float(np.linalg.norm(
                    mistaken_center - initial_acetal)),
                "mistaken_clipping": feature_saturation(data.x[mistaken], model.mean, model.scale),
                "interpretation": "descriptive_overlap_and_shift_not_causal_proof",
            })
        batches.append({"batch": batch, "rows": len(data.ids), "metrics": metrics,
                        "saturation": saturation,
                        "hidden_abs_over_099_fraction": float((np.abs(hidden) >= 0.99).mean()),
                        "confusion_matrix": details["confusion_matrix"],
                        "errors": confusion_errors(details["confusion_matrix"], classes),
                        "acetaldehyde": details["Acetaldehyde"],
                        "acetaldehyde_comparisons": attribution})
    result = {"format": "zhixun.readonly-diagnosis.v1",
              "source_experiment": report["experiment_id"],
              "source_report_sha256": digest(run / "run.json"), "classes": classes,
              "batch1_reference": feature_saturation(train.x, original.mean, original.scale),
              "batches": batches, "training_executed": False,
              "conclusion_scope": "statistical diagnosis; no single causal root established"}
    output = root / "artifacts/diagnosis-v3"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnosis.json", result)
    return result
