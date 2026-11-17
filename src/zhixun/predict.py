from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import numpy as np

from zhixun.vendor.data import flatten, json_value
from zhixun.vendor.model import Model


def predict_stream(checkpoint: Path, source: TextIO, destination: TextIO) -> None:
    model = Model.load(checkpoint)
    for line in source:
        if not line.strip():
            continue
        request = json_value(line)
        if (
            not isinstance(request, dict) or "inputs" not in request
            or set(request) - {"inputs", "task"}
        ):
            raise ValueError("Prediction accepts only inputs and optional task; never labels")
        if request.get("task", model.meta["task"]) != model.meta["task"]:
            raise ValueError("Prediction task differs from checkpoint")
        paths, values = flatten(request["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError("Prediction feature schema differs from checkpoint")
        result = model.predict(np.asarray([values], dtype=np.float64))[0]
        outputs = (
            [{"label": label, "score": float(value)}
             for label, value in zip(model.meta["classes"], result, strict=True)]
            if model.meta["task"] == "classification" else {"value": float(result)}
        )
        destination.write(json.dumps({"outputs": outputs}, allow_nan=False) + "\n")
        destination.flush()


def predict_batch(checkpoint: Path, adapter: Path, source: TextIO, destination: TextIO) -> None:
    """Experimental whole-batch interface; strictly label-free JSONL requests."""
    from zhixun.domain_views import load_domain_adapter

    model = Model.load(checkpoint)
    correction = load_domain_adapter(adapter)
    features = []
    for line in source:
        if not line.strip():
            continue
        request = json_value(line)
        if (not isinstance(request, dict) or "inputs" not in request
                or set(request) - {"inputs", "task"}
                or request.get("task", "classification") != "classification"):
            raise ValueError("Batch prediction accepts only inputs and optional "
                             "classification task")
        paths, values = flatten(request["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError("Batch prediction feature schema differs from checkpoint")
        features.append(values)
        if len(features) > 100000:
            raise ValueError("Batch prediction is limited to 100000 requests")
    if len(features) < 2:
        raise ValueError("CORAL needs at least two unlabeled feature rows in one batch")
    predictions = correction.predict(model, np.asarray(features, dtype=np.float64))
    for values in predictions:
        outputs = [{"label": c, "score": float(p)} for c, p in zip(
            model.meta["classes"], values, strict=True)]
        destination.write(json.dumps({"outputs": outputs}, allow_nan=False) + "\n")
    destination.flush()
