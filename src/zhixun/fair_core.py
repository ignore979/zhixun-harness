"""Shared trainer and observation contract for oracle generation and live fair replay."""
from __future__ import annotations

import copy
import json
import math
from time import monotonic, perf_counter

import numpy as np

from zhixun.data import ROOT, digest, load_dataset, verify_inputs, verify_parent
from zhixun.diagnosis import feature_saturation
from zhixun.drift import detect_drift
from zhixun.llm_policy import parse_decision
from zhixun.loop import EvaluationBatch, verify_sequence
from zhixun.metrics import promotion, score
from zhixun.model_extension import ClassWeightedModel
from zhixun.prepare_gas import verify_replay
from zhixun.training_schedule import lr_at, schedule_epochs
from zhixun.vendor.model import Model


def load_fair_config(root=ROOT, path=None):
    config = json.loads((path or root / "configs/replay_fair.json").read_text(encoding="utf-8"))
    if (config["protocol"] not in {"uci-gas-fair-agent-v3", "uci-gas-fair-agent-v4"}
            or config["total_epoch_budget"] != 45
            or sum(config["flat_allocation"]) != 45 or len(config["flat_allocation"]) != 9
            or config["flat_allocation"][-1] != 0 or config["terminal_batch_training"] is not False
            or config["oracle_training_decision_batches"] != list(range(2, 7))
            or config["oracle_evaluation_decision_batches"] != [7, 8]
            or config["oracle_training_next_batch_max"] != 7):
        raise ValueError("Unsupported fair protocol or temporal split")
    for action in config["action_catalog"]:
        parse_decision(json.dumps(action))
    if config["action_catalog"][0]["action"] != "hold":
        raise ValueError("The catalog must start with hold")
    return config


def load_fair_data(root=ROOT):
    verify_inputs("A", root)
    verify_replay(root)
    initial = root / "data/A/checkpoint.npz"
    model = Model.load(initial)
    train = load_dataset(root / "data/A/train.jsonl", model, "A")
    verify_parent(model, "A", train)
    paths = [root / ("data/A/validation.jsonl" if i == 2
                      else f"data/replay-gas/batch{i}.jsonl") for i in range(2, 11)]
    batches = [EvaluationBatch(i, f"gas-b{i + 2:02d}", load_dataset(path, model, "A"), digest(path))
               for i, path in enumerate(paths)]
    verify_sequence(batches, train)
    return initial, model, train, batches


def observe(model, data, pool, batch, remaining, reference, history, ref, config):
    metrics, details, _ = score(model, data)
    signal = detect_drift(model, ref, data.x)
    saturation = feature_saturation(data.x, model.mean, model.scale)
    observation = {
        "batch": batch, "labels_available": True,
        "label_timing": "prediction_saved_before_labels_are_used_for_current_decision",
        "remaining_epochs": remaining, "remaining_decisions": 10 - batch,
        "pool_rows": sum(len(chunk.ids) for chunk in pool.chunks),
        "current_metrics": metrics, "reference_macro_f1": reference,
        "performance_drop": reference - metrics["macro_f1"],
        "mean_psi": signal.mean_psi, "confidence_drop": signal.confidence_drop,
        "clipped_cell_fraction": saturation["clipped_cell_fraction"],
        "per_class_recall": {c: details[c]["recall"] for c in model.meta["classes"]},
        "history": history[-3:], "terminal": batch == 10,
    }
    return observation


def build_prompt(observation, config):
    # Target/future outcome fields are never accepted into this observation schema.
    forbidden = {"next_metrics", "candidate_results", "chosen", "completion", "future_labels"}
    if forbidden.intersection(observation):
        raise ValueError("Future or answer fields cannot enter the agent observation")
    allowed = {"batch", "labels_available", "label_timing", "remaining_epochs",
               "remaining_decisions", "pool_rows", "current_metrics", "reference_macro_f1",
               "performance_drop", "mean_psi", "confidence_drop", "clipped_cell_fraction",
               "per_class_recall", "history", "terminal"}
    if set(observation) - allowed:
        raise ValueError("Unknown fields cannot enter the agent observation")
    reward = ("Offline teacher first requires next-batch accuracy >=99% of the SAME-STATE "
              "flat-action reference; only feasible actions compete on accuracy minus 0.001 "
              "per epoch. Excessive hold that violates this floor is rejected. This is an "
              "offline teaching constraint, not a guarantee of future accuracy. "
              if config.get("reward_floor_ratio") == 0.99 else
              "Offline teacher utility is next-batch accuracy minus 0.001 per epoch.")
    return ("You allocate industrial-model continuation under a 45-epoch cap. "
            "Return ONLY the JSON of one affordable action from action_catalog, no prose. "
            "Never train at terminal=true. Budget is an upper limit, not a spending target. "
            "Use current diagnostics and past outcomes to preserve future accuracy while saving "
            "training. " + reward + "\n"
            + json.dumps({"action_catalog": config["action_catalog"], "observation": observation},
                         sort_keys=True, separators=(",", ":"), allow_nan=False))


def validate_action(action, observation, config):
    parsed = parse_decision(json.dumps(action))
    if parsed not in config["action_catalog"]:
        raise ValueError("Proposal is not in the registered finite action catalog")
    if parsed["epochs"] > observation["remaining_epochs"]:
        raise ValueError("Proposal exceeds remaining epoch budget")
    if observation["terminal"] and parsed["action"] != "hold":
        raise ValueError("All policies must hold at the terminal batch")
    return parsed


def rule_action(observation, config, *, flat=False):
    if observation["terminal"]:
        return copy.deepcopy(config["action_catalog"][0])
    if flat:
        wanted = config["flat_allocation"][observation["batch"] - 2]
        affordable = [a for a in config["action_catalog"] if a["epochs"] <= min(
            wanted, observation["remaining_epochs"])]
        return copy.deepcopy(max(affordable, key=lambda a: a["epochs"]))
    desired = schedule_epochs(observation["mean_psi"], observation["performance_drop"],
                              observation["pool_rows"])
    # Preserve at least two epochs for each later actionable decision.
    allowance = max(0, observation["remaining_epochs"] - 2 * (
        observation["remaining_decisions"] - 1))
    affordable = [a for a in config["action_catalog"] if 0 < a["epochs"] <= allowance]
    if not affordable:
        return copy.deepcopy(config["action_catalog"][0])
    return copy.deepcopy(min(affordable, key=lambda a: (abs(a["epochs"] - desired), a["epochs"])))


def train_action(parent, pool, current, action, config, *, on_epoch=None, on_candidate=None):
    """No next-batch argument exists on this runtime trainer."""
    pool.verify_future(current)
    before, _, _ = score(parent, current)
    if action["action"] == "hold":
        return copy.deepcopy(parent), {"accepted": False, "before": before, "after": before,
                                       "training_seconds": 0.0, "optimizer_steps": 0}
    candidate = copy.deepcopy(parent)
    candidate.__class__ = ClassWeightedModel
    x, y = pool.as_train()
    steps, epochs = parent.meta["steps"], parent.meta["epochs"]
    started, seconds = monotonic(), 0.0
    for epoch in range(action["epochs"]):
        if monotonic() - started >= config["max_seconds_per_round"]:
            raise TimeoutError("Fair trainer cooperative deadline reached")
        candidate.meta["learning_rate"] = lr_at(epoch, action["epochs"], action["learning_rate"],
                                                decay=config["lr_decay"])
        began = perf_counter()
        loss = candidate.fit(x, y, 1, cw_power=config["cw_power"])[0]
        seconds += perf_counter() - began
        if on_epoch:
            on_epoch(epoch, float(loss), candidate.meta["learning_rate"], seconds)
    if monotonic() - started >= config["max_seconds_per_round"]:
        raise TimeoutError("Fair trainer exceeded its cooperative deadline")
    if (not np.array_equal(candidate.mean, parent.mean)
            or not np.array_equal(candidate.scale, parent.scale)
            or candidate.meta["epochs"] != epochs + action["epochs"]
            or candidate.meta["steps"] != steps + action["epochs"] * math.ceil(
                len(y) / candidate.meta["batch_size"])):
        raise ValueError("Fair continuation invariants failed")
    after, _, _ = score(candidate, current)
    accepted, gate = promotion(before, after, "A")
    if on_candidate:
        on_candidate(candidate)
    selected = candidate if accepted else copy.deepcopy(parent)
    return selected, {"accepted": accepted, "before": before, "after": after,
                      "training_seconds": seconds,
                      "optimizer_steps": candidate.meta["steps"] - steps,
                      "gate": gate}
