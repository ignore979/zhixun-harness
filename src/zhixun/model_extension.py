"""Extensions derived from the supplied vendor Model/Adam; vendor files remain untouched."""
from __future__ import annotations

import copy
import math

import numpy as np

from zhixun.vendor.model import Model


def class_weights(y: np.ndarray, classes: int, power: float) -> np.ndarray:
    if (not math.isfinite(power) or not 0 <= power <= 2 or y.ndim != 1 or not len(y)
            or y.dtype.kind not in "iu" or y.min() < 0 or y.max() >= classes):
        raise ValueError("Invalid class-weight inputs")
    counts = np.bincount(y, minlength=classes).astype(float)
    weights = (len(y) / np.maximum(counts, 1.0)) ** power
    return weights * len(y) / float(weights @ counts)


class ClassWeightedModel(Model):
    def fit(self, x, y, epochs, *, cw_power=1.5):
        if self.meta["task"] != "classification":
            return super().fit(x, y, epochs)
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),):
            raise ValueError("Target shape mismatch")
        weights = class_weights(y, len(self.meta["classes"]), cw_power)
        epoch_losses = []
        for _ in range(epochs):
            rng = np.random.default_rng(np.random.SeedSequence([
                self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], y[indices]
                hidden, logits = self.forward(batch)
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                loss_sum -= float((weights[expected] * np.log(np.maximum(
                    probabilities[np.arange(len(indices)), expected], 1e-300))).sum())
                delta = probabilities
                delta[np.arange(len(indices)), expected] -= 1
                delta = delta * weights[expected][:, None]
                delta /= len(indices)
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden**2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("Nonfinite weighted gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient**2
                    first = self.first[name] / (1 - .9**self.meta["steps"])
                    second = self.second[name] / (1 - .999**self.meta["steps"])
                    self.params[name] -= (
                        self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                    )
                if not all(np.isfinite(p).all() for p in self.params.values()):
                    raise ValueError("Nonfinite weighted model state")
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses


def expand_hidden(model: Model, new_hidden: int, *, seed: int = 42,
                  noise: float = 0.001) -> Model:
    old = model.meta["hidden"]
    if type(new_hidden) is not int or not old < new_hidden <= 128:
        raise ValueError("Expanded width must exceed parent width and be at most 128")
    if not math.isfinite(noise) or not 0 <= noise <= 0.01:
        raise ValueError("Invalid widening perturbation")
    result = copy.deepcopy(model)  # Never invoke Model.__init__ or mutate the parent.
    rng = np.random.default_rng(seed)
    extra = np.resize(rng.permutation(old), new_hidden - old)
    mapping = np.concatenate([np.arange(old), extra])
    copies = np.bincount(mapping, minlength=old)
    result.params["w1"] = model.params["w1"][:, mapping].copy()
    result.params["b1"] = model.params["b1"][mapping].copy()
    result.params["w2"] = model.params["w2"][mapping] / copies[mapping, None]
    # Zero-sum outgoing perturbations break clone symmetry while preserving predictions.
    for source in range(old):
        indices = np.flatnonzero(mapping == source)
        perturbation = rng.normal(0, noise, (len(indices), model.params["w2"].shape[1]))
        perturbation -= perturbation.mean(axis=0)
        result.params["w2"][indices] += perturbation
    for name in ("first", "second"):
        previous = getattr(model, name)
        moments = {key: np.zeros_like(value) for key, value in result.params.items()}
        moments["w1"][:, :old] = previous["w1"]
        moments["b1"][:old] = previous["b1"]
        moments["w2"][:old] = previous["w2"]
        moments["b2"] = previous["b2"].copy()
        setattr(result, name, moments)
    result.meta["hidden"] = new_hidden
    result.meta["parameter_count"] = sum(p.size for p in result.params.values())
    result.meta["width_migration"] = {
        "from_hidden": old, "to_hidden": new_hidden, "source_unit_mapping": mapping.tolist(),
        "noise": noise, "seed": seed,
        "prediction_preservation": "clone_inputs_and_zero_sum_outgoing_perturbations",
        "adam_transfer": "old_prefix_preserved_new_moments_zero; not_trajectory_equivalence",
    }
    result.validate()
    return result
