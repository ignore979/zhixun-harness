"""Delayed-feedback selection among domain views. Never consumes current labels in predict."""
from __future__ import annotations

import numpy as np

from zhixun.ensemble import combine_predictions
from zhixun.metrics import classification_metrics


class ViewSelector:
    def __init__(self, classes, *, baseline, experts, mode="leader", window=2, eta=1.0):
        if (mode not in {"leader", "hedge"} or type(window) is not int or window < 1
                or not 0 < eta <= 2 or not 0 <= baseline < experts):
            raise ValueError("Invalid delayed-view selector settings")
        self.classes, self.baseline, self.experts = classes, baseline, experts
        self.mode, self.window, self.eta = mode, window, eta
        self.history, self.pending, self.last_batch = [], None, None

    def weights(self):
        weights = np.zeros(self.experts)
        if not self.history:
            weights[self.baseline] = 1.
        elif self.mode == "leader":
            quality = np.mean([r['quality'] for r in self.history[-self.window:]], axis=0)
            # Stable baseline wins exact ties; no audit-dependent threshold or batch rule.
            order = [self.baseline, *[i for i in range(self.experts) if i != self.baseline]]
            chosen = max(order, key=lambda i: quality[i])
            weights[chosen] = 1.
        else:
            losses = np.mean([r['balanced_log_loss'] for r in self.history[-self.window:]], axis=0)
            prior = np.full(self.experts, .5 / max(1, self.experts - 1))
            prior[self.baseline] = .5
            logits = np.log(prior) - self.eta * losses
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
        return weights

    def predict(self, predictions, *, batch):
        values = np.asarray(predictions)
        if (self.pending is not None or values.shape[0] != self.experts
                or (self.last_batch is not None and batch != self.last_batch + 1)):
            raise ValueError("Selector batches must predict then release in order")
        weights = self.weights()
        out = combine_predictions(values, weights, 'classification')
        self.pending = {'batch': batch, 'predictions': values.copy()}
        self.last_weights = weights.tolist()
        return out

    def release(self, labels, *, batch):
        if self.pending is None or self.pending['batch'] != batch:
            raise ValueError("Selector labels must follow the matching prediction")
        quality, losses = [], []
        y = np.asarray(labels)
        for p in self.pending['predictions']:
            metrics, _ = classification_metrics(p, y, self.classes)
            quality.append(.5 * (metrics['accuracy'] + metrics['macro_f1']))
            active = np.unique(y)
            losses.append(float(np.mean([-np.log(np.maximum(p[y == c, c], 1e-12)).mean()
                                         for c in active])))
        self.history.append({'batch': batch, 'quality': quality, 'balanced_log_loss': losses})
        self.history = self.history[-self.window:]
        self.last_batch, self.pending = batch, None
