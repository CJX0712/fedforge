"""Pure-numpy multinomial logistic regression (softmax) — the offline model.

Full weight control (W, b) enables exact federated aggregation. Mini-batch
SGD with deterministic shuffling under seed. L2 regularization (biases exempt).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import StrategyError


def init_weights(num_features: int, num_classes: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.01, size=(num_features, num_classes))
    b = np.zeros(num_classes)
    return [W, b]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(weights: list, X) -> np.ndarray:
    W, b = weights
    return softmax(X @ W + b)


def predict(weights: list, X) -> np.ndarray:
    return np.argmax(predict_proba(weights, X), axis=1)


def cross_entropy(weights: list, X, y, l2: float = 0.0) -> float:
    P = predict_proba(weights, X)
    n = X.shape[0]
    eps = 1e-12
    nll = -np.log(P[np.arange(n), y] + eps).mean()
    W = weights[0]
    return float(nll + 0.5 * l2 * float(np.sum(W * W)) / n)


def _grad_batch(weights: list, Xb, yb, l2: float) -> list:
    n, d = Xb.shape
    k = weights[0].shape[1]
    P = predict_proba(weights, Xb)
    Y = np.zeros((n, k))
    Y[np.arange(n), yb] = 1.0
    G = (P - Y) / n
    gW = Xb.T @ G + l2 * weights[0] / n
    gb = G.sum(axis=0)
    return [gW, gb]


def train_local(X, y, weights: list, lr: float, epochs: int, batch_size: int,
                seed: int, l2: float = 0.0) -> list:
    """Run `epochs` of mini-batch SGD on local data starting from `weights`."""
    W, b = weights[0].copy(), weights[1].copy()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    bs = min(batch_size, n)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            gW, gb = _grad_batch([W, b], X[idx], y[idx], l2)
            W = W - lr * gW
            b = b - lr * gb
    return [W, b]


def weights_bytes(weights: list) -> int:
    """Communication size of a weight vector in bytes (float64 payload)."""
    return int(sum(arr.nbytes for arr in weights))


def validate_weights(weights: list, num_features: int, num_classes: int) -> None:
    if len(weights) != 2:
        raise StrategyError("weights must be [W, b]")
    W, b = weights
    if W.shape != (num_features, num_classes) or b.shape != (num_classes,):
        raise StrategyError(
            f"expected W {(num_features, num_classes)} and b {(num_classes,)}, "
            f"got {W.shape} / {b.shape}")
