"""Evaluation metrics. Implementations are self-contained (no sklearn import
inside metric functions) to keep the offline path dependency-free."""

from __future__ import annotations

import numpy as np

from ..core.errors import EvalError


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise EvalError("accuracy requires non-empty equal-shape inputs")
    return float(np.mean(y_true == y_pred))


def worst_client_accuracy(per_client_accs: list) -> float:
    """Fairness metric: the minimum accuracy across clients."""
    if not per_client_accs:
        raise EvalError("per_client_accs must be non-empty")
    return float(min(per_client_accs))


def communication_mb(total_bytes: int) -> float:
    return round(total_bytes / (1024.0 * 1024.0), 4)


def rounds_to_target(history: list, target: float) -> int:
    """First round whose global_acc >= target; -1 if never reached."""
    for row in history:
        if row.global_acc >= target:
            return int(row.round)
    return -1
