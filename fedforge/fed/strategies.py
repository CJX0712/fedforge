"""Aggregation strategies: FedAvg / FedMedian / FedTrimmedMean.

All operate element-wise over the [W, b] weight list and are pure functions
of (client_weights, client_sample_counts).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import StrategyError
from ..core.interfaces import AggregationStrategy


def _check_pair(weights: list, n_samples: list) -> None:
    if len(weights) == 0:
        raise StrategyError("no client updates to aggregate")
    if len(weights) != len(n_samples):
        raise StrategyError("weights/n_samples length mismatch")
    if any(int(n) <= 0 for n in n_samples):
        raise StrategyError("all clients must contribute n_samples > 0")
    shapes = [(w[0].shape, w[1].shape) for w in weights]
    if any(s != shapes[0] for s in shapes):
        raise StrategyError("client weight shapes disagree")


class FedAvg:
    """Weighted average by sample count (McMahan et al., 2017)."""

    name = "fedavg"

    def aggregate(self, weights: list, n_samples: list) -> list:
        _check_pair(weights, n_samples)
        total = float(sum(n_samples))
        W = np.zeros_like(weights[0][0])
        b = np.zeros_like(weights[0][1])
        for w, n in zip(weights, n_samples):
            W += (n / total) * w[0]
            b += (n / total) * w[1]
        return [W, b]


class FedMedian:
    """Coordinate-wise median — Byzantine-robust (Yin et al., 2018)."""

    name = "fedmedian"

    def aggregate(self, weights: list, n_samples: list) -> list:
        _check_pair(weights, n_samples)
        W = np.median(np.stack([w[0] for w in weights]), axis=0)
        b = np.median(np.stack([w[1] for w in weights]), axis=0)
        return [W, b]


class FedTrimmedMean:
    """Coordinate-wise trimmed mean — robust (Blanchard et al., 2017;
    Yin et al., 2018). Trims `trim_ratio` extremes per coordinate, then
    averages the remainder (sample-weighted)."""

    name = "trimmedmean"

    def __init__(self, trim_ratio: float = 0.1) -> None:
        if not 0.0 < trim_ratio < 0.5:
            raise StrategyError("trim_ratio must be in (0, 0.5)")
        self.trim_ratio = trim_ratio

    def aggregate(self, weights: list, n_samples: list) -> list:
        _check_pair(weights, n_samples)
        m = len(weights)
        k = max(1, int(np.floor(self.trim_ratio * m)))
        if m - 2 * k < 1:
            raise StrategyError(
                f"too few clients ({m}) for trim_ratio={self.trim_ratio}")
        Ws = np.stack([w[0] for w in weights])
        Bs = np.stack([w[1] for w in weights])
        order_w = np.argsort(Ws, axis=0)
        order_b = np.argsort(Bs, axis=0)
        keep = np.arange(k, m - k)
        # Canonical coordinate-wise trimmed mean (Yin et al., 2018):
        # unweighted mean over the kept middle coordinates.
        W = np.take_along_axis(Ws, order_w[keep], axis=0).mean(axis=0)
        b = np.take_along_axis(Bs, order_b[keep], axis=0).mean(axis=0)
        return [W, b]


def get_strategy(name: str, trim_ratio: float = 0.1) -> AggregationStrategy:
    registry = {
        "fedavg": FedAvg,
        "fedmedian": FedMedian,
        "trimmedmean": FedTrimmedMean,
    }
    if name not in registry:
        raise StrategyError(f"unknown strategy {name!r}; "
                            f"available: {sorted(registry)}")
    if name == "trimmedmean":
        return FedTrimmedMean(trim_ratio)
    return registry[name]()
