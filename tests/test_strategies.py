import numpy as np
import pytest

from fedforge.core.errors import StrategyError
from fedforge.fed.strategies import (FedAvg, FedMedian, FedTrimmedMean,
                                     get_strategy)


def _mk(n_clients=5, d=3, k=2, poison_idx=None, poison_scale=1e6, seed=0):
    rng = np.random.default_rng(seed)
    ws = [[rng.normal(0, 1, size=(d, k)), rng.normal(0, 1, size=k)]
          for _ in range(n_clients)]
    ns = [rng.integers(50, 200) for _ in range(n_clients)]
    if poison_idx is not None:
        ws[poison_idx] = [np.full((d, k), poison_scale), np.full(k, poison_scale)]
    return ws, ns


def test_fedavg_equals_weighted_mean():
    ws, ns = _mk()
    out = FedAvg().aggregate(ws, ns)
    total = sum(ns)
    expect_W = sum(w[0] * n for w, n in zip(ws, ns)) / total
    expect_b = sum(w[1] * n for w, n in zip(ws, ns)) / total
    assert np.allclose(out[0], expect_W)
    assert np.allclose(out[1], expect_b)


def test_fedmedian_robust_to_single_poisoner():
    ws, ns = _mk(poison_idx=0)
    median = FedMedian().aggregate(ws, ns)[0]
    avg = FedAvg().aggregate(ws, ns)[0]
    # median stays near clean-coordinate scale, average is blown up
    assert np.abs(median).max() < np.abs(avg).max() / 100
    # robustness property: median lies within the clean clients' range
    clean_stack = np.stack([w[0] for w in ws[1:]])
    assert np.all(median >= clean_stack.min(axis=0) - 1e-12)
    assert np.all(median <= clean_stack.max(axis=0) + 1e-12)


def test_trimmed_mean_excludes_extremes():
    ws, ns = _mk(n_clients=6, poison_idx=5)
    tm = FedTrimmedMean(trim_ratio=0.2).aggregate(ws, ns)[0]  # k=1 each side
    assert np.abs(tm).max() < 100  # poisoner removed
    # reference: unweighted mean of the middle coordinates (same as impl)
    Ws = np.stack([w[0] for w in ws])
    order = np.argsort(Ws, axis=0)
    expect = np.take_along_axis(Ws, order[1:5], axis=0).mean(axis=0)
    assert np.allclose(tm, expect, atol=1e-9)


def test_trimmed_mean_too_few_clients_raises():
    ws, ns = _mk(n_clients=2)
    with pytest.raises(StrategyError):
        FedTrimmedMean(trim_ratio=0.4).aggregate(ws, ns)


def test_unknown_strategy_raises():
    with pytest.raises(StrategyError):
        get_strategy("nope")


def test_empty_updates_raise():
    with pytest.raises(StrategyError):
        FedAvg().aggregate([], [])
    ws, ns = _mk()
    with pytest.raises(StrategyError):
        FedAvg().aggregate(ws, ns[:-1])
    bad = _mk(seed=2)[0]
    with pytest.raises(StrategyError):
        FedAvg().aggregate(bad, [0] * len(bad))


def test_strategy_protocol_satisfied():
    from fedforge.core.interfaces import AggregationStrategy
    assert isinstance(FedAvg(), AggregationStrategy)
    assert isinstance(FedMedian(), AggregationStrategy)
    assert isinstance(FedTrimmedMean(), AggregationStrategy)
