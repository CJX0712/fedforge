import pytest

from fedforge.core.types import FedConfig
from fedforge.hpo.optimize import optimize
from fedforge.pipeline.pipeline import FedPipeline


def _cfg():
    return FedConfig(num_clients=5, clients_per_round=3, num_rounds=3,
                     samples_per_client=60, num_classes=3, num_features=5,
                     class_sep=2.5, seed=42, partition="dirichlet",
                     dirichlet_alpha=0.5)


def test_optimize_finds_params_and_is_deterministic():
    clients, test_X, test_y = FedPipeline(_cfg()).build()
    best1, v1, _ = optimize(clients, test_X, test_y, _cfg(),
                            n_trials=3, eval_rounds=3)
    best2, v2, _ = optimize(clients, test_X, test_y, _cfg(),
                            n_trials=3, eval_rounds=3)
    assert best1 == best2
    assert v1 == pytest.approx(v2)
    assert set(best1) == {"lr", "local_epochs", "batch_size", "strategy"}
    assert 0.0 < v1 <= 1.0


def test_optimize_honors_search_space():
    clients, test_X, test_y = FedPipeline(_cfg()).build()
    best, _, _ = optimize(clients, test_X, test_y, _cfg(),
                          n_trials=3, eval_rounds=3)
    assert best["batch_size"] in (16, 32, 64)
    assert best["strategy"] in ("fedavg", "fedmedian", "trimmedmean")
    assert 1 <= best["local_epochs"] <= 4
