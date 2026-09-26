import numpy as np
import pytest

from fedforge.core.types import FedConfig
from fedforge.flwr_backend import (available_flwr, flwr_runtime_usable,
                                   run_flwr_fedavg)
from fedforge.pipeline.pipeline import FedPipeline


def test_available_flwr_consistent():
    try:
        import flwr  # noqa: F401
        installed = True
    except Exception:
        installed = False
    assert available_flwr() == installed


def test_flwr_client_contract_without_ray():
    """NumPyClient adapter must satisfy the flwr contract without ray."""
    from fedforge.flwr_backend.backend import _make_client

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 2, size=30)
    client = _make_client(0, X, y, l2=0.0)
    # raw NumPyClient contract: evaluate(parameters, config)
    loss, n, metrics = client.evaluate([np.zeros((4, 2)), np.zeros(2)], {})
    assert n == 30 and np.isfinite(loss)
    params, num, _ = client.fit([np.zeros((4, 2)), np.zeros(2)],
                                {"seed": 42, "lr": 0.05, "local_epochs": 1,
                                 "batch_size": 16})
    assert num == 30 and len(params) == 2


def test_saving_strategy_instantiates():
    from fedforge.flwr_backend.backend import _SavingFedAvg

    saver = _SavingFedAvg(num_clients=4, clients_per_round=2)
    assert saver.strategy is not None
    assert callable(saver.strategy.aggregate_fit)


@pytest.mark.skipif(not flwr_runtime_usable(),
                    reason="flwr/ray simulation runtime unavailable here")
def test_flwr_fedavg_smoke():
    cfg = FedConfig(num_clients=3, clients_per_round=3, num_rounds=2,
                    samples_per_client=50, num_classes=3, num_features=5,
                    class_sep=3.0, seed=42, partition="dirichlet",
                    dirichlet_alpha=0.5)
    clients, test_X, test_y = FedPipeline(cfg).build()
    clientsXy = [(c.X, c.y) for c in clients]
    res = run_flwr_fedavg(clientsXy, test_X, test_y, cfg)
    assert res.backend == "flwr"
    assert 0.0 < res.final_acc <= 1.0
    assert res.total_comm_bytes > 0
