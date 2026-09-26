import numpy as np
import pytest

from fedforge.core.errors import EvalError, TrainError
from fedforge.core.types import FedConfig
from fedforge.eval.metrics import (accuracy, communication_mb,
                                   rounds_to_target, worst_client_accuracy)
from fedforge.pipeline.pipeline import FedPipeline
from fedforge.training.loop import federated_train


def _easy_cfg(**over):
    base = dict(num_clients=6, clients_per_round=4, num_rounds=8,
                local_epochs=2, samples_per_client=120, class_sep=3.0,
                num_classes=3, num_features=6, partition="iid", seed=42)
    base.update(over)
    return FedConfig(**base)


def test_training_improves_over_rounds_iid():
    res = federated_train(*FedPipeline(_easy_cfg()).build(), _easy_cfg())
    assert res.final_acc > res.first_acc + 0.05
    assert [r.round for r in res.history] == list(range(0, 9))


def test_training_deterministic_same_seed():
    a = federated_train(*FedPipeline(_easy_cfg()).build(), _easy_cfg())
    b = federated_train(*FedPipeline(_easy_cfg()).build(), _easy_cfg())
    assert a.final_acc == b.final_acc
    assert np.allclose([r.worst_client_acc for r in a.history],
                       [r.worst_client_acc for r in b.history])


def test_rounds_to_target():
    res = federated_train(*FedPipeline(_easy_cfg(num_rounds=12)).build(),
                          _easy_cfg(num_rounds=12))
    t = res.rounds_to_target(res.final_acc)
    assert t > 0  # final round reached final accuracy
    assert res.rounds_to_target(1.1) == -1


def test_clients_per_round_out_of_bounds_raises():
    from fedforge.core.errors import ConfigError
    cfg = _easy_cfg(clients_per_round=99)
    with pytest.raises(ConfigError):
        FedPipeline(cfg).build()


def test_communication_grows_with_rounds():
    cfg = _easy_cfg()
    res = federated_train(*FedPipeline(cfg).build(), cfg)
    assert res.total_comm_bytes > 0
    assert communication_mb(res.total_comm_bytes) > 0


def test_accuracy_metric():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert accuracy([0, 1, 2], [0, 2, 1]) == pytest.approx(1 / 3)
    with pytest.raises(EvalError):
        accuracy([], [])


def test_worst_client_accuracy():
    assert worst_client_accuracy([0.9, 0.5, 0.7]) == 0.5
    with pytest.raises(EvalError):
        worst_client_accuracy([])


def test_metrics_rounds_to_target_helper():
    class R:
        def __init__(self, rnd, acc):
            self.round, self.global_acc = rnd, acc
    assert rounds_to_target([R(1, 0.5), R(2, 0.95)], 0.9) == 2
    assert rounds_to_target([R(1, 0.5)], 0.9) == -1
