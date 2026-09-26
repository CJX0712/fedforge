import json

import pytest

from fedforge.core.types import FedConfig
from fedforge.pipeline.pipeline import FedPipeline


def _cfg(**over):
    base = dict(num_clients=6, clients_per_round=3, num_rounds=6,
                samples_per_client=80, num_classes=3, num_features=6,
                class_sep=2.5, seed=42)
    base.update(over)
    return FedConfig(**base)


def test_run_returns_full_history():
    res = FedPipeline(_cfg()).run()
    assert res.backend == "numpy"
    assert len(res.history) == 7  # round 0 + 6 rounds
    assert res.history[0].round == 0


def test_run_dirichlet_hurts_worst_client():
    iid = FedPipeline(_cfg(partition="iid")).run()
    nidi = FedPipeline(_cfg(partition="dirichlet", dirichlet_alpha=0.1)).run()
    # label skew must hurt the worst client under FedAvg
    assert nidi.worst_client_acc < iid.worst_client_acc + 1e-9


def test_benchmark_grid_shape():
    rows = FedPipeline(_cfg(num_rounds=4)).benchmark(
        strategies=("fedavg", "fedmedian"),
        scenarios=[("iid", 0.0), ("dirichlet", 0.1)],
        with_flwr=False)
    assert len(rows) == 4  # 2 scenarios x 2 strategies
    for r in rows:
        assert 0.0 <= r.final_acc <= 1.0
        assert r.worst_client_acc <= r.final_acc + 1e-9


def test_benchmark_json_serializable(tmp_path):
    rows = FedPipeline(_cfg(num_rounds=3)).benchmark(
        strategies=("fedavg",), scenarios=[("iid", 0.0)], with_flwr=False)
    out = tmp_path / "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in rows], f)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["strategy"] == "fedavg"


def test_pipeline_rejects_invalid_cfg():
    with pytest.raises(Exception):
        FedPipeline(FedConfig(num_clients=1))


def test_flwr_row_present_when_requested():
    rows = FedPipeline(_cfg(num_rounds=2, samples_per_client=40)).benchmark(
        strategies=("fedavg",), scenarios=[("iid", 0.0)], with_flwr=True)
    flwr_rows = [r for r in rows if r.backend == "flwr"]
    assert len(flwr_rows) == 1
    # either it ran (acc>0) or it was skipped with an explicit note
    assert flwr_rows[0].final_acc > 0 or flwr_rows[0].note.startswith("skipped")
