"""FedPipeline: build federated environment and run end-to-end workflows."""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import BenchmarkRow, FedConfig, RunResult
from ..data.partition import dirichlet_partition, iid_partition
from ..data.synthetic import make_global_mixture, sample_dataset
from ..fed.client import NumpyFedClient
from ..training.loop import federated_train


class FedPipeline:
    """Single entry point: data -> partition -> clients -> training."""

    def __init__(self, cfg: FedConfig) -> None:
        self.cfg = cfg
        cfg.validate()

    def build(self) -> tuple:
        """Generate pooled data and partition it; returns (clients, test_X, test_y)."""
        cfg = self.cfg
        total = cfg.num_clients * cfg.samples_per_client
        means, cov, rng = make_global_mixture(
            cfg.num_classes, cfg.num_features, cfg.class_sep,
            cfg.class_overlap_noise, cfg.seed)
        X, y = sample_dataset(means, cov, rng, total)
        if cfg.partition == "iid":
            clients_ds = iid_partition(X, y, cfg.num_clients, cfg.seed + 1)
        elif cfg.partition == "dirichlet":
            clients_ds = dirichlet_partition(
                X, y, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed + 1)
        else:
            raise DataError(f"unknown partition {cfg.partition!r}")
        test_X, test_y = sample_dataset(means, cov, rng, 10 * cfg.samples_per_client)
        clients = [NumpyFedClient(ds) for ds in clients_ds]
        return clients, test_X, test_y

    def run(self, cfg: FedConfig | None = None) -> RunResult:
        cfg = cfg or self.cfg
        clients, test_X, test_y = FedPipeline(cfg).build()
        return federated_train(clients, test_X, test_y, cfg)

    # ---------------- benchmark ----------------

    def benchmark(self, strategies=("fedavg", "fedmedian", "trimmedmean"),
                  scenarios=None, with_flwr: bool = True) -> list:
        """Grid benchmark: scenarios x strategies, fixed seed. Returns rows."""
        from ..eval.metrics import communication_mb

        scenarios = scenarios or [
            ("iid", 0.0),
            ("dirichlet", 0.5),
            ("dirichlet", 0.1),
        ]
        rows: list[BenchmarkRow] = []
        target = self.cfg.rounds_to_target_acc
        for partition, alpha in scenarios:
            scfg = FedConfig(**{**self.cfg.__dict__, "partition": partition,
                                "dirichlet_alpha": alpha})
            clients, test_X, test_y = FedPipeline(scfg).build()
            for strat in strategies:
                rcfg = FedConfig(**{**scfg.__dict__, "strategy": strat})
                res = federated_train(clients, test_X, test_y, rcfg)
                rows.append(BenchmarkRow(
                    backend="numpy", partition=partition, alpha=alpha,
                    strategy=strat, final_acc=round(res.final_acc, 4),
                    worst_client_acc=round(res.worst_client_acc, 4),
                    rounds_to_target=res.rounds_to_target(target),
                    total_comm_mb=communication_mb(res.total_comm_bytes)))
        # Optional SOTA backend: single run on the hardest scenario.
        if with_flwr:
            flwr_clients, flwr_X, flwr_y = FedPipeline(
                FedConfig(**{**self.cfg.__dict__, "partition": "dirichlet",
                             "dirichlet_alpha": 0.1})).build()
            rows.extend(self._flwr_rows(flwr_clients, flwr_X, flwr_y, target,
                                        communication_mb))
        return rows

    def _flwr_rows(self, clients, test_X, test_y, target, comm_mb_fn) -> list:
        """Try the SOTA flwr backend on the hardest scenario; skip on failure."""
        from ..flwr_backend import flwr_runtime_usable, run_flwr_fedavg

        if not flwr_runtime_usable():
            return [BenchmarkRow(
                backend="flwr", partition="dirichlet", alpha=0.1,
                strategy="fedavg", final_acc=0.0, worst_client_acc=0.0,
                rounds_to_target=-1, total_comm_mb=0.0,
                note="skipped: flwr/ray runtime unavailable")]
        try:
            cfg = FedConfig(**{**self.cfg.__dict__,
                               "partition": "dirichlet", "dirichlet_alpha": 0.1})
            clientsXy = [(np.asarray(c.X), np.asarray(c.y)) for c in clients]
            res = run_flwr_fedavg(clientsXy, test_X, test_y, cfg)
            return [BenchmarkRow(
                backend="flwr", partition="dirichlet", alpha=0.1,
                strategy="fedavg", final_acc=round(res.final_acc, 4),
                worst_client_acc=round(res.worst_client_acc, 4),
                rounds_to_target=res.rounds_to_target(target),
                total_comm_mb=comm_mb_fn(res.total_comm_bytes))]
        except Exception as exc:
            return [BenchmarkRow(
                backend="flwr", partition="dirichlet", alpha=0.1,
                strategy="fedavg", final_acc=0.0, worst_client_acc=0.0,
                rounds_to_target=-1, total_comm_mb=0.0,
                note=f"skipped: flwr simulation failed ({type(exc).__name__})")]
