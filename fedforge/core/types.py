"""Shared dataclass types used across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientDataset:
    """Data held by one federated client."""

    client_id: str
    X: Any  # np.ndarray, shape (n, d)
    y: Any  # np.ndarray, shape (n,)

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class FedConfig:
    """Single source of truth for a federated run."""

    num_clients: int = 8
    num_classes: int = 3
    num_features: int = 12
    samples_per_client: int = 200
    # partition
    partition: str = "dirichlet"  # iid | dirichlet
    dirichlet_alpha: float = 0.5
    # training
    num_rounds: int = 25
    clients_per_round: int = 4
    local_epochs: int = 2
    batch_size: int = 32
    lr: float = 0.1
    l2: float = 1e-3
    # strategy
    strategy: str = "fedavg"  # fedavg | fedmedian | trimmedmean
    trim_ratio: float = 0.1
    # experiment
    seed: int = 42
    class_sep: float = 1.8
    class_overlap_noise: float = 1.0
    rounds_to_target_acc: float = 0.9
    applied_env: dict = field(default_factory=dict)

    def validate(self) -> None:
        from .errors import ConfigError

        if self.num_clients < 2:
            raise ConfigError("num_clients must be >= 2")
        if self.clients_per_round < 1 or self.clients_per_round > self.num_clients:
            raise ConfigError("clients_per_round must be in [1, num_clients]")
        if self.num_rounds < 1:
            raise ConfigError("num_rounds must be >= 1")
        if self.partition not in ("iid", "dirichlet"):
            raise ConfigError("partition must be 'iid' or 'dirichlet'")
        if self.strategy not in ("fedavg", "fedmedian", "trimmedmean"):
            raise ConfigError("strategy must be fedavg|fedmedian|trimmedmean")
        if not 0.0 < self.trim_ratio < 0.5:
            raise ConfigError("trim_ratio must be in (0, 0.5)")
        if self.partition == "dirichlet" and self.dirichlet_alpha <= 0:
            raise ConfigError("dirichlet_alpha must be > 0")
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        if self.local_epochs < 1 or self.batch_size < 1:
            raise ConfigError("local_epochs and batch_size must be >= 1")


@dataclass
class RoundResult:
    """Metrics snapshot after one federated round."""

    round: int
    global_acc: float
    worst_client_acc: float
    mean_client_acc: float
    comm_bytes: int


@dataclass
class RunResult:
    """Outcome of one full federated training run."""

    strategy: str
    backend: str
    partition: str
    dirichlet_alpha: float
    num_clients: int
    config: FedConfig
    history: list = field(default_factory=list)  # list[RoundResult]

    @property
    def final_acc(self) -> float:
        return self.history[-1].global_acc if self.history else 0.0

    @property
    def first_acc(self) -> float:
        return self.history[0].global_acc if self.history else 0.0

    @property
    def worst_client_acc(self) -> float:
        return self.history[-1].worst_client_acc if self.history else 0.0

    @property
    def total_comm_bytes(self) -> int:
        return sum(r.comm_bytes for r in self.history)

    def rounds_to_target(self, target: float) -> int:
        """Rounds needed to first reach `target` accuracy; -1 if never reached."""
        for r in self.history:
            if r.global_acc >= target:
                return r.round
        return -1


@dataclass
class BenchmarkRow:
    """One row of the benchmark table."""

    backend: str
    partition: str
    alpha: float
    strategy: str
    final_acc: float
    worst_client_acc: float
    rounds_to_target: int
    total_comm_mb: float
    note: str = ""
