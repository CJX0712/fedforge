"""Protocols (structural contracts) shared by pipeline, server and clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Weights format: list[np.ndarray] == [W (d x k), b (k,)]
Weights = list


@runtime_checkable
class AggregationStrategy(Protocol):
    """Contract: aggregate client weight updates into a global model.

    All strategies must be pure functions of (weights, n_samples).
    """

    name: str

    def aggregate(self, weights: list, n_samples: list) -> list:
        ...


@runtime_checkable
class FederatedClient(Protocol):
    """Contract for any federated client backend (numpy / flwr / sklearn)."""

    client_id: str

    def get_weights(self) -> list:
        ...

    def num_samples(self) -> int:
        ...

    def fit(self, global_weights: list, local_epochs: int, batch_size: int,
            lr: float, l2: float, seed: int) -> tuple:
        """Returns (new_weights, n_samples, comm_bytes_up)."""
        ...

    def evaluate(self, global_weights: list) -> tuple:
        """Returns (loss, accuracy, n_samples)."""
        ...
