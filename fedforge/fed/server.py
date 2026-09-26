"""Federated server: orchestrates one round (sample -> collect -> aggregate)."""

from __future__ import annotations

import numpy as np

from ..core.errors import TrainError
from ..core.interfaces import AggregationStrategy, FederatedClient
from .model import init_weights, validate_weights, weights_bytes


class FedServer:
    def __init__(self, strategy: AggregationStrategy, num_features: int,
                 num_classes: int, seed: int) -> None:
        self.strategy = strategy
        self.num_features = num_features
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        self.global_weights = init_weights(num_features, num_classes,
                                           seed + 7)

    def select_clients(self, clients: list, k: int) -> list:
        if k < 1 or k > len(clients):
            raise TrainError(f"cannot sample k={k} of {len(clients)} clients")
        idx = self.rng.permutation(len(clients))[:k]
        return [clients[i] for i in idx]

    def run_round(self, clients: list, cfg) -> int:
        """One federated round; returns communication bytes consumed."""
        selected = self.select_clients(clients, cfg.clients_per_round)
        updates, n_samples = [], []
        down_bytes = 0
        for c in selected:
            new_w, n, up = c.fit(
                self.global_weights,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                l2=cfg.l2,
                seed=cfg.seed,
            )
            validate_weights(new_w, self.num_features, self.num_classes)
            updates.append(new_w)
            n_samples.append(n)
            down_bytes += weights_bytes(self.global_weights)
        self.global_weights = self.strategy.aggregate(updates, n_samples)
        up_bytes = sum(weights_bytes(w) for w in updates)
        return int(up_bytes + down_bytes)

    def evaluate_global(self, X, y) -> float:
        from .model import predict
        y = np.asarray(y, dtype=np.int64)
        return float(np.mean(predict(self.global_weights, X) == y))

    def evaluate_clients(self, clients: list) -> list:
        return [c.evaluate(self.global_weights) for c in clients]
