"""Federated client: holds local shard, runs local SGD, reports metrics."""

from __future__ import annotations

import numpy as np

from ..core.types import ClientDataset
from .model import cross_entropy, predict, train_local, weights_bytes


class NumpyFedClient:
    """Offline-first client implementing the FederatedClient contract."""

    def __init__(self, dataset: ClientDataset) -> None:
        self.client_id = dataset.client_id
        self._X = np.asarray(dataset.X, dtype=np.float64)
        self._y = np.asarray(dataset.y, dtype=np.int64)
        self._base_seed = 1000 + (abs(hash(self.client_id)) % 9000)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def num_samples(self) -> int:
        return int(self._X.shape[0])

    def get_weights(self) -> list:
        return []  # raw-data client; global weights come from the server

    def fit(self, global_weights: list, local_epochs: int, batch_size: int,
            lr: float, l2: float, seed: int) -> tuple:
        local_seed = seed + self._base_seed
        new_w = train_local(self._X, self._y, global_weights, lr,
                            local_epochs, batch_size, local_seed, l2)
        return new_w, self.num_samples(), weights_bytes(new_w)

    def evaluate(self, global_weights: list) -> tuple:
        loss = cross_entropy(global_weights, self._X, self._y)
        acc = float(np.mean(predict(global_weights, self._X) == self._y))
        return loss, acc, self.num_samples()
