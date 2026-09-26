"""Partitioners: split a pooled dataset across clients (IID / Dirichlet)."""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import ClientDataset


def iid_partition(X, y, num_clients: int, seed: int) -> list:
    """Uniform random split; sizes differ by at most 1."""
    if num_clients < 1:
        raise DataError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    shards = np.array_split(idx, num_clients)
    return [
        ClientDataset(client_id=f"client_{i:03d}", X=X[s], y=y[s])
        for i, s in enumerate(shards)
    ]


def dirichlet_partition(X, y, num_clients: int, alpha: float, seed: int,
                        min_size: int = 10) -> list:
    """Non-IID label-skew partition (Hsu et al., 2019).

    For each client, class proportions ~ Dirichlet(alpha). Small alpha ->
    clients specialize on few classes. Implementation is loop-safe: wanted
    allocations are clipped to remaining per-class stock and any deficit is
    topped up from classes that still have samples, so no rejection sampling
    is needed (guaranteed termination; total conserved exactly).
    """
    if alpha <= 0:
        raise DataError("alpha must be > 0")
    if num_clients < 1:
        raise DataError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    k = len(classes)
    per_class = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    cursor = {c: 0 for c in classes}
    stock = {c: len(v) for c, v in per_class.items()}

    total = len(y)
    sizes = _client_sizes(total, num_clients, rng)
    assignments: list[list] = [[] for _ in range(num_clients)]

    for i, take in enumerate(sizes):
        props = rng.dirichlet(np.full(k, alpha))
        want = np.floor(props * take).astype(int)
        for j, c in enumerate(classes):  # clip to remaining stock
            want[j] = min(int(want[j]), stock[c] - cursor[c])
        got = int(want.sum())
        deficit = take - got
        if deficit > 0:  # top up from classes that still have samples
            for j in rng.permutation(k):
                if deficit <= 0:
                    break
                c = classes[j]
                add = min(stock[c] - cursor[c], deficit)
                want[j] += add
                deficit -= add
        for j, c in enumerate(classes):
            lo = cursor[c]
            hi = cursor[c] + int(want[j])
            assignments[i].extend(per_class[c][lo:hi].tolist())
            cursor[c] = hi

    leftover: list = []
    for c in classes:
        leftover.extend(per_class[c][cursor[c]:].tolist())
    rng.shuffle(leftover)
    for j, idx in enumerate(leftover):
        assignments[j % num_clients].append(idx)

    clients = []
    for i, ids in enumerate(assignments):
        ids = np.array(sorted(ids), dtype=int)
        if len(ids) < min_size:
            raise DataError(
                f"client {i} got {len(ids)} samples (< min_size={min_size}); "
                "increase total samples or alpha")
        clients.append(ClientDataset(client_id=f"client_{i:03d}",
                                     X=X[ids], y=y[ids]))
    return clients


def _client_sizes(total: int, num_clients: int, rng) -> list:
    base = [total // num_clients] * num_clients
    for i in range(total - sum(base)):
        base[i] += 1
    return base
