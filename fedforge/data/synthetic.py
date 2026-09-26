"""Synthetic federated classification data.

Each class is a Gaussian cluster with shared global means, so a well-defined
global model exists. Difficulty knobs: class_sep (cluster separation) and
class_overlap_noise (within-class std). Deterministic under seed.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import ClientDataset


def make_global_mixture(num_classes: int, num_features: int, class_sep: float,
                        noise_std: float, seed: int):
    """Return (means (k,d), cov (d,d), rng) defining the global distribution."""
    if num_classes < 2 or num_features < 2:
        raise DataError("num_classes and num_features must be >= 2")
    if class_sep <= 0 or noise_std <= 0:
        raise DataError("class_sep and noise_std must be > 0")
    rng = np.random.default_rng(seed)
    # Well-spread class centers on a hypersphere scaled by class_sep.
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    means = np.zeros((num_classes, num_features))
    means[:, 0] = np.cos(angles) * class_sep * num_classes / (2 * np.pi) * 2
    means[:, 1] = np.sin(angles) * class_sep * num_classes / (2 * np.pi) * 2
    if num_features > 2:
        means[:, 2:] = rng.normal(0, 0.3 * class_sep, size=(num_classes, num_features - 2))
    cov = (noise_std ** 2) * np.eye(num_features)
    return means, cov, rng


def sample_dataset(means, cov, rng, n_samples: int):
    """Draw n_samples (X, y) from the global mixture with uniform class priors."""
    k = means.shape[0]
    y = rng.integers(0, k, size=n_samples)
    X = np.stack(
        [rng.multivariate_normal(means[c], cov) for c in y]
    ).astype(np.float64)
    return X, y


def make_federated_classification(cfg, seed_offset: int = 0):
    """Generate the full federated dataset: per-client draws from the global
    mixture (natural heterogeneity via sampling), plus a pooled global test set.

    Returns (clients: list[ClientDataset], test_X, test_y).
    """
    means, cov, rng = make_global_mixture(
        cfg.num_classes, cfg.num_features, cfg.class_sep,
        cfg.class_overlap_noise, cfg.seed + seed_offset,
    )
    clients: list[ClientDataset] = []
    for i in range(cfg.num_clients):
        X, y = sample_dataset(means, cov, rng, cfg.samples_per_client)
        clients.append(ClientDataset(client_id=f"client_{i:03d}", X=X, y=y))
    test_X, test_y = sample_dataset(means, cov, rng, 10 * cfg.samples_per_client)
    return clients, test_X, test_y
