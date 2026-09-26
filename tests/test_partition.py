import numpy as np
import pytest

from fedforge.core.errors import DataError
from fedforge.core.types import FedConfig
from fedforge.data.partition import dirichlet_partition, iid_partition
from fedforge.data.synthetic import make_global_mixture, sample_dataset


def _pooled(n=1200, seed=7):
    means, cov, rng = make_global_mixture(3, 6, 1.5, 1.0, seed)
    return sample_dataset(means, cov, rng, n)


def test_iid_partition_conserves_samples():
    X, y = _pooled()
    clients = iid_partition(X, y, 8, seed=1)
    assert sum(len(c) for c in clients) == len(y)
    merged = np.concatenate([c.y for c in clients])
    assert np.array_equal(np.sort(merged), np.sort(y))


def test_iid_sizes_balanced():
    X, y = _pooled(900)
    clients = iid_partition(X, y, 8, seed=2)
    sizes = [len(c) for c in clients]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_conserves_samples():
    X, y = _pooled()
    clients = dirichlet_partition(X, y, 10, alpha=0.5, seed=3)
    assert sum(len(c) for c in clients) == len(y)
    merged = np.concatenate([c.y for c in clients])
    assert np.array_equal(np.sort(merged), np.sort(y))


def test_dirichlet_low_alpha_creates_label_skew():
    X, y = _pooled(3000)
    clients = dirichlet_partition(X, y, 10, alpha=0.05, seed=4)
    max_frac = max(
        np.bincount(c.y, minlength=3).max() / len(c) for c in clients)
    assert max_frac > 0.8  # low alpha -> clients specialize on few classes


def test_dirichlet_high_alpha_close_to_iid():
    X, y = _pooled(3000)
    clients = dirichlet_partition(X, y, 10, alpha=100.0, seed=5)
    max_frac = max(
        np.bincount(c.y, minlength=3).max() / len(c) for c in clients)
    assert max_frac < 0.55


def test_dirichlet_invalid_alpha_raises():
    X, y = _pooled(50)
    with pytest.raises(DataError):
        dirichlet_partition(X, y, 5, alpha=0.0, seed=6)


def test_dirichlet_min_size_guard():
    X, y = _pooled(60)
    with pytest.raises(DataError):
        dirichlet_partition(X, y, 20, alpha=0.1, seed=6, min_size=10)


def test_synthetic_rejects_bad_params():
    cfg = FedConfig(num_classes=1)
    with pytest.raises(DataError):
        make_global_mixture(1, 5, 1.0, 1.0, 0)
    means, cov, rng = make_global_mixture(3, 5, 1.0, 1.0, 0)
    X, y = sample_dataset(means, cov, rng, 100)
    assert X.shape == (100, 5) and set(np.unique(y)) <= {0, 1, 2}
    del cfg
