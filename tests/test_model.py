import numpy as np
import pytest

from fedforge.core.errors import StrategyError
from fedforge.fed.model import (cross_entropy, init_weights, predict,
                                softmax, train_local, validate_weights,
                                weights_bytes)


def _blob(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = np.zeros((n, 4))
    for c in range(3):
        X[y == c] = rng.normal(1.5 * c, 0.5, size=((y == c).sum(), 4))
    return X, y


def test_softmax_rows_sum_to_one():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 0.0]])  # second row: overflow probe
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p))


def test_init_weights_deterministic():
    a, b = init_weights(5, 3, 42), init_weights(5, 3, 42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_train_local_decreases_loss():
    X, y = _blob()
    w0 = init_weights(4, 3, 0)
    l0 = cross_entropy(w0, X, y)
    w1 = train_local(X, y, w0, lr=0.05, epochs=10, batch_size=32, seed=1)
    l1 = cross_entropy(w1, X, y)
    assert l1 < l0 - 0.3


def test_train_local_deterministic_given_seed():
    X, y = _blob()
    w0 = init_weights(4, 3, 0)
    a = train_local(X, y, w0, 0.1, 2, 32, 7)
    b = train_local(X, y, w0, 0.1, 2, 32, 7)
    assert np.array_equal(a[0], b[0])


def test_predict_accuracy_high_on_separable_blob():
    X, y = _blob(400, seed=3)
    w0 = init_weights(4, 3, 0)
    w = train_local(X, y, w0, 0.1, 40, 32, 5)
    acc = float(np.mean(predict(w, X) == y))
    assert acc > 0.95


def test_weights_bytes():
    w = init_weights(6, 4, 0)
    assert weights_bytes(w) == 6 * 4 * 8 + 4 * 8


def test_validate_weights_raises_on_bad_shape():
    with pytest.raises(StrategyError):
        validate_weights([np.zeros((3, 3)), np.zeros(3)], 4, 3)
    with pytest.raises(StrategyError):
        validate_weights([np.zeros((4, 3))], 4, 3)
