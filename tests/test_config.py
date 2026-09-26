import pytest

from fedforge.core.config import load_config
from fedforge.core.errors import ConfigError


def test_defaults_valid():
    cfg = load_config()
    cfg.validate()  # must not raise


def test_env_overrides_applied(monkeypatch):
    monkeypatch.setenv("ENV_FED_NUM_CLIENTS", "12")
    monkeypatch.setenv("ENV_FED_STRATEGY", "fedmedian")
    monkeypatch.setenv("ENV_FED_LR", "0.05")
    cfg = load_config()
    assert cfg.num_clients == 12
    assert cfg.strategy == "fedmedian"
    assert cfg.lr == 0.05


def test_bad_env_value_raises(monkeypatch):
    monkeypatch.setenv("ENV_FED_NUM_CLIENTS", "not-an-int")
    with pytest.raises(ConfigError):
        load_config()


def test_explicit_overrides_highest_priority(monkeypatch):
    monkeypatch.setenv("ENV_FED_NUM_CLIENTS", "12")
    cfg = load_config({"num_clients": 5})
    assert cfg.num_clients == 5


def test_unknown_override_key_raises():
    with pytest.raises(ConfigError):
        load_config({"no_such_field": 1})


@pytest.mark.parametrize("field,value", [
    ("num_clients", 1),
    ("clients_per_round", 0),
    ("num_rounds", 0),
    ("partition", "bogus"),
    ("strategy", "bogus"),
    ("trim_ratio", 0.6),
    ("dirichlet_alpha", 0.0),
    ("lr", 0.0),
])
def test_invalid_values_raise(field, value):
    with pytest.raises(ConfigError):
        load_config({field: value})


def test_clients_per_round_bounds():
    with pytest.raises(ConfigError):
        load_config({"num_clients": 4, "clients_per_round": 5})
