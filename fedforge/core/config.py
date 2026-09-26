"""Configuration loading with ENV_FED_* environment overrides."""

from __future__ import annotations

import os

from .errors import ConfigError
from .types import FedConfig

_ENV_MAP = {
    "ENV_FED_NUM_CLIENTS": ("num_clients", int),
    "ENV_FED_NUM_ROUNDS": ("num_rounds", int),
    "ENV_FED_CLIENTS_PER_ROUND": ("clients_per_round", int),
    "ENV_FED_LOCAL_EPOCHS": ("local_epochs", int),
    "ENV_FED_BATCH_SIZE": ("batch_size", int),
    "ENV_FED_SEED": ("seed", int),
    "ENV_FED_NUM_CLASSES": ("num_classes", int),
    "ENV_FED_NUM_FEATURES": ("num_features", int),
    "ENV_FED_SAMPLES_PER_CLIENT": ("samples_per_client", int),
    "ENV_FED_STRATEGY": ("strategy", str),
    "ENV_FED_PARTITION": ("partition", str),
    "ENV_FED_DIRICHLET_ALPHA": ("dirichlet_alpha", float),
    "ENV_FED_LR": ("lr", float),
    "ENV_FED_TRIM_RATIO": ("trim_ratio", float),
    "ENV_FED_CLASS_SEP": ("class_sep", float),
}


def load_config(overrides: dict | None = None) -> FedConfig:
    """Build a FedConfig from defaults, then ENV_FED_* overrides, then
    explicit `overrides` (highest priority). Raises ConfigError when invalid.
    """
    cfg = FedConfig()
    applied: dict[str, str] = {}
    for env_key, (field_name, cast) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, field_name, cast(raw))
            applied[env_key] = raw
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"bad value for {env_key}={raw!r}: {exc}") from exc
    if overrides:
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ConfigError(f"unknown config field: {key}")
            setattr(cfg, key, value)
    cfg.validate()
    cfg.applied_env = applied
    return cfg
