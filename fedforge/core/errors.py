"""Central error hierarchy. Codes E100~E500."""

from __future__ import annotations


class FedForgeError(Exception):
    """Base error for all FedForge failures."""

    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class ConfigError(FedForgeError):
    """Invalid configuration / environment overrides."""

    code = "E100"


class DataError(FedForgeError):
    """Synthetic data generation or partitioning failures."""

    code = "E200"


class StrategyError(FedForgeError):
    """Unknown aggregation strategy or malformed weights."""

    code = "E300"


class TrainError(FedForgeError):
    """Federated training loop failures."""

    code = "E400"


class EvalError(FedForgeError):
    """Evaluation metric failures."""

    code = "E500"
