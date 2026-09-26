"""Optional Flower backend (SOTA federated framework)."""

from .backend import available_flwr, flwr_runtime_usable, run_flwr_fedavg

__all__ = ["available_flwr", "flwr_runtime_usable", "run_flwr_fedavg"]
