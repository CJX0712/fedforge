"""Synthetic data generation and partitioning."""

from .partition import dirichlet_partition, iid_partition
from .synthetic import make_federated_classification, make_global_mixture

__all__ = [
    "make_federated_classification",
    "make_global_mixture",
    "iid_partition",
    "dirichlet_partition",
]
