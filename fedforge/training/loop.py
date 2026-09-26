"""Federated training loop (numpy backend)."""

from __future__ import annotations

import numpy as np

from ..core.types import FedConfig, RoundResult, RunResult
from ..eval.metrics import worst_client_accuracy
from ..fed.server import FedServer
from ..fed.strategies import get_strategy


def evaluate_round(server: FedServer, clients: list, test_X, test_y,
                   round_no: int, comm_bytes: int) -> RoundResult:
    global_acc = server.evaluate_global(test_X, test_y)
    per_client = server.evaluate_clients(clients)
    client_accs = [acc for (_l, acc, _n) in per_client]
    return RoundResult(
        round=round_no,
        global_acc=global_acc,
        worst_client_acc=worst_client_accuracy(client_accs),
        mean_client_acc=float(np.mean(client_accs)) if client_accs else 0.0,
        comm_bytes=int(comm_bytes),
    )


def federated_train(clients: list, test_X, test_y, cfg: FedConfig,
                    eval_every: int = 1) -> RunResult:
    """Run `num_rounds` of FedAvg-family training; returns full history."""
    cfg.validate()
    strategy = get_strategy(cfg.strategy, cfg.trim_ratio)
    num_features = np.asarray(clients[0].X).shape[1]
    server = FedServer(strategy, num_features, cfg.num_classes, cfg.seed)

    # Round 0: evaluation before any training (baseline).
    history = [evaluate_round(server, clients, test_X, test_y, 0, 0)]
    for r in range(1, cfg.num_rounds + 1):
        comm = server.run_round(clients, cfg)
        if r % eval_every == 0 or r == cfg.num_rounds:
            history.append(
                evaluate_round(server, clients, test_X, test_y, r, comm))

    return RunResult(
        strategy=cfg.strategy,
        backend="numpy",
        partition=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        num_clients=cfg.num_clients,
        config=cfg,
        history=history,
    )
