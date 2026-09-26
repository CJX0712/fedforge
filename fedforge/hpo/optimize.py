"""Optuna HPO over federated hyperparameters (offline numpy backend)."""

from __future__ import annotations

import numpy as np

from ..core.types import FedConfig, RunResult
from ..training.loop import federated_train

_SILENCE_TRIED = {"optuna.logging": None}


def _objective_factory(clients, test_X, test_y, base_cfg: FedConfig,
                       eval_rounds: int):
    def objective(trial):
        cfg = FedConfig(**{k: v for k, v in base_cfg.__dict__.items()})
        cfg.lr = trial.suggest_float("lr", 1e-2, 1.0, log=True)
        cfg.local_epochs = trial.suggest_int("local_epochs", 1, 4)
        cfg.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        cfg.strategy = trial.suggest_categorical(
            "strategy", ["fedavg", "fedmedian", "trimmedmean"])
        cfg.num_rounds = eval_rounds
        cfg.clients_per_round = max(2, base_cfg.clients_per_round)
        result: RunResult = federated_train(clients, test_X, test_y, cfg)
        # Objective: mean global accuracy over the last 3 eval points so the
        # metric rewards stable convergence, not a single lucky round.
        tail = result.history[-3:]
        return float(np.mean([r.global_acc for r in tail]))

    return objective


def optimize(clients: list, test_X, test_y, base_cfg: FedConfig,
             n_trials: int = 15, eval_rounds: int = 8) -> tuple:
    """Run an Optuna TPE study; returns (best_params: dict, best_value: float,
    study). Deterministic under base_cfg.seed."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=base_cfg.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _objective_factory(clients, test_X, test_y, base_cfg, eval_rounds),
        n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value), study
