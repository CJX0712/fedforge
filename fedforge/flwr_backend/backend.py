"""Optional Flower (flwr) backend — SOTA federated framework integration.

Targets flwr 1.38 simulation API (Context-based client_fn + Ray engine).
Degrades gracefully: if flwr/ray are missing or simulation fails,
`available_flwr()` returns False / callers catch and fall back to numpy.
"""

from __future__ import annotations

import numpy as np

from ..core.types import FedConfig, RoundResult, RunResult
from ..fed.model import train_local, weights_bytes

_flwr_cache: dict = {}
_runtime_probe: dict = {"usable": None}


def available_flwr() -> bool:
    try:
        import flwr  # noqa: F401
        import ray  # noqa: F401  # flwr simulation engine requires ray
        return True
    except Exception:
        return False


def flwr_runtime_usable() -> bool:
    """True only if a real ray cluster can start on this machine (probed once).

    Some restricted Windows environments fail raylet startup entirely; in that
    case callers must skip the flwr backend and use the numpy fallback.
    """
    if _runtime_probe["usable"] is not None:
        return _runtime_probe["usable"]
    usable = False
    if available_flwr():
        try:
            import ray
            ray.init(include_dashboard=False, logging_level="ERROR",
                     num_cpus=1, _node_ip_address="127.0.0.1")
            ray.shutdown()
            usable = True
        except Exception:
            usable = False
    _runtime_probe["usable"] = usable
    return usable


def _load_flwr():
    if "mod" not in _flwr_cache:
        import flwr as fl
        _flwr_cache["mod"] = fl
    return _flwr_cache["mod"]


def _make_client(cid: int, X, y, l2: float):
    fl = _load_flwr()
    from flwr.client import NumPyClient
    from fedforge.fed.model import cross_entropy, predict

    class _Client(NumPyClient):
        def get_parameters(self, config):
            return []

        def fit(self, parameters, config):
            W, b = np.asarray(parameters[0]), np.asarray(parameters[1])
            local_seed = int(config.get("seed", 42)) + cid * 17
            new_w = train_local(
                X, y, [W, b],
                lr=float(config.get("lr", 0.1)),
                epochs=int(config.get("local_epochs", 2)),
                batch_size=int(config.get("batch_size", 32)),
                seed=local_seed, l2=l2)
            return new_w, len(y), {}

        def evaluate(self, parameters, config):
            W, b = np.asarray(parameters[0]), np.asarray(parameters[1])
            loss = cross_entropy([W, b], X, y)
            acc = float(np.mean(predict([W, b], X) == y))
            return loss, len(y), {"accuracy": acc}

    return _Client()


class _SavingFedAvg:
    """FedAvg strategy that records the latest aggregated global weights."""

    def __init__(self, num_clients: int, clients_per_round: int):
        from flwr.server.strategy import FedAvg
        self._store: dict = {}
        outer = self

        class _S(FedAvg):
            def aggregate_fit(self, server_round, parameters, fit_res):
                agg = super().aggregate_fit(server_round, parameters, fit_res)
                if agg is not None:
                    params, _metrics = agg
                    fl = _load_flwr()
                    nd = fl.common.parameters_to_ndarrays(params)
                    outer._store[server_round] = [np.asarray(a) for a in nd]
                return agg

        self.strategy = _S(
            fraction_fit=clients_per_round / num_clients,
            min_fit_clients=clients_per_round,
            min_available_clients=num_clients,
            min_evaluate_clients=0,
            fraction_evaluate=0.0,
        )

    def final_weights(self) -> list:
        return self._store[max(self._store)]


def run_flwr_fedavg(clients_Xy: list, test_X, test_y, cfg: FedConfig) -> RunResult:
    """Run FedAvg through Flower simulation. Raises on failure; callers
    should catch and fall back to the numpy backend."""
    fl = _load_flwr()
    from flwr.server import ServerConfig
    from flwr.simulation import start_simulation

    data_by_id = {i: (X, y) for i, (X, y) in enumerate(clients_Xy)}

    def client_fn(context):
        pid = int(context.node_config.get("partition-id", 0))
        X, y = data_by_id[pid % len(data_by_id)]
        return _make_client(pid, X, y, cfg.l2).to_client()

    saver = _SavingFedAvg(cfg.num_clients, cfg.clients_per_round)
    start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=ServerConfig(num_rounds=cfg.num_rounds),
        strategy=saver.strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"include_dashboard": False, "logging_level": "ERROR"},
    )

    W, b = saver.final_weights()
    from ..fed.model import predict
    y_test = np.asarray(test_y, dtype=np.int64)
    final_acc = float(np.mean(predict([W, b], test_X) == y_test))
    per_client = [float(np.mean(predict([W, b], X) == np.asarray(y)))
                  for X, y in clients_Xy]
    wb = weights_bytes([W, b])
    comm = cfg.num_rounds * cfg.clients_per_round * 2 * wb

    history = [RoundResult(round=cfg.num_rounds, global_acc=final_acc,
                           worst_client_acc=min(per_client),
                           mean_client_acc=float(np.mean(per_client)),
                           comm_bytes=comm)]
    return RunResult(strategy="fedavg(flwr)", backend="flwr",
                     partition=cfg.partition, dirichlet_alpha=cfg.dirichlet_alpha,
                     num_clients=cfg.num_clients, config=cfg, history=history)
