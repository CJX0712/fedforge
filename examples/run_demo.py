"""End-to-end demo: benchmark grid + HPO + final run -> benchmark.json.

Run from the repo root:
    python examples/run_demo.py
or via CLI:
    python -m fedforge.cli demo
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from fedforge.core.config import load_config  # noqa: E402
from fedforge.hpo.optimize import optimize  # noqa: E402
from fedforge.pipeline.pipeline import FedPipeline  # noqa: E402
from fedforge.training.loop import federated_train  # noqa: E402

try:
    import flwr
    FLWR_VERSION = flwr.__version__
except Exception:
    FLWR_VERSION = None


def main() -> int:
    cfg = load_config({"num_rounds": 15})
    print(f"FedForge demo | seed={cfg.seed} | numpy={np.__version__} "
          f"sklearn={sklearn.__version__} flwr={FLWR_VERSION or 'unavailable'}")
    print("=" * 72)

    # 1. Benchmark grid: scenarios x strategies + optional flwr backend.
    pipe = FedPipeline(cfg)
    rows = pipe.benchmark()
    header = (f"{'backend':<10}{'partition':<12}{'alpha':<8}{'strategy':<14}"
              f"{'final_acc':<11}{'worst_acc':<11}{'rounds@90%':<12}"
              f"{'comm_MB':<10}{'note'}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.backend:<10}{r.partition:<12}{r.alpha:<8.2f}{r.strategy:<14}"
              f"{r.final_acc:<11.4f}{r.worst_client_acc:<11.4f}"
              f"{r.rounds_to_target:<12}{r.total_comm_mb:<10.4f}{r.note}")

    # 2. HPO (Optuna) on the hardest scenario, then a final run with best params.
    print("\nOptuna HPO (8 trials, 8 eval rounds)...")
    hcfg = load_config({"partition": "dirichlet", "dirichlet_alpha": 0.1,
                        "num_rounds": 8})
    clients, test_X, test_y = FedPipeline(hcfg).build()
    best_params, best_value, _ = optimize(clients, test_X, test_y, hcfg,
                                          n_trials=8, eval_rounds=8)
    print(f"best_params={best_params} best_tail_acc={best_value:.4f}")

    final_cfg = load_config({"partition": "dirichlet", "dirichlet_alpha": 0.1,
                             "num_rounds": 15, **best_params})
    final_res = federated_train(*FedPipeline(final_cfg).build(), final_cfg)
    print(f"final run: acc={final_res.final_acc:.4f} "
          f"worst={final_res.worst_client_acc:.4f}")

    # 3. Persist benchmark.json (reproducible: fixed seed, no wall-clock data).
    out = {
        "system": "FedForge",
        "author": "晨星 (CJX0712)",
        "seed": cfg.seed,
        "rounds_to_target_acc": cfg.rounds_to_target_acc,
        "env": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                "flwr": FLWR_VERSION},
        "hpo": {"best_params": best_params, "best_tail_acc": round(best_value, 4)},
        "final_run": {
            "partition": final_res.partition,
            "alpha": final_res.dirichlet_alpha,
            "final_acc": round(final_res.final_acc, 4),
            "worst_client_acc": round(final_res.worst_client_acc, 4),
            "total_comm_mb": round(final_res.total_comm_bytes / 1048576, 4),
        },
        "benchmark_rows": [r.__dict__ for r in rows],
    }
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
