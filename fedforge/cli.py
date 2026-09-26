"""FedForge CLI: demo / train / hpo / benchmark.

All table output uses fixed-width columns for alignment safety.
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import load_config
from .core.types import BenchmarkRow
from .pipeline.pipeline import FedPipeline


def _print_table(rows: list) -> None:
    header = (f"{'backend':<10}{'partition':<12}{'alpha':<8}{'strategy':<14}"
              f"{'final_acc':<11}{'worst_acc':<11}{'rounds@90%':<12}"
              f"{'comm_MB':<10}{'note'}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.backend:<10}{r.partition:<12}{r.alpha:<8.2f}{r.strategy:<14}"
              f"{r.final_acc:<11.4f}{r.worst_client_acc:<11.4f}"
              f"{r.rounds_to_target:<12}{r.total_comm_mb:<10.4f}{r.note}")


def _rows_to_dicts(rows: list) -> list:
    return [r.__dict__ | {} for r in rows]


def cmd_demo(args) -> int:
    cfg = load_config({"num_rounds": 15, "num_clients": 8,
                       "clients_per_round": 4} if not args.fast else
                      {"num_rounds": 8, "num_clients": 5,
                       "clients_per_round": 3})
    pipe = FedPipeline(cfg)
    rows = pipe.benchmark()
    _print_table(rows)
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"system": "FedForge", "seed": cfg.seed,
                   "rounds_to_target_acc": cfg.rounds_to_target_acc,
                   "rows": _rows_to_dicts(rows)}, f, indent=2, ensure_ascii=False)
    print("\nSaved benchmark.json")
    return 0


def cmd_benchmark(args) -> int:
    cfg = load_config()
    if args.rounds:
        cfg.num_rounds = args.rounds
    rows = FedPipeline(cfg).benchmark()
    _print_table(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"system": "FedForge", "seed": cfg.seed,
                   "rounds_to_target_acc": cfg.rounds_to_target_acc,
                   "rows": _rows_to_dicts(rows)}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}")
    return 0


def cmd_train(args) -> int:
    overrides = {}
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.partition:
        overrides["partition"] = args.partition
    if args.alpha is not None:
        overrides["dirichlet_alpha"] = args.alpha
    if args.rounds:
        overrides["num_rounds"] = args.rounds
    cfg = load_config(overrides or None)
    res = FedPipeline(cfg).run()
    print(f"backend={res.backend} strategy={res.strategy} "
          f"partition={res.partition}(alpha={res.dirichlet_alpha})")
    print(f"final_acc={res.final_acc:.4f} worst_client_acc={res.worst_client_acc:.4f} "
          f"comm={res.total_comm_bytes / 1048576:.4f}MB")
    for row in res.history[-3:]:
        print(f"  round {row.round}: acc={row.global_acc:.4f} "
              f"worst={row.worst_client_acc:.4f}")
    return 0


def cmd_hpo(args) -> int:
    from .hpo.optimize import optimize

    cfg = load_config()
    clients, test_X, test_y = FedPipeline(cfg).build()
    best, value, _study = optimize(clients, test_X, test_y, cfg,
                                   n_trials=args.trials)
    print("best_params:", json.dumps(best))
    print(f"best_value(mean tail acc): {value:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fedforge",
                                description="FedForge: modular federated learning")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="quick end-to-end benchmark -> benchmark.json")
    d.add_argument("--fast", action="store_true", help="smaller config")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("benchmark", help="full benchmark grid")
    b.add_argument("--rounds", type=int, default=None)
    b.add_argument("--out", default="benchmark.json")
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train", help="single federated run")
    t.add_argument("--strategy", default=None)
    t.add_argument("--partition", default=None)
    t.add_argument("--alpha", type=float, default=None)
    t.add_argument("--rounds", type=int, default=None)
    t.set_defaults(func=cmd_train)

    h = sub.add_parser("hpo", help="Optuna hyperparameter search")
    h.add_argument("--trials", type=int, default=10)
    h.set_defaults(func=cmd_hpo)
    return p


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface [Exxx] messages cleanly
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
