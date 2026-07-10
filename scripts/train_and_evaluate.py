#!/usr/bin/env python3
"""Train a deep hedging policy and compare it against the classical
Black-Scholes delta-hedge baseline (and a naive "never hedge" floor) on
a fresh, out-of-sample set of simulated paths.

Example:
    python scripts/train_and_evaluate.py --cost-rate 0.02 --n-epochs 300 \
        --n-paths 3000 --n-steps 30 --plot --out results.csv

Supports checkpointing for long runs:
    python scripts/train_and_evaluate.py --cost-rate 0.02 --n-epochs 100 \
        --checkpoint runs/cost2pct.npz
    # run again with the same --checkpoint path to keep training further
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deephedge.baseline import bs_delta_hedge_deltas, no_hedge_deltas  # noqa: E402
from deephedge.hedge import rollout_deltas_no_grad  # noqa: E402
from deephedge.market import simulate_gbm_paths  # noqa: E402
from deephedge.nn import MLPPolicy  # noqa: E402
from deephedge.pnl import simulate_hedge_loss  # noqa: E402
from deephedge.risk import entropic_risk  # noqa: E402
from deephedge.train import (  # noqa: E402
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    train_policy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--maturity-days", type=float, default=30.0)
    p.add_argument("--n-steps", type=int, default=30, help="rebalancing dates")
    p.add_argument("--sigma", type=float, default=0.2, help="GBM volatility")
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=None, help="defaults to r (risk-neutral simulation)")
    p.add_argument("--cost-rate", type=float, default=0.01, help="proportional transaction cost")
    p.add_argument("--lam", type=float, default=8.0, help="entropic risk-aversion")
    p.add_argument("--hidden", type=str, default="16,16")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--n-paths", type=int, default=2000, help="training paths per epoch")
    p.add_argument("--n-test-paths", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-every", type=int, default=20)
    p.add_argument("--checkpoint", type=str, default=None, help="path to .npz checkpoint; resumes if it exists")
    p.add_argument("--out", type=str, default=None, help="CSV path to append summary results")
    p.add_argument("--plot", action="store_true", help="save a P&L histogram + training curve PNG")
    p.add_argument("--plot-path", type=str, default="deep_hedge_results.png")
    return p.parse_args()


def main():
    args = parse_args()
    mu = args.r if args.mu is None else args.mu
    hidden = tuple(int(x) for x in args.hidden.split(","))
    cfg = TrainConfig(
        S0=args.S0, K=args.K, T_mat=args.maturity_days / 365, r=args.r, mu=mu,
        sigma=args.sigma, n_steps=args.n_steps, cost_rate=args.cost_rate,
        lam=args.lam, n_epochs=args.n_epochs, n_paths=args.n_paths, lr=args.lr,
        seed=args.seed,
    )

    policy = MLPPolicy(n_features=3, hidden_sizes=hidden, seed=args.seed, output_scale=1.5)
    adam = None
    epoch_offset = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        policy, adam, epoch_offset = load_checkpoint(args.checkpoint, policy)
        print(f"resumed from {args.checkpoint} at epoch {epoch_offset}")

    t0 = time.time()
    history, adam = train_policy(policy, cfg, verbose_every=args.verbose_every,
                                  adam=adam, epoch_offset=epoch_offset)
    elapsed = time.time() - t0
    total_epochs = epoch_offset + args.n_epochs
    print(f"trained {args.n_epochs} epochs ({total_epochs} total) in {elapsed:.1f}s")

    if args.checkpoint:
        save_checkpoint(args.checkpoint, policy, adam, total_epochs)
        print(f"checkpoint saved to {args.checkpoint}")

    # ---- out-of-sample evaluation ------------------------------------
    test_paths = simulate_gbm_paths(args.n_test_paths, cfg.n_steps, cfg.S0, cfg.mu,
                                     cfg.sigma, cfg.T_mat, seed=args.seed + 777_777)

    deltas_dh = rollout_deltas_no_grad(test_paths, policy, cfg.K, cfg.T_mat)
    deltas_bs = bs_delta_hedge_deltas(test_paths, cfg.K, cfg.T_mat, cfg.r, cfg.sigma)
    deltas_zero = no_hedge_deltas(test_paths)

    results = {}
    for name, deltas in [("deep_hedge", deltas_dh), ("bs_delta", deltas_bs), ("no_hedge", deltas_zero)]:
        detail = simulate_hedge_loss(test_paths, deltas, cfg.K, cfg.cost_rate, return_details=True)
        loss = detail["loss"]
        results[name] = {
            "entropic_risk": entropic_risk(loss, cfg.lam),
            "mean_loss": float(loss.mean()),
            "std_loss": float(loss.std()),
            "mean_total_cost": float(detail["total_cost"].mean()),
            "mean_turnover": float(detail["total_turnover"].mean()),
        }

    print(f"\n{'strategy':<12} {'entropic_risk':>14} {'mean_loss':>11} {'std_loss':>10} "
          f"{'mean_cost':>10} {'mean_turnover':>13}")
    for name, r in results.items():
        print(f"{name:<12} {r['entropic_risk']:14.5f} {r['mean_loss']:11.5f} {r['std_loss']:10.5f} "
              f"{r['mean_total_cost']:10.5f} {r['mean_turnover']:13.4f}")

    improvement = (results["bs_delta"]["entropic_risk"] - results["deep_hedge"]["entropic_risk"])
    print(f"\ndeep hedge vs BS delta: entropic-risk improvement = {improvement:+.5f} "
          f"({'better' if improvement > 0 else 'worse'})")

    if args.out:
        import csv
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["cost_rate", "n_epochs", "n_steps", "n_paths", "strategy",
                            "entropic_risk", "mean_loss", "std_loss", "mean_total_cost", "mean_turnover"])
            for name, r in results.items():
                w.writerow([cfg.cost_rate, total_epochs, cfg.n_steps, args.n_paths, name,
                            r["entropic_risk"], r["mean_loss"], r["std_loss"],
                            r["mean_total_cost"], r["mean_turnover"]])
        print(f"results appended to {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(history)
        axes[0].set_title("Training entropic risk per epoch")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel(r"$\rho_\lambda(\mathrm{Loss})$")

        detail_dh = simulate_hedge_loss(test_paths, deltas_dh, cfg.K, cfg.cost_rate, return_details=True)
        detail_bs = simulate_hedge_loss(test_paths, deltas_bs, cfg.K, cfg.cost_rate, return_details=True)
        axes[1].hist(detail_bs["loss"], bins=60, alpha=0.55, label="BS delta hedge", density=True)
        axes[1].hist(detail_dh["loss"], bins=60, alpha=0.55, label="Deep hedge", density=True)
        axes[1].set_title(f"Out-of-sample hedging loss (cost={cfg.cost_rate:.1%})")
        axes[1].set_xlabel("Loss = payoff - wealth")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.plot_path, dpi=140)
        print(f"plot saved to {args.plot_path}")


if __name__ == "__main__":
    main()
