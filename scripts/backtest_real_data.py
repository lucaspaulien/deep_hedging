#!/usr/bin/env python3
r"""OPTIONAL, network-dependent bonus script: evaluate an ALREADY-TRAINED
deep hedging policy (and the Black-Scholes delta baseline) on REAL
historical S&P 500 / SPY daily returns, via a stationary block bootstrap,
instead of the simulated GBM paths used for training.

This does NOT retrain anything. Deep Hedging fundamentally needs many
thousands of simulated market scenarios to train on (see README "Known
limitations" -- one clean simulator is part of the method, not a
shortcut), so training stays on GBM (`scripts/train_and_evaluate.py`).
This script only asks a narrower, honest question: does a policy learned
purely on simulated GBM paths still beat the BS delta hedge once
evaluated on resampled REAL daily returns, which have fatter tails and
short-range autocorrelation (volatility clustering) that GBM does not?

Requires network access and `pip install yfinance` (an OPTIONAL
dependency, not required to train/test the core package -- install with
`pip install -e ".[realdata]"` or `pip install yfinance` directly). If
the download fails for any reason (no network, bad ticker, empty
response, ...) this script prints the error and exits WITHOUT
fabricating any numbers.

Example (evaluate the 1%-cost checkpoint produced by train_and_evaluate.py):
    python scripts/backtest_real_data.py --checkpoint runs/cost1.npz \
        --cost-rate 0.01 --ticker SPY --n-steps 30 --n-bootstrap 20000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deephedge.baseline import bs_delta_hedge_deltas  # noqa: E402
from deephedge.hedge import rollout_deltas_no_grad  # noqa: E402
from deephedge.nn import MLPPolicy  # noqa: E402
from deephedge.pnl import simulate_hedge_loss  # noqa: E402
from deephedge.risk import entropic_risk  # noqa: E402
from deephedge.train import load_checkpoint  # noqa: E402

TRADING_DAYS_PER_YEAR = 252


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                    help="trained policy checkpoint (.npz) from train_and_evaluate.py --checkpoint")
    p.add_argument("--ticker", type=str, default="SPY",
                    help="Yahoo Finance ticker for the underlying")
    p.add_argument("--history-period", type=str, default="10y",
                    help="yfinance `period` for the daily close history the returns are drawn from")
    p.add_argument("--K-moneyness", type=float, default=1.0,
                    help="strike as a multiple of S0 (bootstrap start price)")
    p.add_argument("--n-steps", type=int, default=30,
                    help="rebalancing dates per bootstrapped path (trading days)")
    p.add_argument("--cost-rate", type=float, default=0.01,
                    help="should match the cost the checkpoint was trained for")
    p.add_argument("--lam", type=float, default=8.0)
    p.add_argument("--n-bootstrap", type=int, default=20000,
                    help="number of bootstrapped paths to evaluate on")
    p.add_argument("--block-len", type=int, default=5,
                    help="stationary block-bootstrap block length in trading days (preserves "
                         "short-range autocorrelation / vol clustering within a block)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fetch_daily_log_returns(ticker: str, period: str) -> np.ndarray:
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit(
            "yfinance is not installed. This is an optional dependency of this "
            "bonus script only:\n    pip install yfinance\n"
            f"(original error: {e})"
        ) from e
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise SystemExit(
            f"yfinance returned no data for ticker={ticker!r}, period={period!r}. "
            "Check network access / the ticker symbol. Refusing to fabricate "
            "results -- rerun once data is available."
        )
    close = df["Close"].to_numpy().reshape(-1)
    log_ret = np.diff(np.log(close))
    log_ret = log_ret[np.isfinite(log_ret)]
    if len(log_ret) < 500:
        raise SystemExit(
            f"only {len(log_ret)} usable daily returns downloaded for {ticker!r}, "
            "too few for a meaningful bootstrap (need >= 500)."
        )
    return log_ret


def block_bootstrap_paths(log_returns: np.ndarray, n_paths: int, n_steps: int,
                           block_len: int, S0: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary block bootstrap: build each path's return sequence by
    concatenating randomly-positioned blocks of `block_len` consecutive
    HISTORICAL daily log-returns (preserving their real short-range
    autocorrelation / vol clustering within a block), until n_steps
    returns are filled, then exponentiate the cumulative sum into a price
    path. This is deliberately NOT the same as sampling i.i.d. increments
    from a fitted GBM: no distributional assumption is made beyond "the
    future may resemble resampled chunks of the past" -- exactly the kind
    of non-Gaussian, fat-tailed input GBM-trained policies do not see
    during training.
    """
    n_hist = len(log_returns)
    if block_len >= n_hist:
        raise SystemExit(
            f"block_len={block_len} must be smaller than the {n_hist} available historical returns."
        )
    n_blocks_needed = -(-n_steps // block_len)  # ceil division
    starts = rng.integers(0, n_hist - block_len, size=(n_paths, n_blocks_needed))
    offsets = np.arange(block_len)
    idx = starts[:, :, None] + offsets[None, None, :]           # (n_paths, n_blocks, block_len)
    rets = log_returns[idx].reshape(n_paths, n_blocks_needed * block_len)[:, :n_steps]

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(rets, axis=1))
    return paths


def main():
    args = parse_args()
    print(f"downloading {args.ticker} daily history (period={args.history_period}) "
          "from Yahoo Finance...")
    log_returns = fetch_daily_log_returns(args.ticker, args.history_period)
    ann_vol = float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    print(f"got {len(log_returns)} daily log-returns; realized annualized vol = {ann_vol:.1%}")

    S0 = 100.0
    K = args.K_moneyness * S0
    rng = np.random.default_rng(args.seed)
    paths = block_bootstrap_paths(log_returns, args.n_bootstrap, args.n_steps,
                                   args.block_len, S0, rng)

    # hidden_sizes here is irrelevant: load_checkpoint overwrites Ws/bs
    # with the checkpoint's own (saved) shapes.
    policy = MLPPolicy(n_features=3, hidden_sizes=(1,), seed=0)
    policy, _adam, trained_epochs = load_checkpoint(args.checkpoint, policy)
    print(f"loaded policy from {args.checkpoint} "
          f"(trained for {trained_epochs} epochs on simulated GBM)")

    # trading-day convention, matches the bootstrapped daily blocks
    T_mat = args.n_steps / TRADING_DAYS_PER_YEAR
    r = 0.0

    deltas_dh = rollout_deltas_no_grad(paths, policy, K, T_mat)
    deltas_bs = bs_delta_hedge_deltas(paths, K, T_mat, r, ann_vol)

    print(f"\n{'strategy':<12} {'entropic_risk':>14} {'mean_loss':>11} {'std_loss':>10} "
          f"{'mean_cost':>10} {'mean_turnover':>13}")
    for name, deltas in [("deep_hedge", deltas_dh), ("bs_delta", deltas_bs)]:
        detail = simulate_hedge_loss(paths, deltas, K, args.cost_rate, return_details=True)
        loss = detail["loss"]
        rho = entropic_risk(loss, args.lam)
        print(f"{name:<12} {rho:14.5f} {loss.mean():11.5f} {loss.std():10.5f} "
              f"{detail['total_cost'].mean():10.5f} {detail['total_turnover'].mean():13.4f}")

    print(
        "\nNote: this is an EVALUATION-ONLY bonus check on resampled real "
        f"{args.ticker} returns (block bootstrap, block_len={args.block_len} "
        "trading days) using a policy trained purely on simulated GBM paths -- "
        "see README 'Known limitations'. The BS delta baseline uses the "
        "realized historical volatility as its sigma input, since there is no "
        "single 'true' sigma for a bootstrapped path."
    )


if __name__ == "__main__":
    main()
