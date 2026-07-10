"""Classical Black-Scholes delta-hedge baseline: rebalance to the
closed-form BS delta at every date, ignoring transaction costs when
deciding the hedge ratio (because BS delta hedging is only optimal in
a frictionless, continuous-time world). This is the natural benchmark
Deep Hedging is compared against -- the whole point of the exercise is
that once trading is costly, re-hedging to the theoretical
frictionless delta at every discrete date is no longer optimal, and a
policy that is aware of transaction costs (and of its own current
position) can do better.
"""
from __future__ import annotations

import numpy as np

from .market import bs_call_delta


def bs_delta_hedge_deltas(paths: np.ndarray, K: float, T_mat: float, r: float,
                           sigma: float) -> np.ndarray:
    """paths: (N, n_steps+1). Returns deltas: (N, n_steps), the BS
    delta at each rebalancing date t=0..n_steps-1, using the true
    time-to-maturity remaining at that date."""
    N, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1
    dt = T_mat / n_steps
    deltas = np.zeros((N, n_steps))
    for t in range(n_steps):
        tau = T_mat - t * dt
        deltas[:, t] = bs_call_delta(paths[:, t], K, tau, r, sigma)
    return deltas


def no_hedge_deltas(paths: np.ndarray) -> np.ndarray:
    """Trivial baseline: never hedge (deltas always 0). Useful as a
    sanity floor -- any real hedge should beat this by a wide margin."""
    N, n_steps_p1 = paths.shape
    return np.zeros((N, n_steps_p1 - 1))
