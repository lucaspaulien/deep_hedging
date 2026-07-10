"""Market model: GBM path simulation and closed-form Black-Scholes
price/delta for a European call, used both as the simulator that
generates training/test paths and as the classical hedging baseline.

No scipy in this sandbox, so the normal CDF is built from
`math.erf` (exact, not an approximation), exactly as in the
spx-vol-surface project.
"""
from __future__ import annotations

import math

import numpy as np

_erf = np.vectorize(math.erf)


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def simulate_gbm_paths(n_paths: int, n_steps: int, S0: float, mu: float,
                        sigma: float, T: float, seed: int = 0,
                        antithetic: bool = True) -> np.ndarray:
    """Simulate `n_paths` GBM paths of the underlying, sampled at
    `n_steps` equally spaced dates over [0, T]. Returns an array of
    shape (n_paths, n_steps + 1), column 0 = S0.

    `antithetic=True` pairs each Brownian increment path with its
    mirror image (-Z), which does not change the model but noticeably
    reduces Monte-Carlo noise in the risk-measure estimates used both
    for training and evaluation, for a fixed path budget.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(size=(half, n_steps))
        z = np.concatenate([z, -z], axis=0)[:n_paths]
    else:
        z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
    return S0 * np.exp(log_paths)


def bs_call_price(S, K, tau, r, sigma):
    """Black-Scholes price of a European call. tau = time to maturity
    (years), can be an array. Handles tau -> 0 (intrinsic value)."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    tau_safe = np.where(tau > 1e-12, tau, np.nan)
    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau_safe) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    price = S * norm_cdf(d1) - K * np.exp(-r * tau_safe) * norm_cdf(d2)
    return np.where(tau > 1e-12, price, intrinsic)


def bs_call_delta(S, K, tau, r, sigma):
    """Black-Scholes delta of a European call, dP/dS. tau -> 0 gives
    the (subgradient) 0/1 intrinsic delta."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.where(tau > 1e-12, tau, np.nan)
    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau_safe) / (sigma * sqrt_tau)
    delta = norm_cdf(d1)
    intrinsic_delta = np.where(S > K, 1.0, 0.0)
    return np.where(tau > 1e-12, delta, intrinsic_delta)
