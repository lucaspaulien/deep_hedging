"""Entropic risk measure (exponential-utility indifference risk), the
convex risk measure used to train and evaluate hedging strategies,
following Follmer & Schied's convex risk measure framework and
Buehler, Gonon, Teichmann & Wood (2019), "Deep Hedging".

    rho_lambda(L) = (1/lambda) * log( E[exp(lambda * L)] )

with L the hedging Loss (see `deephedge.pnl`). rho is:
  - convex in L,
  - translation invariant: rho(L + c) = rho(L) + c,
  - monotone: L1 <= L2 pointwise implies rho(L1) <= rho(L2),
  - and recovers the risk-neutral expectation as lambda -> 0
    (rho_lambda(L) -> E[L]), while larger lambda penalizes tail risk
    more (more risk-averse).

`rho_lambda(Loss*)` at the optimal hedge is exactly the classical
exponential-utility indifference price of the option: the minimum
premium a risk-averse seller who hedges optimally would require.
"""
from __future__ import annotations

import numpy as np


def entropic_risk(loss: np.ndarray, lam: float) -> float:
    """Numerically stable (1/lam) * log(mean(exp(lam * loss)))."""
    x = lam * loss
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m)))) / lam


def entropic_risk_grad_weights(loss: np.ndarray, lam: float) -> np.ndarray:
    """d(rho)/d(loss_i), i.e. the softmax weights exp(lam*loss_i) /
    sum_j exp(lam*loss_j). Sums to 1 over the batch."""
    x = lam * loss
    m = np.max(x)
    w = np.exp(x - m)
    w /= np.sum(w)
    return w


def mean_variance_risk(loss: np.ndarray, gamma: float) -> float:
    """Simple alternative risk measure (mean + gamma * std), provided
    for comparison in scripts/plots; NOT used for training (not smooth
    enough to be a nice illustration of the entropic-risk gradient,
    though it is differentiable almost everywhere)."""
    return float(np.mean(loss) + gamma * np.std(loss))
