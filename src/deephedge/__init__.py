"""deephedge: a dependency-free (numpy-only) replication of the core
idea in Buehler, Gonon, Teichmann & Wood (2019), "Deep Hedging" --
train a neural-network hedging policy by directly minimizing a convex
risk measure of the hedging P&L under proportional transaction costs,
via backprop-through-time, and compare it against the classical
Black-Scholes delta-hedge baseline.
"""
from .baseline import bs_delta_hedge_deltas, no_hedge_deltas
from .hedge import forward_and_grad, rollout_deltas_no_grad
from .market import bs_call_delta, bs_call_price, simulate_gbm_paths
from .nn import Adam, MLPPolicy
from .pnl import simulate_hedge_loss
from .risk import entropic_risk, entropic_risk_grad_weights

__all__ = [
    "simulate_gbm_paths",
    "bs_call_price",
    "bs_call_delta",
    "MLPPolicy",
    "Adam",
    "forward_and_grad",
    "rollout_deltas_no_grad",
    "simulate_hedge_loss",
    "entropic_risk",
    "entropic_risk_grad_weights",
    "bs_delta_hedge_deltas",
    "no_hedge_deltas",
]

__version__ = "0.1.0"
