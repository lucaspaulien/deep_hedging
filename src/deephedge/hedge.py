"""The core of the project: rolling the policy network forward across
every rebalancing date to build the hedging P&L, and backpropagating
the entropic-risk loss all the way back through time (BPTT,
backprop-through-time) to every weight of the shared network.

State / features
----------------
At each rebalancing date t (t = 0 .. n_steps-1) the policy observes:
  1. log-moneyness            log(S_t / K)
  2. time remaining, scaled   (T_mat - t*dt) / T_mat   in [0, 1]
  3. current position         delta_{t-1}  (0 at t=0)

Feature 3 is what makes this genuinely recurrent: delta_{t-1} is
BOTH (a) an algebraic term in the trade at date t-1 (trade_t = delta_t
- delta_{t-1}) AND (b) an input the network at date t reads to decide
how much to trade. Backprop has to add up both contributions -- this
is exactly backprop-through-time for a weight-shared network, done by
hand below. `tests/test_hedge_bptt.py` verifies the resulting
gradient against plain finite differences on a small random case;
that test is the real proof this is implemented correctly, not this
docstring.

Position is closed out at maturity via one final trade back to zero
(itself subject to transaction costs), handled uniformly with the
same trade/cash/cost algebra as every other rebalancing date -- see
`deephedge.pnl.simulate_hedge_loss`, which independently recomputes
the same Loss from a completed (S_t) path and a completed (delta_t)
grid, used here as a cross-check (see `rollout_deltas_no_grad` +
`pnl.simulate_hedge_loss` giving the same number as `forward_and_grad`'s
own loss, tested in test_hedge_bptt.py).
"""
from __future__ import annotations

import numpy as np

from .risk import entropic_risk, entropic_risk_grad_weights

N_FEATURES = 3
_PREV_DELTA_IDX = 2


def rollout_deltas_no_grad(paths: np.ndarray, policy, K: float, T_mat: float) -> np.ndarray:
    """Forward-only rollout (no cache kept): returns deltas (N, n_steps)
    for the given policy on the given paths. Used for evaluation /
    out-of-sample testing, NOT for training."""
    N, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1
    dt = T_mat / n_steps
    prev_delta = np.zeros(N)
    deltas = np.zeros((N, n_steps))
    for t in range(n_steps):
        S_t = paths[:, t]
        tau_t = T_mat - t * dt
        logm = np.log(S_t / K)
        feats = np.stack([logm, np.full(N, tau_t / T_mat), prev_delta], axis=1)
        delta_t, _ = policy.forward(feats)
        deltas[:, t] = delta_t
        prev_delta = delta_t
    return deltas


def forward_and_grad(paths: np.ndarray, policy, K: float, T_mat: float,
                      cost_rate: float, lam: float):
    """Full forward rollout + entropic-risk loss + BPTT backward pass.

    Returns (rho, grads_W, grads_b, loss) where grads_W/grads_b are
    lists matching policy.Ws / policy.bs (summed, not averaged, over
    rebalancing dates -- the batch averaging is already folded into
    the entropic-risk softmax weights, see below).
    """
    N, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1
    dt = T_mat / n_steps

    prev_delta = np.zeros(N)
    caches = []
    trades = []  # trade_0 .. trade_{n_steps-1}, then closeout trade_{n_steps}
    S_at_trade = []

    for t in range(n_steps):
        S_t = paths[:, t]
        tau_t = T_mat - t * dt
        logm = np.log(S_t / K)
        feats = np.stack([logm, np.full(N, tau_t / T_mat), prev_delta], axis=1)
        delta_t, cache_t = policy.forward(feats)
        caches.append(cache_t)
        trades.append(delta_t - prev_delta)
        S_at_trade.append(S_t)
        prev_delta = delta_t

    S_T = paths[:, n_steps]
    trades.append(-prev_delta)          # closeout trade
    S_at_trade.append(S_T)

    # ---- forward: cash flows, wealth, loss -----------------------------
    wealth = np.zeros(N)
    for t in range(n_steps + 1):
        trade = trades[t]
        S = S_at_trade[t]
        cost = cost_rate * np.abs(trade) * S
        cash = -trade * S - cost
        wealth += cash

    payoff = np.maximum(S_T - K, 0.0)
    loss = payoff - wealth
    rho = entropic_risk(loss, lam)

    # ---- backward: d(rho)/d(trade_t) for every t -----------------------
    w = entropic_risk_grad_weights(loss, lam)      # d(rho)/d(loss_i), sums to 1
    dC = -w                                        # d(rho)/d(cash_t), same at every t

    dTrades = []
    for t in range(n_steps + 1):
        trade = trades[t]
        S = S_at_trade[t]
        # np.sign(0) == 0 here, i.e. we pick the 0 subgradient of |trade|
        # at trade == 0 -- a valid subgradient choice, and the one
        # test_bptt_gradient_check_holds_with_zero_transaction_cost_too
        # exercises directly (cost_rate=0 removes this term entirely, so
        # that test alone can't catch a wrong subgradient here; the
        # weights/biases finite-difference tests do, since real training
        # paths pass through trade == 0 with positive probability).
        dcash_dtrade = -S - cost_rate * S * np.sign(trade)
        dTrades.append(dC * dcash_dtrade)

    # d(rho)/d(delta_t) algebraic part, t = 0..n_steps-1:
    #   delta_t enters trade_t with +1 and trade_{t+1} with -1
    dDeltaAlgebraic = [dTrades[t] - dTrades[t + 1] for t in range(n_steps)]

    # ---- BPTT: walk backward through time, accumulate param grads ------
    # Textbook BPTT recurrence: at each step t (walking from maturity back
    # to t=0), the gradient reaching delta_t is the algebraic contribution
    # computed above PLUS whatever flowed back from step t+1 through the
    # "prev_delta" input feature (d_from_future). policy.backward returns
    # dx, the gradient w.r.t. ALL 3 input features at date t; only the
    # prev-delta column (index _PREV_DELTA_IDX) continues the recurrence
    # into step t-1 -- the other two features (log-moneyness, time
    # remaining) are market data, not something earlier steps produced.
    grads_W = [np.zeros_like(Wi) for Wi in policy.Ws]
    grads_b = [np.zeros_like(bi) for bi in policy.bs]
    d_from_future = np.zeros(N)
    for t in reversed(range(n_steps)):
        dtotal = dDeltaAlgebraic[t] + d_from_future
        gW, gb, dx = policy.backward(dtotal, caches[t])
        for i in range(policy.n_layers):
            grads_W[i] += gW[i]
            grads_b[i] += gb[i]
        d_from_future = dx[:, _PREV_DELTA_IDX]

    return rho, grads_W, grads_b, loss
