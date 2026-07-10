"""The single most important test file in this repository.

`deephedge.hedge.forward_and_grad` hand-implements backprop-through-time
for a weight-shared recurrent policy. This file proves that
implementation correct two independent ways:

1. `test_forward_matches_independent_rollout`: the rho value returned by
   forward_and_grad's own (fused, grad-tracking) forward pass must
   exactly match rho computed via the completely independent code path
   `rollout_deltas_no_grad` -> `pnl.simulate_hedge_loss` ->
   `risk.entropic_risk`, which shares no code with the backward pass.

2. `test_bptt_gradient_matches_finite_difference`: every weight and bias
   gradient returned by forward_and_grad must match a numerical
   (central finite-difference) derivative of that SAME independent
   rollout's rho, perturbing one parameter at a time. If BPTT had a bug
   anywhere in the recurrence (the classic failure mode being the
   "previous position is also a feature" pathway), this test would fail.
"""
import numpy as np

from deephedge.hedge import forward_and_grad, rollout_deltas_no_grad
from deephedge.market import simulate_gbm_paths
from deephedge.nn import MLPPolicy
from deephedge.pnl import simulate_hedge_loss
from deephedge.risk import entropic_risk


def _independent_rho(paths, policy, K, T_mat, cost_rate, lam):
    deltas = rollout_deltas_no_grad(paths, policy, K, T_mat)
    loss = simulate_hedge_loss(paths, deltas, K, cost_rate)
    return entropic_risk(loss, lam)


def _make_case(n_paths=12, n_steps=5, cost_rate=0.01, seed=0):
    paths = simulate_gbm_paths(n_paths, n_steps, 100.0, 0.0, 0.25, 20 / 365, seed=seed)
    policy = MLPPolicy(n_features=3, hidden_sizes=(4, 3), seed=seed + 1, output_scale=1.5)
    K, T_mat, lam = 100.0, 20 / 365, 5.0
    return paths, policy, K, T_mat, cost_rate, lam


def test_forward_matches_independent_rollout():
    paths, policy, K, T_mat, cost_rate, lam = _make_case()
    rho, _gW, _gb, loss = forward_and_grad(paths, policy, K, T_mat, cost_rate, lam)
    rho_indep = _independent_rho(paths, policy, K, T_mat, cost_rate, lam)
    assert abs(rho - rho_indep) < 1e-10

    deltas = rollout_deltas_no_grad(paths, policy, K, T_mat)
    loss_indep = simulate_hedge_loss(paths, deltas, K, cost_rate)
    assert np.allclose(loss, loss_indep, atol=1e-10)


def test_bptt_gradient_matches_finite_difference_weights():
    paths, policy, K, T_mat, cost_rate, lam = _make_case()
    _rho, grads_W, grads_b, _loss = forward_and_grad(paths, policy, K, T_mat, cost_rate, lam)

    eps = 1e-5
    for layer in range(policy.n_layers):
        W = policy.Ws[layer]
        # subsample entries for speed if the layer is large; here layers
        # are tiny (<=12 entries) so we check all of them.
        for idx in np.ndindex(W.shape):
            orig = W[idx]
            W[idx] = orig + eps
            rho_plus = _independent_rho(paths, policy, K, T_mat, cost_rate, lam)
            W[idx] = orig - eps
            rho_minus = _independent_rho(paths, policy, K, T_mat, cost_rate, lam)
            W[idx] = orig
            numeric = (rho_plus - rho_minus) / (2 * eps)
            analytic = grads_W[layer][idx]
            assert abs(numeric - analytic) < 1e-4, (
                f"layer {layer} idx {idx}: numeric={numeric:.6f} analytic={analytic:.6f}"
            )


def test_bptt_gradient_matches_finite_difference_biases():
    paths, policy, K, T_mat, cost_rate, lam = _make_case(seed=3)
    _rho, grads_W, grads_b, _loss = forward_and_grad(paths, policy, K, T_mat, cost_rate, lam)

    eps = 1e-5
    for layer in range(policy.n_layers):
        b = policy.bs[layer]
        for idx in np.ndindex(b.shape):
            orig = b[idx]
            b[idx] = orig + eps
            rho_plus = _independent_rho(paths, policy, K, T_mat, cost_rate, lam)
            b[idx] = orig - eps
            rho_minus = _independent_rho(paths, policy, K, T_mat, cost_rate, lam)
            b[idx] = orig
            numeric = (rho_plus - rho_minus) / (2 * eps)
            analytic = grads_b[layer][idx]
            assert abs(numeric - analytic) < 1e-4, (
                f"layer {layer} idx {idx}: numeric={numeric:.6f} analytic={analytic:.6f}"
            )


def test_bptt_gradient_check_holds_with_zero_transaction_cost_too():
    # cost_rate=0 removes the |trade| kink from the graph -> good to check
    # the smooth branch is also correct, independently of the cost path.
    paths, policy, K, T_mat, _cost, lam = _make_case(cost_rate=0.0, seed=11)
    _rho, grads_W, grads_b, _loss = forward_and_grad(paths, policy, K, T_mat, 0.0, lam)
    eps = 1e-5
    W = policy.Ws[0]
    idx = (0, 0)
    orig = W[idx]
    W[idx] = orig + eps
    rho_plus = _independent_rho(paths, policy, K, T_mat, 0.0, lam)
    W[idx] = orig - eps
    rho_minus = _independent_rho(paths, policy, K, T_mat, 0.0, lam)
    W[idx] = orig
    numeric = (rho_plus - rho_minus) / (2 * eps)
    assert abs(numeric - grads_W[0][idx]) < 1e-4
