"""Finite-difference gradient check of MLPPolicy.forward/backward in
isolation (a single call, not the recurrent BPTT case -- that is
tested separately in test_hedge_bptt.py). This is the first line of
defense: if the network's own backward pass is wrong, nothing built
on top of it can be right.
"""
import numpy as np

from deephedge.nn import MLPPolicy


def _numeric_grad_W(policy, x, dout, layer, eps=1e-6):
    W = policy.Ws[layer]
    grad = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        orig = W[idx]
        W[idx] = orig + eps
        d_plus, _ = policy.forward(x)
        loss_plus = np.sum(dout * d_plus)
        W[idx] = orig - eps
        d_minus, _ = policy.forward(x)
        loss_minus = np.sum(dout * d_minus)
        W[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def _numeric_grad_b(policy, x, dout, layer, eps=1e-6):
    b = policy.bs[layer]
    grad = np.zeros_like(b)
    for idx in np.ndindex(b.shape):
        orig = b[idx]
        b[idx] = orig + eps
        d_plus, _ = policy.forward(x)
        loss_plus = np.sum(dout * d_plus)
        b[idx] = orig - eps
        d_minus, _ = policy.forward(x)
        loss_minus = np.sum(dout * d_minus)
        b[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def _numeric_grad_x(policy, x, dout, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        d_plus, _ = policy.forward(x)
        loss_plus = np.sum(dout * d_plus)
        x[idx] = orig - eps
        d_minus, _ = policy.forward(x)
        loss_minus = np.sum(dout * d_minus)
        x[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def test_mlp_backward_matches_finite_difference():
    rng = np.random.default_rng(42)
    policy = MLPPolicy(n_features=3, hidden_sizes=(5, 4), seed=7, output_scale=1.5)
    N = 6
    x = rng.normal(size=(N, 3))
    dout = rng.normal(size=N)

    delta, cache = policy.forward(x)
    grads_W, grads_b, dx = policy.backward(dout, cache)

    for i in range(policy.n_layers):
        num_W = _numeric_grad_W(policy, x.copy(), dout, i)
        assert np.max(np.abs(num_W - grads_W[i])) < 1e-5, f"W[{i}] grad mismatch"

        num_b = _numeric_grad_b(policy, x.copy(), dout, i)
        assert np.max(np.abs(num_b - grads_b[i])) < 1e-5, f"b[{i}] grad mismatch"

    num_x = _numeric_grad_x(policy, x.copy(), dout)
    assert np.max(np.abs(num_x - dx)) < 1e-5, "d(loss)/d(x) mismatch"


def test_mlp_output_bounded():
    rng = np.random.default_rng(1)
    policy = MLPPolicy(n_features=3, hidden_sizes=(8,), seed=3, output_scale=1.5)
    x = rng.normal(size=(100, 3)) * 5.0
    delta, _ = policy.forward(x)
    assert np.all(np.abs(delta) <= 1.5 + 1e-9)


def test_mlp_param_roundtrip():
    policy = MLPPolicy(n_features=3, hidden_sizes=(4, 4), seed=2)
    flat = policy.get_params()
    policy2 = MLPPolicy(n_features=3, hidden_sizes=(4, 4), seed=99)
    policy2.set_params(flat)
    x = np.random.default_rng(0).normal(size=(5, 3))
    d1, _ = policy.forward(x)
    d2, _ = policy2.forward(x)
    assert np.allclose(d1, d2)
