"""A tiny, dependency-free feed-forward network with hand-written
forward and backward passes, plus an Adam optimizer.

Why hand-rolled instead of PyTorch/TensorFlow?
-----------------------------------------------
This project has no network access to `pip install torch`, but more
importantly: writing the backward pass by hand for a *recurrent*
rollout (see `deephedge.hedge`, which reuses this network's weights at
every rebalancing date and has to backpropagate through the whole
path, i.e. backprop-through-time / BPTT) is exactly the kind of thing
an autodiff framework normally hides from you. Doing it by hand here
is both a hard constraint (no framework available) and, honestly, a
better demonstration of understanding what "training a hedging
policy" actually involves under the hood.

Everything is verified against numerical (finite-difference)
gradients in tests/test_nn.py and tests/test_hedge_bptt.py -- see
those files for the actual correctness proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class MLPPolicy:
    """A small MLP mapping a hedging *state* to a scalar hedge ratio.

    Architecture: (Linear -> tanh) x (n_hidden_layers) -> Linear -> tanh,
    with the final tanh scaled by `output_scale` so the hedge ratio is
    bounded in [-output_scale, output_scale] (a call's delta lives in
    [0, 1], so `output_scale=1.5` gives generous headroom).

    The SAME weights are reused at every rebalancing date (a Markovian,
    time-homogeneous policy that additionally sees time-to-maturity as
    an input feature, rather than a separate network per date as in
    some published implementations). This is a deliberate simplicity
    choice: fewer parameters, faster to train in a dependency-free
    sandbox, and it is exactly what is verified against finite
    differences.
    """

    def __init__(self, n_features: int, hidden_sizes=(16, 16), seed: int = 0,
                 output_scale: float = 1.5):
        rng = np.random.default_rng(seed)
        sizes = [n_features] + list(hidden_sizes) + [1]
        self.Ws = [_xavier(rng, fin, fout) for fin, fout in zip(sizes[:-1], sizes[1:], strict=True)]
        self.bs = [np.zeros(fout) for fout in sizes[1:]]
        self.output_scale = float(output_scale)
        self.n_layers = len(self.Ws)

    # -- parameter (de)serialization, used by Adam and by save/load ----
    def get_params(self):
        return [w.copy() for w in self.Ws] + [b.copy() for b in self.bs]

    def set_params(self, flat):
        n = self.n_layers
        self.Ws = [p.copy() for p in flat[:n]]
        self.bs = [p.copy() for p in flat[n:]]

    def param_shapes(self):
        return [w.shape for w in self.Ws] + [b.shape for b in self.bs]

    # -- forward ---------------------------------------------------------
    def forward(self, x: np.ndarray):
        """x: (N, n_features) -> delta: (N,), cache for backward()."""
        cache = {"x": x, "z": [], "a": []}
        a = x
        for i in range(self.n_layers):
            z = a @ self.Ws[i] + self.bs[i]
            cache["z"].append(z)
            if i < self.n_layers - 1:
                a = np.tanh(z)
            else:
                a = self.output_scale * np.tanh(z)
            cache["a"].append(a)
        delta = a[:, 0]
        return delta, cache

    # -- backward ---------------------------------------------------------
    def backward(self, dout: np.ndarray, cache: dict):
        """dout: (N,) = d(loss)/d(delta_output). Returns (grads_W, grads_b, dx)
        where dx: (N, n_features) = d(loss)/d(input features), needed by the
        caller to propagate gradients further back in time (BPTT)."""
        n = self.n_layers
        grads_W = [None] * n
        grads_b = [None] * n
        da = dout[:, None]
        for i in reversed(range(n)):
            z = cache["z"][i]
            if i == n - 1:
                dz = da * self.output_scale * (1.0 - np.tanh(z) ** 2)
            else:
                dz = da * (1.0 - np.tanh(z) ** 2)
            a_prev = cache["x"] if i == 0 else cache["a"][i - 1]
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = dz.sum(axis=0)
            da = dz @ self.Ws[i].T
        dx = da
        return grads_W, grads_b, dx


@dataclass
class Adam:
    """Textbook Adam optimizer, operating on a flat list of numpy arrays
    (the shapes of `MLPPolicy.get_params()`)."""
    shapes: list
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: list = field(default_factory=list)
    v: list = field(default_factory=list)
    t: int = 0

    def __post_init__(self):
        self.m = [np.zeros(s) for s in self.shapes]
        self.v = [np.zeros(s) for s in self.shapes]

    def step(self, params: list, grads: list) -> list:
        self.t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_params.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return new_params
