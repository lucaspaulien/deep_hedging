"""Training loop: fresh Monte-Carlo GBM paths every epoch (so the
policy never "memorizes" a fixed path set), full-batch Adam gradient
step from `hedge.forward_and_grad` at each epoch. This is a Monte
Carlo analogue of SGD -- the randomness comes from re-sampling the
market, not from mini-batching a fixed dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .hedge import forward_and_grad
from .market import simulate_gbm_paths
from .nn import Adam, MLPPolicy


@dataclass
class TrainConfig:
    S0: float = 100.0
    K: float = 100.0
    T_mat: float = 30 / 365
    r: float = 0.0
    mu: float = 0.0          # simulate under the risk-neutral measure by
                              # default (mu = r); see README for why using
                              # mu != r is a natural, easy extension.
    sigma: float = 0.2
    n_steps: int = 30
    cost_rate: float = 0.0
    lam: float = 10.0
    n_epochs: int = 200
    n_paths: int = 2000
    lr: float = 3e-3
    seed: int = 0


def train_policy(policy: MLPPolicy, cfg: TrainConfig, verbose_every: int = 0,
                  adam: Adam | None = None, epoch_offset: int = 0):
    """Runs cfg.n_epochs further optimizer steps on `policy` in place.

    Pass in an existing `adam` (and `epoch_offset`, the number of epochs
    already trained) to resume training across multiple calls -- e.g. to
    checkpoint a long run across several sandbox time budgets, or simply
    to keep going after inspecting intermediate results. Path randomness
    is keyed off `cfg.seed` and the *global* epoch index
    (epoch_offset + local epoch), so resumed runs see genuinely fresh
    paths rather than repeating a chunk.

    Returns (history, adam) so the optimizer state (Adam moment
    estimates) can be checkpointed and reused by the caller.
    """
    if adam is None:
        adam = Adam(shapes=policy.param_shapes(), lr=cfg.lr)
    history = []
    for local_epoch in range(cfg.n_epochs):
        global_epoch = epoch_offset + local_epoch
        paths = simulate_gbm_paths(
            cfg.n_paths, cfg.n_steps, cfg.S0, cfg.mu, cfg.sigma, cfg.T_mat,
            seed=cfg.seed * 1_000_003 + global_epoch, antithetic=True,
        )
        rho, grads_W, grads_b, _loss = forward_and_grad(
            paths, policy, cfg.K, cfg.T_mat, cfg.cost_rate, cfg.lam,
        )
        params = policy.get_params()
        grads = grads_W + grads_b
        new_params = adam.step(params, grads)
        policy.set_params(new_params)
        history.append(rho)
        if verbose_every and (local_epoch % verbose_every == 0 or local_epoch == cfg.n_epochs - 1):
            print(f"epoch {global_epoch:5d}  train rho={rho:.6f}")
    return np.array(history), adam


def save_checkpoint(path: str, policy: MLPPolicy, adam: Adam, epoch: int):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    flat = policy.get_params()
    n = policy.n_layers
    np.savez(
        path,
        epoch=epoch,
        n_layers=n,
        **{f"W{i}": flat[i] for i in range(n)},
        **{f"b{i}": flat[n + i] for i in range(n)},
        **{f"m{i}": adam.m[i] for i in range(len(adam.m))},
        **{f"v{i}": adam.v[i] for i in range(len(adam.v))},
        adam_t=adam.t,
        adam_lr=adam.lr,
    )


def load_checkpoint(path: str, policy: MLPPolicy):
    """Loads weights, biases and Adam state into `policy` in place.

    `policy.n_layers` is reset to the checkpoint's own layer count before
    unflattening, so `policy`'s hidden-layer sizes at construction time
    don't need to match the checkpoint's architecture -- the checkpoint's
    own (saved) shapes always win. This matters for callers that load a
    checkpoint without already knowing what architecture it was trained
    with (e.g. an evaluation-only script)."""
    data = np.load(path)
    n = int(data["n_layers"])
    flat = [data[f"W{i}"] for i in range(n)] + [data[f"b{i}"] for i in range(n)]
    policy.n_layers = n
    policy.set_params(flat)
    shapes = policy.param_shapes()
    adam = Adam(shapes=shapes, lr=float(data["adam_lr"]))
    adam.m = [data[f"m{i}"] for i in range(len(shapes))]
    adam.v = [data[f"v{i}"] for i in range(len(shapes))]
    adam.t = int(data["adam_t"])
    epoch = int(data["epoch"])
    return policy, adam, epoch
