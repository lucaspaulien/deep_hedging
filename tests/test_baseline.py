import numpy as np

from deephedge.baseline import bs_delta_hedge_deltas, no_hedge_deltas
from deephedge.market import simulate_gbm_paths
from deephedge.pnl import simulate_hedge_loss


def test_bs_delta_hedge_shape_and_bounds():
    paths = simulate_gbm_paths(50, 15, 100.0, 0.0, 0.2, 30 / 365, seed=0)
    deltas = bs_delta_hedge_deltas(paths, K=100.0, T_mat=30 / 365, r=0.0, sigma=0.2)
    assert deltas.shape == (50, 15)
    assert np.all(deltas >= 0.0) and np.all(deltas <= 1.0)


def test_bs_delta_hedge_massively_reduces_variance_vs_no_hedge_frictionless():
    # This is the textbook result: in a frictionless world, rebalancing to
    # the BS delta at each date removes almost all of the variance of the
    # option seller's hedging loss, relative to not hedging at all.
    paths = simulate_gbm_paths(4000, 60, 100.0, 0.0, 0.2, 30 / 365, seed=7)
    K = 100.0
    deltas_bs = bs_delta_hedge_deltas(paths, K, T_mat=30 / 365, r=0.0, sigma=0.2)
    deltas_zero = no_hedge_deltas(paths)
    loss_bs = simulate_hedge_loss(paths, deltas_bs, K, cost_rate=0.0)
    loss_zero = simulate_hedge_loss(paths, deltas_zero, K, cost_rate=0.0)
    assert loss_bs.std() < 0.15 * loss_zero.std()


def test_no_hedge_deltas_are_all_zero():
    paths = simulate_gbm_paths(10, 5, 100.0, 0.0, 0.2, 30 / 365, seed=0)
    deltas = no_hedge_deltas(paths)
    assert deltas.shape == (10, 5)
    assert np.all(deltas == 0.0)
