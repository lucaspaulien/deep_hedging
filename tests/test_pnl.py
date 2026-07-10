import numpy as np

from deephedge.market import simulate_gbm_paths
from deephedge.pnl import simulate_hedge_loss


def test_zero_hedge_loss_equals_payoff():
    paths = simulate_gbm_paths(200, 10, 100.0, 0.0, 0.2, 30 / 365, seed=0)
    K = 100.0
    deltas = np.zeros((200, 10))
    loss = simulate_hedge_loss(paths, deltas, K, cost_rate=0.0)
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    assert np.allclose(loss, payoff)


def test_static_full_share_hedge_changes_loss_profile_as_expected():
    # Holding exactly 1 share throughout (bought at S0=K, sold at S_T) turns
    # the loss into max(S_T - K, 0) - (S_T - S0) = max(S0 - S_T, 0) when
    # S0 == K: algebraically a put-like payoff. Check the identity holds
    # exactly (a strong, deterministic sanity check on the accounting).
    paths = simulate_gbm_paths(500, 20, 100.0, 0.0, 0.2, 30 / 365, seed=1)
    K = 100.0
    always_one = np.ones((500, 20))
    loss_one = simulate_hedge_loss(paths, always_one, K, cost_rate=0.0)
    expected = np.maximum(K - paths[:, -1], 0.0)
    assert np.allclose(loss_one, expected, atol=1e-8)


def test_transaction_costs_increase_loss_mean():
    paths = simulate_gbm_paths(2000, 20, 100.0, 0.0, 0.2, 30 / 365, seed=2)
    K = 100.0
    rng = np.random.default_rng(0)
    deltas = np.clip(rng.normal(0.5, 0.2, size=(2000, 20)), 0, 1)
    loss_no_cost = simulate_hedge_loss(paths, deltas, K, cost_rate=0.0)
    loss_with_cost = simulate_hedge_loss(paths, deltas, K, cost_rate=0.01)
    # same trades, but costly ones cost more (loss = payoff - wealth,
    # so extra costs REDUCE wealth and thus INCREASE loss)
    assert loss_with_cost.mean() > loss_no_cost.mean()


def test_no_trading_no_cost_regardless_of_rate():
    paths = simulate_gbm_paths(50, 5, 100.0, 0.0, 0.2, 30 / 365, seed=3)
    K = 100.0
    deltas = np.zeros((50, 5))
    loss_a = simulate_hedge_loss(paths, deltas, K, cost_rate=0.0)
    loss_b = simulate_hedge_loss(paths, deltas, K, cost_rate=0.05)
    assert np.allclose(loss_a, loss_b)


def test_return_details_keys():
    paths = simulate_gbm_paths(20, 4, 100.0, 0.0, 0.2, 30 / 365, seed=4)
    K = 100.0
    deltas = np.full((20, 4), 0.3)
    out = simulate_hedge_loss(paths, deltas, K, cost_rate=0.01, return_details=True)
    for key in ("loss", "wealth", "payoff", "total_cost", "total_turnover"):
        assert key in out
    assert np.all(out["total_cost"] >= 0)
    assert np.all(out["total_turnover"] >= 0)
