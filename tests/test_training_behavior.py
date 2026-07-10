"""End-to-end behavioral tests: does training actually make the policy
better, and does it reproduce the two qualitative results the whole
project is built to demonstrate?

1. Frictionless sanity check: with cost_rate=0, the classical result is
   that the Black-Scholes delta is (essentially) optimal, so a trained
   deep hedge should land in the same ballpark of risk as the BS
   baseline -- not dramatically worse. It is allowed to not exactly match
   given the small demo-scale training budget used in this fast test
   (see scripts/train_and_evaluate.py for a longer, better-converged run
   whose results are reported in the README).

2. Cost-benefit result (the headline claim of Buehler et al. 2019):
   with cost_rate > 0, a policy trained to be cost-aware should achieve a
   LOWER entropic risk than naively re-hedging to the BS delta at every
   date, because BS delta hedging over-trades under costs.

Both tests use tiny, fast settings so the full suite runs in seconds;
they are still real training runs on freshly simulated paths, not
mocked.
"""
import numpy as np

from deephedge.baseline import bs_delta_hedge_deltas
from deephedge.hedge import rollout_deltas_no_grad
from deephedge.market import simulate_gbm_paths
from deephedge.nn import MLPPolicy
from deephedge.pnl import simulate_hedge_loss
from deephedge.risk import entropic_risk
from deephedge.train import TrainConfig, train_policy


def _evaluate(policy, cfg, cost_rate, seed=999, n_paths=3000):
    test_paths = simulate_gbm_paths(n_paths, cfg.n_steps, cfg.S0, cfg.mu,
                                     cfg.sigma, cfg.T_mat, seed=seed)
    deltas_dh = rollout_deltas_no_grad(test_paths, policy, cfg.K, cfg.T_mat)
    loss_dh = simulate_hedge_loss(test_paths, deltas_dh, cfg.K, cost_rate)
    deltas_bs = bs_delta_hedge_deltas(test_paths, cfg.K, cfg.T_mat, cfg.r, cfg.sigma)
    loss_bs = simulate_hedge_loss(test_paths, deltas_bs, cfg.K, cost_rate)
    rho_dh = entropic_risk(loss_dh, cfg.lam)
    rho_bs = entropic_risk(loss_bs, cfg.lam)
    return rho_dh, rho_bs


def test_frictionless_deep_hedge_is_in_the_same_ballpark_as_bs():
    cfg = TrainConfig(cost_rate=0.0, n_epochs=60, n_paths=700, n_steps=15,
                       lam=8.0, lr=5e-3, seed=0)
    policy = MLPPolicy(n_features=3, hidden_sizes=(12, 12), seed=0, output_scale=1.5)
    train_policy(policy, cfg)
    rho_dh, rho_bs = _evaluate(policy, cfg, cost_rate=0.0)
    # Same ballpark: not more than ~40% worse than the (near-)optimal
    # frictionless benchmark. A generous bound given the small training
    # budget used here for test speed.
    assert rho_dh < rho_bs + 0.4 * abs(rho_bs) + 0.05


def test_deep_hedge_beats_bs_delta_under_transaction_costs():
    cost_rate = 0.02
    cfg = TrainConfig(cost_rate=cost_rate, n_epochs=80, n_paths=700, n_steps=15,
                       lam=8.0, lr=5e-3, seed=1)
    policy = MLPPolicy(n_features=3, hidden_sizes=(12, 12), seed=1, output_scale=1.5)
    train_policy(policy, cfg)
    rho_dh, rho_bs = _evaluate(policy, cfg, cost_rate=cost_rate)
    assert rho_dh < rho_bs, f"deep hedge rho={rho_dh:.4f} should beat BS rho={rho_bs:.4f}"


def test_deep_hedge_trades_less_than_bs_under_costs():
    # A cost-aware policy should reduce total turnover relative to
    # blindly re-hedging to the BS delta every date.
    cost_rate = 0.02
    cfg = TrainConfig(cost_rate=cost_rate, n_epochs=80, n_paths=700, n_steps=15,
                       lam=8.0, lr=5e-3, seed=2)
    policy = MLPPolicy(n_features=3, hidden_sizes=(12, 12), seed=2, output_scale=1.5)
    train_policy(policy, cfg)
    test_paths = simulate_gbm_paths(3000, cfg.n_steps, cfg.S0, cfg.mu, cfg.sigma,
                                     cfg.T_mat, seed=123)
    deltas_dh = rollout_deltas_no_grad(test_paths, policy, cfg.K, cfg.T_mat)
    deltas_bs = bs_delta_hedge_deltas(test_paths, cfg.K, cfg.T_mat, cfg.r, cfg.sigma)
    detail_dh = simulate_hedge_loss(test_paths, deltas_dh, cfg.K, cost_rate, return_details=True)
    detail_bs = simulate_hedge_loss(test_paths, deltas_bs, cfg.K, cost_rate, return_details=True)
    assert detail_dh["total_turnover"].mean() < detail_bs["total_turnover"].mean()
