import numpy as np

from deephedge.market import (
    bs_call_delta,
    bs_call_price,
    norm_cdf,
    simulate_gbm_paths,
)


def test_norm_cdf_matches_known_values():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(norm_cdf(1.959963985) - 0.975) < 1e-6
    assert abs(norm_cdf(-1.959963985) - 0.025) < 1e-6


def test_bs_call_price_deep_itm_is_intrinsic_like():
    # Deep ITM, short maturity: price ~ S - K*exp(-rT)
    price = bs_call_price(S=200.0, K=100.0, tau=1 / 365, r=0.0, sigma=0.2)
    assert abs(price - 100.0) < 1.0


def test_bs_call_price_deep_otm_is_near_zero():
    price = bs_call_price(S=50.0, K=200.0, tau=30 / 365, r=0.0, sigma=0.2)
    assert price < 1e-3


def test_bs_call_delta_bounds():
    S = np.array([50.0, 100.0, 100.0, 300.0])
    K = 100.0
    delta = bs_call_delta(S, K, tau=30 / 365, r=0.0, sigma=0.2)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
    # deep ITM delta close to 1, deep OTM close to 0
    assert delta[3] > 0.95
    assert delta[0] < 0.30


def test_bs_call_delta_atm_near_half():
    delta = bs_call_delta(100.0, 100.0, tau=30 / 365, r=0.0, sigma=0.2)
    assert 0.45 < float(delta) < 0.60


def test_bs_delta_is_derivative_of_price_finite_difference():
    S0, K, tau, r, sigma = 105.0, 100.0, 45 / 365, 0.01, 0.25
    eps = 1e-3
    p_plus = bs_call_price(S0 + eps, K, tau, r, sigma)
    p_minus = bs_call_price(S0 - eps, K, tau, r, sigma)
    numeric_delta = (p_plus - p_minus) / (2 * eps)
    analytic_delta = bs_call_delta(S0, K, tau, r, sigma)
    assert abs(float(numeric_delta) - float(analytic_delta)) < 1e-4


def test_simulate_gbm_paths_shape_and_start():
    paths = simulate_gbm_paths(n_paths=101, n_steps=20, S0=100.0, mu=0.0,
                                sigma=0.2, T=30 / 365, seed=0)
    assert paths.shape == (101, 21)
    assert np.allclose(paths[:, 0], 100.0)
    assert np.all(paths > 0)


def test_simulate_gbm_paths_mean_matches_drift():
    # E[S_T] = S0 * exp(mu*T) under the simulated (mu, sigma) measure.
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    paths = simulate_gbm_paths(n_paths=40000, n_steps=50, S0=S0, mu=mu,
                                sigma=sigma, T=T, seed=1, antithetic=True)
    mean_ST = paths[:, -1].mean()
    expected = S0 * np.exp(mu * T)
    assert abs(mean_ST - expected) / expected < 0.01


def test_antithetic_reduces_variance_of_terminal_price():
    S0, mu, sigma, T = 100.0, 0.0, 0.3, 1.0
    n = 4000
    paths_anti = simulate_gbm_paths(n, 50, S0, mu, sigma, T, seed=5, antithetic=True)
    paths_plain = simulate_gbm_paths(n, 50, S0, mu, sigma, T, seed=5, antithetic=False)
    mean_anti = paths_anti[:, -1].mean()
    mean_plain = paths_plain[:, -1].mean()
    expected = S0 * np.exp(mu * T)
    # antithetic sampling should not be worse (usually much better) at
    # estimating the true mean than plain iid sampling of the same size
    assert abs(mean_anti - expected) <= abs(mean_plain - expected) + 0.5
