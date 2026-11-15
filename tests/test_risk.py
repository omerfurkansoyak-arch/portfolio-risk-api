import math

import numpy as np
import pytest
from scipy.stats import norm

from risk_api import risk


def normal_returns(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, 0.00012], [0.00012, 0.0009]])
    return rng.multivariate_normal([0.0, 0.0], cov, size=n), cov


def test_parametric_var_matches_closed_form_for_one_asset():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=(500, 1))
    result = risk.parametric([1_000_000.0], r, confidence=0.99)
    sigma, mean = r.std(ddof=1), r.mean()
    assert result.var == pytest.approx(1_000_000 * (norm.ppf(0.99) * sigma - mean))
    assert result.expected_shortfall > result.var


def test_three_methods_agree_on_normal_data():
    returns, _ = normal_returns()
    exposures = [600_000.0, 400_000.0]
    h = risk.historical(exposures, returns, 0.99)
    p = risk.parametric(exposures, returns, 0.99)
    m = risk.monte_carlo(exposures, returns, 0.99, n_sims=400_000, rng=np.random.default_rng(2))
    assert h.var == pytest.approx(p.var, rel=0.03)
    assert m.var == pytest.approx(p.var, rel=0.03)
    assert h.expected_shortfall == pytest.approx(p.expected_shortfall, rel=0.03)
    assert m.expected_shortfall == pytest.approx(p.expected_shortfall, rel=0.03)


def test_diversification_reduces_var():
    returns, _ = normal_returns(50_000)
    together = risk.parametric([500_000.0, 500_000.0], returns).var
    separate = (risk.parametric([500_000.0, 0.0], returns).var
                + risk.parametric([0.0, 500_000.0], returns).var)
    assert together < separate


def test_hedge_reduces_var():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 5000)
    returns = np.column_stack([base, base + rng.normal(0, 0.001, 5000)])
    long_only = risk.historical([1_000_000.0, 0.0], returns).var
    hedged = risk.historical([1_000_000.0, -1_000_000.0], returns).var
    assert hedged < long_only / 5


def test_component_var_sums_to_parametric_var():
    returns, _ = normal_returns(20_000)
    exposures = np.array([700_000.0, -300_000.0])
    total = risk.parametric(exposures, returns, 0.975, horizon_days=10).var
    parts = risk.component_var(exposures, returns, 0.975, horizon_days=10)
    assert parts.sum() == pytest.approx(total)


def test_horizon_scales_volatility_by_square_root_of_time():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.01, size=(100_000, 1))
    r -= r.mean()                                    # remove drift so only volatility scales
    one = risk.parametric([1.0], r, horizon_days=1).var
    ten = risk.parametric([1.0], r, horizon_days=10).var
    assert ten == pytest.approx(one * math.sqrt(10))


def test_var_rises_with_confidence():
    returns, _ = normal_returns(20_000)
    for fn in (risk.historical, risk.parametric):
        assert fn([1.0, 1.0], returns, 0.99).var > fn([1.0, 1.0], returns, 0.95).var


def test_input_validation():
    r = np.zeros((10, 2))
    with pytest.raises(ValueError):
        risk.historical([1.0], r)                    # wrong number of exposures
    with pytest.raises(ValueError):
        risk.parametric([1.0, 1.0], r, confidence=1.5)
    with pytest.raises(ValueError):
        risk.historical([1.0, 1.0], r, horizon_days=0)


def test_kupiec_statistic_known_value():
    # 250 days at 99%: 2.5 exceptions expected. 8 observed -> LR = 7.73 (rejected at 5%).
    stat, p = risk.kupiec_test(8, 250, 0.99)
    assert stat == pytest.approx(7.73, abs=0.01)
    assert p < 0.01


def test_kupiec_accepts_expected_rate_and_handles_zero():
    stat, p = risk.kupiec_test(10, 1000, 0.99)
    assert stat == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    stat0, p0 = risk.kupiec_test(0, 250, 0.99)
    assert stat0 == pytest.approx(-2 * 250 * math.log(0.99))
    assert 0 < p0 < 1


def test_backtest_on_iid_data_is_not_rejected():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, size=(3000, 1))
    result = risk.backtest_historical([1.0], returns, 0.99, window=500)
    assert result.observations == 2500
    assert result.expected_exceptions == pytest.approx(25.0)
    assert result.p_value > 0.05


def test_backtest_detects_volatility_regime_change():
    rng = np.random.default_rng(6)
    calm = rng.normal(0, 0.005, size=(600, 1))
    stressed = rng.normal(0, 0.02, size=(300, 1))
    result = risk.backtest_historical([1.0], np.vstack([calm, stressed]), 0.99, window=500)
    assert result.exceptions > 3 * result.expected_exceptions
    assert result.p_value < 0.01
