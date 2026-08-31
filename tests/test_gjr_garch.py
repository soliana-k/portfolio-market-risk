"""
Coverage for the GJR-GARCH module (Phase 7):
- GJR-GARCH fitting and conditional volatility estimation
- 1-day-ahead volatility forecast
- Leverage (asymmetry) parameter extraction and sign interpretation
- GARCH vs GJR-GARCH conditional volatility comparison
- Scaling historical returns by GJR-GARCH volatility
- GJR-GARCH-scaled Historical Simulation VaR / ES correctness
- Invalid-input rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gjr_garch.gjr import fit_gjr_garch, gjr_garch_scaled_historical_var


@pytest.fixture
def gjr_returns() -> pd.Series:
    rng = np.random.default_rng(123)
    n = 800
    vol = 0.01 * np.exp(np.sin(np.arange(n) / 25.0))
    r = pd.Series(
        0.0003 + rng.standard_normal(n) * vol,
        index=pd.bdate_range("2023-01-02", periods=n),
        name="portfolio_returns",
    )
    # Inject persistent negative shocks to create a leverage signature.
    r.iloc[::35] -= 0.05
    return r


def test_fit_gjr_garch_result_shape(gjr_returns):
    res = fit_gjr_garch(gjr_returns, confidence_level=0.95)
    n = len(gjr_returns.dropna())
    assert len(res.conditional_volatility) == n
    assert len(res.standardized_residuals) == n
    assert len(res.garch_conditional_volatility) == n
    assert np.isfinite(res.sigma_forecast) and res.sigma_forecast > 0
    assert res.variance_forecast > 0


def test_conditional_volatility_positive(gjr_returns):
    res = fit_gjr_garch(gjr_returns)
    assert (res.conditional_volatility > 0).all()
    reconstructed = (
        res.params.get("mu", 0.0)
        + res.conditional_volatility * res.standardized_residuals
    )
    assert np.allclose(reconstructed, res.returns, atol=1e-5)


def test_leverage_parameter_present(gjr_returns):
    res = fit_gjr_garch(gjr_returns)
    # gamma should be numeric and bounded in [-1, 1].
    assert np.isfinite(res.leverage_parameter)
    assert -1.0 <= res.leverage_parameter <= 1.0
    # The asymmetry increases negative-shock sensitivity to (alpha + gamma).
    total_neg_sensitivity = res.alpha + res.leverage_parameter
    assert np.isfinite(total_neg_sensitivity)


def test_leverage_stats_populated(gjr_returns):
    res = fit_gjr_garch(gjr_returns)
    ls = res.leverage_stats
    assert ls["n_negative_days"] + ls["n_positive_days"] == len(res.returns)
    assert np.isfinite(ls["avg_vol_after_negative"])
    assert np.isfinite(ls["avg_vol_after_positive"])


def test_garch_vs_gjr_overlap(gjr_returns):
    res = fit_gjr_garch(gjr_returns)
    both = res.conditional_volatility.dropna()
    garch = res.garch_conditional_volatility.dropna()
    # Both series must be defined over the same sample.
    assert len(both) == len(garch)
    assert (garch.index == both.index).all()


def test_one_day_ahead_forecast_consistency(gjr_returns):
    res = fit_gjr_garch(gjr_returns)
    scaled = res.sigma_forecast * res.standardized_residuals
    pd.testing.assert_series_equal(scaled, res.scaled_returns, check_names=False)


def test_var_sign_and_magnitude(gjr_returns):
    res = fit_gjr_garch(gjr_returns, confidence_level=0.95, portfolio_value=1_000_000)
    assert res.var_pct < 0
    assert res.var_dollar == abs(res.var_pct) * 1_000_000
    assert res.es_pct <= res.var_pct
    assert res.confidence_level == 0.95


def test_higher_confidence_stricter_var(gjr_returns):
    low = fit_gjr_garch(gjr_returns, confidence_level=0.95)
    high = fit_gjr_garch(gjr_returns, confidence_level=0.99)
    assert high.var_pct <= low.var_pct


def test_wrapper_returns_dict(gjr_returns):
    out = gjr_garch_scaled_historical_var(gjr_returns, confidence_level=0.99)
    assert "var_pct" in out and "es_pct" in out and "leverage_parameter" in out
    assert out["var_pct"] < 0


def test_empty_returns_rejected():
    with pytest.raises(ValueError):
        fit_gjr_garch(pd.Series([], dtype=float))


def test_bad_confidence_rejected(gjr_returns):
    with pytest.raises(ValueError):
        fit_gjr_garch(gjr_returns, confidence_level=1.5)
