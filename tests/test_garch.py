"""
Coverage for the GARCH(1,1) module:
- GARCH(1,1) fitting and conditional volatility estimation
- 1-day-ahead volatility forecast
- Scaling historical returns by GARCH volatility (standardised residuals)
- GARCH-scaled Historical Simulation VaR / ES correctness
- Invalid-input rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.garch_var.garch import fit_garch, garch_scaled_historical_var


@pytest.fixture
def garch_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    n = 600
    vol = 0.01 * np.exp(np.sin(np.arange(n) / 25.0))
    r = pd.Series(
        0.0004 + rng.standard_normal(n) * vol,
        index=pd.bdate_range("2023-01-02", periods=n),
        name="portfolio_returns",
    )
    return r


def test_fit_garch_returns_result(garch_returns):
    res = fit_garch(garch_returns, confidence_level=0.95)
    assert isinstance(res, object)
    assert len(res.conditional_volatility) == len(garch_returns.dropna())
    assert len(res.standardized_residuals) == len(garch_returns.dropna())
    assert np.isfinite(res.sigma_forecast)
    assert res.sigma_forecast > 0
    assert res.variance_forecast > 0


def test_conditional_volatility_positive(garch_returns):
    res = fit_garch(garch_returns)
    assert (res.conditional_volatility > 0).all()
    mu = res.params.get("mu", 0.0)
    reconstructed = mu + res.conditional_volatility * res.standardized_residuals
    assert np.allclose(reconstructed, res.returns, atol=1e-6)


def test_standardized_residuals_roughly_unit_variance(garch_returns):
    res = fit_garch(garch_returns)
    assert abs(res.standardized_residuals.std() - 1.0) < 0.2


def test_one_day_ahead_forecast_consistency(garch_returns):
    res = fit_garch(garch_returns)
    scaled = res.sigma_forecast * res.standardized_residuals
    pd.testing.assert_series_equal(
        scaled, res.scaled_returns, check_names=False
    )


def test_var_sign_and_magnitude(garch_returns):
    res = fit_garch(garch_returns, confidence_level=0.95, portfolio_value=1_000_000)
    assert res.var_pct < 0
    assert res.var_dollar == abs(res.var_pct) * 1_000_000
    assert res.es_pct <= res.var_pct
    assert res.confidence_level == 0.95


def test_higher_confidence_stricter_var(garch_returns):
    low = fit_garch(garch_returns, confidence_level=0.95)
    high = fit_garch(garch_returns, confidence_level=0.99)
    assert high.var_pct <= low.var_pct


def test_wrapper_returns_dict(garch_returns):
    out = garch_scaled_historical_var(garch_returns, confidence_level=0.99)
    assert "var_pct" in out and "es_pct" in out
    assert out["var_pct"] < 0


def test_empty_returns_rejected():
    with pytest.raises(ValueError):
        fit_garch(pd.Series([], dtype=float))


def test_bad_confidence_rejected(garch_returns):
    with pytest.raises(ValueError):
        fit_garch(garch_returns, confidence_level=1.5)
