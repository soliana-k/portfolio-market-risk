"""
Coverage for the EWMA (RiskMetrics) scaled Historical VaR module (Phase 9):
- EWMA volatility recursion and conditional-volatility positivity
- 1-day-ahead volatility forecast
- Scaling historical returns by EWMA volatility
- EWMA-scaled Historical Simulation VaR / ES correctness
- Invalid-input rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ewma_var.ewma import fit_ewma, ewma_volatility


@pytest.fixture
def ewma_returns() -> pd.Series:
    rng = np.random.default_rng(11)
    n = 600
    vol = 0.01 * np.exp(np.sin(np.arange(n) / 25.0))
    r = pd.Series(
        0.0004 + rng.standard_normal(n) * vol,
        index=pd.bdate_range("2023-01-02", periods=n),
        name="portfolio_returns",
    )
    return r


def test_ewma_volatility_positive_and_finite():
    r = np.random.default_rng(0).standard_normal(200) / 100
    sigma = ewma_volatility(r, lambda_=0.94)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)


def test_fit_ewma_result_shape(ewma_returns):
    res = fit_ewma(ewma_returns, confidence_level=0.95)
    n = len(ewma_returns.dropna())
    assert len(res.conditional_volatility) == n
    assert len(res.standardized_residuals) == n
    assert np.isfinite(res.sigma_forecast) and res.sigma_forecast > 0


def test_ewma_conditional_volatility_positive_and_reconstructs(ewma_returns):
    res = fit_ewma(ewma_returns)
    assert (res.conditional_volatility > 0).all()
    recon = res.conditional_volatility * res.standardized_residuals
    assert np.allclose(recon, res.returns, atol=1e-9)


def test_ewma_lambda_invariant_to_scaling(ewma_returns):
    # EWMA variance recursion is scale-invariant in the return ratio: scaling the
    # inputs by c scales sigma (and therefore VaR) by |c|, so the VaR expressed in
    # the *original* (economic) units is identical.  big = 100x, small = 0.01x.
    big = ewma_returns * 100.0
    small = ewma_returns * 0.01
    v_big = fit_ewma(big).var_pct
    v_small = fit_ewma(small).var_pct
    # Convert each back to original-unit VaR (divide by the scaling factor).
    assert abs(v_big / 100.0 - v_small / 0.01) < 1e-8


def test_ewma_var_sign_and_magnitude(ewma_returns):
    res = fit_ewma(ewma_returns, confidence_level=0.95, portfolio_value=1_000_000)
    assert res.var_pct < 0
    assert res.var_dollar == abs(res.var_pct) * 1_000_000
    assert res.es_pct <= res.var_pct


def test_ewma_higher_confidence_stricter_var(ewma_returns):
    low = fit_ewma(ewma_returns, confidence_level=0.95)
    high = fit_ewma(ewma_returns, confidence_level=0.99)
    assert high.var_pct <= low.var_pct


def test_ewma_empty_rejected():
    with pytest.raises(ValueError):
        fit_ewma(pd.Series([], dtype=float))


def test_ewma_bad_confidence_rejected(ewma_returns):
    with pytest.raises(ValueError):
        fit_ewma(ewma_returns, confidence_level=1.5)
