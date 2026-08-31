"""
Coverage for the Stress Testing module (Phase 10):
- Historical scenarios use actual realised loss when the window is in-sample
- Hypothetical equity / volatility / correlation / sector shocks
- Stressed VaR, ES, $ loss and worst-case flagging
- Scenario ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stress_testing.stress import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
    run_stress_tests,
)


@pytest.fixture
def stress_inputs():
    # Returns that DO cover the COVID window so the historical path is exercised.
    dates = pd.bdate_range("2020-01-02", "2021-06-30")
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(len(dates)) * 0.01, index=dates)
    # Inject the COVID crash: -34% over the real window.
    cov_start = pd.Timestamp("2020-02-19")
    cov_end = pd.Timestamp("2020-03-23")
    mask = (r.index >= cov_start) & (r.index <= cov_end)
    r.loc[mask] = -0.02
    weights = pd.Series({"A": 0.6, "B": 0.4})
    asset_returns = pd.DataFrame(
        {a: rng.standard_normal(len(dates)) * 0.01 for a in weights.index},
        index=dates,
    )
    return dict(
        portfolio_value=1_000_000.0,
        weights=weights,
        portfolio_returns=r,
        asset_returns=asset_returns,
        baseline_var_pct=-0.02,
        baseline_es_pct=-0.03,
        confidence_level=0.95,
    )


def test_all_scenarios_present(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    names = set(res.scenarios.keys())
    assert names == set(HISTORICAL_SCENARIOS) | set(HYPOTHETICAL_SCENARIOS)


def test_historical_uses_actual_loss_when_in_sample(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    covid = res.scenarios["COVID Crash 2020"]
    # Actual cumulative return over the injected -2%/day crash window (~25 days).
    actual = float((1.0 + stress_inputs["portfolio_returns"].loc["2020-02-19":"2020-03-23"]).prod() - 1.0)
    assert covid.source == "actual portfolio return"
    assert covid.shock_return_pct == pytest.approx(actual, rel=1e-6)
    assert covid.stressed_var_pct == pytest.approx(abs(actual), rel=1e-6)


def test_hypothetical_equity_shock(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    assert res.scenarios["Equity -20%"].shock_return_pct == pytest.approx(-0.20)
    assert res.scenarios["Equity -20%"].stressed_var_pct == pytest.approx(0.20)
    assert res.scenarios["Equity -5%"].stressed_loss_value == pytest.approx(50_000.0)


def test_volatility_and_correlation_scale_var(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    base = abs(stress_inputs["baseline_var_pct"])
    assert res.scenarios["Volatility x2"].stressed_var_pct == pytest.approx(2.0 * base)
    assert res.scenarios["Correlation Spike"].stressed_var_pct == pytest.approx(1.5 * base)


def test_sector_shock_hits_largest_position(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    sec = res.scenarios["Sector Shock"]
    assert sec.sector_contrib is not None
    # Only the largest-weight asset ("A", 0.6) is shocked.
    assert sec.sector_contrib["A"] == pytest.approx(1_000_000.0 * 0.6 * 0.15)
    assert sec.sector_contrib["B"] == 0.0
    assert sec.shock_return_pct == pytest.approx(0.6 * -0.15)


def test_stressed_es_scales_with_var(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    for sc in res.scenarios.values():
        assert sc.stressed_es_pct >= sc.stressed_var_pct  # ES >= VaR by construction


def test_worst_case_marked(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    # 2008 representative shock (-45%) is the most severe.
    assert res.worst_case_name == "2008 Financial Crisis"
    assert res.scenarios[res.worst_case_name].worst_case is True
    # Exactly one worst-case flagged.
    assert sum(1 for sc in res.scenarios.values() if sc.worst_case) == 1


def test_ranking_descending_by_loss(stress_inputs):
    res = run_stress_tests(**stress_inputs)
    losses = [res.scenarios[n].stressed_loss_value for n in res.ranking]
    assert losses == sorted(losses, reverse=True)
