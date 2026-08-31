"""
Coverage for the VaR Backtesting module (Phase 8):
- Rolling out-of-sample backtest driver and exception counting
- Kupiec POF (proportion-of-failures) test correctness
- Christoffersen independence & conditional-coverage tests
- Basel Traffic-Light zoning (canonical 99% / 250-obs boundaries)
- BacktestResult fields (exceptions, expected, ratio, p-values, zone, pass/fail)
- Invalid-input rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.var_backtesting.backtest import (
    BacktestResult,
    backtest_model,
    basel_traffic_light,
    christoffersen_tests,
    kupiec_pof_test,
    run_all_backtests,
)


@pytest.fixture
def bt_returns() -> pd.Series:
    rng = np.random.default_rng(2024)
    n = 1100
    vol = 0.01 * np.exp(np.sin(np.arange(n) / 30.0))
    r = pd.Series(
        0.0003 + rng.standard_normal(n) * vol,
        index=pd.bdate_range("2019-01-02", periods=n),
        name="portfolio_returns",
    )
    return r


def test_kupiec_pof_no_exceptions():
    # x=0 failures over many obs -> two-sided POF can still reject (too few),
    # but the statistic must be finite and in [0,1].
    lr, p = kupiec_pof_test(0, 250, 0.99)
    assert np.isfinite(lr) and 0.0 <= p <= 1.0


def test_kupiec_pof_canonical_95():
    # At 95% VaR over 250 obs, 12.5 expected exceptions.
    # 12 exceptions is close to expected -> should not reject (high p-value).
    _, p = kupiec_pof_test(12, 250, 0.95)
    assert p > 0.05
    # A large excess (40) must be strongly rejected.
    _, p_bad = kupiec_pof_test(40, 250, 0.95)
    assert p_bad < 0.05


def test_christoffersen_independence_detects_clustering():
    rng = np.random.default_rng(0)
    independent = (rng.random(600) < 0.05).astype(int)
    clustered = np.zeros(600, dtype=int)
    for i in range(1, 600):
        p = 0.35 if clustered[i - 1] == 1 else 0.02
        clustered[i] = int(rng.random() < p)
    indep = christoffersen_tests(independent, 0.95)
    clust = christoffersen_tests(clustered, 0.95)
    # Independent exceptions: independence test must not reject.
    assert indep["independence_pvalue"] > 0.05
    # Clustered exceptions: independence test must reject.
    assert clust["independence_pvalue"] < 0.05


def test_basel_traffic_light_canonical():
    # Canonical Basel boundaries for 99% VaR over 250 observations.
    assert basel_traffic_light(0, 250, 0.99) == "Green"
    assert basel_traffic_light(4, 250, 0.99) == "Green"
    assert basel_traffic_light(5, 250, 0.99) == "Yellow"
    assert basel_traffic_light(9, 250, 0.99) == "Yellow"
    assert basel_traffic_light(10, 250, 0.99) == "Red"
    assert basel_traffic_light(15, 250, 0.99) == "Red"


def test_backtest_model_result_fields(bt_returns):
    res = backtest_model(
        bt_returns,
        lambda tr, cl, pv: float(np.quantile(tr.values, 1.0 - cl)),
        model_name="Historical",
        confidence_level=0.95,
        estimation_window=750,
    )
    assert isinstance(res, BacktestResult)
    assert res.n_observations == len(bt_returns) - 750
    assert res.n_exceptions == int(res.exceptions.sum())
    assert res.expected_exceptions == pytest.approx(
        (1.0 - 0.95) * res.n_observations, rel=1e-9
    )
    assert res.exception_ratio == pytest.approx(res.n_exceptions / res.n_observations, rel=1e-9)
    assert 0.0 <= res.kupiec_pof_pvalue <= 1.0
    assert 0.0 <= res.christoffersen_independence_pvalue <= 1.0
    assert 0.0 <= res.christoffersen_cc_pvalue <= 1.0
    assert res.basel_zone in {"Green", "Yellow", "Red"}
    assert res.pass_fail in {"Pass", "Fail"}
    # Exceptions are correctly flagged: actual loss worse than VaR.
    exc = res.exceptions.astype(bool)
    assert (res.actual_returns[exc] < res.var_series[exc]).all()


def test_run_all_backtests_three_models(bt_returns):
    est = 850  # -> ~250 out-of-sample days, keeps the GARCH/GJR refits fast
    results = run_all_backtests(bt_returns, confidence_level=0.95, estimation_window=est)
    assert set(results.keys()) == {"Historical", "GARCH", "GJR-GARCH"}
    for r in results.values():
        assert r.n_observations == len(bt_returns) - est


def test_backtest_requires_enough_data():
    short = pd.Series(np.random.default_rng(1).standard_normal(100) / 100)
    with pytest.raises(ValueError):
        backtest_model(
            short,
            lambda tr, cl, pv: -0.01,
            model_name="X",
            estimation_window=750,
        )


def test_backtest_bad_confidence(bt_returns):
    with pytest.raises(ValueError):
        backtest_model(
            bt_returns,
            lambda tr, cl, pv: -0.01,
            model_name="X",
            confidence_level=1.5,
            estimation_window=200,
        )


def test_basel_zone_matches_pof_extremes():
    # Many more exceptions than expected -> Red; far fewer -> Green.
    assert basel_traffic_light(30, 250, 0.99) == "Red"
    assert basel_traffic_light(1, 250, 0.99) == "Green"
