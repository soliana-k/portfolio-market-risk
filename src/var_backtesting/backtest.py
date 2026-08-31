from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2, binom

from src.garch_var.garch import fit_garch
from src.gjr_garch.gjr import fit_gjr_garch

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ESTIMATION_WINDOW = 750
MIN_BACKTEST_DAYS = 30
# Basel Committee Traffic-Light thresholds for the upper-tail exception probability
# P(X >= x). The Basel traffic light is one-sided (it only penalises *too many*
# exceptions, not too few). These thresholds reproduce the canonical Basel
# boundaries for a 99% VaR backtest over 250 observations -> Green 0-4,
# Yellow 5-9, Red >=10 exceptions:
#   * green / yellow boundary: P(X >= x) ~= 0.15
#   * yellow / red  boundary: P(X >= x) ~= 0.0005
BASEL_GREEN_PVALUE = 0.15
BASEL_YELLOW_PVALUE = 0.0005


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    confidence_level: float
    estimation_window: int
    backtest_dates: pd.DatetimeIndex
    actual_returns: pd.Series
    var_series: pd.Series
    exceptions: pd.Series
    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    exception_ratio: float
    kupiec_pof_lr: float
    kupiec_pof_pvalue: float
    christoffersen_independence_lr: float
    christoffersen_independence_pvalue: float
    christoffersen_cc_lr: float
    christoffersen_cc_pvalue: float
    basel_zone: str
    pass_fail: str
    summary: str = ""


# ── VaR model estimators (return a 1-day VaR in return-fraction units) ───────
def _historical_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    alpha = 1.0 - confidence_level
    return float(np.quantile(train.values, alpha))


def _garch_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    return fit_garch(train, confidence_level=confidence_level, portfolio_value=portfolio_value).var_pct


def _gjr_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    # compare_garch=False avoids a second (plain GARCH) fit on every backtest day.
    return fit_gjr_garch(
        train, confidence_level=confidence_level, portfolio_value=portfolio_value,
        compare_garch=False,
    ).var_pct


MODEL_REGISTRY = {
    "Historical": _historical_var_pct,
    "GARCH": _garch_var_pct,
    "GJR-GARCH": _gjr_var_pct,
}


# ── Statistical backtesting tests ────────────────────────────────────────────
def kupiec_pof_test(n_exceptions: int, n_observations: int, confidence_level: float) -> tuple[float, float]:
    """
    Kupiec Proportion-of-Failures (POF) test of correct unconditional coverage.

    H0: the true exception probability equals p = 1 - confidence_level.
    Returns (LR_statistic, p_value).
    """
    p = 1.0 - confidence_level
    x = n_exceptions
    n = n_observations
    if n <= 0:
        return 0.0, 1.0
    if x < 0 or x > n:
        raise ValueError("n_exceptions must lie in [0, n_observations].")

    if x == 0:
        lr = -2.0 * (n * np.log(1.0 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        ll0 = (n - x) * np.log(1.0 - p) + x * np.log(p)
        ll1 = (n - x) * np.log(1.0 - x / n) + x * np.log(x / n)
        lr = -2.0 * (ll0 - ll1)
    return float(lr), float(chi2.sf(lr, df=1))


def _transition_counts(exceptions: np.ndarray) -> tuple[int, int, int, int]:
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(exceptions)):
        i, j = int(exceptions[t - 1]), int(exceptions[t])
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        elif i == 1 and j == 1:
            n11 += 1
    return n00, n01, n10, n11


def christoffersen_tests(exceptions: np.ndarray, confidence_level: float) -> dict:
    """
    Christoffersen (1998) independence and conditional-coverage tests.

    - Independence (LR_ind, df=1): exceptions are not clustered.
    - Conditional Coverage (LR_cc, df=2): correct coverage AND independence.

    Returns a dict with the LR statistics and p-values for both.
    """
    if len(exceptions) <= 1:
        return {
            "independence_lr": 0.0, "independence_pvalue": 1.0,
            "cc_lr": 0.0, "cc_pvalue": 1.0,
        }

    p = 1.0 - confidence_level
    n00, n01, n10, n11 = _transition_counts(exceptions)
    n = n00 + n01 + n10 + n11
    x = n01 + n11  # total exceptions among transitions

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p_hat = x / n if n > 0 else 0.0

    # Safe x*log(y): 0 * log(0) is 0, not NaN.
    def _xlog(coeff: float, y: float) -> float:
        return coeff * np.log(y) if coeff > 0 and y > 0 else 0.0

    # Independence test: unrestricted vs pooled (no-clustering) likelihood.
    if x == 0:
        lr_ind = 0.0
    else:
        ll0 = _xlog(n00 + n10, 1.0 - p_hat) + _xlog(x, p_hat)
        ll1 = (
            _xlog(n00 + n01, 1.0 - p01) + _xlog(n01, p01)
            + _xlog(n10 + n11, 1.0 - p11) + _xlog(n11, p11)
        )
        lr_ind = -2.0 * (ll0 - ll1)
    lr_ind = max(lr_ind, 0.0)

    # Conditional coverage test: restricted (correct p, independence) vs unrestricted.
    if x == 0:
        lr_cc = 0.0
    else:
        ll0 = _xlog(n00 + n10, 1.0 - p) + _xlog(x, p)
        ll1 = (
            _xlog(n00 + n01, 1.0 - p01) + _xlog(n01, p01)
            + _xlog(n10 + n11, 1.0 - p11) + _xlog(n11, p11)
        )
        lr_cc = -2.0 * (ll0 - ll1)
    lr_cc = max(lr_cc, 0.0)

    return {
        "independence_lr": float(lr_ind),
        "independence_pvalue": float(chi2.sf(lr_ind, df=1)),
        "cc_lr": float(lr_cc),
        "cc_pvalue": float(chi2.sf(lr_cc, df=2)),
    }


def basel_traffic_light(n_exceptions: int, n_observations: int, confidence_level: float) -> str:
    """
    Basel Committee Traffic-Light zone.

    The published Basel zones (99% VaR, 250 obs) — Green 0–4, Yellow 5–9,
    Red ≥10 exceptions — are reproduced by comparing the upper-tail probability
    of observing x or more exceptions, P(X ≥ x) under Binomial(N, 1-cl), against
    the calibrated thresholds BASEL_GREEN_PVALUE / BASEL_YELLOW_PVALUE. The test
    is one-sided: it only flags models with *too many* exceptions.
    """
    p = 1.0 - confidence_level
    x = int(n_exceptions)
    n = int(n_observations)
    p_upper = float(binom.sf(x - 1, n, p)) if n > 0 else 1.0
    if p_upper > BASEL_GREEN_PVALUE:
        return "Green"
    if p_upper >= BASEL_YELLOW_PVALUE:
        return "Yellow"
    return "Red"


# ── Rolling out-of-sample backtest driver ────────────────────────────────────
def backtest_model(
    returns: pd.Series,
    model_func,
    *,
    model_name: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    portfolio_value: float = 1.0,
) -> BacktestResult:
    """
    Rolling out-of-sample VaR backtest.

    For each day t in the backtest period (from `estimation_window` to the end),
    estimate the 1-day VaR using only data up to t-1 (a window of at most
    `estimation_window` trading days), then compare the realised return at t to
    that VaR. An *exception* occurs when the realised loss exceeds the VaR
    (return_t < VaR_t).
    """
    if returns is None or returns.empty:
        raise ValueError("returns cannot be empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1).")

    r = returns.dropna().astype(float)
    r.index = pd.to_datetime(r.index)
    n_total = len(r)

    if n_total <= estimation_window + 1:
        raise ValueError(
            f"Need more than {estimation_window} observations for the estimation "
            f"window; only {n_total} available. Reduce the estimation window or "
            f"supply a longer return history."
        )

    alpha = 1.0 - confidence_level
    var_vals: list[float] = []
    actual_vals: list[float] = []
    dates: list[pd.Timestamp] = []

    logger.info(
        "Backtesting '{}' | est_window={} | backtest days={}",
        model_name, estimation_window, n_total - estimation_window,
    )

    for t in range(estimation_window, n_total):
        train = r.iloc[t - estimation_window: t]
        var_pct = model_func(train, confidence_level, portfolio_value)
        actual = float(r.iloc[t])
        var_vals.append(var_pct)
        actual_vals.append(actual)
        dates.append(r.index[t])

    var_series = pd.Series(var_vals, index=pd.Index(dates, name="Date"), name=f"VaR_{model_name}")
    actual_series = pd.Series(actual_vals, index=pd.Index(dates, name="Date"), name="actual_return")
    exceptions = (actual_series < var_series).astype(int)
    exceptions.name = "exception"

    n_obs = len(exceptions)
    n_exc = int(exceptions.sum())
    expected = (1.0 - confidence_level) * n_obs
    ratio = n_exc / n_obs if n_obs else 0.0

    kupiec_lr, kupiec_pval = kupiec_pof_test(n_exc, n_obs, confidence_level)
    chr_tests = christoffersen_tests(exceptions.values.astype(int), confidence_level)
    zone = basel_traffic_light(n_exc, n_obs, confidence_level)

    # Pass if the model is in the Green zone and the independence test does not
    # reject (no unacceptable clustering of exceptions).
    pass_fail = (
        "Pass"
        if (zone == "Green" and chr_tests["independence_pvalue"] > 0.05)
        else "Fail"
    )

    summary = (
        f"{model_name}: {n_exc}/{n_obs} exceptions (expected {expected:.1f}, "
        f"ratio {ratio:.2%}) | Kupiec p={kupiec_pval:.3g} | "
        f"Indep p={chr_tests['independence_pvalue']:.3g} | "
        f"CC p={chr_tests['cc_pvalue']:.3g} | Basel={zone} -> {pass_fail}"
    )

    return BacktestResult(
        model_name=model_name,
        confidence_level=confidence_level,
        estimation_window=estimation_window,
        backtest_dates=pd.DatetimeIndex(dates),
        actual_returns=actual_series,
        var_series=var_series,
        exceptions=exceptions,
        n_observations=n_obs,
        n_exceptions=n_exc,
        expected_exceptions=expected,
        exception_ratio=ratio,
        kupiec_pof_lr=kupiec_lr,
        kupiec_pof_pvalue=kupiec_pval,
        christoffersen_independence_lr=chr_tests["independence_lr"],
        christoffersen_independence_pvalue=chr_tests["independence_pvalue"],
        christoffersen_cc_lr=chr_tests["cc_lr"],
        christoffersen_cc_pvalue=chr_tests["cc_pvalue"],
        basel_zone=zone,
        pass_fail=pass_fail,
        summary=summary,
    )


def run_all_backtests(
    returns: pd.Series,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    portfolio_value: float = 1.0,
    models: list[str] | None = None,
) -> dict[str, BacktestResult]:
    """
    Run the rolling backtest for every registered VaR model (Historical, GARCH,
    GJR-GARCH) and return a {model_name: BacktestResult} dictionary.
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown VaR model(s): {unknown}. Choose from {list(MODEL_REGISTRY)}.")

    results: dict[str, BacktestResult] = {}
    for name in models:
        results[name] = backtest_model(
            returns,
            MODEL_REGISTRY[name],
            model_name=name,
            confidence_level=confidence_level,
            estimation_window=estimation_window,
            portfolio_value=portfolio_value,
        )
    return results
