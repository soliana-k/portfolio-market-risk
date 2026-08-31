from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.garch_var.garch import fit_garch
from src.gjr_garch.gjr import fit_gjr_garch
from src.ewma_var.ewma import fit_ewma
from src.var_backtesting.backtest import (
    BacktestResult,
    backtest_model,
    DEFAULT_ESTIMATION_WINDOW,
    DEFAULT_CONFIDENCE_LEVEL,
)

# Fixed display order for the four compared VaR models.
MODEL_ORDER = ["Historical", "EWMA", "GARCH", "GJR-GARCH"]

ZONE_RANK = {"Green": 0, "Yellow": 1, "Red": 2}


@dataclass(frozen=True)
class ModelComparisonResult:
    model_name: str
    backtest: BacktestResult
    average_var: float
    var_volatility: float
    worst_daily_loss: float
    expected_shortfall: float
    rank: int = 0
    rank_reason: str = ""


# ── Per-model VaR estimators used by the rolling backtest ─────────────────────
def _historical_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    return float(np.quantile(train.values, 1.0 - confidence_level))


def _ewma_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    return fit_ewma(train, confidence_level=confidence_level, portfolio_value=portfolio_value).var_pct


def _garch_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    return fit_garch(train, confidence_level=confidence_level, portfolio_value=portfolio_value).var_pct


def _gjr_var_pct(train: pd.Series, confidence_level: float, portfolio_value: float) -> float:
    return fit_gjr_garch(
        train, confidence_level=confidence_level, portfolio_value=portfolio_value,
        compare_garch=False,
    ).var_pct


MODEL_FUNCS = {
    "Historical": _historical_var_pct,
    "EWMA": _ewma_var_pct,
    "GARCH": _garch_var_pct,
    "GJR-GARCH": _gjr_var_pct,
}


def _rank_models(results: dict[str, BacktestResult], confidence_level: float) -> dict[str, tuple[int, str]]:
    """
    Rank models from best (1) to worst. Primary key is the Basel traffic-light
    zone (Green best), secondary is the absolute deviation of the realised
    exception ratio from the target (1 - cl), tertiary is the Kupiec POF p-value
    (higher = closer to correct coverage).
    """
    target = 1.0 - confidence_level
    scored = []
    for name, btr in results.items():
        dev = abs(btr.exception_ratio - target)
        scored.append((name, ZONE_RANK.get(btr.basel_zone, 3), dev, btr.kupiec_pof_pvalue))
    # Sort: zone asc, deviation asc, kupiec p-value desc.
    scored.sort(key=lambda x: (x[1], x[2], -x[3]))
    ranking: dict[str, tuple[int, str]] = {}
    for i, (name, zone_rank, dev, kupiec_p) in enumerate(scored, start=1):
        reason = (
            f"Basel={results[name].basel_zone}, "
            f"ratio偏差={dev:.2%}, Kupiec p={kupiec_p:.3g}"
        )
        ranking[name] = (i, reason)
    return ranking


def assemble_comparison(
    backtest_results: dict[str, BacktestResult],
    confidence_level: float,
) -> dict[str, ModelComparisonResult]:
    """
    Augment a dict of BacktestResult (one per model) with the comparison metrics
    required by Phase 9 and a model ranking.

    Extra metrics (computed over the out-of-sample backtest period):
      * average_var        – mean of the VaR series
      * var_volatility     – std of the VaR series
      * worst_daily_loss   – minimum realised return
      * expected_shortfall – empirical ES = mean realised return on exception days
    """
    ranking = _rank_models(backtest_results, confidence_level)
    out: dict[str, ModelComparisonResult] = {}
    for name in backtest_results:
        btr = backtest_results[name]
        var_series = btr.var_series
        actual = btr.actual_returns
        exc = btr.exceptions.astype(bool)
        es = float(actual[exc].mean()) if exc.any() else float(actual.min())
        out[name] = ModelComparisonResult(
            model_name=name,
            backtest=btr,
            average_var=float(var_series.mean()),
            var_volatility=float(var_series.std()),
            worst_daily_loss=float(actual.min()),
            expected_shortfall=es,
            rank=ranking[name][0],
            rank_reason=ranking[name][1],
        )
    return out


def compare_models(
    returns: pd.Series,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    portfolio_value: float = 1.0,
    models: list[str] | None = None,
) -> dict[str, ModelComparisonResult]:
    """
    Run the rolling out-of-sample backtest for every compared VaR model
    (Historical, EWMA, GARCH, GJR-GARCH) and return the full ModelComparisonResult
    set including ranking.
    """
    if models is None:
        models = list(MODEL_ORDER)
    unknown = [m for m in models if m not in MODEL_FUNCS]
    if unknown:
        raise ValueError(f"Unknown VaR model(s): {unknown}. Choose from {MODEL_ORDER}.")

    backtests: dict[str, BacktestResult] = {}
    for name in models:
        backtests[name] = backtest_model(
            returns,
            MODEL_FUNCS[name],
            model_name=name,
            confidence_level=confidence_level,
            estimation_window=estimation_window,
            portfolio_value=portfolio_value,
        )
    return assemble_comparison(backtests, confidence_level)
