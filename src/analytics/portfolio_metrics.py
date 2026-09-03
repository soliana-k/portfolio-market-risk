"""
Portfolio analytics and dashboard metric helpers.

Provides pure, testable computations used by the app dashboard:
  * correlation / covariance structure
  * diversification & concentration statistics
  * multi-confidence Value-at-Risk / Expected Shortfall
  * per-asset return attribution
  * drawdown analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def asset_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of asset returns (columns = assets)."""
    return returns.corr()


def expected_shortfall_from_pnl_losses(losses: np.ndarray, alpha: float) -> float:
    """ES = mean of losses that are beyond (worse than) the alpha-quantile."""
    if losses.size == 0:
        return 0.0
    var_level = np.quantile(losses, alpha)  # negative P&L
    tail = losses[losses <= var_level]
    return float(tail.mean()) if tail.size else float(var_level)


def multi_confidence_var_es(
    portfolio_returns: pd.Series,
    portfolio_values: pd.Series,
    confidences: list[float],
) -> pd.DataFrame:
    """
    Historical VaR & ES at several confidence levels.

    Returns a tidy DataFrame with columns:
      [confidence, var_pct, var_dollar, es_pct, es_dollar]
    """
    current_val = float(portfolio_values.iloc[-1]) if len(portfolio_values) else 1.0
    rows = []
    for conf in confidences:
        alpha = 1.0 - conf
        var_pct = float(np.quantile(portfolio_returns.values, alpha))
        es_pct = expected_shortfall_from_pnl_losses(portfolio_returns.values, alpha)
        rows.append({
            "confidence": f"{conf:.1%}",
            "conf_level": conf,
            "var_pct": abs(var_pct),
            "var_dollar": abs(var_pct) * current_val,
            "es_pct": abs(es_pct),
            "es_dollar": abs(es_pct) * current_val,
        })
    return pd.DataFrame(rows)


def herfindahl_index(weights: pd.Series) -> float:
    """HHI of the weight vector (1/N for equal weights, 1.0 for a single asset)."""
    w = weights.astype(float)
    if w.sum() == 0:
        return np.nan
    norm = w / w.sum()
    return float((norm ** 2).sum())


def effective_number_of_assets(weights: pd.Series) -> float:
    """Effective N = 1 / HHI. Ranges from 1 to the actual number of assets."""
    hhi = herfindahl_index(weights)
    return float(1.0 / hhi) if hhi and hhi > 0 else np.nan


def concentration_metrics(weights: pd.Series) -> dict:
    """Concentration / diversification statistics given target weights."""
    weights = weights.astype(float)
    hhi = herfindahl_index(weights)
    eff_n = effective_number_of_assets(weights)
    top1 = float(weights.abs().max())
    return {
        "HHI": hhi,
        "Effective N": eff_n,
        "Largest weight": top1,
    }


def portfolio_variance(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Variance of the portfolio given weights aligned to the covariance matrix."""
    w = weights.reindex(cov.columns).fillna(0.0).values
    return float(w @ cov.values @ w)


def asset_return_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Per-asset contribution to total portfolio return.

    Weighted return of each asset plus proportional share of the buy-and-hold
    portfolio return, expressed in percentage points (so values sum to the
    total period return in %).

    Returns a DataFrame with columns [asset, total_return, contribution_pct,
    contribution_share].
    """
    w = weights.reindex(returns.columns).fillna(0.0)
    asset_returns = (1 + returns).prod() - 1.0  # cumulative per asset
    weighted = (asset_returns * w)

    total = float(weighted.sum())
    if abs(total) < 1e-12:
        shares = np.full(len(asset_returns), np.nan)
    else:
        shares = (weighted / total).values

    out = pd.DataFrame({
        "asset": asset_returns.index,
        "total_return": asset_returns.values,
        "weight": w.values,
        "contribution_pct": weighted.values * 100.0,
        "contribution_share": shares,
    })
    return out.sort_values("contribution_share", ascending=False).reset_index(drop=True)


def drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    """Drawdown time series (negative values)."""
    return portfolio_value / portfolio_value.cummax() - 1.0


def rolling_exception_series(exceptions: pd.Series, window: int = 60) -> pd.Series:
    """Rolling exception rate (share of VaR breaches) over a window."""
    return exceptions.astype(float).rolling(window, min_periods=1).mean()
