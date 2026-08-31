from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_LAMBDA = 0.94  # RiskMetrics EWMA decay factor


@dataclass(frozen=True)
class EwmaResult:
    returns: pd.Series
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    sigma_forecast: float
    variance_forecast: float
    scaled_returns: pd.Series
    var_pct: float
    var_dollar: float
    es_pct: float
    es_dollar: float
    confidence_level: float
    lambda_: float
    params: dict = field(default_factory=dict)
    model_summary: str = ""


def ewma_volatility(returns: np.ndarray, lambda_: float = DEFAULT_LAMBDA) -> np.ndarray:
    """
    Exponentially Weighted Moving-Average (RiskMetrics) variance recursion:

        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Initialised with the full-sample variance so the recursion is stable.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    var = np.zeros(n)
    var[0] = float(np.var(r)) if n > 1 else float(r[0] ** 2)
    for t in range(1, n):
        var[t] = lambda_ * var[t - 1] + (1.0 - lambda_) * r[t - 1] ** 2
    return np.sqrt(np.clip(var, 1e-300, None))


def fit_ewma(
    returns: pd.Series,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
) -> EwmaResult:
    """
    Fit an EWMA (RiskMetrics) volatility model and compute an EWMA-scaled
    Historical Simulation VaR.

    Steps:
      1. Estimate the EWMA conditional volatility sigma_t.
      2. Standardise returns z_t = r_t / sigma_t (i.i.d.-like innovations).
      3. Forecast 1-day-ahead volatility sigma_{t+1|t} from the recursion
         applied at t (lambda * sigma_t^2 + (1 - lambda) * r_t^2).
      4. Scale historical innovations by the forecast: r*_t = sigma_{t+1|t} * z_t.
      5. Read VaR / ES from the simulated r*_t distribution quantiles.
    """
    if returns is None or returns.empty:
        raise ValueError("returns cannot be empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1).")

    r = returns.dropna().astype(float)
    r.index = pd.to_datetime(r.index)
    r.name = "portfolio_returns"
    logger.info(
        "EWMA fitting on {} observations | lambda={} | {} -> {}",
        len(r), lambda_, r.index.min().date(), r.index.max().date(),
    )

    sigma = ewma_volatility(r.values, lambda_)
    cond_vol = pd.Series(sigma, index=r.index, name="ewma_volatility")
    std_resid = pd.Series(r.values / sigma, index=r.index, name="standardized_residuals")

    last_var = lambda_ * sigma[-1] ** 2 + (1.0 - lambda_) * r.values[-1] ** 2
    variance_forecast = float(last_var)
    sigma_forecast = float(np.sqrt(variance_forecast))

    scaled_returns = pd.Series(
        sigma_forecast * std_resid.values,
        index=r.index,
        name="ewma_scaled_returns",
    )

    alpha = 1.0 - confidence_level
    var_pct = float(np.quantile(scaled_returns.values, alpha))
    tail = scaled_returns[scaled_returns <= var_pct]
    es_pct = float(tail.mean()) if len(tail) > 0 else var_pct

    if portfolio_value is None:
        portfolio_value = 1.0
    var_dollar = abs(var_pct) * portfolio_value
    es_dollar = abs(es_pct) * portfolio_value

    logger.info(
        "EWMA-scaled Historical VaR ({:.0%}) = {:.4%} | ES = {:.4%} | "
        "1-day sigma forecast={:.4%}",
        confidence_level, var_pct, es_pct, sigma_forecast,
    )

    return EwmaResult(
        returns=r,
        conditional_volatility=cond_vol,
        standardized_residuals=std_resid,
        sigma_forecast=sigma_forecast,
        variance_forecast=variance_forecast,
        scaled_returns=scaled_returns,
        var_pct=var_pct,
        var_dollar=var_dollar,
        es_pct=es_pct,
        es_dollar=es_dollar,
        confidence_level=confidence_level,
        lambda_=lambda_,
        params={"lambda": float(lambda_), "omega_init_var": float(np.var(r.values))},
        model_summary=f"EWMA (RiskMetrics) lambda={lambda_}",
    )


def ewma_scaled_historical_var(
    returns: pd.Series,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
) -> dict:
    """Convenience wrapper returning just the EWMA-scaled Historical VaR metrics."""
    res = fit_ewma(
        returns,
        lambda_=lambda_,
        confidence_level=confidence_level,
        portfolio_value=portfolio_value,
    )
    return {
        "var_pct": res.var_pct,
        "var_dollar": res.var_dollar,
        "es_pct": res.es_pct,
        "es_dollar": res.es_dollar,
        "sigma_forecast": res.sigma_forecast,
        "conditional_volatility": res.conditional_volatility,
        "standardized_residuals": res.standardized_residuals,
        "scaled_returns": res.scaled_returns,
    }
