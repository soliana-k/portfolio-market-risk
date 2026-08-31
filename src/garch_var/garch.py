from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger

TRADING_DAYS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_DISTRIBUTION = "normal"


@dataclass(frozen=True)
class GarchResult:
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
    params: dict = field(default_factory=dict)
    model_summary: str = ""


def fit_garch(
    returns: pd.Series,
    *,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = DEFAULT_DISTRIBUTION,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
) -> GarchResult:
    """
    Fit a GARCH(1,1) model (default) to portfolio returns and compute a
    GARCH-scaled Historical Simulation VaR.

    Steps:
      1. Fit GARCH(p,q) to estimate conditional volatility sigma_t.
      2. Standardise returns z_t = r_t / sigma_t (i.i.d.-like innovations).
      3. Forecast 1-day-ahead conditional volatility sigma_{t+1|t}.
      4. Scale historical innovations by the forecast volatility to build a
         simulated 1-day return distribution: r*_t = sigma_{t+1|t} * z_t.
      5. Read VaR / ES from the simulated distribution quantiles.
    """
    if returns is None or returns.empty:
        raise ValueError("returns cannot be empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1).")

    r = returns.dropna().astype(float)
    r.index = pd.to_datetime(r.index)
    r.name = "portfolio_returns"
    logger.info("GARCH fitting on {} observations | {} → {}", len(r), r.index.min().date(), r.index.max().date())

    scale = 100.0 if abs(r.std()) < 0.1 else 1.0
    scaled_r = r * scale

    model = arch_model(scaled_r, vol="Garch", p=p, q=q, mean=mean, dist=dist, rescale=False)
    fitted = model.fit(disp="off")

    conditional_vol = pd.Series(fitted.conditional_volatility, index=r.index, name="conditional_volatility") / scale
    std_resid = pd.Series(fitted.std_resid, index=r.index, name="standardized_residuals")

    forecast = fitted.forecast(horizon=1)
    variance_forecast = float(forecast.variance.values[-1, -1]) / (scale ** 2)
    sigma_forecast = float(np.sqrt(variance_forecast))

    logger.info(
        "GARCH({},{}) fit | omega={:.3e} alpha={:.4f} beta={:.4f} | "
        "1-day sigma forecast={:.4%}",
        p, q,
        float(fitted.params.get("omega", np.nan)),
        float(fitted.params.get("alpha[1]", np.nan)),
        float(fitted.params.get("beta[1]", np.nan)),
        sigma_forecast,
    )

    scaled_returns = pd.Series(
        sigma_forecast * std_resid.values,
        index=r.index,
        name="garch_scaled_returns",
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
        "GARCH-scaled Historical VaR ({:.0%}) = {:.4%} | ES = {:.4%}",
        confidence_level, var_pct, es_pct,
    )

    return GarchResult(
        returns=r,
        conditional_volatility=conditional_vol,
        standardized_residuals=std_resid,
        sigma_forecast=sigma_forecast,
        variance_forecast=variance_forecast,
        scaled_returns=scaled_returns,
        var_pct=var_pct,
        var_dollar=var_dollar,
        es_pct=es_pct,
        es_dollar=es_dollar,
        confidence_level=confidence_level,
        params=_rescale_params(fitted.params, scale),
        model_summary=str(fitted.summary()),
    )


def _rescale_params(params: "dict", scale: float) -> dict:
    """
    Convert GARCH parameters estimated on `returns * scale` back to the
    original (unscaled) return units.

    For a model y* = scale * y:
      mu*      = scale * mu        -> mu      = mu* / scale
      omega*   = scale**2 * omega  -> omega   = omega* / scale**2
      alpha, beta are scale-invariant.
    """
    if scale == 1.0:
        return {k: float(v) for k, v in params.items()}
    out = {}
    for k, v in params.items():
        if k == "mu":
            out[k] = float(v) / scale
        elif k == "omega":
            out[k] = float(v) / (scale ** 2)
        else:
            out[k] = float(v)
    return out


def garch_scaled_historical_var(
    returns: pd.Series,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
) -> dict:
    """
    Convenience wrapper returning just the GARCH-scaled Historical VaR metrics.
    """
    res = fit_garch(
        returns,
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
