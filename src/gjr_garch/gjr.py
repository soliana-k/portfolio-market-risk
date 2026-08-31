from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger

from src.garch_var.garch import fit_garch

TRADING_DAYS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_DISTRIBUTION = "normal"


@dataclass(frozen=True)
class GjrGarchResult:
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
    leverage_parameter: float
    alpha: float
    beta: float
    omega: float
    garch_conditional_volatility: pd.Series
    leverage_stats: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    model_summary: str = ""


def _rescale_params(params: "dict", scale: float) -> dict:
    """
    Convert GJR-GARCH parameters estimated on `returns * scale` back to the
    original (unscaled) return units.

    For a model y* = scale * y:
      mu*      = scale * mu        -> mu      = mu* / scale
      omega*   = scale**2 * omega  -> omega   = omega* / scale**2
      alpha, beta, gamma are scale-invariant.
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


def _leverage_diagnostics(
    returns: pd.Series, cond_vol: pd.Series, std_resid: pd.Series
) -> dict:
    """
    Summarise the leverage effect in the fitted data:

    - The GJR-GARCH asymmetry implies that the response of next-period variance
      to a negative shock is (alpha + gamma) while to a positive shock it is
      alpha. We report both response coefficients.
    - We also compare the *realised* average next-day conditional volatility
      following negative vs positive return days as an empirical sanity check.
    """
    r = returns
    neg = r < 0
    pos = r >= 0

    vol_after_neg = float(cond_vol[neg].mean()) if neg.any() else float("nan")
    vol_after_pos = float(cond_vol[pos].mean()) if pos.any() else float("nan")

    return {
        "n_negative_days": int(neg.sum()),
        "n_positive_days": int(pos.sum()),
        "avg_vol_after_negative": vol_after_neg,
        "avg_vol_after_positive": vol_after_pos,
        "vol_ratio_neg_over_pos": (
            vol_after_neg / vol_after_pos if vol_after_pos else float("nan")
        ),
    }


def fit_gjr_garch(
    returns: pd.Series,
    *,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = DEFAULT_DISTRIBUTION,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
    compare_garch: bool = True,
) -> GjrGarchResult:
    """
    Fit a GJR-GARCH(p,o,q) model (default 1,1,1, the threshold-GARCH of
    Glosten-Jagannathan-Runkle) to portfolio returns and compute a
    GJR-GARCH-scaled Historical Simulation VaR.

    The GJR-GARCH variance recursion is

        sigma_t^2 = omega
                    + alpha * eps_{t-1}^2
                    + gamma * eps_{t-1}^2 * I(eps_{t-1} < 0)
                    + beta  * sigma_{t-1}^2

    The asymmetry (leverage) parameter `gamma` makes negative shocks
    (eps_{t-1} < 0) raise future volatility more than positive shocks of the
    same magnitude: total negative-shock sensitivity is (alpha + gamma).

    Steps:
      1. Fit GJR-GARCH(p,o,q) to estimate conditional volatility sigma_t.
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
    logger.info(
        "GJR-GARCH fitting on {} observations | {} → {}",
        len(r), r.index.min().date(), r.index.max().date(),
    )

    scale = 100.0 if abs(r.std()) < 0.1 else 1.0
    scaled_r = r * scale

    # Allow the user to override the asymmetry order; o=0 reduces to plain GARCH.
    vol_kwargs = {"vol": "GARCH", "p": p, "o": o, "q": q}
    model = arch_model(scaled_r, mean=mean, dist=dist, rescale=False, **vol_kwargs)
    fitted = model.fit(disp="off")

    conditional_vol = (
        pd.Series(
            fitted.conditional_volatility,
            index=r.index,
            name="gjr_conditional_volatility",
        )
        / scale
    )
    std_resid = pd.Series(
        fitted.std_resid, index=r.index, name="standardized_residuals"
    )

    forecast = fitted.forecast(horizon=1)
    variance_forecast = float(forecast.variance.values[-1, -1]) / (scale ** 2)
    sigma_forecast = float(np.sqrt(variance_forecast))

    # GJR asymmetry / leverage parameter (gamma[1]). For o=0 it is 0.
    gamma_val = float(fitted.params.get("gamma[1]", 0.0))
    alpha_val = float(fitted.params.get("alpha[1]", np.nan))
    beta_val = float(fitted.params.get("beta[1]", np.nan))
    omega_val = float(fitted.params.get("omega", np.nan))

    logger.info(
        "GJR-GARCH({},{},{}) fit | omega={:.3e} alpha={:.4f} gamma={:.4f} beta={:.4f} | "
        "1-day sigma forecast={:.4%}",
        p, o, q, omega_val, alpha_val, gamma_val, beta_val, sigma_forecast,
    )

    scaled_returns = pd.Series(
        sigma_forecast * std_resid.values,
        index=r.index,
        name="gjr_scaled_returns",
    )

    alpha_q = 1.0 - confidence_level
    var_pct = float(np.quantile(scaled_returns.values, alpha_q))
    tail = scaled_returns[scaled_returns <= var_pct]
    es_pct = float(tail.mean()) if len(tail) > 0 else var_pct

    if portfolio_value is None:
        portfolio_value = 1.0
    var_dollar = abs(var_pct) * portfolio_value
    es_dollar = abs(es_pct) * portfolio_value

    # Plain GARCH(1,1) conditional volatility for side-by-side comparison.
    # Skipped during rolling backtests (compare_garch=False) to halve the cost.
    if compare_garch:
        garch_res = fit_garch(
            r, p=1, q=1, mean=mean, dist=dist,
            confidence_level=confidence_level, portfolio_value=portfolio_value,
        )
        garch_cond_vol = garch_res.conditional_volatility.reindex(r.index)
    else:
        garch_cond_vol = pd.Series(index=r.index, dtype=float, name="garch_conditional_volatility")

    leverage_stats = _leverage_diagnostics(r, conditional_vol, std_resid)

    logger.info(
        "GJR-GARCH-scaled Historical VaR ({:.0%}) = {:.4%} | ES = {:.4%} | "
        "leverage gamma={:.4f}",
        confidence_level, var_pct, es_pct, gamma_val,
    )

    return GjrGarchResult(
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
        leverage_parameter=gamma_val,
        alpha=alpha_val,
        beta=beta_val,
        omega=omega_val,
        garch_conditional_volatility=garch_cond_vol,
        leverage_stats=leverage_stats,
        params=_rescale_params(fitted.params, scale),
        model_summary=str(fitted.summary()),
    )


def gjr_garch_scaled_historical_var(
    returns: pd.Series,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    portfolio_value: float | None = None,
) -> dict:
    """
    Convenience wrapper returning just the GJR-GARCH-scaled Historical VaR metrics.
    """
    res = fit_gjr_garch(
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
        "leverage_parameter": res.leverage_parameter,
        "conditional_volatility": res.conditional_volatility,
        "garch_conditional_volatility": res.garch_conditional_volatility,
        "standardized_residuals": res.standardized_residuals,
        "scaled_returns": res.scaled_returns,
        "leverage_stats": res.leverage_stats,
    }
