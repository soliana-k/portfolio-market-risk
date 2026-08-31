from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_CONFIDENCE_LEVEL = 0.95

# ── Historical stress scenarios ──────────────────────────────────────────────
# `equity_shock` is the representative broad-equity drawdown used when the
# portfolio's own return history does NOT cover the scenario window (fallback).
# When the window IS covered, the portfolio's actual cumulative return is used.
HISTORICAL_SCENARIOS = {
    "2008 Financial Crisis": dict(
        start="2008-09-01", end="2009-03-09", equity_shock=-0.45,
        desc="Global financial crisis; S&P 500 fell ~50% peak-to-trough.",
    ),
    "COVID Crash 2020": dict(
        start="2020-02-19", end="2020-03-23", equity_shock=-0.34,
        desc="Pandemic crash; S&P 500 dropped ~34% in five weeks.",
    ),
    "Inflation Shock 2022": dict(
        start="2022-01-03", end="2022-09-30", equity_shock=-0.25,
        desc="Rate-hike / inflation shock; S&P 500 down ~25% YTD.",
    ),
    "Banking Stress 2023": dict(
        start="2023-03-01", end="2023-03-13", equity_shock=-0.10,
        desc="Regional-banking crisis (SVB / Credit Suisse); broad equity lower, banks -30%+.",
    ),
    "Tech Selloff": dict(
        start="2022-01-03", end="2022-10-14", equity_shock=-0.30,
        desc="Growth / tech de-rating; Nasdaq fell ~35% through 2022.",
    ),
}

# ── Hypothetical stress scenarios ────────────────────────────────────────────
HYPOTHETICAL_SCENARIOS = {
    "Equity -5%": dict(equity_shock=-0.05),
    "Equity -10%": dict(equity_shock=-0.10),
    "Equity -20%": dict(equity_shock=-0.20),
    "Volatility x2": dict(vol_multiplier=2.0),
    "Correlation Spike": dict(corr_multiplier=1.5),
    "Sector Shock": dict(sector_shock=-0.15),
}


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    scenario_type: str
    description: str
    shock_return_pct: float
    stressed_var_pct: float
    stressed_es_pct: float
    stressed_loss_value: float
    baseline_var_pct: float
    worst_case: bool = False
    sector_contrib: "pd.Series | None" = None
    source: str = "representative"


@dataclass(frozen=True)
class StressTestResults:
    scenarios: dict = field(default_factory=dict)
    ranking: list = field(default_factory=list)  # scenario names, worst first
    worst_case_name: str = ""
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL


def _historical_window_return(portfolio_returns: pd.Series, start: str, end: str):
    """Cumulative portfolio return over [start, end] if covered by the data."""
    idx = portfolio_returns.index
    if len(idx) == 0:
        return None
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if idx.min() <= lo and idx.max() >= hi:
        sub = portfolio_returns.loc[lo:hi]
        if len(sub) > 1:
            return float((1.0 + sub).prod() - 1.0)
    return None


def run_stress_tests(
    portfolio_value: float,
    weights: pd.Series,
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    baseline_var_pct: float,
    baseline_es_pct: float,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> StressTestResults:
    """
    Apply every historical and hypothetical stress scenario and compute:

      * shock_return_pct  – instantaneous portfolio return impact (<= 0)
      * stressed_var_pct  – stressed 1-day VaR (loss, positive number)
      * stressed_es_pct   – stressed 1-day Expected Shortfall
      * stressed_loss_value – potential $ loss = portfolio_value * stressed_var_pct
      * worst_case        – the single most severe scenario

    Historical scenarios use the portfolio's *actual* realised loss over the
    crisis window when available, otherwise a representative equity drawdown.
    Volatility / correlation shocks scale the baseline (model) VaR.
    The sector shock hits the largest single position.
    """
    if abs(baseline_var_pct) < 1e-12:
        baseline_var_pct = -0.02
    es_ratio = (baseline_es_pct / baseline_var_pct) if baseline_var_pct < 0 else 1.3

    results: dict[str, ScenarioResult] = {}

    # ── Historical ──
    for name, spec in HISTORICAL_SCENARIOS.items():
        actual = _historical_window_return(portfolio_returns, spec["start"], spec["end"])
        if actual is not None:
            shock = actual
            source = "actual portfolio return"
        else:
            shock = spec["equity_shock"]
            source = "representative equity drawdown (out-of-sample)"
        shock = min(shock, 0.0)
        var_pct = abs(shock)
        es_pct = var_pct * es_ratio
        results[name] = ScenarioResult(
            name=name, scenario_type="historical", description=spec["desc"],
            shock_return_pct=shock, stressed_var_pct=var_pct,
            stressed_es_pct=es_pct,
            stressed_loss_value=portfolio_value * var_pct,
            baseline_var_pct=baseline_var_pct, source=source,
        )
        logger.info("Stress [{}]: shock={:.2%} stressedVaR={:.2%} ({}).", name, shock, var_pct, source)

    # ── Hypothetical ──
    for name, spec in HYPOTHETICAL_SCENARIOS.items():
        sector_contrib = None
        if "equity_shock" in spec:
            shock = spec["equity_shock"]
            var_pct = abs(shock)
            source = "equity shock"
        elif "vol_multiplier" in spec:
            shock = 0.0
            var_pct = abs(baseline_var_pct) * spec["vol_multiplier"]
            source = f"volatility x{spec['vol_multiplier']}"
        elif "corr_multiplier" in spec:
            shock = 0.0
            var_pct = abs(baseline_var_pct) * spec["corr_multiplier"]
            source = f"correlation spike x{spec['corr_multiplier']}"
        elif "sector_shock" in spec:
            w = weights.abs().sort_values(ascending=False)
            top_asset = w.index[0]
            top_w = float(w.iloc[0])
            shock = top_w * spec["sector_shock"]
            var_pct = abs(shock)
            # Per-asset contribution (only the largest position is shocked).
            contrib = pd.Series(0.0, index=weights.index, name="shock_loss")
            contrib[top_asset] = portfolio_value * abs(shock)
            sector_contrib = contrib
            source = f"sector shock on {top_asset}"
        else:
            continue
        es_pct = var_pct * es_ratio
        results[name] = ScenarioResult(
            name=name, scenario_type="hypothetical", description=source,
            shock_return_pct=shock, stressed_var_pct=var_pct,
            stressed_es_pct=es_pct,
            stressed_loss_value=portfolio_value * var_pct,
            baseline_var_pct=baseline_var_pct, source=source,
            sector_contrib=sector_contrib,
        )
        logger.info("Stress [{}]: shock={:.2%} stressedVaR={:.2%} ({}).", name, shock, var_pct, source)

    # ── Ranking (most severe first) ──
    ranking = sorted(results.values(), key=lambda r: r.stressed_loss_value, reverse=True)
    for i, r in enumerate(ranking):
        # Rebuild frozen dataclass with worst_case flag set.
        results[r.name] = ScenarioResult(
            name=r.name, scenario_type=r.scenario_type, description=r.description,
            shock_return_pct=r.shock_return_pct, stressed_var_pct=r.stressed_var_pct,
            stressed_es_pct=r.stressed_es_pct, stressed_loss_value=r.stressed_loss_value,
            baseline_var_pct=r.baseline_var_pct, worst_case=(i == 0),
            sector_contrib=r.sector_contrib, source=r.source,
        )
    worst_name = ranking[0].name if ranking else ""

    return StressTestResults(
        scenarios={r.name: results[r.name] for r in ranking},
        ranking=[r.name for r in ranking],
        worst_case_name=worst_name,
        confidence_level=confidence_level,
    )
