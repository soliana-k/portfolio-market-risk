"""
Phase 3 – Portfolio Construction

Supports:
  • Equal-weighted
  • User-defined weights
  • Market-cap weighted
  • Long-short (optional)

Produces daily portfolio value, P&L and returns with optional rebalancing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data_collection.download_portfolio import download_market_caps

RebalanceFreq = Literal["none", "daily", "weekly", "monthly", "quarterly"]
WeightScheme = Literal["equal", "user", "market_cap", "long_short"]


@dataclass
class PortfolioResult:
    portfolio_value: pd.Series
    portfolio_pnl: pd.Series
    portfolio_returns: pd.Series
    weights_history: pd.DataFrame
    holdings: pd.DataFrame
    target_weights: pd.Series
    rebalance_dates: pd.DatetimeIndex
    metadata: dict = field(default_factory=dict)


def _normalize_weights(w: pd.Series, allow_negative: bool = False) -> pd.Series:
    w = w.astype(float).copy()
    if allow_negative:
        gross = w.abs().sum()
        if gross < 1e-12:
            raise ValueError("Long-short weights have zero gross exposure.")
        return w / gross
    total = w.sum()
    if abs(total) < 1e-12:
        raise ValueError("Weights sum to zero.")
    return w / total


def make_equal_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    if n == 0:
        raise ValueError("No tickers supplied for equal weights.")
    return pd.Series(1.0 / n, index=tickers, dtype=float)


def make_market_cap_weights(tickers: list[str]) -> pd.Series:
    caps = download_market_caps(tickers)
    caps = caps.reindex(tickers).fillna(0.0)
    if caps.sum() <= 0:
        raise ValueError("All market caps are zero / missing – cannot build weights.")
    return (caps / caps.sum()).astype(float)


def make_user_weights(tickers: list[str], weight_dict: Dict[str, float]) -> pd.Series:
    w = pd.Series({t: float(weight_dict.get(t, 0.0)) for t in tickers})
    return _normalize_weights(w, allow_negative=False)


def make_long_short_weights(
    tickers: list[str],
    long_tickers: list[str],
    short_tickers: list[str],
    long_weight: float = 0.5,
    short_weight: float = 0.5,
) -> pd.Series:
    if not long_tickers and not short_tickers:
        raise ValueError("Need at least one long or short ticker.")
    w = pd.Series(0.0, index=tickers, dtype=float)
    if long_tickers:
        w[long_tickers] = long_weight / len(long_tickers)
    if short_tickers:
        w[short_tickers] = -short_weight / len(short_tickers)
    return _normalize_weights(w, allow_negative=True)


def _rebalance_mask(dates: pd.DatetimeIndex, freq: RebalanceFreq) -> pd.Series:
    if freq == "none":
        mask = pd.Series(False, index=dates)
        mask.iloc[0] = True
        return mask

    if freq == "daily":
        return pd.Series(True, index=dates)

    if freq == "weekly":
        period = dates.to_period("W")
    elif freq == "monthly":
        period = dates.to_period("M")
    elif freq == "quarterly":
        period = dates.to_period("Q")
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")

    is_first = ~pd.Series(period, index=dates).duplicated(keep="first")
    mask = is_first.copy()
    mask.iloc[0] = True
    return mask


def construct_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    initial_value: float = 100_000.0,
    rebalance_freq: RebalanceFreq = "monthly",
    transaction_cost_bps: float = 0.0,
) -> PortfolioResult:
    if prices.empty:
        raise ValueError("prices cannot be empty.")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")

    prices = prices.sort_index()
    w = target_weights.reindex(prices.columns).astype(float)
    if w.isna().any():
        missing = w[w.isna()].index.tolist()
        raise ValueError(f"Missing target weights for: {missing}")

    if (w < 0).any():
        pass  # long-short already normalised
    else:
        if not np.isclose(w.sum(), 1.0, atol=1e-4):
            w = w / w.sum()

    dates = prices.index
    rebal = _rebalance_mask(dates, rebalance_freq)
    rebal_dates = dates[rebal]

    n = len(dates)
    n_assets = prices.shape[1]

    port_value = np.zeros(n)
    holdings = np.zeros((n, n_assets))
    weights_hist = np.zeros((n, n_assets))
    daily_ret = np.zeros(n)
    daily_pnl = np.zeros(n)

    px0 = prices.iloc[0].values
    px0 = np.where(px0 <= 0, np.nan, px0)
    if np.isnan(px0).any():
        raise ValueError("Non-positive prices on first day.")

    cash = initial_value
    shares = (w.values * cash) / px0
    holdings[0] = shares
    port_value[0] = float(np.nansum(shares * px0))
    weights_hist[0] = w.values
    daily_ret[0] = 0.0
    daily_pnl[0] = 0.0

    prev_value = port_value[0]

    for i in range(1, n):
        px = prices.iloc[i].values
        px = np.where(px <= 0, np.nan, px)

        mtm = float(np.nansum(holdings[i - 1] * px))
        if np.isnan(mtm) or mtm <= 0:
            mtm = prev_value

        if rebal.iloc[i]:
            target_dollars = w.values * mtm
            new_shares = target_dollars / px
            turnover = np.nansum(np.abs(new_shares - holdings[i - 1]) * px)
            cost = turnover * (transaction_cost_bps / 10_000.0)
            mtm_after_cost = mtm - cost
            if mtm_after_cost > 0:
                new_shares = (w.values * mtm_after_cost) / px
            holdings[i] = new_shares
            port_value[i] = float(np.nansum(new_shares * px))
        else:
            holdings[i] = holdings[i - 1]
            port_value[i] = mtm

        daily_pnl[i] = port_value[i] - prev_value
        daily_ret[i] = daily_pnl[i] / prev_value if prev_value > 0 else 0.0
        weights_hist[i] = (
            (holdings[i] * px) / port_value[i] if port_value[i] > 0 else w.values
        )
        prev_value = port_value[i]

    result = PortfolioResult(
        portfolio_value=pd.Series(port_value, index=dates, name="portfolio_value"),
        portfolio_pnl=pd.Series(daily_pnl, index=dates, name="portfolio_pnl"),
        portfolio_returns=pd.Series(daily_ret, index=dates, name="portfolio_returns"),
        weights_history=pd.DataFrame(weights_hist, index=dates, columns=prices.columns),
        holdings=pd.DataFrame(holdings, index=dates, columns=prices.columns),
        target_weights=w,
        rebalance_dates=rebal_dates,
        metadata={
            "initial_value": initial_value,
            "rebalance_freq": rebalance_freq,
            "transaction_cost_bps": transaction_cost_bps,
            "n_rebalances": int(rebal.sum()),
            "final_value": float(port_value[-1]),
            "total_return": float(port_value[-1] / initial_value - 1.0),
            "n_days": n,
        },
    )
    logger.info(
        "Portfolio constructed | final={:,.2f} | total_return={:.2%} | rebalances={}",
        result.metadata["final_value"],
        result.metadata["total_return"],
        result.metadata["n_rebalances"],
    )
    return result


def resolve_target_weights(
    scheme: WeightScheme,
    tickers: list[str],
    user_weights: Optional[Dict[str, float]] = None,
    long_tickers: Optional[list[str]] = None,
    short_tickers: Optional[list[str]] = None,
    long_weight: float = 0.5,
    short_weight: float = 0.5,
) -> pd.Series:
    if scheme == "equal":
        return make_equal_weights(tickers)
    if scheme == "user":
        if not user_weights:
            raise ValueError("user_weights required for scheme='user'")
        return make_user_weights(tickers, user_weights)
    if scheme == "market_cap":
        return make_market_cap_weights(tickers)
    if scheme == "long_short":
        return make_long_short_weights(
            tickers,
            long_tickers or [],
            short_tickers or [],
            long_weight=long_weight,
            short_weight=short_weight,
        )
    raise ValueError(f"Unknown weight scheme: {scheme}")