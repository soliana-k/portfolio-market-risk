from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats.mstats import winsorize

TRADING_DAYS_PER_YEAR = 252
DEFAULT_STALE_WINDOW = 3
DEFAULT_WINSOR_LIMITS = (0.01, 0.01)
DEFAULT_ROLLING_VOL_WINDOW = 21
WEIGHT_SUM_TOLERANCE = 1e-6




@dataclass(frozen=True)
class DataOverview:
    n_rows: int
    n_columns: int
    date_start: str
    date_end: str
    tickers: list[str]
    missing_count: int
    missing_pct: float
    missing_by_ticker: pd.Series
    non_positive_count: int
    describe: pd.DataFrame
    index_freq_hint: str

    def summary_text(self) -> str:
        lines = [
            "=" * 70,
            "DATA OVERVIEW",
            "=" * 70,
            f"Shape            : {self.n_rows} rows × {self.n_columns} columns",
            f"Date range       : {self.date_start} → {self.date_end}",
            f"Tickers          : {', '.join(self.tickers)}",
            f"Missing cells    : {self.missing_count} ({self.missing_pct:.2f}%)",
            f"Non-positive     : {self.non_positive_count}",
            f"Index freq hint  : {self.index_freq_hint}",
            "-" * 70,
            "Missing by ticker:",
            self.missing_by_ticker.to_string(),
            "-" * 70,
            "Describe:",
            self.describe.to_string(),
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CleanedDataResult:
    clean_prices: pd.DataFrame
    clean_returns: pd.DataFrame
    portfolio_returns: pd.Series
    rolling_volatility: pd.Series
    weights: pd.Series
    n_missing_filled: int = 0
    n_stale_removed: int = 0
    n_returns_winsorized: int = 0
    metadata: dict = field(default_factory=dict)


def data_overview(df: pd.DataFrame) -> DataOverview:
    
    if df is None or df.empty:
        raise ValueError("df cannot be empty for data overview.")

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    missing_by_ticker = work.isna().sum().astype(int)
    missing_count = int(missing_by_ticker.sum())
    total = work.shape[0] * work.shape[1]
    missing_pct = (missing_count / total * 100.0) if total else 0.0
    non_positive = int((work <= 0).sum().sum())

    if len(work.index) >= 3:
        inferred = pd.infer_freq(work.index)
        freq_hint = inferred if inferred else "irregular / mixed"
    else:
        freq_hint = "too short to infer"

    return DataOverview(
        n_rows=work.shape[0],
        n_columns=work.shape[1],
        date_start=str(work.index.min().date()),
        date_end=str(work.index.max().date()),
        tickers=list(work.columns.astype(str)),
        missing_count=missing_count,
        missing_pct=round(missing_pct, 4),
        missing_by_ticker=missing_by_ticker,
        non_positive_count=non_positive,
        describe=work.describe().T,
        index_freq_hint=freq_hint,
    )


def print_overview(df: pd.DataFrame) -> DataOverview:
    
    ov = data_overview(df)
    print(ov.summary_text())
    return ov


def validate_prices(prices: pd.DataFrame) -> None:
    
    if prices is None or prices.empty:
        raise ValueError("prices cannot be empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            f"prices index must be a DatetimeIndex. Got {type(prices.index).__name__}"
        )
    if not all(pd.api.types.is_numeric_dtype(prices[c]) for c in prices.columns):
        raise ValueError("All price columns must be numeric.")
    if (prices <= 0).any().any():
        bad = int(prices.le(0).sum().sum())
        raise ValueError(f"prices must be strictly positive. Found {bad} bad value(s).")


def _validate_weights(weights: pd.Series, tickers: pd.Index) -> pd.Series:
    if weights is None or weights.empty:
        raise ValueError("weights cannot be empty.")
    aligned = weights.reindex(tickers)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"Missing weights for ticker(s): {missing}")
    total = float(aligned.sum())
    if not np.isclose(total, 1.0, atol=WEIGHT_SUM_TOLERANCE):
        raise ValueError(
            f"Portfolio weights must sum to 1.0. Received total weight: {total:.6f}"
        )
    return aligned.astype(float)



def align_trading_dates(prices: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    
    if prices.empty:
        raise ValueError("prices cannot be empty for date alignment.")
    df = prices.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    return df.reindex(full_range)


def handle_missing_values(prices: pd.DataFrame, method: str = "ffill_bfill") -> tuple[pd.DataFrame, int]:

    if prices.empty:
        raise ValueError("prices cannot be empty.")
    n_missing = int(prices.isna().sum().sum())
    df = prices.copy()
    if method == "ffill_bfill":
        df = df.ffill().bfill()
    elif method == "ffill":
        df = df.ffill()
    elif method == "drop":
        df = df.ffill().dropna(how="any")
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'ffill_bfill', 'ffill', or 'drop'."
        )
    residual = int(df.isna().sum().sum())
    if residual > 0:
        raise ValueError(f"After '{method}', {residual} missing value(s) remain.")
    return df, n_missing


def remove_stale_prices(prices: pd.DataFrame, window: int = DEFAULT_STALE_WINDOW) -> tuple[pd.DataFrame, int]:
    if prices.empty:
        raise ValueError("prices cannot be empty.")
    if window < 2:
        raise ValueError("stale window must be >= 2.")
    df = prices.copy()
    unchanged = df.diff() == 0
    run_threshold = window - 1
    stale_mask = (unchanged.rolling(window=run_threshold, min_periods=run_threshold).sum()>= run_threshold)
    n_stale = int(stale_mask.sum().sum())
    if n_stale > 0:
        df = df.mask(stale_mask).ffill()
        if int(df.isna().sum().sum()) > 0:
            df = df.bfill()
    return df, n_stale


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    validate_prices(prices)
    return np.log(prices / prices.shift(1)).dropna(how="all")


def winsorize_returns(returns: pd.DataFrame, limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS) -> tuple[pd.DataFrame, int]:
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    out = returns.copy()
    n_clipped = 0
    for col in out.columns:
        series = out[col].dropna()
        if series.empty:
            continue
        clipped = winsorize(series.to_numpy(), limits=limits)
        n_clipped += int(np.sum(series.to_numpy() != np.asarray(clipped)))
        out.loc[series.index, col] = np.asarray(clipped)
    return out, n_clipped


def build_weighted_portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty.")
    aligned = _validate_weights(weights, asset_returns.columns)
    port = asset_returns.mul(aligned, axis=1).sum(axis=1)
    port.name = "portfolio_returns"
    return port


def calculate_rolling_volatility(returns: pd.Series, window: int = DEFAULT_ROLLING_VOL_WINDOW, annualize: bool = True) -> pd.Series:
    
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    if window < 2:
        raise ValueError("rolling window must be >= 2.")
    vol = returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol.name = "rolling_volatility"
    return vol


def run_data_cleaning_pipeline(
    prices: pd.DataFrame,
    weights: pd.Series,
    *,
    align_dates: bool = True,
    missing_method: str = "ffill_bfill",
    stale_window: int = DEFAULT_STALE_WINDOW,
    apply_winsorize: bool = False,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
    vol_window: int = DEFAULT_ROLLING_VOL_WINDOW,
) -> CleanedDataResult:
    
    logger.info("=" * 60)
    logger.info("Phase 2 – Data Cleaning pipeline started")
    logger.info("=" * 60)

    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, dtype=float)

    df = prices.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info(
        "Input prices: {} rows × {} tickers | {} → {}",
        df.shape[0],
        df.shape[1],
        df.index.min().date(),
        df.index.max().date(),
    )
    logger.info("Weights sum: {:.6f} | tickers: {}", float(weights.sum()), list(weights.index))

    if align_dates:
        logger.info("[1/7] Aligning trading dates (freq='B') ...")
        before = df.shape[0]
        df = align_trading_dates(df)
        logger.info(
            "        {} → {} rows ({} calendar gaps introduced)",
            before,
            df.shape[0],
            df.shape[0] - before,
        )
    else:
        logger.info("[1/7] Date alignment skipped")

    logger.info("[2/7] Handling missing values (method='{}') ...", missing_method)
    df, n_missing = handle_missing_values(df, method=missing_method)
    logger.info("        Filled {} missing cell(s)", n_missing)

    logger.info("[3/7] Removing stale prices (window={}) ...", stale_window)
    df, n_stale = remove_stale_prices(df, window=stale_window)
    if n_stale:
        logger.warning("        Replaced {} stale cell(s)", n_stale)
    else:
        logger.info("        No stale runs found")

    logger.info("[4/7] Validating clean prices ...")
    validate_prices(df)
    logger.info("        OK – all prices positive, numeric, DatetimeIndex")

    logger.info("[5/7] Computing log returns ...")
    log_returns = calculate_log_returns(df)
    logger.info(
        "        {} observations | mean={:.6f} | std={:.6f}",
        len(log_returns),
        float(log_returns.stack().mean()),
        float(log_returns.stack().std()),
    )

    n_winsor = 0
    if apply_winsorize:
        logger.info("[6/7] Winsorising returns (limits={}) ...", winsor_limits)
        log_returns, n_winsor = winsorize_returns(log_returns, limits=winsor_limits)
        logger.info("        Clipped {} observation(s)", n_winsor)
    else:
        logger.info("[6/7] Winsorisation skipped")

    logger.info("[7/7] Building portfolio returns & rolling volatility ...")
    port_returns = build_weighted_portfolio_returns(log_returns, weights)
    rolling_vol = calculate_rolling_volatility(port_returns, window=vol_window)
    logger.info(
        "        Portfolio return mean={:.6f} | std={:.6f} | "
        "worst day={:.4%} | best day={:.4%}",
        float(port_returns.mean()),
        float(port_returns.std()),
        float(port_returns.min()),
        float(port_returns.max()),
    )
    vol_valid = rolling_vol.dropna()
    if not vol_valid.empty:
        logger.info(
            "        Rolling vol (ann.): last={:.2%} | mean={:.2%} | max={:.2%}",
            float(vol_valid.iloc[-1]),
            float(vol_valid.mean()),
            float(vol_valid.max()),
        )

    result = CleanedDataResult(
        clean_prices=df,
        clean_returns=log_returns,
        portfolio_returns=port_returns,
        rolling_volatility=rolling_vol,
        weights=_validate_weights(weights, df.columns),
        n_missing_filled=n_missing,
        n_stale_removed=n_stale,
        n_returns_winsorized=n_winsor,
        metadata={
            "align_dates": align_dates,
            "missing_method": missing_method,
            "stale_window": stale_window,
            "apply_winsorize": apply_winsorize,
            "vol_window": vol_window,
            "n_assets": df.shape[1],
            "n_dates": df.shape[0],
            "return_start": str(log_returns.index.min().date()),
            "return_end": str(log_returns.index.max().date()),
        },
    )

    logger.info("=" * 60)
    logger.info(
        "Phase 2 complete | missing_filled={} | stale_removed={} | winsorised={}",
        n_missing,
        n_stale,
        n_winsor,
    )
    logger.info("=" * 60)
    return result