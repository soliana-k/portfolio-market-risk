"""Offline / stale-data fallback tests (Phase: production hardening).

Verifies that when a live download fails, the cached wrappers fall back to the
last cached data frame and report staleness through the ``on_stale`` callback,
so the app can keep rendering instead of crashing.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_collection.download_portfolio import (
    download_asset_prices_cached,
    download_market_caps_cached,
)

SYMBOLS = ["AAPL"]
START, END = "2023-01-01", "2023-06-01"


def _fake_download(sym, **kwargs):
    idx = pd.bdate_range(START, END)
    return pd.DataFrame({"Close": pd.Series(100.0, index=idx)}, index=idx)


def _seed_prices(tmp: str) -> None:
    with (
        patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
        patch(
            "src.data_collection.download_portfolio.yf.download",
            side_effect=_fake_download,
        ),
    ):
        download_asset_prices_cached(
            tickers=SYMBOLS,
            start_date=START,
            end_date=END,
            force_refresh=True,
        )


def test_prices_fallback_to_stale_on_download_failure():
    with TemporaryDirectory() as tmp:
        with (
            patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
            patch(
                "src.data_collection.download_portfolio.yf.download",
                side_effect=_fake_download,
            ),
        ):
            download_asset_prices_cached(
                tickers=SYMBOLS,
                start_date=START,
                end_date=END,
                force_refresh=True,
            )

        stale_reported = []
        with (
            patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
            patch(
                "src.data_collection.download_portfolio.download_asset_prices",
                side_effect=RuntimeError("network down"),
            ),
        ):
            frame = download_asset_prices_cached(
                tickers=SYMBOLS,
                start_date=START,
                end_date=END,
                force_refresh=True,
                on_stale=lambda m: stale_reported.append(m.get("saved_at")),
            )

    assert len(frame) > 0
    assert "AAPL" in frame.columns
    assert len(stale_reported) == 1
    assert isinstance(stale_reported[0], float)  # epoch seconds


def test_prices_raise_when_no_cache_and_download_fails():
    with TemporaryDirectory() as tmp:
        with (
            patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
            patch(
                "src.data_collection.download_portfolio.download_asset_prices",
                side_effect=RuntimeError("network down"),
            ),
        ):
            with pytest.raises(RuntimeError):
                download_asset_prices_cached(
                    tickers=SYMBOLS,
                    start_date=START,
                    end_date=END,
                    force_refresh=True,
                )


def test_market_caps_fallback_to_stale_on_download_failure():
    with TemporaryDirectory() as tmp:
        with (
            patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
            patch(
                "src.data_collection.download_portfolio.download_market_caps",
                return_value=pd.Series({"AAPL": 2_900_000_000_000.0}),
            ),
        ):
            download_market_caps_cached(SYMBOLS, force_refresh=True)

        stale_reported = []
        with (
            patch("src.data_collection.cache.DEFAULT_CACHE_DIR", Path(tmp)),
            patch(
                "src.data_collection.download_portfolio.download_market_caps",
                side_effect=ValueError("rate limited"),
            ),
        ):
            caps = download_market_caps_cached(
                SYMBOLS,
                force_refresh=True,
                on_stale=lambda m: stale_reported.append(m.get("saved_at")),
            )

    assert caps.loc["AAPL"] > 0
    assert len(stale_reported) == 1