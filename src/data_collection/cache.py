"""
File-based caching for downloaded market data.

Downloads are expensive and rate-limit prone (Yahoo Finance), so we cache the
raw price and market-cap data under ``data/raw`` and only refresh when the
inputs change or the cache is stale.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
DEFAULT_TTL_HOURS = 24.0


def _cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _meta_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _data_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.csv"


def _is_fresh(meta_path: Path, ttl_hours: float) -> bool:
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return bool(meta.get("fresh")) or (time.time() - float(meta.get("saved_at", 0))) < ttl_hours * 3600
    except Exception:
        return False


def load_cached(cache_key: str, cache_dir: Optional[Path] = None, ttl_hours: float = DEFAULT_TTL_HOURS):
    """Return (cached_frame, meta) if a fresh cache entry exists, else (None, None)."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    data_p, meta_p = _data_path(cache_dir, cache_key), _meta_path(cache_dir, cache_key)
    if not data_p.exists() or not _is_fresh(meta_p, ttl_hours):
        return None, None
    try:
        df = pd.read_csv(data_p, index_col=0, parse_dates=True)
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta
    except Exception:
        return None, None


def load_cached_stale(cache_key: str, cache_dir: Optional[Path] = None):
    """Load a cache entry even if it is stale (TTL expired or refresh failed).

    Returns ``(frame, meta)`` if any cached data file exists, else ``(None,
    None)``. Used as a graceful fallback when a live download fails and only
    older cached data is available, so the app can keep working offline.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    data_p, meta_p = _data_path(cache_dir, cache_key), _meta_path(cache_dir, cache_key)
    if not data_p.exists() or not meta_p.exists():
        return None, None
    try:
        df = pd.read_csv(data_p, index_col=0, parse_dates=True)
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta
    except Exception:
        return None, None


def save_cache(cache_key: str, frame: pd.DataFrame, meta: dict, cache_dir: Optional[Path] = None) -> None:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta["saved_at"] = time.time()
    meta["fresh"] = True
    frame.to_csv(_data_path(cache_dir, cache_key))
    with open(_meta_path(cache_dir, cache_key), "w", encoding="utf-8") as f:
        json.dump(meta, f, default=str)


def invalidate(cache_key: str, cache_dir: Optional[Path] = None) -> None:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    for p in (_data_path(cache_dir, cache_key), _meta_path(cache_dir, cache_key)):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
