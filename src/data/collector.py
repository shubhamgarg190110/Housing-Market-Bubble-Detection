from __future__ import annotations

"""
Data collection from FRED (Federal Reserve), Zillow, and BLS.
All data is cached locally so repeated runs are instant.
"""

import io
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _fred_url(series_id: str, start: str, end: str) -> str:
    return (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&vintage_date={end[:10]}"
    )


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Download one FRED series via the public CSV endpoint (no API key needed)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.columns = [series_id]
        df = df.replace(".", np.nan).astype(float)
        return df[series_id].loc[start:end]
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", series_id, exc)
        return pd.Series(dtype=float, name=series_id)


def _try_pandas_datareader(series_id: str, start: str, end: str) -> pd.Series:
    try:
        import pandas_datareader.data as web
        s = web.DataReader(series_id, "fred", start=start, end=end)[series_id]
        s.name = series_id
        return s
    except Exception:
        return pd.Series(dtype=float, name=series_id)


def collect_fred_data(
    series_map: dict,
    start: str,
    end: str,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Download all FRED series, rename columns, merge into a single DataFrame,
    and cache to CSV.
    """
    cache_file = cache_path / "fred_raw.csv"
    if cache_file.exists():
        logger.info("Loading FRED data from cache: %s", cache_file)
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    logger.info("Downloading %d FRED series …", len(series_map))
    frames = {}
    for fred_id, col_name in series_map.items():
        s = fetch_fred_series(fred_id, start, end)
        if s.empty:
            s = _try_pandas_datareader(fred_id, start, end)
        if not s.empty:
            frames[col_name] = s
            logger.info("  ✓ %s (%s)", col_name, fred_id)
        else:
            logger.warning("  ✗ %s (%s) – skipped", col_name, fred_id)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.to_csv(cache_file)
    logger.info("FRED data cached to %s", cache_file)
    return df


def _download_zillow(url: str, label: str, cache_path: Path) -> pd.DataFrame | None:
    cache_file = cache_path / f"zillow_{label}.csv"
    if cache_file.exists():
        logger.info("Loading Zillow %s from cache", label)
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
        raw.to_csv(cache_file, index=False)
        logger.info("Zillow %s cached", label)
        return raw
    except Exception as exc:
        logger.warning("Could not fetch Zillow %s: %s", label, exc)
        return None


def collect_zillow_data(zhvi_url: str, zori_url: str, cache_path: Path) -> dict:
    """
    Download Zillow ZHVI (home values) and ZORI (rents) metro-level data.
    Returns dict with keys 'zhvi' and 'zori', each a tidy time-series DataFrame.
    """
    results = {}
    for label, url in [("zhvi", zhvi_url), ("zori", zori_url)]:
        raw = _download_zillow(url, label, cache_path)
        if raw is None:
            continue
        date_cols = [c for c in raw.columns if c[:2] in ("19", "20") or c[4:5] == "-"]
        id_cols = [c for c in raw.columns if c not in date_cols]
        metro_col = next((c for c in id_cols if "region" in c.lower()), id_cols[0] if id_cols else None)
        if metro_col is None:
            continue
        tidy = raw.melt(id_vars=[metro_col], value_vars=date_cols, var_name="date", value_name=label)
        tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
        tidy = tidy.dropna(subset=["date"]).set_index("date").sort_index()
        results[label] = tidy
    return results


def _national_zhvi(zillow: dict) -> pd.Series:
    """Extract a national-level median home value series from ZHVI."""
    if "zhvi" not in zillow:
        return pd.Series(dtype=float, name="zhvi_national")
    df = zillow["zhvi"]
    region_col = df.columns[0]
    national_keywords = ("United States", "national", "us", "usa")
    mask = df[region_col].str.lower().str.contains("|".join(national_keywords), na=False)
    national = df[mask]["zhvi"].dropna()
    if national.empty:
        national = df.groupby("date")["zhvi"].median()
    else:
        national = national.resample("MS").last()
    national.name = "zhvi_national"
    return national


def _nyc_zori(zillow: dict) -> pd.Series:
    """Extract NYC rent index from ZORI."""
    if "zori" not in zillow:
        return pd.Series(dtype=float, name="zori_nyc")
    df = zillow["zori"]
    region_col = df.columns[0]
    mask = df[region_col].str.contains("New York", na=False)
    nyc = df[mask]["zori"].dropna()
    nyc.name = "zori_nyc"
    return nyc.resample("MS").last() if not nyc.empty else nyc


def build_master_dataset(
    fred_df: pd.DataFrame,
    zillow: dict,
    start: str,
    end: str,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Merge FRED and Zillow series into a single monthly DataFrame
    aligned to month-start frequency.
    """
    cache_file = cache_path / "master_monthly.csv"
    if cache_file.exists():
        logger.info("Loading master dataset from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    fred_monthly = fred_df.resample("MS").interpolate(method="time")

    zhvi = _national_zhvi(zillow)
    zori = _nyc_zori(zillow)

    parts = [fred_monthly]
    for s in (zhvi, zori):
        if not s.empty:
            parts.append(s.resample("MS").last().rename(s.name))

    master = pd.concat(parts, axis=1).loc[start:end]
    master.index.name = "date"
    master.to_csv(cache_file)
    logger.info("Master dataset saved: %s rows × %s cols", *master.shape)
    return master


def load_or_build(config) -> pd.DataFrame:
    """
    Top-level entry point: load from cache if available, otherwise fetch everything.
    Returns the merged monthly DataFrame.
    """
    from config import (
        FRED_SERIES, ZILLOW_ZHVI_URL, ZILLOW_ZORI_URL,
        START_DATE, END_DATE, DATA_RAW, DATA_PROCESSED,
    )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    fred_df = collect_fred_data(FRED_SERIES, START_DATE, END_DATE, DATA_RAW)
    zillow = collect_zillow_data(ZILLOW_ZHVI_URL, ZILLOW_ZORI_URL, DATA_RAW)
    master = build_master_dataset(fred_df, zillow, START_DATE, END_DATE, DATA_PROCESSED)
    return master
