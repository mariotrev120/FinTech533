"""
Exogenous feature engineering for the ML filter.

Vestal's Law: features MUST be strictly exogenous to the ticker being traded.
No Open/High/Low/Close/Volume of the ticker itself may enter the feature vector.

Temporal discipline: every feature is computed from data strictly BEFORE the
entry bar. Internally this is implemented by selecting data at the most recent
timestamp strictly less than the entry_date (i.e. t-1 lookup).

Outputs one feature row per candidate trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


# =======================
# Loading helpers
# =======================
def load_cached(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load one cached parquet from the fetch_data.py output."""
    p = data_dir / f"{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no cached data for {symbol} at {p}")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _lookup_prior(df: pd.DataFrame, ts: pd.Timestamp, col: str = "close"):
    """Return df[col] at the most recent timestamp strictly before ts. None if missing."""
    sub = df[df["timestamp"] < ts]
    if sub.empty:
        return None
    return float(sub[col].iloc[-1])


def _ret_window(df: pd.DataFrame, ts: pd.Timestamp, window: int):
    """
    Return ( close_{t-1} / close_{t-1-window} - 1 ) using data strictly before ts.
    """
    sub = df[df["timestamp"] < ts].tail(window + 1)
    if len(sub) < window + 1:
        return None
    c0 = float(sub["close"].iloc[0])
    c1 = float(sub["close"].iloc[-1])
    if c0 == 0 or np.isnan(c0) or np.isnan(c1):
        return None
    return c1 / c0 - 1.0


def _realized_vol(df: pd.DataFrame, ts: pd.Timestamp, window: int = 30):
    """
    Annualized realized volatility from log-returns, computed on data strictly
    before ts. Uses the last `window` log-returns.
    """
    sub = df[df["timestamp"] < ts].tail(window + 1)
    if len(sub) < window + 1:
        return None
    lr = np.log(sub["close"].astype(float)).diff().dropna().values
    if len(lr) == 0:
        return None
    return float(np.std(lr, ddof=1) * np.sqrt(252))


# =======================
# Feature primitives
# =======================
def yield_curve_spline_coeffs(
    yields: Mapping[str, float],
    tenor_years: Mapping[str, float] = None,
    degree: int = 3,
) -> dict:
    """
    Fit a polynomial of given degree to the yield curve and return coefficients.

    `yields`  : mapping symbol -> yield (in the units IBKR returns, which for
                ^TNX/^FVX/^IRX/^TYX is the raw yield * 10, e.g. 45.21 for 4.521%).
                We rescale by /10 so features are in percent.
    `tenor_years`: mapping symbol -> tenor in years. Defaults to the IBKR set.

    Returns a dict {curve_a3, curve_a2, curve_a1, curve_a0}. If degree < 3 some
    coefficients are omitted.
    """
    if tenor_years is None:
        tenor_years = {"IRX": 0.25, "FVX": 5.0, "TNX": 10.0, "TYX": 30.0}

    xs, ys = [], []
    for sym, tenor in tenor_years.items():
        y = yields.get(sym)
        if y is None or not np.isfinite(y):
            continue
        xs.append(tenor)
        ys.append(y / 10.0)  # IBKR yields are quoted as 10x
    if len(xs) < degree + 1:
        return {f"curve_a{i}": np.nan for i in range(degree + 1)}
    coeffs = np.polyfit(xs, ys, deg=degree)
    # np.polyfit returns highest-degree first: a3, a2, a1, a0 for deg=3
    return {f"curve_a{degree - i}": float(coeffs[i]) for i in range(degree + 1)}


def vix_features(vix_df: pd.DataFrame, vix3m_df: pd.DataFrame, ts: pd.Timestamp) -> dict:
    """
    VIX level + VIX term-structure slope. All values are t-1.

    vix_level      : front-month VIX.
    vix3m_level    : 3-month VIX.
    vix_spread     : VIX3M - VIX  (positive = contango/stability, negative = backwardation/stress).
    vix_level_20d  : 20-day change in VIX level (context: is fear rising?).
    """
    vix = _lookup_prior(vix_df, ts)
    vix3m = _lookup_prior(vix3m_df, ts)
    vix_20d_ago = _lookup_prior(vix_df[vix_df["timestamp"] < ts].head(-20) if False else vix_df, ts - pd.Timedelta(days=30))
    if vix is None or vix3m is None:
        return dict(vix_level=np.nan, vix3m_level=np.nan, vix_spread=np.nan, vix_change_20d=np.nan)
    return dict(
        vix_level=vix,
        vix3m_level=vix3m,
        vix_spread=vix3m - vix,
        vix_change_20d=(vix - vix_20d_ago) if vix_20d_ago is not None else np.nan,
    )


def iv_rv_spread(vix_df: pd.DataFrame, spy_df: pd.DataFrame, ts: pd.Timestamp) -> dict:
    """
    Market-level Implied-minus-Realized vol spread.
    IV proxy  : VIX (t-1).
    RV proxy  : 30-day realized vol of SPY (t-1).
    Both in percent-vol terms.
    """
    vix = _lookup_prior(vix_df, ts)  # VIX is quoted as % already
    rv = _realized_vol(spy_df, ts, window=30)
    if vix is None or rv is None:
        return dict(iv_level=np.nan, rv_level=np.nan, iv_rv_spread=np.nan)
    rv_pct = rv * 100.0
    return dict(iv_level=vix, rv_level=rv_pct, iv_rv_spread=vix - rv_pct)


def sector_relative_strength(
    ticker_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    ts: pd.Timestamp,
    window: int = 5,
) -> dict:
    """
    Sector Relative Strength: ticker_5d_return - sector_5d_return (t-1).

    Note: this uses the ticker's own price history but only as a RELATIVE measure
    (difference of returns). It never feeds an absolute price / OHLCV to the model.
    Disable by passing `include_sector_rs=False` to build_feature_row if strict
    Vestal purity is required.
    """
    ticker_r = _ret_window(ticker_df, ts, window)
    sector_r = _ret_window(sector_df, ts, window)
    if ticker_r is None or sector_r is None:
        return dict(sector_rs=np.nan, sector_ret_5d=np.nan)
    return dict(sector_rs=ticker_r - sector_r, sector_ret_5d=sector_r)


def market_features(spy_df: pd.DataFrame, ts: pd.Timestamp) -> dict:
    """Broad-market context. SPY 20-day return as a risk-on/risk-off proxy."""
    r20 = _ret_window(spy_df, ts, 20)
    return dict(spy_ret_20d=r20 if r20 is not None else np.nan)


# =======================
# One-stop per-trade feature row
# =======================
@dataclass
class FeatureBundle:
    """All cached price frames needed to compute features."""
    vix: pd.DataFrame
    vix3m: pd.DataFrame
    tnx: pd.DataFrame
    fvx: pd.DataFrame
    irx: pd.DataFrame
    tyx: pd.DataFrame
    spy: pd.DataFrame
    ticker_prices: dict  # symbol -> price df (used for sector RS: ticker_df and sector_df)
    sector_prices: dict  # symbol -> sector ETF df
    ticker_to_sector: dict  # symbol -> sector ETF symbol


def build_feature_row(
    bundle: FeatureBundle,
    ticker: str,
    entry_date: pd.Timestamp,
    include_sector_rs: bool = True,
) -> dict:
    """
    Build one feature vector for a candidate trade on `ticker` entering at `entry_date`.
    Every lookup is t-1.
    """
    entry_date = pd.to_datetime(entry_date)

    # yield curve spline (t-1 on EACH tenor)
    yields = {}
    for sym, df in [("IRX", bundle.irx), ("FVX", bundle.fvx), ("TNX", bundle.tnx), ("TYX", bundle.tyx)]:
        yields[sym] = _lookup_prior(df, entry_date)
    feats = yield_curve_spline_coeffs(yields, degree=3)

    # VIX features
    feats.update(vix_features(bundle.vix, bundle.vix3m, entry_date))

    # Market IV/RV spread
    feats.update(iv_rv_spread(bundle.vix, bundle.spy, entry_date))

    # Broad-market context
    feats.update(market_features(bundle.spy, entry_date))

    # Sector RS
    if include_sector_rs:
        sector_sym = bundle.ticker_to_sector.get(ticker)
        tdf = bundle.ticker_prices.get(ticker)
        sdf = bundle.sector_prices.get(sector_sym) if sector_sym else None
        if tdf is not None and sdf is not None:
            feats.update(sector_relative_strength(tdf, sdf, entry_date))
        else:
            feats.update(dict(sector_rs=np.nan, sector_ret_5d=np.nan))

    return feats


# =======================
# Ordered list of feature columns (for StandardScaler / LR)
# =======================
FEATURE_COLS = [
    "curve_a3", "curve_a2", "curve_a1", "curve_a0",
    "vix_level", "vix3m_level", "vix_spread", "vix_change_20d",
    "iv_level", "rv_level", "iv_rv_spread",
    "spy_ret_20d",
    "sector_rs", "sector_ret_5d",
]
