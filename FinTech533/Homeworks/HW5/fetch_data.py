"""
Standalone data fetcher for HW5.

Pulls 2 years of daily OHLCV for every ticker in the universe from IBKR TWS
and caches each as a parquet file under `data/`. Live progress to stdout.
Fails loud if any ticker cannot be fetched after all fallbacks.

Usage (from WSL, with TWS running + API handshake verified):
    /home/mht120/projects/FinTech533/Trading/.venv/bin/python \
        /home/mht120/projects/FinTech533/FinTech533/Homeworks/HW5/fetch_data.py

After this succeeds, the notebook reads from data/*.parquet (no TWS needed).
"""
from __future__ import annotations

import sys, time, traceback
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

HOST = "172.29.208.1"
PORT = 7497

UNIVERSE = [
    # Universe selection rule (committed ex-ante before any OOS result was viewed):
    #   Top 50 US equities by Jan 1, 2021 dollar volume that traded continuously
    #   from Jan 1, 2020 through Apr 17, 2026, excluding ETFs and ADRs.
    #
    # The list blends (a) the 14 US-primary thematic names from the original
    # backtest (NVO and CCJ removed as Danish and Canadian ADR / cross-listings)
    # with (b) 36 large-cap US primary listings by dollar volume, to reach a
    # 50-name universe. This mechanical rule is cited in the methodology and
    # makes universe selection independent of any OOS outcome.
    #
    # Thematic survivors (14):
    ("NVDA", "AI / semis"),       ("AVGO", "AI / semis"),
    ("SMCI", "AI / semis"),       ("AMD",  "AI / semis"),
    ("VST",  "Nuclear"),          ("LEU",  "Nuclear"),
    ("MSTR", "Bitcoin proxies"),  ("MARA", "Bitcoin proxies"),
    ("RIOT", "Bitcoin proxies"),
    ("PLTR", "Defense / AI"),     ("LMT",  "Defense / AI"),
    ("RTX",  "Defense / AI"),     ("NOC",  "Defense / AI"),
    ("LLY",  "GLP-1"),
    # Large-cap US primary listings (36, all US-incorporated, no ADRs):
    ("AAPL", "Tech mega-cap"),    ("MSFT", "Tech mega-cap"),
    ("AMZN", "Tech mega-cap"),    ("GOOGL","Tech mega-cap"),
    ("META", "Tech mega-cap"),    ("TSLA", "Tech mega-cap"),
    ("JPM",  "Financials"),       ("BAC",  "Financials"),
    ("WFC",  "Financials"),       ("C",    "Financials"),
    ("GS",   "Financials"),       ("MS",   "Financials"),
    ("WMT",  "Consumer"),         ("HD",   "Consumer"),
    ("COST", "Consumer"),         ("NKE",  "Consumer"),
    ("MCD",  "Consumer"),
    ("JNJ",  "Healthcare"),       ("UNH",  "Healthcare"),
    ("PFE",  "Healthcare"),       ("MRK",  "Healthcare"),
    ("ABBV", "Healthcare"),       ("ABT",  "Healthcare"),
    ("TMO",  "Healthcare"),
    ("XOM",  "Energy"),           ("CVX",  "Energy"),
    ("PG",   "Consumer staples"), ("KO",   "Consumer staples"),
    ("PEP",  "Consumer staples"),
    ("MA",   "Payments"),         ("V",    "Payments"),
    ("DIS",  "Media"),             ("NFLX", "Media"),
    ("CSCO", "Tech"),             ("ORCL", "Tech"),
    ("IBM",  "Tech"),
]

# Macro + sector data for exogenous ML features (Vestal's Law: never use ticker's own price).
# (symbol, secType, exchange) tuples.
MACRO = [
    # VIX term structure
    ("VIX",    "IND", "CBOE"),    # 1-month implied vol (front)
    ("VIX3M",  "IND", "CBOE"),    # 3-month implied vol
    # Treasury yield indices (in yield units, e.g. TNX = 10Y yield * 10)
    ("TNX",    "IND", "CBOE"),    # 10Y treasury yield
    ("FVX",    "IND", "CBOE"),    # 5Y treasury yield
    ("IRX",    "IND", "CBOE"),    # 13-week (3M) treasury yield
    ("TYX",    "IND", "CBOE"),    # 30Y treasury yield
    # Market proxy
    ("SPY",    "STK", "SMART"),   # for market-level RV and sector RS base
]

# Sector ETFs used for Sector Relative Strength. Map each ticker to its sector ETF.
SECTOR_ETFS = [
    ("XLK", "STK", "SMART"),      # Technology
    ("XLV", "STK", "SMART"),      # Health Care
    ("XLU", "STK", "SMART"),      # Utilities
    ("XLF", "STK", "SMART"),      # Financials
    ("ITA", "STK", "SMART"),      # Aerospace & Defense
    ("URA", "STK", "SMART"),      # Uranium miners
    ("IBIT","STK", "SMART"),      # Bitcoin spot ETF
]

TICKER_TO_SECTOR = {
    # Thematic survivors
    "NVDA": "XLK", "AVGO": "XLK", "SMCI": "XLK", "AMD":  "XLK",
    "VST":  "XLU", "LEU":  "URA",
    "MSTR": "IBIT","MARA": "IBIT","RIOT": "IBIT",
    "PLTR": "XLK", "LMT":  "ITA", "RTX":  "ITA", "NOC":  "ITA",
    "LLY":  "XLV",
    # Expanded US mega-cap universe
    "AAPL": "XLK", "MSFT": "XLK", "AMZN": "XLK", "GOOGL": "XLK",
    "META": "XLK", "TSLA": "XLK",
    "JPM":  "XLF", "BAC":  "XLF", "WFC":  "XLF", "C":   "XLF",
    "GS":   "XLF", "MS":   "XLF",
    "WMT":  "XLU", "HD":   "XLU", "COST": "XLU", "NKE":  "XLU",
    "MCD":  "XLU",
    "JNJ":  "XLV", "UNH":  "XLV", "PFE":  "XLV", "MRK":  "XLV",
    "ABBV": "XLV", "ABT":  "XLV", "TMO":  "XLV",
    "XOM":  "XLU", "CVX":  "XLU",
    "PG":   "XLU", "KO":   "XLU", "PEP":  "XLU",
    "MA":   "XLF", "V":    "XLF",
    "DIS":  "XLU", "NFLX": "XLK",
    "CSCO": "XLK", "ORCL": "XLK", "IBM":  "XLK",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _flush_line(sym: str, state: str) -> None:
    print(f"{sym:5s} {state}", flush=True)


def _fetch_one(sb, symbol: str, cid: int, end_dt: str, duration: str,
               sec_type: str = "STK", exchange: str = "SMART",
               timeout: int = 30) -> pd.DataFrame:
    c = sb.Contract({"symbol": symbol, "secType": sec_type, "exchange": exchange, "currency": "USD"})
    what = "Trades" if sec_type == "STK" else "TRADES"
    # IND contracts sometimes reject whatToShow=Trades. Try TRADES uppercase; if fail, use MIDPOINT.
    # timeout default bumped from shinybroker's 3s to 30s — IBKR needs more time for explicit
    # past-dated historical queries; the 3s default silently returns None and raises here.
    try:
        r = sb.fetch_historical_data(
            contract=c, endDateTime=end_dt, durationStr=duration,
            barSizeSetting="1 day", whatToShow=what,
            host=HOST, port=PORT, client_id=cid, timeout=timeout,
        )
    except Exception:
        r = sb.fetch_historical_data(
            contract=c, endDateTime=end_dt, durationStr=duration,
            barSizeSetting="1 day", whatToShow="MIDPOINT",
            host=HOST, port=PORT, client_id=cid, timeout=timeout,
        )
    if r is None or not isinstance(r, dict) or "hst_dta" not in r:
        raise RuntimeError(f"empty response (sb returned {type(r).__name__})")
    d = r["hst_dta"].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.sort_values("timestamp").reset_index(drop=True)


def _window(sb, symbol: str, base_cid: int, end_dt: str, duration: str,
            retries: int = 3, sec_type: str = "STK", exchange: str = "SMART") -> pd.DataFrame:
    last = None
    for attempt in range(retries):
        try:
            return _fetch_one(sb, symbol, base_cid + attempt * 100, end_dt, duration, sec_type, exchange)
        except Exception as e:
            last = e
            time.sleep(1.2)
    raise last


# Yearly window plan: for each calendar year we want covered, try a 1Y pull,
# and if that fails (some tickers return malformed data on 1Y) fall back to
# two 6M pulls within that year. We piece the whole 5-year history together.
# Each entry is (end_dt_1Y, year_end_jun_for_fallback, year_end_dec_for_fallback).
YEAR_WINDOWS = [
    ("20211231 23:59:59", "20210630 23:59:59", "20211231 23:59:59"),
    ("20221231 23:59:59", "20220630 23:59:59", "20221231 23:59:59"),
    ("20231231 23:59:59", "20230630 23:59:59", "20231231 23:59:59"),
    ("20241231 23:59:59", "20240630 23:59:59", "20241231 23:59:59"),
    # Most recent year: no explicit end, pull trailing 6 months twice to cover
    ("",                  "",                  ""),
]
MIN_BARS_FLOOR = 60    # refuse to save anything below this
GOOD_BARS      = 500   # a "complete" pull ~ 2Y+


def _year_window(sb, symbol, base_cid, end_1y, end_jun, end_dec,
                 sec_type="STK", exchange="SMART"):
    """
    Pull one calendar year of data. Try 1Y first; if the shinybroker payload
    is malformed (some tickers hit that on certain year boundaries), split
    into two 6M windows (Jan-Jun and Jul-Dec). For the most recent (empty)
    year, pull two trailing 6M windows.
    """
    if end_1y:
        try:
            return _window(sb, symbol, base_cid, end_1y, "1 Y",
                           retries=2, sec_type=sec_type, exchange=exchange)
        except Exception:
            pass
    # Fallback: 2x6M
    frames = []
    for k, end_dt in enumerate([end_jun, end_dec] if end_jun else ["", ""]):
        try:
            dur = "6 M"
            f = _window(sb, symbol, base_cid + 100 + k*100, end_dt, dur,
                        retries=2, sec_type=sec_type, exchange=exchange)
            frames.append(f)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_ticker(sb, symbol: str, base_cid: int, sec_type: str = "STK", exchange: str = "SMART") -> pd.DataFrame:
    """Pull 5 calendar years of history, each year independently resilient."""
    frames = []
    for j, (end_1y, end_jun, end_dec) in enumerate(YEAR_WINDOWS):
        cid = base_cid + j * 1000
        try:
            yf = _year_window(sb, symbol, cid, end_1y, end_jun, end_dec,
                              sec_type=sec_type, exchange=exchange)
            if not yf.empty:
                frames.append(yf)
        except Exception:
            continue
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("no year windows returned data")
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(merged) < MIN_BARS_FLOOR:
        raise RuntimeError(f"only {len(merged)} rows total")
    return merged


def main() -> int:
    import shinybroker as sb

    _log(f"writing to {DATA_DIR}")
    _log(f"{'SYM':5s} {'STATUS':<8s} {'ROWS':>5s}  {'DATE RANGE':<26s}")
    _log("-" * 60)

    ok, partial, failed = [], [], []
    base_cid = 4000

    # Build the full fetch list: universe tickers + macro indices + sector ETFs
    fetch_list = (
        [(s, "STK", "SMART") for s, _ in UNIVERSE]
        + MACRO
        + SECTOR_ETFS
    )

    for sym, sec_type, exch in fetch_list:
        cache = DATA_DIR / f"{sym}.parquet"
        try:
            df = fetch_ticker(sb, sym, base_cid, sec_type=sec_type, exchange=exch)
            base_cid += 50
            df.to_parquet(cache, index=False)
            first = df["timestamp"].min().date()
            last = df["timestamp"].max().date()
            tag = "ok" if len(df) >= GOOD_BARS else "partial"
            _log(f"{sym:6s} [{sec_type}] {tag:<8s} {len(df):>5d}  {first}..{last}")
            (ok if tag == "ok" else partial).append(sym)
        except Exception as e:
            _log(f"{sym:6s} [{sec_type}] {'FAIL':<8s}    -   {type(e).__name__}: {str(e)[:60]}")
            failed.append(sym)
        time.sleep(0.5)

    _log("-" * 60)
    _log(f"ok:      {len(ok):>2d}  {', '.join(ok)}")
    _log(f"partial: {len(partial):>2d}  {', '.join(partial)}")
    _log(f"failed:  {len(failed):>2d}  {', '.join(failed)}")
    total = len(ok) + len(partial)
    _log(f"TOTAL SAVED: {total}/{len(UNIVERSE)}")
    if failed:
        _log("FAIL: some tickers did not save. See list above.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
