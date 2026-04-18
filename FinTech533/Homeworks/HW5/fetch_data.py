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
    ("NVDA", "AI / semis"),       ("AVGO", "AI / semis"),
    ("SMCI", "AI / semis"),       ("AMD",  "AI / semis"),
    ("CCJ",  "Nuclear"),          ("VST",  "Nuclear"),
    ("OKLO", "Nuclear"),          ("SMR",  "Nuclear"),
    ("MSTR", "Bitcoin proxies"),  ("COIN", "Bitcoin proxies"),
    ("MARA", "Bitcoin proxies"),
    ("IONQ", "Quantum"),          ("RGTI", "Quantum"),
    ("QBTS", "Quantum"),
    ("PLTR", "Defense / AI"),     ("LMT",  "Defense / AI"),
    ("RTX",  "Defense / AI"),
    ("LLY",  "GLP-1"),            ("NVO",  "GLP-1"),
    ("RKLB", "Space"),
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
    "NVDA": "XLK", "AVGO": "XLK", "SMCI": "XLK", "AMD":  "XLK",
    "CCJ":  "URA", "VST":  "XLU", "OKLO": "URA", "SMR":  "URA",
    "MSTR": "IBIT","COIN": "IBIT","MARA": "IBIT",
    "IONQ": "XLK", "RGTI": "XLK", "QBTS": "XLK",
    "PLTR": "XLK", "LMT":  "ITA", "RTX":  "ITA",
    "LLY":  "XLV", "NVO":  "XLV",
    "RKLB": "ITA",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _flush_line(sym: str, state: str) -> None:
    print(f"{sym:5s} {state}", flush=True)


def _fetch_one(sb, symbol: str, cid: int, end_dt: str, duration: str,
               sec_type: str = "STK", exchange: str = "SMART") -> pd.DataFrame:
    c = sb.Contract({"symbol": symbol, "secType": sec_type, "exchange": exchange, "currency": "USD"})
    what = "Trades" if sec_type == "STK" else "TRADES"
    # IND contracts sometimes reject whatToShow=Trades. Try TRADES uppercase; if fail, use MIDPOINT.
    try:
        r = sb.fetch_historical_data(
            contract=c, endDateTime=end_dt, durationStr=duration,
            barSizeSetting="1 day", whatToShow=what,
            host=HOST, port=PORT, client_id=cid,
        )
    except Exception:
        r = sb.fetch_historical_data(
            contract=c, endDateTime=end_dt, durationStr=duration,
            barSizeSetting="1 day", whatToShow="MIDPOINT",
            host=HOST, port=PORT, client_id=cid,
        )
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


# Ordered list of fetch plans (each plan = list of (end_dt, duration) windows).
# We try plans in order; first one to produce >= MIN_BARS rows wins.
FETCH_PLANS = [
    # Plan A: 1 Y 2024 + 6 M recent  -> ~376 rows
    [("20241231 23:59:59", "1 Y"), ("", "6 M")],
    # Plan B: 6 M + 6 M for 2024 + 6 M recent  (for tickers that fail on 1 Y)
    [("20240630 23:59:59", "6 M"), ("20241231 23:59:59", "6 M"), ("", "6 M")],
    # Plan C: 9 M end Sep 2024 + 6 M recent  (minimal usable 2024 coverage)
    [("20240930 23:59:59", "9 M"), ("", "6 M")],
    # Plan D: try just recent 12 M as a floor  (at least some data)
    [("", "1 Y")],
]
MIN_BARS_FLOOR = 100   # refuse to save anything below this
GOOD_BARS      = 300   # a "complete" pull


def fetch_ticker(sb, symbol: str, base_cid: int, sec_type: str = "STK", exchange: str = "SMART") -> pd.DataFrame:
    last_err = None
    for i, plan in enumerate(FETCH_PLANS):
        try:
            frames = []
            for j, (end_dt, dur) in enumerate(plan):
                frames.append(_window(sb, symbol, base_cid + i * 1000 + j * 100, end_dt, dur,
                                      sec_type=sec_type, exchange=exchange))
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(merged) >= MIN_BARS_FLOOR:
                return merged
            last_err = RuntimeError(f"plan {i} returned only {len(merged)} rows")
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err or RuntimeError("all plans failed")


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
