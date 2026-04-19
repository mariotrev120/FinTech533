"""
Completion pass (parallel): audit every ticker parquet against the expected
2021-01-04..today coverage per calendar year, and for any ticker missing a
year's worth of bars, refetch only those year-windows via a ThreadPoolExecutor
and merge into the cache.

Why this exists:
  1. fetch_parallel.py tolerates partial-year failures silently. A ticker like
     CVX or PG can come back short because one 1Y pull plus its 6M fallback
     both timed out under parallel load.
  2. fetch_data.py's YEAR_WINDOWS jumps straight from the 2024 end-of-year
     window to an empty "current-year" trailing-6M window. When this was
     written, "current year" was 2025 and the trailing 6M covered H2-2025.
     Now that today is in 2026, the trailing 6M covers ~Oct 2025..Apr 2026,
     leaving ~Jan..Oct 2025 completely unfetched across every ticker. That
     silent hole is why "Fold 4 has 0 OOS candidates" — it's a fetcher
     artifact, not a calendar fact.

This script explicitly defines per-year IBKR query windows (including 2025
and a trailing 2026 window) and fills every year where coverage is below a
threshold. Workers run in parallel with distinct client_ids.
"""
from __future__ import annotations

import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_data import (
    UNIVERSE, MACRO, SECTOR_ETFS, DATA_DIR,
    _year_window,
)

WORKERS = 4

# Explicit per-year windows. Each tuple is (end_1y, end_jun, end_dec) fed into
# _year_window, which tries the 1Y pull first and falls back to two 6M pulls.
# For 2026 (the current partial year), we use empty strings so shinybroker
# defaults to "now" and pulls trailing 6M twice (which deduplicates to the
# ticker's full YTD).
YEAR_MAP: dict[int, tuple[str, str, str]] = {
    2021: ("20211231 23:59:59", "20210630 23:59:59", "20211231 23:59:59"),
    2022: ("20221231 23:59:59", "20220630 23:59:59", "20221231 23:59:59"),
    2023: ("20231231 23:59:59", "20230630 23:59:59", "20231231 23:59:59"),
    2024: ("20241231 23:59:59", "20240630 23:59:59", "20241231 23:59:59"),
    2025: ("20251231 23:59:59", "20250630 23:59:59", "20251231 23:59:59"),
    2026: ("",                  "",                  ""),
}
YEAR_FLOOR = 200
CURRENT_YEAR = datetime.now(timezone.utc).year
CURRENT_YEAR_FLOOR = 50


def first_covered_year(df: pd.DataFrame) -> int | None:
    if df.empty:
        return None
    by_year = df.groupby(df["timestamp"].dt.year).size()
    for yr in sorted(YEAR_MAP):
        if by_year.get(yr, 0) >= 20:
            return yr
    return None


def years_missing(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return sorted(YEAR_MAP.keys())
    first = first_covered_year(df)
    by_year = df.groupby(df["timestamp"].dt.year).size()
    missing = []
    for yr in sorted(YEAR_MAP):
        if first is not None and yr < first:
            continue
        floor = CURRENT_YEAR_FLOOR if yr == CURRENT_YEAR else YEAR_FLOOR
        if by_year.get(yr, 0) < floor:
            missing.append(yr)
    return missing


def _refetch_one(sb_mod, symbol: str, cid_base: int, years: list[int],
                 sec_type: str, exchange: str) -> tuple[str, pd.DataFrame]:
    frames = []
    for j, yr in enumerate(years):
        end_1y, end_jun, end_dec = YEAR_MAP[yr]
        cid = cid_base + j * 50
        try:
            f = _year_window(sb_mod, symbol, cid, end_1y, end_jun, end_dec,
                             sec_type=sec_type, exchange=exchange)
            if not f.empty:
                frames.append(f)
        except Exception:
            pass
        time.sleep(0.4)
    if not frames:
        return symbol, pd.DataFrame()
    return symbol, pd.concat(frames, ignore_index=True)


def merge_and_save(existing: pd.DataFrame, new: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if new.empty:
        return existing
    merged = pd.concat([existing, new], ignore_index=True) if not existing.empty else new.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = (
        merged.drop_duplicates("timestamp")
              .sort_values("timestamp")
              .reset_index(drop=True)
    )
    merged.to_parquet(cache, index=False)
    return merged


def main() -> int:
    import shinybroker as sb

    tickers = (
        [(s, "STK", "SMART") for s, _ in UNIVERSE]
        + list(MACRO)
        + list(SECTOR_ETFS)
    )

    jobs = []
    for sym, sec_type, exch in tickers:
        cache = DATA_DIR / f"{sym}.parquet"
        existing = pd.DataFrame()
        if cache.exists():
            try:
                existing = pd.read_parquet(cache)
                if "timestamp" in existing.columns:
                    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
                else:
                    existing = pd.DataFrame()
            except Exception:
                existing = pd.DataFrame()
        missing = years_missing(existing)
        if missing:
            jobs.append((sym, sec_type, exch, missing, existing))

    print(f"[completion] audited {len(tickers)} tickers, {len(jobs)} need refetch",
          flush=True)
    for sym, _, _, yrs, _ in jobs:
        print(f"    {sym:6s} missing years: {yrs}", flush=True)
    if not jobs:
        print("[completion] nothing to do — all tickers cover every year.",
              flush=True)
        return 0

    print("-" * 60, flush=True)
    print(f"[completion] running with {WORKERS} parallel workers", flush=True)

    base_cid = 40000
    fixed, still_bad = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for i, (sym, sec_type, exch, years, existing) in enumerate(jobs):
            cid = base_cid + i * 500
            fut = ex.submit(_refetch_one, sb, sym, cid, years, sec_type, exch)
            futs[fut] = (sym, sec_type, exch, years, existing)

        done = 0
        for fut in as_completed(futs):
            sym, sec_type, exch, years, existing = futs[fut]
            done += 1
            try:
                _, new = fut.result()
            except Exception as e:
                print(f"[{done:>2d}/{len(jobs)}] {sym:6s} EXC: {type(e).__name__}: {e}",
                      flush=True)
                still_bad.append(sym)
                continue
            merged = merge_and_save(existing, new,
                                    DATA_DIR / f"{sym}.parquet")
            remaining = years_missing(merged)
            if remaining:
                print(f"[{done:>2d}/{len(jobs)}] {sym:6s} still missing {remaining} "
                      f"(have {len(merged)} rows)", flush=True)
                still_bad.append(sym)
            else:
                print(f"[{done:>2d}/{len(jobs)}] {sym:6s} OK {len(merged)} rows "
                      f"{merged['timestamp'].min().date()}..{merged['timestamp'].max().date()}",
                      flush=True)
                fixed.append(sym)

    print("-" * 60, flush=True)
    print(f"fixed:     {len(fixed):>2d}  {', '.join(fixed)}", flush=True)
    print(f"still bad: {len(still_bad):>2d}  {', '.join(still_bad)}", flush=True)
    return 0 if not still_bad else 1


if __name__ == "__main__":
    sys.exit(main())
