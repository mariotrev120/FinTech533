"""
Long / short Donchian-breakout backtest engine.

Produces two outputs matching the HW4 shape:

    blotter : one row per completed trade, all cost components itemized
    ledger  : one row per trading day, position / cash / mark / mkt_value

The engine is a single function so the notebook can call it for the 2024
training pass, any parameter-sweep pass, and the 2025 out-of-sample pass
without duplicating logic.
"""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from breakout import (
    ALLOW_LONG,
    ALLOW_SHORT,
    POSITION_NOTIONAL,
    PROFIT_ATR_MULT,
    STARTING_CAPITAL,
    STOP_ATR_MULT,
    TIMEOUT_DAYS,
    add_indicators,
    detect_breakouts,
)
from costs import CostConfig, round_trip_costs, slippage


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    *,
    lookback: int = None,
    stop_atr_mult: float = STOP_ATR_MULT,
    profit_atr_mult: float = PROFIT_ATR_MULT,
    timeout_days: int = TIMEOUT_DAYS,
    position_notional: float = POSITION_NOTIONAL,
    starting_capital: float = STARTING_CAPITAL,
    allow_long: bool = ALLOW_LONG,
    allow_short: bool = ALLOW_SHORT,
    cost_cfg: CostConfig | None = None,
    signals_override: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a Donchian breakout backtest on a daily OHLCV frame.

    Execution conventions (documented so another student can replicate):
      * Signals are generated on the CLOSE of day t.
      * Orders execute on the OPEN of day t+1 ("next bar open" — no lookahead).
      * Stops and profit targets are checked against HIGH/LOW of each day the
        trade is live. If both are touched on the same bar we assume the STOP
        triggers first (conservative).
      * Timeout exit = market-on-close of the (TIMEOUT_DAYS)-th day.
      * Position sizing fixes USD notional per trade (rounded down to whole
        shares). One open position at a time.

    Returns
    -------
    blotter : DataFrame, one row per trade.
    ledger  : DataFrame, one row per bar.
    """
    if cost_cfg is None:
        cost_cfg = CostConfig()

    lb = lookback if lookback is not None else 20
    d = add_indicators(df, lookback=lb).reset_index(drop=True)
    if signals_override is not None:
        sig = signals_override.reindex(df.index).fillna(0).astype(int).values
        d["signal"] = sig
    else:
        d["signal"] = detect_breakouts(
            df, lookback=lb, allow_long=allow_long, allow_short=allow_short
        ).values

    cash = float(starting_capital)
    pos = 0                 # signed: +long, -short
    in_trade = False
    info: dict | None = None

    trades: list[dict] = []
    ledger_rows: list[dict] = []
    trade_id = 0

    for i, row in d.iterrows():
        date = row["timestamp"]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

        exit_type: str | None = None
        exit_price: float | None = None

        # -------- manage existing trade on this bar --------
        if in_trade:
            info["hold_days"] += 1
            direction = info["direction"]
            stop_px = info["stop_price"]
            target_px = info["target_price"]

            if direction == "long":
                # stop first (conservative if both hit)
                if l <= stop_px:
                    exit_type, exit_price = "stop_loss", stop_px
                elif h >= target_px:
                    exit_type, exit_price = "profit_target", target_px
            else:  # short
                if h >= stop_px:
                    exit_type, exit_price = "stop_loss", stop_px
                elif l <= target_px:
                    exit_type, exit_price = "profit_target", target_px

            # timeout check (only if no other exit already)
            if exit_type is None and info["hold_days"] >= timeout_days:
                exit_type, exit_price = "timeout", c

            if exit_type is not None:
                qty_abs = abs(pos)
                direction = info["direction"]
                ep = info["entry_price"]
                gross = (exit_price - ep) * qty_abs * (1 if direction == "long" else -1)
                costs = round_trip_costs(
                    qty_abs, ep, exit_price, info["hold_days"], direction, cost_cfg
                )
                net = gross - costs["total_cost"]

                # cash settlement: release any reserved margin and add net pnl
                cash += info["reserved_cash"] + net

                trades.append({
                    "trade_id": info["trade_id"],
                    "entry_date": info["entry_date"],
                    "exit_date": date,
                    "direction": direction,
                    "qty": qty_abs,
                    "entry_price": ep,
                    "exit_price": exit_price,
                    "gross_pnl": gross,
                    "commission": costs["commission"],
                    "reg_fees": costs["reg_fees"],
                    "slippage": costs["slippage"],
                    "borrow_cost": costs["borrow_cost"],
                    "total_cost": costs["total_cost"],
                    "net_pnl": net,
                    "trade_return": net / (ep * qty_abs) if qty_abs > 0 else 0.0,
                    "hold_days": info["hold_days"],
                    "exit_type": exit_type,
                    "good_trade": int(net > 0),
                    "atr_at_entry": info["atr_at_entry"],
                    "adx_at_entry": info["adx_at_entry"],
                    "stop_price": stop_px,
                    "target_price": target_px,
                })
                pos = 0
                in_trade = False
                info = None

        # -------- look for new entry (only if flat) --------
        # Signal generated at close of bar i, but we use the *previous* bar's
        # signal so execution happens at today's OPEN (no lookahead).
        if (not in_trade) and i > 0:
            prev_sig = int(d.loc[i - 1, "signal"])
            prev_atr = float(d.loc[i - 1, "atr"])
            prev_adx = float(d.loc[i - 1, "adx"])

            if prev_sig != 0 and not np.isnan(prev_atr) and prev_atr > 0:
                qty = int(floor(position_notional / o))
                if qty > 0:
                    direction = "long" if prev_sig == 1 else "short"
                    entry_px = o + (o * cost_cfg.slippage_bps / 10_000.0) * (
                        1 if direction == "long" else -1
                    )

                    if direction == "long":
                        stop = entry_px - stop_atr_mult * prev_atr
                        target = entry_px + profit_atr_mult * prev_atr
                        pos = qty
                        reserved = qty * entry_px
                    else:
                        stop = entry_px + stop_atr_mult * prev_atr
                        target = entry_px - profit_atr_mult * prev_atr
                        pos = -qty
                        # Short sale: borrow and sell; cash INCREASES by
                        # proceeds. We record the reserved cash as the
                        # negative so cash settles symmetrically at exit.
                        reserved = -qty * entry_px

                    cash -= reserved  # for longs this debits cash for the buy; for shorts, reserved<0 so cash increases by proceeds
                    trade_id += 1
                    in_trade = True
                    info = {
                        "trade_id": trade_id,
                        "direction": direction,
                        "entry_date": date,
                        "entry_price": entry_px,
                        "stop_price": stop,
                        "target_price": target,
                        "reserved_cash": reserved,
                        "hold_days": 0,
                        "atr_at_entry": prev_atr,
                        "adx_at_entry": prev_adx,
                    }

        # -------- mark-to-market ledger row --------
        if pos == 0:
            mkt_value = cash
            unrealized = 0.0
        else:
            # For longs cash was already reduced by entry notional (reserved>0);
            # position mark = pos * c. For shorts reserved<0 so cash was boosted;
            # position mark = pos * c which is negative for shorts.
            mkt_value = cash + pos * c
            unrealized = mkt_value - starting_capital  # change vs start

        ledger_rows.append({
            "date": date,
            "position": pos,
            "cash": cash,
            "mark": c,
            "mkt_value": mkt_value,
            "unrealized_pnl": unrealized,
            "signal": int(row["signal"]),
            "kc_upper": float(row["kc_upper"]) if "kc_upper" in row and not pd.isna(row["kc_upper"]) else np.nan,
            "kc_lower": float(row["kc_lower"]) if "kc_lower" in row and not pd.isna(row["kc_lower"]) else np.nan,
            "bb_upper": float(row["bb_upper"]) if "bb_upper" in row and not pd.isna(row["bb_upper"]) else np.nan,
            "bb_lower": float(row["bb_lower"]) if "bb_lower" in row and not pd.isna(row["bb_lower"]) else np.nan,
            "squeeze_on": bool(row["squeeze_on"]) if "squeeze_on" in row and not pd.isna(row["squeeze_on"]) else False,
            "atr": float(row["atr"]) if not np.isnan(row["atr"]) else np.nan,
            "adx": float(row["adx"]) if not np.isnan(row["adx"]) else np.nan,
        })

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["mkt_value"].pct_change()

    blotter = pd.DataFrame(trades)
    return blotter, ledger
