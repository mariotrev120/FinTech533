"""
Performance metrics for the breakout backtest.

All functions take a trades DataFrame (blotter) and/or a daily ledger and
return a dict of labeled metrics. Risk-free rate and trading-day count are
pulled from `breakout.RISK_FREE` and 252 by default so the whole repo stays
internally consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from breakout import RISK_FREE


TRADING_DAYS = 252


def _safe_div(n, d):
    return n / d if d not in (0, 0.0, None) and not pd.isna(d) else np.nan


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drop in the equity curve (negative number)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def sharpe_ratio(daily_rets: pd.Series, rf: float = RISK_FREE) -> float:
    """Annualized Sharpe. Uses daily returns and 252-day annualization."""
    r = daily_rets.dropna()
    if r.empty:
        return np.nan
    excess = r - rf / TRADING_DAYS
    sigma = r.std(ddof=1)
    if sigma == 0 or pd.isna(sigma):
        return np.nan
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS))


def sortino_ratio(daily_rets: pd.Series, rf: float = RISK_FREE) -> float:
    """
    Annualized Sortino. Same as Sharpe but uses downside deviation (returns
    below the daily RF threshold) in the denominator.
    """
    r = daily_rets.dropna()
    if r.empty:
        return np.nan
    excess = r - rf / TRADING_DAYS
    downside = r[r < rf / TRADING_DAYS] - rf / TRADING_DAYS
    d_std = downside.std(ddof=1)
    if d_std == 0 or pd.isna(d_std):
        return np.nan
    return float(excess.mean() / d_std * np.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate from start to end of the equity series."""
    if len(equity) < 2:
        return np.nan
    n_days = len(equity) - 1
    if n_days == 0 or equity.iloc[0] <= 0:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return np.nan
    return float(total_return ** (TRADING_DAYS / n_days) - 1)


def win_rate(blotter: pd.DataFrame) -> float:
    if blotter.empty:
        return np.nan
    return float((blotter["net_pnl"] > 0).mean())


def profit_factor(blotter: pd.DataFrame) -> float:
    """Sum of winners / |sum of losers|. >1 means profitable after costs."""
    if blotter.empty:
        return np.nan
    wins = blotter.loc[blotter["net_pnl"] > 0, "net_pnl"].sum()
    losses = blotter.loc[blotter["net_pnl"] < 0, "net_pnl"].sum()
    return _safe_div(wins, abs(losses))


def expectancy(blotter: pd.DataFrame) -> float:
    """Expected $ PnL per trade."""
    if blotter.empty:
        return np.nan
    return float(blotter["net_pnl"].mean())


def avg_return_per_trade(blotter: pd.DataFrame) -> float:
    if blotter.empty:
        return np.nan
    return float(blotter["trade_return"].mean())


def cost_drag(blotter: pd.DataFrame) -> dict:
    """Total cost as a fraction of gross PnL and of notional traded."""
    if blotter.empty:
        return {"cost_pct_of_gross": np.nan, "cost_pct_of_notional": np.nan}
    gross = blotter["gross_pnl"].sum()
    cost = blotter["total_cost"].sum()
    notional = (blotter["qty"] * blotter["entry_price"]).sum() + (
        blotter["qty"] * blotter["exit_price"]
    ).sum()
    return {
        "cost_pct_of_gross": _safe_div(cost, abs(gross)),
        "cost_pct_of_notional": _safe_div(cost, notional),
    }


def summarize(blotter: pd.DataFrame, ledger: pd.DataFrame, rf: float = RISK_FREE) -> dict:
    """One-stop metrics dict, suitable for dumping into a Plotly table."""
    equity = ledger["mkt_value"] if "mkt_value" in ledger else pd.Series(dtype=float)
    daily = ledger["daily_return"] if "daily_return" in ledger else pd.Series(dtype=float)

    cd = cost_drag(blotter)
    n = len(blotter)

    return {
        "n_trades": n,
        "win_rate": win_rate(blotter),
        "avg_return_per_trade": avg_return_per_trade(blotter),
        "avg_hold_days": float(blotter["hold_days"].mean()) if n else np.nan,
        "expectancy_usd": expectancy(blotter),
        "profit_factor": profit_factor(blotter),
        "total_net_pnl_usd": float(blotter["net_pnl"].sum()) if n else 0.0,
        "total_gross_pnl_usd": float(blotter["gross_pnl"].sum()) if n else 0.0,
        "total_cost_usd": float(blotter["total_cost"].sum()) if n else 0.0,
        "cost_pct_of_gross": cd["cost_pct_of_gross"],
        "cost_pct_of_notional": cd["cost_pct_of_notional"],
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(daily, rf),
        "sortino": sortino_ratio(daily, rf),
        "max_drawdown": max_drawdown(equity),
        "rf_rate": rf,
    }


def exit_type_breakdown(blotter: pd.DataFrame) -> pd.DataFrame:
    """Counts and mean PnL for each exit_type (profit_target/stop_loss/timeout)."""
    if blotter.empty:
        return pd.DataFrame(
            columns=["exit_type", "count", "pct", "avg_net_pnl", "avg_return", "win_rate"]
        )
    grp = blotter.groupby("exit_type").agg(
        count=("trade_id", "count"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_return=("trade_return", "mean"),
        win_rate=("good_trade", "mean"),
    ).reset_index()
    grp["pct"] = grp["count"] / grp["count"].sum()
    return grp[["exit_type", "count", "pct", "avg_net_pnl", "avg_return", "win_rate"]]
