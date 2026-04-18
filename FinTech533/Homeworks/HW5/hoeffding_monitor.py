"""
Hoeffding Regime Monitor (Egger & Vestal, 2025).

Two parallel one-sided monitors run after each completed OOS trade:

  1. Bernoulli Win-rate monitor W, bounded on [0, 1]:
         P(mu_W - Xbar_W >= t) <= exp(-2 * t^2 * N_eff)

  2. Bounded continuously-compounded trade-return monitor R, bounded on [a, b]:
         P(mu_R - Xbar_R >= t) <= exp(-2 * t^2 * N_eff / (b - a) ** 2)
     where [a, b] is the trade-return range implied by the strategy's exits.
     For ATR-symmetric exits (+PROFIT_ATR, -STOP_ATR), the span in ATR units is
     (PROFIT_ATR + STOP_ATR). Feed (b - a) directly.

Direction convention: we test the UNDERPERFORMANCE hypothesis. The tracked
quantity is mu - Xbar: positive values mean realized is worse than expected.
When Xbar falls, t grows, P shrinks, and H0 (favorable regime persists) becomes
less plausible.

Discipline (Anti-Leaking Manifest, Section 4):
  * Xbar is an EXPANDING mean (all trades to date). Rolling windows discard
    statistical power and violate the inequality's assumption set.
  * Autocorrelation correction uses the first-order autocorrelation of the
    trade-return series. If rho > 0, apply N_eff = N * (1 - rho) / (1 + rho)
    and clip to N. N_eff > N is forbidden.
  * mu (expected rate) MUST come from the training period only. No leakage.

Alert thresholds used by convention: 50% / 25% / 10%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ALERT_AMBER = 0.50   # "more likely than not beliefs are off"
ALERT_ORANGE = 0.25  # "significant risk beliefs no longer justified"
ALERT_RED = 0.10     # "almost certain regime change. halt."


# =======================
# Primitive: probability bound
# =======================
def hoeffding_bound(t: float, N_eff: float, ba_range: float = 1.0) -> float:
    """
    One-sided Hoeffding upper bound:
        P(Xbar - mu >= t) <= exp(-2 * t^2 * N_eff / (b - a) ** 2)

    `t` is the positive deviation we are bounding. For the UNDER-performance
    monitor, feed t = mu - Xbar.
    `ba_range` = b - a is the RV's support width (default 1.0 for [0, 1]).
    """
    if t <= 0 or N_eff <= 0 or ba_range <= 0:
        return 1.0
    expo = -2.0 * (t ** 2) * N_eff / (ba_range ** 2)
    # exp underflow protection
    if expo < -700:
        return 0.0
    return float(np.exp(expo))


# =======================
# Autocorrelation + N_eff
# =======================
def effective_N(series: pd.Series, N: int) -> float:
    """
    Compute effective sample size from first-order autocorrelation.

    N_eff = min( N * (1 - rho) / (1 + rho),  N )

    Requires at least 3 observations to estimate rho. For rho <= 0 or NaN,
    returns N unchanged (standard lower-bound behavior).
    """
    s = pd.Series(series).dropna().astype(float).reset_index(drop=True)
    if len(s) < 3:
        return float(N)
    rho = s.autocorr(lag=1)
    if rho is None or not np.isfinite(rho) or rho <= 0:
        return float(N)
    denom = 1.0 + rho
    if denom <= 0:
        return float(N)
    N_eff = N * (1.0 - rho) / denom
    return float(max(0.0, min(N_eff, N)))


# =======================
# Two monitors running together
# =======================
@dataclass
class MonitorConfig:
    mu_W: float            # expected win rate (training baseline)
    mu_R: float            # expected per-trade return (training baseline)
    R_range: float         # (b - a) for the return series in the same units as mu_R
    label_W: str = "good_trade"
    label_R: str = "trade_return"


def run_monitor(trades: pd.DataFrame, cfg: MonitorConfig) -> pd.DataFrame:
    """
    Run the two-track Hoeffding monitor over an ordered sequence of trades.

    `trades` must contain at least the columns specified by cfg.label_W (binary
    good-trade indicator) and cfg.label_R (per-trade return). Rows are assumed
    to be ordered by trade completion time.

    Returns a DataFrame with one row per trade and the following columns:
        trade_n            : cumulative trade count (1..N)
        Xbar_W, t_W, P_W   : expanding mean, deviation from mu_W, bound
        Xbar_R, t_R, P_R   : expanding mean, deviation from mu_R, bound
        rho, N_eff         : AR(1) autocorrelation on labels and adjusted N
        P_min              : min(P_W, P_R) (the triggering monitor)
        alert              : "GREEN" / "AMBER" / "ORANGE" / "RED"
    """
    if trades.empty:
        return pd.DataFrame(columns=[
            "trade_n","Xbar_W","t_W","P_W","Xbar_R","t_R","P_R",
            "rho","N_eff","P_min","alert"
        ])

    w = pd.Series(trades[cfg.label_W].astype(int).values)
    r = pd.Series(trades[cfg.label_R].astype(float).values)

    rows = []
    for k in range(1, len(trades) + 1):
        w_k = w.iloc[:k]
        r_k = r.iloc[:k]

        Xbar_W = float(w_k.mean())
        Xbar_R = float(r_k.mean())

        t_W = max(0.0, cfg.mu_W - Xbar_W)   # underperformance on win rate
        t_R = max(0.0, cfg.mu_R - Xbar_R)   # underperformance on return

        # N_eff from the binary label series (Egger & Vestal). This adjusts
        # both monitors with the same N_eff so the regime signal is coherent.
        N_eff_k = effective_N(w_k, N=k)
        rho_k = w_k.autocorr(lag=1) if k >= 3 else np.nan

        P_W = hoeffding_bound(t_W, N_eff_k, ba_range=1.0)
        P_R = hoeffding_bound(t_R, N_eff_k, ba_range=cfg.R_range)
        P_min = min(P_W, P_R)

        if P_min < ALERT_RED:
            alert = "RED"
        elif P_min < ALERT_ORANGE:
            alert = "ORANGE"
        elif P_min < ALERT_AMBER:
            alert = "AMBER"
        else:
            alert = "GREEN"

        rows.append({
            "trade_n": k,
            "Xbar_W": Xbar_W, "t_W": t_W, "P_W": P_W,
            "Xbar_R": Xbar_R, "t_R": t_R, "P_R": P_R,
            "rho": float(rho_k) if pd.notna(rho_k) else np.nan,
            "N_eff": N_eff_k,
            "P_min": P_min, "alert": alert,
        })

    return pd.DataFrame(rows)
