"""
Keltner Squeeze Breakout signal generation.

Strategy:
  1. Volatility Squeeze: Bollinger Bands (20, 2 std) lie INSIDE the Keltner
     Channels (20, 1.5 ATR). In this regime, vol is unusually compressed.
  2. Squeeze fires: transition from squeeze-on to squeeze-off, i.e. BB widens
     beyond the KC. That's the "coiled spring" release.
  3. Directional confirmation: on the same bar the squeeze fires, the close
     pierces the upper KC (long) or lower KC (short).

t-1 index shift on every indicator so the current bar is never part of its own
reference window (no look-ahead). Signals are generated at close of bar t, the
backtest engine executes at open of bar t+1.

All parameters live as module-level constants at the top. Adjust them here;
the backtest engine, feature engineering, and the website all read from these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Strategy parameters (explicit, named, at the top per the assignment)
# ----------------------------------------------------------------------
LOOKBACK           = 20      # period for Bollinger Bands and Keltner Channels
BB_STD_MULT        = 2.0     # Bollinger Band stddev multiplier
KC_ATR_MULT        = 1.5     # Keltner Channel ATR multiplier
ATR_WINDOW         = 14      # ATR period for exits (stop / target)
ADX_WINDOW         = 14      # ADX period for trend filter
ADX_MIN            = 20.0    # Participation rule: breakout only if ADX(14) >= 20
VOLUME_WINDOW      = 20      # rolling window for average daily volume
VOLUME_MULT        = 1.30    # Participation rule: breakout bar volume >= 1.3 x average
STOP_ATR_MULT      = 2.0     # stop distance in multiples of ATR (downside)
PROFIT_ATR_MULT    = 4.0     # profit target distance in multiples of ATR (upside)
TIMEOUT_DAYS       = 20      # market-close exit after this many bars in trade
POSITION_NOTIONAL  = 10_000  # target USD notional per trade
STARTING_CAPITAL   = 100_000 # starting equity for backtest
RISK_FREE          = 0.0375  # Sharpe risk-free rate (annualized)
ALLOW_LONG         = True
ALLOW_SHORT        = True


# ----------------------------------------------------------------------
# Indicator primitives
# ----------------------------------------------------------------------
def _wilder_ema(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (equivalent to ewm with alpha = 1/window, adjust=False)."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """TR = max(high-low, |high - prev_close|, |low - prev_close|)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range (Wilder)."""
    return _wilder_ema(true_range(df), window)


def adx(df: pd.DataFrame, window: int = ADX_WINDOW) -> pd.Series:
    """
    Average Directional Index (Wilder).

    Measures trend strength regardless of direction. Squeeze breakouts in a
    low-ADX environment are typically fakeouts; ADX >= 20 confirms the squeeze
    is resolving into a real trend.
    """
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = true_range(df)
    atr_s = _wilder_ema(tr, window)
    plus_di = 100.0 * _wilder_ema(plus_dm, window) / atr_s
    minus_di = 100.0 * _wilder_ema(minus_dm, window) / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_ema(dx, window)


def bollinger_bands(df: pd.DataFrame, window: int = LOOKBACK, std_mult: float = BB_STD_MULT):
    """
    Bollinger Bands built on the 20-day SMA ± std_mult * rolling std.

    Returns (mid, upper, lower).
    """
    mid = df["close"].rolling(window, min_periods=window).mean()
    std = df["close"].rolling(window, min_periods=window).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def keltner_channels(df: pd.DataFrame, window: int = LOOKBACK,
                     atr_mult: float = KC_ATR_MULT, atr_window: int = ATR_WINDOW):
    """
    Keltner Channels built on the EMA of close ± atr_mult * ATR.

    Uses EMA (more responsive than SMA) for the mid-line per Carter's
    original formulation.

    Returns (mid, upper, lower).
    """
    mid = df["close"].ewm(span=window, adjust=False, min_periods=window).mean()
    a = atr(df, atr_window)
    return mid, mid + atr_mult * a, mid - atr_mult * a


def add_indicators(df: pd.DataFrame,
                   lookback: int = LOOKBACK,
                   atr_window: int = ATR_WINDOW,
                   adx_window: int = ADX_WINDOW,
                   volume_window: int = VOLUME_WINDOW,
                   bb_std_mult: float = BB_STD_MULT,
                   kc_atr_mult: float = KC_ATR_MULT) -> pd.DataFrame:
    """Return a new frame augmented with BB, KC, ATR, ADX, vol_ma, squeeze flags."""
    out = df.copy()
    bb_mid, bb_up, bb_lo = bollinger_bands(out, lookback, bb_std_mult)
    kc_mid, kc_up, kc_lo = keltner_channels(out, lookback, kc_atr_mult, atr_window)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_lo
    out["kc_mid"] = kc_mid
    out["kc_upper"] = kc_up
    out["kc_lower"] = kc_lo
    out["atr"] = atr(out, atr_window)
    out["adx"] = adx(out, adx_window)
    out["vol_ma"] = out["volume"].rolling(volume_window, min_periods=volume_window).mean()
    # Squeeze: BB lies inside KC on both sides
    out["squeeze_on"] = (out["bb_upper"] < out["kc_upper"]) & (out["bb_lower"] > out["kc_lower"])
    return out


# ----------------------------------------------------------------------
# Keltner Squeeze Breakout detection
# ----------------------------------------------------------------------
def detect_breakouts(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    bb_std_mult: float = BB_STD_MULT,
    kc_atr_mult: float = KC_ATR_MULT,
    atr_window: int = ATR_WINDOW,
    adx_window: int = ADX_WINDOW,
    volume_window: int = VOLUME_WINDOW,
    volume_mult: float = VOLUME_MULT,
    adx_min: float = ADX_MIN,
    allow_long: bool = ALLOW_LONG,
    allow_short: bool = ALLOW_SHORT,
) -> pd.Series:
    """
    Identify Keltner Squeeze Breakouts with Volume + ADX participation filters.

    A signal fires on bar t when ALL of:
      (a) The squeeze was ON as of bar t-1  (BB fully inside KC).
      (b) The squeeze is OFF as of bar t    (BB has expanded beyond KC).
      (c) The close of bar t breaks the upper KC (long) or lower KC (short).
      (d) Volume confirmation: bar t volume >= volume_mult * 20-day avg volume
          (the squeeze fires with real institutional participation, not noise).
      (e) Trend confirmation: ADX(14) on bar t-1 >= adx_min (the squeeze is
          resolving into a trending regime, not choppy mean-reversion).

    All reference bands and indicator values are taken with a t-1 shift so the
    current bar is never part of its own reference. Execution (done by the
    backtest engine) happens at the OPEN of bar t+1.

    Returns
    -------
    pd.Series of {+1, -1, 0} aligned to df.index
        +1 : confirmed long breakout
        -1 : confirmed short breakout
         0 : no signal
    """
    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input frame missing columns: {missing}")

    enriched = add_indicators(df,
                              lookback=lookback,
                              atr_window=atr_window,
                              adx_window=adx_window,
                              volume_window=volume_window,
                              bb_std_mult=bb_std_mult,
                              kc_atr_mult=kc_atr_mult)

    # t-1 shifted reference bands (the channels as known at the START of bar t)
    kc_upper_prev = enriched["kc_upper"].shift(1)
    kc_lower_prev = enriched["kc_lower"].shift(1)
    squeeze_prev = enriched["squeeze_on"].shift(1)
    squeeze_now = enriched["squeeze_on"]
    vol_ma_prev = enriched["vol_ma"].shift(1)
    adx_prev = enriched["adx"].shift(1)

    fired = squeeze_prev.fillna(False) & (~squeeze_now.fillna(True))
    vol_ok = enriched["volume"] >= (volume_mult * vol_ma_prev)
    # ADX is a lagging indicator inside a squeeze (vol is compressed). We
    # measure ADX at the breakout bar itself — it is known at close(t) and
    # execution is at open(t+1), so no look-ahead. If ADX is still depressed
    # at the fire bar, the squeeze is resolving into chop, not a trend.
    trend_ok = enriched["adx"] >= adx_min

    close = enriched["close"]
    long_signal = allow_long & fired & (close > kc_upper_prev) & vol_ok & trend_ok
    short_signal = allow_short & fired & (close < kc_lower_prev) & vol_ok & trend_ok

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_signal.fillna(False).values] = 1
    sig[short_signal.fillna(False).values] = -1
    return sig
