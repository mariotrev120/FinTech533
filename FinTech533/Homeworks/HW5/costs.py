"""
Trading cost and tax model.

Every trade in the blotter is charged realistic round-trip costs, and realized
PnL is adjusted for short-term capital gains tax at the period level. Numbers
reflect IBKR Pro tiered pricing as published in 2024-2025 and the current
SEC / FINRA regulatory fee schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Cost model parameters (explicit, easy to override)
# ----------------------------------------------------------------------
IBKR_PER_SHARE        = 0.0035       # IBKR Pro tiered base rate (USD / share)
IBKR_MIN_PER_ORDER    = 0.35         # min commission per order (USD)
IBKR_MAX_PCT_OF_TRADE = 0.01         # capped at 1% of trade value
SEC_FEE_RATE          = 0.0000278    # sell-side fee, 27.80 bps per million -> 0.00278% notional (2025)
FINRA_TAF_PER_SHARE   = 0.000166     # sell-side FINRA Trading Activity Fee (USD / share)
FINRA_TAF_CAP         = 8.30         # capped at $8.30 per trade
SLIPPAGE_BPS          = 5            # bid-ask / impact slippage in basis points per side
SHORT_BORROW_ANNUAL   = 0.03         # 3% annualized borrow fee for short positions
TRADING_DAYS_PER_YEAR = 252
TAX_RATE              = 0.30         # short-term capital gains (marginal ordinary rate proxy)


@dataclass
class CostConfig:
    """All cost parameters bundled so the backtest engine takes a single object."""

    ibkr_per_share: float = IBKR_PER_SHARE
    ibkr_min_per_order: float = IBKR_MIN_PER_ORDER
    ibkr_max_pct_of_trade: float = IBKR_MAX_PCT_OF_TRADE
    sec_fee_rate: float = SEC_FEE_RATE
    finra_taf_per_share: float = FINRA_TAF_PER_SHARE
    finra_taf_cap: float = FINRA_TAF_CAP
    slippage_bps: float = SLIPPAGE_BPS
    short_borrow_annual: float = SHORT_BORROW_ANNUAL
    tax_rate: float = TAX_RATE
    trading_days: int = TRADING_DAYS_PER_YEAR


# ----------------------------------------------------------------------
# Per-leg cost computations
# ----------------------------------------------------------------------
def commission(qty: int, price: float, cfg: CostConfig) -> float:
    """IBKR Pro tiered: $0.0035/share, min $0.35, cap 1% of trade value."""
    shares = abs(qty)
    if shares == 0:
        return 0.0
    raw = shares * cfg.ibkr_per_share
    raw = max(raw, cfg.ibkr_min_per_order)
    cap = cfg.ibkr_max_pct_of_trade * shares * price
    return min(raw, cap)


def regulatory_fees(qty: int, price: float, side: str, cfg: CostConfig) -> float:
    """
    SEC Section 31 fee + FINRA Trading Activity Fee.

    Both are charged on SELL side only (regardless of long-sell or short-sell).
    """
    if side.lower() != "sell":
        return 0.0
    shares = abs(qty)
    if shares == 0:
        return 0.0
    notional = shares * price
    sec = cfg.sec_fee_rate * notional
    taf = min(shares * cfg.finra_taf_per_share, cfg.finra_taf_cap)
    return sec + taf


def slippage(qty: int, price: float, cfg: CostConfig) -> float:
    """Slippage as a fraction of trade notional, per side."""
    return abs(qty) * price * (cfg.slippage_bps / 10_000.0)


def borrow_cost(
    qty: int, entry_price: float, hold_days: int, direction: str, cfg: CostConfig
) -> float:
    """
    Short-borrow charge. Long positions pay zero.

    Simple linear accrual: annual_rate * notional * days / 252.
    """
    if direction.lower() != "short":
        return 0.0
    notional = abs(qty) * entry_price
    return notional * cfg.short_borrow_annual * hold_days / cfg.trading_days


def apply_tax(net_pnl_total: float, cfg: CostConfig) -> float:
    """
    Apply short-term capital gains tax to aggregate positive net PnL.

    Losses within the same period offset gains naturally because we tax the
    aggregate, not each trade. If the aggregate is a loss, tax owed is zero
    (we don't model carry-forwards or refunds).
    """
    if net_pnl_total <= 0:
        return net_pnl_total
    return net_pnl_total * (1.0 - cfg.tax_rate)


# ----------------------------------------------------------------------
# Round-trip cost bundle (entry + exit, returned as itemized dict)
# ----------------------------------------------------------------------
def round_trip_costs(
    qty: int,
    entry_price: float,
    exit_price: float,
    hold_days: int,
    direction: str,
    cfg: CostConfig,
) -> dict:
    """
    Compute every cost component for one completed trade.

    `qty` is always entered as positive share count (unsigned). `direction`
    determines which side is the opening and which is the closing transaction.
    """
    if direction.lower() == "long":
        entry_side, exit_side = "buy", "sell"
    else:
        entry_side, exit_side = "sell", "buy"  # short sale then buy-to-cover

    comm = commission(qty, entry_price, cfg) + commission(qty, exit_price, cfg)
    reg = regulatory_fees(qty, entry_price, entry_side, cfg) + regulatory_fees(
        qty, exit_price, exit_side, cfg
    )
    slip = slippage(qty, entry_price, cfg) + slippage(qty, exit_price, cfg)
    borrow = borrow_cost(qty, entry_price, hold_days, direction, cfg)

    return {
        "commission": comm,
        "reg_fees": reg,
        "slippage": slip,
        "borrow_cost": borrow,
        "total_cost": comm + reg + slip + borrow,
    }
