"""
Order management — place, monitor, and cancel orders.
"""

import logging
from ib_async import MarketOrder, LimitOrder, StopOrder, StopLimitOrder
from data import make_contract

logger = logging.getLogger(__name__)


def place_market_order(ib, symbol, action, quantity, sec_type='STK', exchange='SMART', currency='USD'):
    """Place a market order. action = 'BUY' or 'SELL'."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    logger.info(f"Market {action} {quantity} {symbol} — Order ID: {trade.order.orderId}")
    return trade


def place_limit_order(ib, symbol, action, quantity, limit_price,
                      sec_type='STK', exchange='SMART', currency='USD'):
    """Place a limit order."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)
    order = LimitOrder(action, quantity, limit_price)
    trade = ib.placeOrder(contract, order)
    logger.info(f"Limit {action} {quantity} {symbol} @ {limit_price} — Order ID: {trade.order.orderId}")
    return trade


def place_stop_order(ib, symbol, action, quantity, stop_price,
                     sec_type='STK', exchange='SMART', currency='USD'):
    """Place a stop order."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)
    order = StopOrder(action, quantity, stop_price)
    trade = ib.placeOrder(contract, order)
    logger.info(f"Stop {action} {quantity} {symbol} @ {stop_price} — Order ID: {trade.order.orderId}")
    return trade


def place_stop_limit_order(ib, symbol, action, quantity, stop_price, limit_price,
                           sec_type='STK', exchange='SMART', currency='USD'):
    """Place a stop-limit order."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)
    order = StopLimitOrder(action, quantity, stop_price, limit_price)
    trade = ib.placeOrder(contract, order)
    logger.info(f"StopLimit {action} {quantity} {symbol} stop={stop_price} limit={limit_price}")
    return trade


def cancel_order(ib, trade):
    """Cancel an open order."""
    ib.cancelOrder(trade.order)
    logger.info(f"Cancelled order {trade.order.orderId}")


def get_open_orders(ib):
    """Get all open orders."""
    return ib.openOrders()


def get_trades(ib):
    """Get all trades (filled and pending)."""
    return ib.trades()
