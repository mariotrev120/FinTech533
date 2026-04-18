"""
Market data fetching — historical bars, real-time streaming, account info.
"""

import logging
import pandas as pd
from ib_async import Stock, Forex, Future, Option, Contract

logger = logging.getLogger(__name__)


def make_contract(symbol, sec_type='STK', exchange='SMART', currency='USD', **kwargs):
    """Create a qualified contract."""
    contract_map = {
        'STK': Stock,
        'CASH': Forex,
        'FUT': Future,
        'OPT': Option,
    }
    cls = contract_map.get(sec_type)
    if cls:
        if sec_type == 'CASH':
            return cls(symbol, exchange=exchange, currency=currency, **kwargs)
        return cls(symbol, exchange, currency, **kwargs)
    # Fallback to generic Contract
    c = Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency=currency, **kwargs)
    return c


def fetch_historical(ib, symbol, duration='1 Y', bar_size='1 day',
                     what_to_show='TRADES', sec_type='STK', exchange='SMART', currency='USD'):
    """Fetch historical bars and return a DataFrame."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=True,
        formatDate=1
    )

    df = pd.DataFrame(bars)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        logger.info(f"Fetched {len(df)} bars for {symbol}")
    else:
        logger.warning(f"No data returned for {symbol}")
    return df


def fetch_realtime(ib, symbol, sec_type='STK', exchange='SMART', currency='USD'):
    """Get a real-time snapshot (delayed if no market data subscription)."""
    contract = make_contract(symbol, sec_type, exchange, currency)
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, snapshot=True)
    ib.sleep(2)  # wait for data to arrive
    return ticker


def get_positions(ib):
    """Get current portfolio positions as a DataFrame."""
    positions = ib.positions()
    if not positions:
        logger.info("No open positions")
        return pd.DataFrame()
    data = [{
        'account': p.account,
        'symbol': p.contract.symbol,
        'secType': p.contract.secType,
        'quantity': p.position,
        'avgCost': p.avgCost,
        'value': p.position * p.avgCost
    } for p in positions]
    return pd.DataFrame(data)


def get_account_summary(ib):
    """Get account summary values."""
    summary = ib.accountSummary()
    data = [{
        'tag': item.tag,
        'value': item.value,
        'currency': item.currency
    } for item in summary]
    return pd.DataFrame(data)
