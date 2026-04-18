"""
IBKR connection manager with auto-reconnect and logging.
"""

import logging
from ib_async import IB
from config import TWS_HOST, TWS_PORT, CLIENT_ID, LOG_LEVEL

logger = logging.getLogger('trading')
logger.setLevel(getattr(logging, LOG_LEVEL))
if not logger.handlers:
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler('trading.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def connect(host=TWS_HOST, port=TWS_PORT, client_id=CLIENT_ID, timeout=20):
    """Connect to TWS/Gateway with error handling."""
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        logger.info(f"Connected to IBKR at {host}:{port} (client {client_id})")
        logger.info(f"Account: {ib.managedAccounts()}")
        return ib
    except ConnectionRefusedError:
        logger.error(f"Connection refused at {host}:{port}. Is TWS running with API enabled?")
        raise
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise


def disconnect(ib):
    """Cleanly disconnect from TWS."""
    if ib and ib.isConnected():
        ib.disconnect()
        logger.info("Disconnected from IBKR")
