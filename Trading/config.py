"""
Trading system configuration.
Update these settings for your environment.
"""

# IBKR TWS Connection
TWS_HOST = '172.29.208.1'  # Windows host IP from WSL
TWS_PORT = 7497            # 7497 = TWS paper, 7496 = TWS live, 4002 = Gateway paper, 4001 = Gateway live
CLIENT_ID = 2

# Logging
LOG_FILE = 'trading.log'
LOG_LEVEL = 'INFO'
