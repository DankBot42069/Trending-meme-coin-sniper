#!/usr/bin/env python3
"""
Configuration for the trading bot.
Contains only essential parameters that are actively used in the code.
"""
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# File Paths - Dynamic detection for cross-platform compatibility
# ------------------------------------------------------------------------------
# Determine the script's location dynamically
if getattr(sys, 'frozen', False):
    # If the application is frozen (executable)
    BASE_PATH = os.path.dirname(sys.executable)
else:
    # If running as a script
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Create paths relative to the script location
DATA_PATH = os.path.join(BASE_PATH, "data")
DB_DIR = os.path.join(DATA_PATH, "database")

# Ensure the directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Database settings
DB_PATH = os.path.join(DB_DIR, "trading_bot.db")

# ------------------------------------------------------------------------------
# Blockchain & API Settings
# ------------------------------------------------------------------------------
SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111112"
MY_ADDRESS = '3C6ApByWpV63zUQBrb9QtRH8am8HGUafmbxCtxswg5yV'
API_KEY = "AtQBE9G-D0kyi7UDs29VHVk3stCX5_wR76waqi5fk9AX"
drpc_url = f"https://lb.drpc.org/ogrpc?network=solana&dkey={API_KEY}"

# ------------------------------------------------------------------------------
# Trading Parameters
# ------------------------------------------------------------------------------
# Performance Tuning
MAX_WORKERS = 25               # Maximum number of threads for token processing
TOKEN_BATCH_SIZE = 10          # Number of tokens to process in parallel 
API_TIMEOUT_SECONDS = 5        # Timeout for API calls
DB_WRITE_INTERVAL = 10         # Seconds between database writes
MIN_API_REQUEST_INTERVAL = 0.1  # Minimum time between API requests per token

# Trade Execution
USDC_BUY_AMOUNT = 0.50         # Amount in USD to spend per purchase
SLIPPAGE = 500                 # 5.00% slippage (in basis points)
MAX_RETRIES = 3                # Maximum retry attempts for transactions

# Trade Strategy
TRAILING_STOP_LOSS_PERCENT = 0.05  # 5% trailing stop
PROFIT_TARGET = 1.00               # 100% profit target (double your entry)
STOP_LOSS = 0.10                   # 10% stop loss
DONT_SELL_IF_BELOW_PNL = -0.60     # Don't sell if below 40% loss as we can just burn the coin for more than the sell most of the time 
PRICE_STALL_THRESHOLD = 0.01       # If price change is less than 0.1% over recent cycles
PRICE_STALL_CYCLES = 40            # Number of cycles to consider for price stalling check
PNL_JUMP_THRESHOLD = 0.15          # If PnL jumps more than 15% in one cycle, trigger sell

# Check intervals
WALLET_UPDATE_INTERVAL = 3         # How often to update wallet/trades in seconds
TOKEN_CHECK_INTERVAL = 0.5         # How frequently each token is checked
PNL_STALL_THRESHOLD = 0.02         # 2% threshold for PnL stalling
PNL_STALL_CYCLES = 25              # Number of iterations to check for PnL stalling
RECENTLY_BOUGHT_WINDOW_HOURS = 5   # Don't rebuy tokens purchased within this timeframe

# Volume Change Thresholds
BUY_VOLUME_DROP_THRESHOLD = 0.35 
SELL_VOLUME_SHARE_INCREASE_THRESHOLD = 0.50
TOTAL_VOLUME_DROP_THRESHOLD = 0.25
TOTAL_VOLUME_SPIKE_THRESHOLD = 1.00
SELL_DIFF_THRESHOLD = 0            # Sell if (buy_volume - sell_volume) is less than 0
DIFF_DROP_THRESHOLD = 0.30

# Top Buyers thresholds (in percentage points)
TOP_HOLD_DROP_THRESHOLD = 3.5      # If highest hold % drops by 3.5 points
TOP_SOLD_RISE_THRESHOLD = 8.0      # If sold % increases by 8 points
TOP_BOUGHT_MORE_RISE_THRESHOLD = 35.0

# Do Not Trade List - Default tokens to never trade
DO_NOT_TRADE_LIST = [
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC address
    "So11111111111111111111111111111111111111111",
    "So11111111111111111111111111111111111111112",
    "7xmFoP1PBhey43HybJYhBxWTP2uFS1M3Z9RzBEw3sLy3"
]

# Initialize database function
def initialize_database():
    """
    Initialize the database structure.
    """
    try:
        # Import inside function to avoid circular imports
        import database as db
        
        # Initialize database
        db.db_init()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")