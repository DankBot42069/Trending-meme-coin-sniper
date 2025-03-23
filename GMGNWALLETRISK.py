#!/usr/bin/env python3
import os
import time
import logging
import asyncio
import threading
import traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
import websocket  # synchronous websocket-client library
import tkinter as tk
from tkinter import ttk

import config as c
import functions as n
import dontshare as d
import database as db  # Import the new database module
from gmgn.client import gmgn

#########################################
# CONFIGURATION VALUES
#########################################
TRAILING_STOP_LOSS_PERCENT = 0.05  # 5% trailing stop
PROFIT_TARGET = 1.00               # 100% profit target (double your entry)
STOP_LOSS = 0.10                   # 10% stop loss
DONT_SELL_IF_BELOW_PNL = -0.40     # Don't sell if below 40% loss
PRICE_STALL_THRESHOLD = 0.01       # e.g., if price change is less than 0.1% over recent cycles
PRICE_STALL_CYCLES = 40            # number of cycles to consider for price stalling check
PNL_JUMP_THRESHOLD = 0.15          # if pnl jumps more than 15% in one cycle, trigger sell

# Check intervals
WALLET_UPDATE_INTERVAL = 5         # How often to update wallet/trades in seconds
TOKEN_CHECK_INTERVAL = 1           # How frequently each token is checked (in its own thread)
PNL_STALL_THRESHOLD = 0.02         # 2% threshold for pnl stalling
PNL_STALL_CYCLES = 25              # Number of iterations to check for pnl stalling

#########################################
# Volume Change Thresholds
#########################################
BUY_VOLUME_DROP_THRESHOLD = 0.35 
SELL_VOLUME_SHARE_INCREASE_THRESHOLD = 0.50
TOTAL_VOLUME_DROP_THRESHOLD = 0.25
TOTAL_VOLUME_SPIKE_THRESHOLD = 1.00
SELL_DIFF_THRESHOLD = 0            # Sell if (buy_volume - sell_volume) is less than 0
DIFF_DROP_THRESHOLD = 0.30
# Set a base hold value and compute a multiplier if current hold_pct is high
base_hold = 75.0

#########################################
# Top Buyers thresholds (in percentage points)
#########################################
TOP_HOLD_DROP_THRESHOLD = 3.5      # e.g., if highest hold % drops by 3.5 points
TOP_SOLD_RISE_THRESHOLD = 8.0      # e.g., if sold % increases by 8 points
TOP_BOUGHT_MORE_RISE_THRESHOLD = 35.0  

# Adjustment percentage for top buyers thresholds (applied as a multiplier)
ADJUST_THRESHOLD_PERCENT = 15      # e.g. if hold_pct > base_hold then thresholds multiply by (1+0.15)

#########################################
# Wallet Holdings Filtering
#########################################
HOLDING_START_FILTER_DAYS = 100

# Initialize the database
db.db_init()

#########################################
# Logging
#########################################
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#########################################
# Global Variables
#########################################
bought_tokens_info = {}  # Key: token_address, Value: dict of tracking information
ignored_tokens = []      # List of tokens to ignore
active_sell_threads = {}
active_sell_threads_lock = threading.Lock()
active_monitor_threads = {}
active_monitor_threads_lock = threading.Lock()
# Updated top buyers history now includes extreme values
top_buyers_history = {}

#########################################
# Custom Colored Print Function (cprint)
#########################################
def cprint(message, color="white"):
    """
    Print colored text to the console.
    
    Args:
        message (str): The message to print
        color (str): Color name (red, green, yellow, blue, magenta, cyan, white)
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{message}{colors['reset']}")

def colored(text, passed: bool):
    """
    Return colored text based on pass/fail status.
    
    Args:
        text (str): The text to color
        passed (bool): True for green (pass), False for red (fail)
        
    Returns:
        str: Colored text with ANSI escape codes
    """
    return f"\033[92m{text}\033[0m" if passed else f"\033[91m{text}\033[0m"

#########################################
# Safe Converters
#########################################
def safe_float(value, default=0.0):
    """
    Safely convert a value to float, returning a default if conversion fails.
    
    Args:
        value: The value to convert
        default (float): Default value to return if conversion fails
        
    Returns:
        float: The converted value or default
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int(value, default=0):
    """
    Safely convert a value to int, returning a default if conversion fails.
    
    Args:
        value: The value to convert
        default (int): Default value to return if conversion fails
        
    Returns:
        int: The converted value or default
    """
    try:
        return int(float(value))
    except Exception:
        return default

#########################################
# Helper: Robust Token Address Extraction
#########################################
def extract_token_address(token):
    """
    Extract a token address from various data formats.
    
    Args:
        token: Token information (dict or str)
        
    Returns:
        str: The token address or empty string if not found
    """
    if isinstance(token, dict):
        return token.get('address', '').strip()
    elif isinstance(token, str):
        return token.strip()
    return ''

#########################################
# Trade Handling
#########################################
def log_trade(trade_details):
    """
    Log a trade to the database.
    
    Args:
        trade_details (dict): Details of the trade
        
    Returns:
        bool: True if successful, False otherwise
    """
    return db.log_trade_to_db(trade_details)

def display_trade_summary():
    """
    Display a summary of recent trades.
    """
    try:
        # Query for total USD from trades
        query = "SELECT SUM(CAST(pnl_usd AS REAL)) AS total_usd FROM trades"
        result = db.query_to_dataframe(query)
        
        if not result.empty and 'total_usd' in result.columns:
            total_usd = result['total_usd'].iloc[0]
            if pd.notna(total_usd):
                cprint(f"Trade Summary: Total USD from trades: {total_usd:.2f}", "cyan")
                return
        
        cprint("No trades recorded or no profit data available.", "yellow")
    except Exception as e:
        cprint(f"Error calculating trade summary: {e}", "yellow")

#########################################
# Print Token Metrics in Chart Format
#########################################
def print_token_chart(token_address, buy_vol, sell_vol, diff, computed_drop_pct,
                      computed_volume_change_pct, current_pnl, total_pnl,
                      current_price, entry_price, trailing_stop, total_usd_pnl,
                      hold_pct, bought_more_pct, sold_pct,
                      highest_buy_pct, lowest_sold_pct):
    """
    Print a formatted chart of token metrics.
    
    Args:
        token_address: Token address
        buy_vol: 1m buy volume
        sell_vol: 1m sell volume
        diff: Difference between buy and sell volumes
        computed_drop_pct: Computed price drop percentage
        computed_volume_change_pct: Computed volume change percentage
        current_pnl: Current PnL (since last sell)
        total_pnl: Overall PnL
        current_price: Current token price
        entry_price: Entry price
        trailing_stop: Trailing stop price
        total_usd_pnl: Total PnL in USD
        hold_pct: Percentage of holders holding
        bought_more_pct: Percentage of holders who bought more
        sold_pct: Percentage of holders who sold
        highest_buy_pct: Highest recorded hold percentage
        lowest_sold_pct: Lowest recorded sold percentage
    """
    diff_pass = (diff >= SELL_DIFF_THRESHOLD)
    drop_pass = (computed_drop_pct is not None and computed_drop_pct < DIFF_DROP_THRESHOLD)
    volume_pass = True  # Always computed.
    pnl_pass = (current_pnl >= PROFIT_TARGET)
    price_vs_stop_pass = (current_price >= trailing_stop)
    
    # Calculate top buyers trigger percentages.
    trigger_hold_pct = highest_buy_pct - TOP_HOLD_DROP_THRESHOLD if highest_buy_pct is not None else 0
    trigger_sold_pct = lowest_sold_pct + TOP_SOLD_RISE_THRESHOLD if lowest_sold_pct is not None else 0

    # Compute sell price thresholds based on entry_price and configuration.
    profit_target_price = entry_price * (1 + PROFIT_TARGET)
    stop_loss_price = entry_price * (1 - STOP_LOSS)
    dont_sell_below_price = entry_price * (1 + DONT_SELL_IF_BELOW_PNL)
    # trailing_stop is already computed as new_max * (1 - TRAILING_STOP_LOSS_PERCENT)

    # --- Volume Averages Section ---
    token_data = bought_tokens_info.get(token_address, {})
    buy_history = token_data.get('buy_vol_history', [])
    sell_history = token_data.get('sell_vol_history', [])
    if len(buy_history) >= 10:
        # Use only the last 50 values if available.
        relevant_buy_history = buy_history[-50:] if len(buy_history) > 50 else buy_history
        relevant_sell_history = sell_history[-50:] if len(sell_history) > 50 else sell_history
        avg_buy_vol = sum(relevant_buy_history) / len(relevant_buy_history)
        avg_sell_vol = sum(relevant_sell_history) / len(relevant_sell_history)
        # Use configuration thresholds instead of fixed multipliers.
        buy_trigger_vol = avg_buy_vol * (1 - BUY_VOLUME_DROP_THRESHOLD)
        sell_trigger_vol = avg_sell_vol * (1 + SELL_VOLUME_SHARE_INCREASE_THRESHOLD)
        vol_section = (
            f"║ Avg Buy Volume         : {avg_buy_vol:15.8f}    (Trigger if below {buy_trigger_vol:15.8f})\n"
            f"║ Avg Sell Volume        : {avg_sell_vol:15.8f}    (Trigger if above {sell_trigger_vol:15.8f})"
        )
    else:
        vol_section = "║ Avg Volumes            : N/A (Need at least 10 data points)                     "

    chart = (
        "╔════════════════════════════════════════════════════════════════════════╗\n"
        f"║ Token: {token_address:<68}║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        f"║ 1m Buy Volume            : {buy_vol:15.8f}                                 ║\n"
        f"║ 1m Sell Volume           : {sell_vol:15.8f}                                 ║\n"
        f"║ 1m Buy-Sell Diff         : {diff:15.8f}   ({colored('Pass' if diff_pass else 'Fail', diff_pass)})      ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        f"║ Diff Drop Percentage     : {(computed_drop_pct*100) if computed_drop_pct is not None else 'N/A':>7}%   ({colored('Pass' if drop_pass else 'Fail', drop_pass)})          ║\n"
        f"║ Total Volume Change      : {computed_volume_change_pct*100:15.2f}%   ({colored('Pass' if volume_pass else 'Fail', volume_pass)})          ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        f"║ Current PnL (since last sell) : {current_pnl*100:15.2f}%   ({colored('Pass' if pnl_pass else 'Fail', pnl_pass)})       ║\n"
        f"║ Total PnL (overall)      : {total_pnl*100:15.2f}%                                 ║\n"
        f"║ Total PnL (USD)          : {total_usd_pnl:15.2f} USD                              ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        f"║ Current Price            : {current_price:15.8f}                                 ║\n"
        f"║ Entry Price              : {entry_price:15.8f}                                 ║\n"
        f"║ Trailing Stop            : {trailing_stop:15.8f}   ({colored('Pass' if price_vs_stop_pass else 'Fail', price_vs_stop_pass)})  ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        "║       SELL PRICE THRESHOLDS (Computed from Entry Price)                  ║\n"
        f"║    Profit Target Price   : {profit_target_price:15.8f}  (Entry x (1+{PROFIT_TARGET:.2f}))         ║\n"
        f"║    Stop Loss Price       : {stop_loss_price:15.8f}  (Entry x (1-{STOP_LOSS:.2f}))         ║\n"
        f"║    Minimum Sell Price    : {dont_sell_below_price:15.8f}  (Entry x (1+{DONT_SELL_IF_BELOW_PNL:.2f}))         ║\n"
        f"║    Trailing Stop Price   : {trailing_stop:15.8f}  (Based on max price & {TRAILING_STOP_LOSS_PERCENT:.2f})         ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        "║               TOP BUYERS TRIGGER VALUES (Based on percentages)           ║\n"
        f"║ Top Buyers: Hold {hold_pct:6.2f}%, Bought More {bought_more_pct:6.2f}%, Sold {sold_pct:6.2f}%       ║\n"
        f"║ Highest Hold %%          : {highest_buy_pct:15.2f}%                              ║\n"
        f"║ Lowest Sold %%           : {lowest_sold_pct if lowest_sold_pct is not None else 'N/A':>15}                              ║\n"
        f"║ Sell Trigger Values:     Hold drops to {trigger_hold_pct:6.2f}%, Sold rises to {trigger_sold_pct:6.2f}%          ║\n"
        f"║ Top PnL: {total_usd_pnl:15.2f} USD                                           ║\n"
        "╠════════════════════════════════════════════════════════════════════════╣\n"
        "║              VOLUME BASED TRIGGER VALUES (Last 50 values, min 10)          ║\n"
        f"{vol_section}\n"
        "╚════════════════════════════════════════════════════════════════════════╝"
    )
    cprint(chart, "cyan")

#########################################
# Log Sell Diagnostics
#########################################
def log_sell_debug_info(token_address, condition, diagnostics):
    """
    Log sell diagnostics to the database.
    
    Args:
        token_address (str): The token address
        condition (str): The condition that triggered the sell
        diagnostics (dict): Diagnostic information
        
    Returns:
        bool: True if successful, False otherwise
    """
    return db.log_sell_debug_info_to_db(token_address, condition, diagnostics)

#########################################
# PERIODIC WALLET UPDATES
#########################################
def update_held_tokens():
    """
    Update the list of tokens currently held in the wallet.
    """
    global ignored_tokens
    try:
        df = n.df_wallet_holdings(c.MY_ADDRESS)
        if df.empty:
            cprint("No data returned from df_wallet_holdings, skipping update.", "yellow")
            return
        
        # Remove tokens no longer in the wallet
        for token in list(bought_tokens_info.keys()):
            matching_rows = df[df['token'].apply(lambda x: x.get('address', '').strip() if isinstance(x, dict) else str(x).strip()) == token.strip()]
            if matching_rows.empty:
                cprint(f"{token} is no longer in wallet. Removing from tracking.", "yellow")
                del bought_tokens_info[token]
                db.remove_buy_price_from_db(token)
            else:
                token_balance = safe_float(matching_rows.iloc[0]['balance'])
                if token_balance <= 0:
                    cprint(f"{token} has 0 balance. Removing from tracking.", "yellow")
                    del bought_tokens_info[token]
                    db.remove_buy_price_from_db(token)
        
        # Add new tokens or update existing ones
        for _, row in df.iterrows():
            token_raw = row.get('token', '')
            token_address = extract_token_address(token_raw)
            balance = safe_float(row.get('balance', 0))
            usd_value = safe_float(row.get('usd_value', 0))
            
            if not token_address or balance <= 0:
                continue
            
            if token_address in ignored_tokens:
                continue
            
            if usd_value < 0.0001:
                ignored_tokens.append(token_address)
                continue
            
            if token_address not in bought_tokens_info:
                stored_buy_price = db.get_buy_price_from_db(token_address)
                token_info = n.get_token_info(token_address)
                current_price = safe_float(token_info.get("price", 0.0))
                buy_price = stored_buy_price if stored_buy_price is not None else current_price
                
                bought_tokens_info[token_address] = {
                    'buy_price': buy_price,
                    'original_buy_price': buy_price,
                    'entry_price': buy_price,
                    'timestamp': datetime.now(),
                    'max_price': buy_price,
                    'quantity': balance,
                    'old_sell_volume_1m': None,
                    'old_diff_1m': None,
                    'old_buy_volume_1m': None,
                    'old_total_volume_1m': None,
                    # Tracking extremes for top buyers
                    'highest_buy_pct': None,
                    'lowest_sold_pct': None,
                }
                
                cprint(f"Added token {token_address} balance={balance}, USD={usd_value:.4f} to tracking.", "yellow")
            else:
                bought_tokens_info[token_address]['quantity'] = balance
                bought_tokens_info[token_address]['timestamp'] = datetime.now()
    except Exception as e:
        logger.error(f"Error updating held tokens: {e}")
        logger.error(traceback.format_exc())

#########################################
# SELL FUNCTION WITH RETRY LOGIC
#########################################
def sell_token(address, reason):
    """
    Sell a token with retry logic.
    
    Args:
        address (str): The token address
        reason (str): The reason for selling
    """
    MAX_RETRIES = 3
    retry_delay = 2  # seconds
    attempt = 0
    
    while attempt < MAX_RETRIES:
        try:
            df = n.df_wallet_holdings(c.MY_ADDRESS)
            if df.empty:
                cprint(f"❌ {address}: No wallet holdings data returned", "red")
                cleanup_token(address)
                return

            df["token_address"] = df["token"].apply(
                lambda t: t.get("address", "").strip() if isinstance(t, dict) else str(t).strip()
            )
            matching_rows = df[df["token_address"] == address.strip()]
            if matching_rows.empty:
                cprint(f"❌ {address}: No token balance found in wallet holdings", "red")
                cleanup_token(address)
                return

            token_balance = safe_float(matching_rows.iloc[0].get("balance", 0))
            if token_balance <= 0:
                cprint(f"❌ {address}: token sold (balance: {token_balance})", "red")
                cleanup_token(address)
                return

            if address.strip() == c.SOL_MINT_ADDRESS.strip():
                decimals_value = 9
            else:
                token_details = n.get_token_info(address)
                decimals_value = token_details.get("decimals", 6)
                
            amount_in_base_units = int(token_balance * (10 ** decimals_value))
            
            if amount_in_base_units <= 0:
                cprint(f"❌ {address}: Calculated amount too small after conversion", "red")
                return

            # Attempt to sell the tokens
            n.market_sell(address, str(amount_in_base_units))
            cprint(f"✅ {address}: Sell attempt for {token_balance:.4f} tokens (base units: {amount_in_base_units}) due to {reason}.", "green")
            
            # Wait briefly for the sell to process and then check wallet for balance
            time.sleep(2)
            df_post = n.df_wallet_holdings(c.MY_ADDRESS)
            matching_post = df_post[df_post["token"].apply(lambda t: extract_token_address(t)) == address.strip()]
            if matching_post.empty or safe_float(matching_post.iloc[0].get("balance", 0)) <= 0:
                # Sell successful
                price_info = n.get_usd_value(address)
                current_price = safe_float(price_info.get("usd_price", 0.0)) if isinstance(price_info, dict) else safe_float(price_info)
                trade_time = datetime.now()
                time_held = (trade_time - bought_tokens_info[address]["timestamp"]).total_seconds()
                entry_price = bought_tokens_info[address]["buy_price"]
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                pnl_usd = (current_price - entry_price) * token_balance

                trade_details = {
                    "token_address": address,
                    "trade_type": "full",
                    "quantity_sold": f"{token_balance:.4f}",
                    "pnl_percent": f"{pnl_percent:.2f}",
                    "pnl_usd": f"{pnl_usd:.2f}",
                    "trade_time": trade_time.isoformat(),
                    "time_held": f"{time_held:.0f} sec"
                }
                log_trade(trade_details)
                cleanup_token(address)
                return  # Exit function since sell succeeded
            else:
                attempt += 1
                cprint(f"❌ {address}: Sell attempt {attempt} failed to clear the position. Retrying in {retry_delay}s...", "red")
                time.sleep(retry_delay)
        except Exception as e:
            cprint(f"❌ {address}: Sell error on attempt {attempt + 1} ({e})", "red")
            attempt += 1
            time.sleep(retry_delay)
    
    cprint(f"❌ {address}: Failed to close position after {MAX_RETRIES} attempts.", "red")
    cleanup_token(address)

def cleanup_token(address):
    """
    Remove a token from all tracking systems.
    
    Args:
        address (str): The token address
    """
    # Remove token from the main tracking dictionary
    if address in bought_tokens_info:
        del bought_tokens_info[address]
        
        # Save trailing data to database for remaining tokens
        save_trailing_data()
        
        # Remove token entry from the database
        db.remove_buy_price_from_db(address)
    
    # Remove token from top buyers tracking, if it exists
    if address in top_buyers_history:
        del top_buyers_history[address]

    # Remove token from any active sell threads tracking
    with active_sell_threads_lock:
        active_sell_threads.pop(address, None)

#########################################
# SELL THREAD HELPER
#########################################
def threaded_sell(address, sell_func):
    """
    Execute a sell function in a separate thread.
    
    Args:
        address (str): The token address
        sell_func (callable): The function to execute
    """
    try:
        sell_func(address)
    except Exception as e:
        cprint(f"Error selling {address} in thread: {e}", "red")
    finally:
        with active_sell_threads_lock:
            active_sell_threads.pop(address, None)

def initiate_sell(address, reason):
    """
    Initiate a sell operation in a separate thread.
    
    Args:
        address (str): The token address
        reason (str): The reason for selling
    """
    with active_sell_threads_lock:
        if address in active_sell_threads:
            cprint(f"Sell thread already active for {address}.", "yellow")
            return
        t_thread = threading.Thread(
            target=threaded_sell,
            args=(address, lambda addr: sell_token(addr, reason))
        )
        active_sell_threads[address] = t_thread
        t_thread.start()

#########################################
# EVALUATE POSITION
#########################################
def evaluate_position(token_address: str):
    """
    Evaluate a token position and determine if it should be sold.
    
    Args:
        token_address (str): The token address
        
    Returns:
        str: "sell_all" if the token should be sold, None otherwise
    """
    if token_address not in bought_tokens_info:
        return None

    # --- Get Top Buyers Data ---
    try:
        summary_df, holder_info_df = n.df_top_buyers(token_address)
        if not holder_info_df.empty:
            total_holders = len(holder_info_df)
            hold_count = (holder_info_df['status'].str.lower() == "hold").sum()
            bought_more_count = (holder_info_df['status'].str.lower() == "bought_more").sum()
            sold_count = total_holders - (hold_count + bought_more_count)
            hold_pct = (hold_count / total_holders) * 100
            bought_more_pct = (bought_more_count / total_holders) * 100
            sold_pct = (sold_count / total_holders) * 100
        else:
            hold_pct, bought_more_pct, sold_pct = 0, 0, 0
    except Exception as e:
        hold_pct, bought_more_pct, sold_pct = 0, 0, 0

    # --- Compute PnL from Wallet Data ---
    wallet_df = n.df_wallet_holdings(c.MY_ADDRESS)
    matching = wallet_df[wallet_df['token'].apply(lambda t: extract_token_address(t)) == token_address]
    if not matching.empty:
        wallet_row = matching.iloc[0]
        usd_value = safe_float(wallet_row.get('usd_value', 0))
        quantity = bought_tokens_info[token_address].get("quantity", 0)
        entry_price = bought_tokens_info[token_address].get("buy_price", 0)
        entry_cost = entry_price * quantity
        current_pnl = (usd_value - entry_cost) / entry_cost if entry_cost > 0 else 0.0
    else:
        token_info = n.get_token_info(token_address)
        current_price = safe_float(token_info.get("price", 0.0))
        entry_price = bought_tokens_info[token_address].get("buy_price", current_price)
        current_pnl = (current_price - entry_price) / entry_price

    # Skip evaluation if the current pnl is below the minimum threshold
    if current_pnl <= DONT_SELL_IF_BELOW_PNL:
        return None

    # --- Get Token Metrics ---
    token_info = n.get_token_info(token_address)
    current_buy_volume_1m = safe_float(token_info.get('buys_1m', 0))
    current_sell_volume_1m = safe_float(token_info.get('sells_1m', 0))
    difference_1m = current_buy_volume_1m - current_sell_volume_1m
    old_diff = bought_tokens_info[token_address].get("old_diff_1m", None)
    computed_drop_pct = (old_diff - difference_1m) / old_diff if old_diff and old_diff > 0 else None

    # --- Compute Total Volume Change Over Past Minute ---
    current_total_volume = current_buy_volume_1m + current_sell_volume_1m
    old_total_vol = bought_tokens_info[token_address].get("old_total_volume_1m", None)
    computed_volume_change_pct = ((current_total_volume - old_total_vol) / old_total_vol 
                                   if old_total_vol and old_total_vol > 0 else 0)
    # Update for next cycle:
    bought_tokens_info[token_address]["old_total_volume_1m"] = current_total_volume
    bought_tokens_info[token_address]["old_diff_1m"] = difference_1m

    # --- Volume History Tracking & Checks ---
    if 'buy_vol_history' not in bought_tokens_info[token_address]:
        bought_tokens_info[token_address]['buy_vol_history'] = []
    if 'sell_vol_history' not in bought_tokens_info[token_address]:
        bought_tokens_info[token_address]['sell_vol_history'] = []
    bought_tokens_info[token_address]['buy_vol_history'].append(current_buy_volume_1m)
    bought_tokens_info[token_address]['sell_vol_history'].append(current_sell_volume_1m)
    relevant_buy_history = (bought_tokens_info[token_address]['buy_vol_history'][-50:]
                            if len(bought_tokens_info[token_address]['buy_vol_history']) > 50 
                            else bought_tokens_info[token_address]['buy_vol_history'])
    relevant_sell_history = (bought_tokens_info[token_address]['sell_vol_history'][-50:]
                             if len(bought_tokens_info[token_address]['sell_vol_history']) > 50 
                             else bought_tokens_info[token_address]['sell_vol_history'])
    if len(relevant_buy_history) >= 25:
        avg_buy_vol = sum(relevant_buy_history) / len(relevant_buy_history)
        avg_sell_vol = sum(relevant_sell_history) / len(relevant_sell_history)
        buy_trigger = avg_buy_vol * (1 - BUY_VOLUME_DROP_THRESHOLD)
        sell_trigger = avg_sell_vol * (1 + SELL_VOLUME_SHARE_INCREASE_THRESHOLD)
        if current_sell_volume_1m > sell_trigger:
            cprint(f"{token_address}: Sell triggered due to sell volume spike: {current_sell_volume_1m:.4f} > avg {avg_sell_vol:.4f} * (1+{SELL_VOLUME_SHARE_INCREASE_THRESHOLD})", "red")
            return "sell_all"
        if current_buy_volume_1m < buy_trigger:
            cprint(f"{token_address}: Sell triggered due to buy volume drop: {current_buy_volume_1m:.4f} < avg {avg_buy_vol:.4f} * (1-{BUY_VOLUME_DROP_THRESHOLD})", "red")
            return "sell_all"

    # --- Update Trailing Stop and Other Metrics ---
    current_price = safe_float(token_info.get("price", 0.0))
    if current_price <= 0:
        cprint(f"[DEBUG] {token_address} => current_price=0 or negative, skipping", "yellow")
        return None
    entry_price = bought_tokens_info[token_address].get("entry_price", current_price)
    if entry_price <= 0:
        bought_tokens_info[token_address]["entry_price"] = current_price
        entry_price = current_price
        db.save_buy_price_to_db(token_address, current_price)
    if "original_buy_price" not in bought_tokens_info[token_address]:
        bought_tokens_info[token_address]["original_buy_price"] = entry_price
    old_max = bought_tokens_info[token_address].get("max_price", entry_price)
    new_max = max(current_price, old_max)
    bought_tokens_info[token_address]["max_price"] = new_max
    bought_tokens_info[token_address]["trailing_stop"] = new_max * (1 - TRAILING_STOP_LOSS_PERCENT)
    current_pnl = (current_price - entry_price) / entry_price
    original_buy_price = bought_tokens_info[token_address].get("original_buy_price", entry_price)
    total_pnl = (current_price - original_buy_price) / original_buy_price
    quantity = bought_tokens_info[token_address].get("quantity", 0)
    total_usd_pnl = (current_price - original_buy_price) * quantity

    cprint(f"[DEBUG] {token_address} => current_price={current_price:.6f}, trailing_stop={bought_tokens_info[token_address]['trailing_stop']:.6f}, current_pnl={current_pnl:.2%}, total_pnl={total_pnl:.2%}", "blue")

    # --- NEW: PnL Stall Check ---
    if 'pnl_history' not in bought_tokens_info[token_address]:
        bought_tokens_info[token_address]['pnl_history'] = []
    bought_tokens_info[token_address]['pnl_history'].append(current_pnl)
    if len(bought_tokens_info[token_address]['pnl_history']) >= PNL_STALL_CYCLES:
        recent_pnls = bought_tokens_info[token_address]['pnl_history'][-PNL_STALL_CYCLES:]
        if max(recent_pnls) - min(recent_pnls) < PNL_STALL_THRESHOLD:
            cprint(f"{token_address}: PnL has stalled over the last {PNL_STALL_CYCLES} iterations (range: {(max(recent_pnls) - min(recent_pnls)):.2%}). Triggering sell.", "red")
            return "sell_all"

    # --- NEW: PnL Jump Check ---
    if 'prev_pnl' not in bought_tokens_info[token_address]:
        bought_tokens_info[token_address]['prev_pnl'] = current_pnl
    else:
        pnl_jump = current_pnl - bought_tokens_info[token_address]['prev_pnl']
        if pnl_jump > PNL_JUMP_THRESHOLD:
            cprint(f"{token_address}: PnL jumped by {pnl_jump:.2%} since last check. Triggering sell.", "red")
            return "sell_all"
        bought_tokens_info[token_address]['prev_pnl'] = current_pnl

    # --- Update Top Buyers Global History ---
    global top_buyers_history
    if token_address not in top_buyers_history:
        top_buyers_history[token_address] = {
            'hold_pct': hold_pct,
            'sold_pct': sold_pct,
            'bought_more_pct': bought_more_pct,
            'total': total_holders if 'total_holders' in locals() else 0,
            'highest_buy_pct': hold_pct,
            'lowest_sold_pct': sold_pct if sold_pct != 0 else None
        }
    else:
        prev = top_buyers_history[token_address]
        if hold_pct > prev.get('highest_buy_pct', hold_pct):
            prev['highest_buy_pct'] = hold_pct
        if sold_pct != 0:
            if prev.get('lowest_sold_pct') is None or sold_pct < prev.get('lowest_sold_pct'):
                prev['lowest_sold_pct'] = sold_pct
        prev['hold_pct'] = hold_pct
        prev['sold_pct'] = sold_pct
        prev['bought_more_pct'] = bought_more_pct
        prev['total'] = total_holders if 'total_holders' in locals() else 0

    tracked_extremes = top_buyers_history.get(token_address, {})
    highest_buy_pct = tracked_extremes.get("highest_buy_pct", hold_pct)
    lowest_sold_pct = tracked_extremes.get("lowest_sold_pct", sold_pct)

    # --- Check Top Buyers Thresholds to Trigger Sell ---
    if hold_pct < highest_buy_pct - TOP_HOLD_DROP_THRESHOLD:
        cprint(
            f"{token_address}: Hold percentage dropped from {highest_buy_pct:.2f}% to {hold_pct:.2f}% exceeding threshold of {TOP_HOLD_DROP_THRESHOLD}%. Triggering sell.",
            "red"
        )
        return "sell_all"
    if lowest_sold_pct is not None and sold_pct > lowest_sold_pct + TOP_SOLD_RISE_THRESHOLD:
        cprint(
            f"{token_address}: Sold percentage increased from {lowest_sold_pct:.2f}% to {sold_pct:.2f}% exceeding threshold of {TOP_SOLD_RISE_THRESHOLD}%. Triggering sell.",
            "red"
        )
        return "sell_all"

    # --- Print Token Chart for Debugging ---
    print_token_chart(
        token_address,
        current_buy_volume_1m,
        current_sell_volume_1m,
        difference_1m,
        computed_drop_pct,
        computed_volume_change_pct,
        current_pnl,
        total_pnl,
        current_price,
        entry_price,
        bought_tokens_info[token_address]["trailing_stop"],
        total_usd_pnl,
        hold_pct,
        bought_more_pct,
        sold_pct,
        highest_buy_pct,
        lowest_sold_pct
    )
    return None

def save_trailing_data():
    """
    Save trailing data for all tracked tokens to the database.
    """
    db.save_trailing_data_to_db(bought_tokens_info)

#########################################
# MONITOR TOKEN THREAD
#########################################
def monitor_token(token_address: str):
    """
    Monitor a token in a separate thread and sell if conditions are met.
    
    Args:
        token_address (str): The token address to monitor
    """
    cprint(f"Started monitoring token {token_address}", "blue")
    try:
        while token_address in bought_tokens_info:
            decision = evaluate_position(token_address)
            if decision == "sell_all":
                initiate_sell(token_address, "condition triggered")
                cprint(f"{token_address}: Condition met for FULL SELL. Selling entire balance.", "red")
            time.sleep(TOKEN_CHECK_INTERVAL)
    except Exception as e:
        cprint(f"Error in monitor_token for {token_address}: {e}", "red")
    finally:
        with active_monitor_threads_lock:
            if token_address in active_monitor_threads:
                del active_monitor_threads[token_address]
        cprint(f"Stopped monitoring token {token_address}", "yellow")

#########################################
# START MONITORING NEW TOKENS
#########################################
def start_monitoring_new_tokens():
    """
    Start monitoring threads for all tokens that aren't already being monitored.
    """
    with active_monitor_threads_lock:
        for token in list(bought_tokens_info.keys()):
            if token in active_monitor_threads:
                if active_monitor_threads[token].is_alive():
                    continue
                else:
                    del active_monitor_threads[token]
            cprint(f"Starting monitoring thread for token {token}", "blue")
            t_thread = threading.Thread(target=monitor_token, args=(token,), daemon=True)
            active_monitor_threads[token] = t_thread
            t_thread.start()

async def monitor_new_tokens_periodically():
    """
    Periodically check for new tokens to monitor.
    """
    while True:
        try:
            await asyncio.to_thread(start_monitoring_new_tokens)
        except Exception as e:
            logger.error(f"Error in monitor_new_tokens_periodically: {e}")
        await asyncio.sleep(10)

#########################################
# Utility Functions for GUI
#########################################
def get_wallet_balance_df_main():
    """
    Get wallet balance information.
    
    Returns:
        pd.DataFrame: DataFrame with wallet information
    """
    return n.df_wallet_info(c.MY_ADDRESS)

def load_top_buyers_tracking():
    """
    Load tracking information for top buyers.
    
    Returns:
        list: List of dictionaries with top buyer tracking information
    """
    results = []
    for token, data in top_buyers_history.items():
        results.append({
            "token_address": token,
            "hold_pct": f"{data.get('hold_pct', 0):.2f}",
            "highest_buy_pct": f"{data.get('highest_buy_pct', data.get('hold_pct', 0)):.2f}",
            "sold_pct": f"{data.get('sold_pct', 0):.2f}",
            "lowest_sold_pct": f"{data.get('lowest_sold_pct', data.get('sold_pct', 0)):.2f}" if data.get('lowest_sold_pct') is not None else "N/A",
            "total_holders": data.get("total", 0)
        })
    return results

def load_wallet_holdings():
    """
    Load current wallet holdings.
    
    Returns:
        list: List of tuples with wallet holdings information
    """
    try:
        df = n.df_wallet_holdings(c.MY_ADDRESS)
        if df.empty:
            return []
        
        # Extract mint_address and token_name using helper functions
        def get_mint(token):
            if isinstance(token, dict):
                return token.get("address", "N/A")
            return token
        def get_token_name(token):
            if isinstance(token, dict):
                return token.get("name", "N/A")
            return "N/A"
            
        df["mint_address"] = df["token"].apply(get_mint)
        df["token_name"] = df["token"].apply(get_token_name)
        
        # Select only the columns needed for the GUI in the desired order
        df = df[["mint_address", "token_name", "balance", "realized_profit", "unrealized_profit", "start_holding_at"]]
        df.sort_values(by="balance", ascending=False, inplace=True)
        
        # Return rows as tuples
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        cprint(f"Error loading wallet holdings: {e}", "yellow")
        return []

def load_recent_trades(recent_only=False):
    """
    Load recent trades from the database.
    
    Args:
        recent_only (bool): If True, only return trades from the last 24 hours
        
    Returns:
        list: List of trade dictionaries
    """
    return db.load_recent_trades(recent_only)

#########################################
# ASYNC PERIODIC TOKEN UPDATE
#########################################
async def periodic_token_update():
    """
    Periodically update token information and display summaries.
    """
    while True:
        try:
            await asyncio.to_thread(update_held_tokens)
            info_df = get_wallet_balance_df_main()
            display_trade_summary()
        except Exception as e:
            logger.error(f"Error in periodic_token_update: {e}")
        await asyncio.sleep(WALLET_UPDATE_INTERVAL)

async def main_async():
    """
    Main asynchronous function that sets up periodic tasks.
    """
    update_task = asyncio.create_task(periodic_token_update())
    monitor_new_task = asyncio.create_task(monitor_new_tokens_periodically())
    await asyncio.gather(update_task, monitor_new_task)

async def run_forever():
    """
    Run the main async function forever, restarting if it fails.
    """
    while True:
        try:
            await main_async()
        except Exception as e:
            logger.error(f"main_async died: {e}. Restarting in 5s...")
            await asyncio.sleep(5)

def initialize_database() -> None:
    """
    Initialize the database.
    """
    try:
        # Initialize database schema
        db.db_init()
        cprint("Database initialized successfully", "green")
        
        # Simply ensure the database directory exists
        import config as c
        os.makedirs(os.path.dirname(c.DB_PATH), exist_ok=True)
    except Exception as e:
        cprint(f"Error initializing database: {e}", "red")
        logger.error(f"Database initialization error: {e}")
        logger.error(traceback.format_exc())

#########################################
# Function to Show Recent Trades in GUI
#########################################
def show_recent_trades_gui() -> None:
    """
    Display a GUI with recent trades, wallet holdings, and top buyers tracking.
    Optimized for better performance with caching and efficient data retrieval.
    """
    import tkinter as tk
    import tkinter.messagebox as msgbox
    from tkinter import ttk
    
    # Define UI refresh rate for responsiveness
    UI_REFRESH_INTERVAL = 1500  # milliseconds
    
    # Cache for data to reduce database queries
    data_cache = {
        "trades": {"data": None, "timestamp": 0},
        "wallet": {"data": None, "timestamp": 0},
        "tracking": {"data": None, "timestamp": 0},
        "profit": {"data": None, "timestamp": 0}
    }
    
    # Cache expiration time in milliseconds
    CACHE_EXPIRY = 5000  # 5 seconds
    
    window = tk.Tk()
    window.title("Dashboard: Recent Trades, Live Wallet Holdings & Top Buyers Tracking")
    window.geometry("1300x900")

    # --- TOP FRAME: Total Wallet Profit, Last 24hr Profit, & Net Combined Profit ---
    top_frame = ttk.Frame(window)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    total_profit_label = ttk.Label(top_frame, text="Total Profit: Calculating...", font=("Helvetica", 14, "bold"))
    total_profit_label.pack(side=tk.LEFT, padx=(0, 20))

    last24_profit_label = ttk.Label(top_frame, text="Last 24hr Profit: Calculating...", font=("Helvetica", 14, "bold"))
    last24_profit_label.pack(side=tk.LEFT, padx=(0, 20))

    # Label for "Net Combined Profit" (all-time realized + current unrealized)
    net_combined_label = ttk.Label(top_frame, text="Net Combined Profit: Calculating...", font=("Helvetica", 14, "bold"))
    net_combined_label.pack(side=tk.LEFT, padx=(0, 20))

    # --- RECENT TRADES FRAME (Aggregated per token) ---
    trades_frame = ttk.LabelFrame(window, text="Recent Trades (Aggregated by Token)")
    trades_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    trades_columns = ("token_address", "trade_count", "total_pnl_usd", "total_held_time")
    trades_tree = ttk.Treeview(trades_frame, columns=trades_columns, show='headings')
    
    # Add scrollbars
    trades_vsb = ttk.Scrollbar(trades_frame, orient="vertical", command=trades_tree.yview)
    trades_hsb = ttk.Scrollbar(trades_frame, orient="horizontal", command=trades_tree.xview)
    trades_tree.configure(yscrollcommand=trades_vsb.set, xscrollcommand=trades_hsb.set)
    
    # Grid layout with scrollbars
    trades_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    trades_hsb.pack(side=tk.BOTTOM, fill=tk.X)
    trades_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    for col in trades_columns:
        trades_tree.heading(col, text=col.replace("_", " ").title())
        trades_tree.column(col, anchor=tk.CENTER, width=200)

    # --- LIVE WALLET HOLDINGS FRAME ---
    wallet_frame = ttk.LabelFrame(window, text="Live Wallet Holdings (Only Tokens with Positive Balance)")
    wallet_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    wallet_columns = ("mint_address", "token_name", "balance", "realized_profit", "unrealized_profit", "start_holding_at")
    wallet_tree = ttk.Treeview(wallet_frame, columns=wallet_columns, show='headings')
    
    # Add scrollbars
    wallet_vsb = ttk.Scrollbar(wallet_frame, orient="vertical", command=wallet_tree.yview)
    wallet_hsb = ttk.Scrollbar(wallet_frame, orient="horizontal", command=wallet_tree.xview)
    wallet_tree.configure(yscrollcommand=wallet_vsb.set, xscrollcommand=wallet_hsb.set)
    
    # Grid layout with scrollbars
    wallet_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    wallet_hsb.pack(side=tk.BOTTOM, fill=tk.X)
    wallet_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    for col in wallet_columns:
        wallet_tree.heading(col, text=col.replace("_", " ").title())
        wallet_tree.column(col, anchor=tk.CENTER, width=150)

    # --- TOP BUYERS TRACKING FRAME ---
    tracking_frame = ttk.LabelFrame(window, text="Top Buyers Tracking (Only for Tokens Currently Held)")
    tracking_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    tracking_columns = ("token_address", "hold_pct", "highest_buy_pct", "sold_pct", "lowest_sold_pct", "total_holders")
    tracking_tree = ttk.Treeview(tracking_frame, columns=tracking_columns, show='headings')
    
    # Add scrollbars
    tracking_vsb = ttk.Scrollbar(tracking_frame, orient="vertical", command=tracking_tree.yview)
    tracking_hsb = ttk.Scrollbar(tracking_frame, orient="horizontal", command=tracking_tree.xview)
    tracking_tree.configure(yscrollcommand=tracking_vsb.set, xscrollcommand=tracking_hsb.set)
    
    # Grid layout with scrollbars
    tracking_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    tracking_hsb.pack(side=tk.BOTTOM, fill=tk.X)
    tracking_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    for col in tracking_columns:
        tracking_tree.heading(col, text=col.replace("_", " ").title())
        tracking_tree.column(col, anchor=tk.CENTER, width=150)

    # --- Double-click event: Show detailed trade info for selected token ---
    def on_trade_double_click(event) -> None:
        """
        Handle double-click on a trade to show details.
        Uses cached data when possible to improve performance.
        """
        selected_item = trades_tree.focus()
        if not selected_item:
            return
        token = trades_tree.item(selected_item, "values")[0]

        # Create window immediately to improve perceived responsiveness
        detail_window = tk.Toplevel(window)
        detail_window.title(f"Trade Details for {token}")
        text_widget = tk.Text(detail_window, wrap=tk.WORD, width=100, height=20)
        text_widget.pack(padx=10, pady=10)
        
        # Add a loading message
        text_widget.insert(tk.END, "Loading trade details...")
        
        # Fetch trade details in background to keep UI responsive
        def fetch_trade_details(token, text_widget, detail_window):
            try:
                # Query all trade records for this token directly from database
                conn = db.get_conn()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE token_address = ? 
                    ORDER BY trade_time DESC
                """, (token,))
                
                token_trades = []
                for row in cursor:
                    token_trades.append(dict(row))
                conn.close()
                
                details = f"Token: {token}\n\n"
                details += f"Total Trades: {len(token_trades)}\n"
                
                # Aggregate data in Python instead of multiple database queries
                total_pnl = sum(safe_float(trade.get("pnl_usd", 0)) for trade in token_trades)
                details += f"Total Profit (USD): {total_pnl:.2f}\n"
                
                # Calculate total held time
                total_held = 0.0
                for trade in token_trades:
                    held_str = trade.get("time_held", "0").replace(" sec", "")
                    try:
                        total_held += float(held_str)
                    except Exception:
                        continue
                
                details += f"Total Held Time: {total_held:.0f} sec\n\n"
                details += "Trade Details:\n"
                for trade in token_trades:
                    reason = trade.get("reason", "Not recorded")
                    details += (
                        f"- Time: {trade.get('trade_time')}, "
                        f"Qty Sold: {trade.get('quantity_sold')}, "
                        f"PnL: {trade.get('pnl_percent')}%, Reason: {reason}\n"
                    )
                
                # Update UI in main thread
                window.after(0, lambda: update_text_widget(text_widget, details))
            
            except Exception as e:
                window.after(0, lambda: update_text_widget(text_widget, f"Error loading trade details: {e}"))
        
        def update_text_widget(text_widget, content):
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
        
        # Start background thread to fetch data
        threading.Thread(target=fetch_trade_details, args=(token, text_widget, detail_window), daemon=True).start()

    trades_tree.bind("<Double-1>", on_trade_double_click)

    # --- Efficient Data Refresh Function ---
    def refresh_data() -> None:
        """
        Refresh the GUI data efficiently using background tasks and caching.
        """
        current_time = datetime.now().timestamp() * 1000  # current time in milliseconds
        
        try:
            # Define functions to fetch data in background
            def fetch_total_profit():
                # Check cache first
                if (current_time - data_cache["profit"].get("timestamp", 0) < CACHE_EXPIRY 
                    and data_cache["profit"].get("data") is not None):
                    return data_cache["profit"]["data"]
                    
                try:
                    # Use a direct, optimized SQL query instead of dataframe operations
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT SUM(realized_profit) as realized, SUM(unrealized_profit) as unrealized
                        FROM (
                            SELECT 
                                CAST(IFNULL(realized_profit, '0') AS REAL) as realized_profit,
                                CAST(IFNULL(unrealized_profit, '0') AS REAL) as unrealized_profit
                            FROM wallet_info
                            WHERE balance > 0
                        )
                    """)
                    result = cursor.fetchone()
                    conn.close()
                    
                    realized = result["realized"] if result["realized"] is not None else 0
                    unrealized = result["unrealized"] if result["unrealized"] is not None else 0
                    total_profit = realized + unrealized
                    
                    # Cache the result
                    data_cache["profit"]["data"] = f"Total Profit: {total_profit:.2f} USD"
                    data_cache["profit"]["timestamp"] = current_time
                    
                    return data_cache["profit"]["data"]
                except Exception as e:
                    return f"Total Profit: Error ({str(e)})"
                
            def fetch_last24hr_profit():
                # Use a direct SQL query that joins trades and wallet info
                try:
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    
                    # Last 24 hours cutoff
                    cutoff = (datetime.now() - timedelta(days=1)).isoformat()
                    
                    # Get total profit from trades in last 24h
                    cursor.execute("""
                        SELECT SUM(CAST(pnl_usd AS REAL)) as recent_profit
                        FROM trades
                        WHERE trade_time >= ?
                    """, (cutoff,))
                    trade_result = cursor.fetchone()
                    last24_profit = trade_result["recent_profit"] if trade_result["recent_profit"] is not None else 0
                    
                    # Add current unrealized pnl from open positions
                    cursor.execute("""
                        SELECT SUM(CAST(IFNULL(unrealized_profit, '0') AS REAL)) as open_unrealized
                        FROM wallet_holdings
                        WHERE balance > 0
                    """)
                    unrealized_result = cursor.fetchone()
                    open_unrealized = unrealized_result["open_unrealized"] if unrealized_result["open_unrealized"] is not None else 0
                    
                    # Calculate total profit
                    last24_profit += open_unrealized
                    conn.close()
                    
                    return f"Last 24hr Profit: {last24_profit:.2f} USD"
                except Exception as e:
                    return f"Last 24hr Profit: Error ({str(e)})"
                
            def fetch_net_combined():
                try:
                    # Query database for total realized profit
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    
                    # Get all-time realized profit
                    cursor.execute("SELECT SUM(CAST(pnl_usd AS REAL)) AS total_realized FROM trades")
                    result = cursor.fetchone()
                    all_time_realized = result["total_realized"] if result and result["total_realized"] is not None else 0.0
                    
                    # Get unrealized profit from wallet
                    cursor.execute("""
                        SELECT SUM(CAST(IFNULL(unrealized_profit, '0') AS REAL)) as total_unrealized
                        FROM wallet_holdings 
                        WHERE balance > 0
                    """)
                    unrealized_result = cursor.fetchone()
                    total_unrealized = unrealized_result["total_unrealized"] if unrealized_result and unrealized_result["total_unrealized"] is not None else 0.0
                    
                    net_combined = all_time_realized + total_unrealized
                    conn.close()
                    
                    return f"Net Combined Profit: {net_combined:.2f} USD"
                except Exception as e:
                    return f"Net Combined Profit: Error ({str(e)})"
                
            def fetch_trades_data():
                # Check cache first
                if (current_time - data_cache["trades"].get("timestamp", 0) < CACHE_EXPIRY 
                    and data_cache["trades"].get("data") is not None):
                    return data_cache["trades"]["data"]
                    
                try:
                    # Use a direct, optimized SQL query with grouping
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    
                    # Last 24 hours cutoff
                    cutoff = (datetime.now() - timedelta(days=1)).isoformat()
                    
                    # Query with aggregation directly in SQL
                    cursor.execute("""
                        SELECT 
                            token_address,
                            COUNT(*) as trade_count,
                            SUM(CAST(IFNULL(pnl_usd, '0') AS REAL)) as total_pnl_usd,
                            SUM(CASE 
                                WHEN time_held LIKE '%sec%' 
                                THEN CAST(REPLACE(time_held, ' sec', '') AS REAL) 
                                ELSE 0 
                            END) as total_held_time
                        FROM trades
                        WHERE trade_time >= ?
                        GROUP BY token_address
                        ORDER BY total_pnl_usd DESC
                    """, (cutoff,))
                    
                    aggregated_trades = []
                    for row in cursor:
                        aggregated_trades.append((
                            row["token_address"],
                            row["trade_count"],
                            f"{row['total_pnl_usd']:.2f}",
                            f"{row['total_held_time']:.0f} sec"
                        ))
                    
                    conn.close()
                    
                    # Cache the result
                    data_cache["trades"]["data"] = aggregated_trades
                    data_cache["trades"]["timestamp"] = current_time
                    
                    return aggregated_trades
                except Exception as e:
                    cprint(f"Error fetching trades data: {e}", "red")
                    return []
                    
            def fetch_wallet_data():
                # Check cache first
                if (current_time - data_cache["wallet"].get("timestamp", 0) < CACHE_EXPIRY 
                    and data_cache["wallet"].get("data") is not None):
                    return data_cache["wallet"]["data"]
                
                try:
                    # Direct database query for wallet holdings
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    
                    # Query for wallet holdings with token details
                    cursor.execute("""
                        SELECT 
                            h.token as mint_address,
                            IFNULL(t.name, 'Unknown') as token_name,
                            h.balance,
                            h.realized_profit,
                            h.unrealized_profit,
                            h.start_holding_at
                        FROM 
                            wallet_holdings as h
                        LEFT JOIN 
                            tokens_full_details as t ON h.token = t.address
                        WHERE 
                            CAST(h.balance AS REAL) > 0
                        ORDER BY 
                            CAST(h.balance AS REAL) DESC
                    """)
                    
                    wallet_data = []
                    for row in cursor:
                        wallet_data.append((
                            row["mint_address"],
                            row["token_name"],
                            row["balance"],
                            row["realized_profit"],
                            row["unrealized_profit"],
                            row["start_holding_at"]
                        ))
                    
                    conn.close()
                    
                    # Cache the result
                    data_cache["wallet"]["data"] = wallet_data
                    data_cache["wallet"]["timestamp"] = current_time
                    
                    return wallet_data
                except Exception as e:
                    cprint(f"Error fetching wallet data: {e}", "red")
                    return []
                    
            def fetch_tracking_data():
                # Check cache first  
                if (current_time - data_cache["tracking"].get("timestamp", 0) < CACHE_EXPIRY 
                    and data_cache["tracking"].get("data") is not None):
                    return data_cache["tracking"]["data"]
                
                try:
                    # Direct database query for top buyers tracking
                    conn = db.get_conn()
                    cursor = conn.cursor()
                    
                    # Get current wallet holdings
                    cursor.execute("""
                        SELECT DISTINCT token as token_address
                        FROM wallet_holdings
                        WHERE CAST(balance AS REAL) > 0
                    """)
                    held_tokens = [row["token_address"] for row in cursor.fetchall()]
                    
                    if not held_tokens:
                        return []
                    
                    # Format for SQL IN clause
                    placeholders = ','.join(['?'] * len(held_tokens))
                    
                    # Get top buyers data for held tokens
                    cursor.execute(f"""
                        SELECT 
                            token_address,
                            hold_pct,
                            highest_buy_pct,
                            sold_pct,
                            IFNULL(lowest_sold_pct, 'N/A') as lowest_sold_pct,
                            total_holders
                        FROM 
                            current_top_buyers
                        WHERE 
                            token_address IN ({placeholders})
                    """, held_tokens)
                    
                    tracking_data = []
                    for row in cursor:
                        tracking_data.append((
                            row["token_address"],
                            f"{row['hold_pct']:.2f}" if isinstance(row['hold_pct'], (int, float)) else row['hold_pct'],
                            f"{row['highest_buy_pct']:.2f}" if isinstance(row['highest_buy_pct'], (int, float)) else row['highest_buy_pct'],
                            f"{row['sold_pct']:.2f}" if isinstance(row['sold_pct'], (int, float)) else row['sold_pct'],
                            row["lowest_sold_pct"],
                            row["total_holders"]
                        ))
                    
                    conn.close()
                    
                    # Cache the result
                    data_cache["tracking"]["data"] = tracking_data
                    data_cache["tracking"]["timestamp"] = current_time
                    
                    return tracking_data
                except Exception as e:
                    cprint(f"Error fetching tracking data: {e}", "red")
                    return []
            
            # Start data fetching operations in separate threads
            total_profit_thread = threading.Thread(target=lambda: total_profit_label.config(text=fetch_total_profit()))
            last24_profit_thread = threading.Thread(target=lambda: last24_profit_label.config(text=fetch_last24hr_profit()))
            net_combined_thread = threading.Thread(target=lambda: net_combined_label.config(text=fetch_net_combined()))
            
            # Start the threads
            total_profit_thread.start()
            last24_profit_thread.start()
            net_combined_thread.start()
            
            # Update trades tree
            trades_data = fetch_trades_data()
            for item in trades_tree.get_children():
                trades_tree.delete(item)
            for row in trades_data:
                trades_tree.insert("", tk.END, values=row)
                
            # Update wallet tree
            wallet_data = fetch_wallet_data()
            for item in wallet_tree.get_children():
                wallet_tree.delete(item)
            for row in wallet_data:
                wallet_tree.insert("", tk.END, values=row)
                
            # Update tracking tree
            tracking_data = fetch_tracking_data()
            for item in tracking_tree.get_children():
                tracking_tree.delete(item)
            for row in tracking_data:
                tracking_tree.insert("", tk.END, values=row)
            
            # Wait for threads to complete
            total_profit_thread.join()
            last24_profit_thread.join()
            net_combined_thread.join()
            
        except Exception as e:
            cprint(f"Error in refresh_data: {e}", "red")
        
        # Schedule next refresh
        window.after(UI_REFRESH_INTERVAL, refresh_data)

    # Initial data load
    refresh_data()
    window.mainloop()
#########################################
# MAIN ENTRY POINT
#########################################
if __name__ == "__main__":
    # Initialize database first
    initialize_database()
    
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=lambda: asyncio.run(run_forever()), daemon=True)
    monitoring_thread.start()
    
    # Show the GUI
    show_recent_trades_gui()