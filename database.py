"""
Database module for the trading bot.
"""
import os
import sqlite3
import logging
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Dict, List, Optional, Union, Any, Set

# Import config for paths
import config as c

# Setup logging
logger = logging.getLogger(__name__)

# Global lock for database operations to prevent concurrent writes
db_lock = threading.Lock()

# Database location - use the path defined in config
DB_DIR = Path(os.path.dirname(c.DB_PATH))
DB_PATH = Path(c.DB_PATH)

# Ensure the database directory exists
DB_DIR.mkdir(parents=True, exist_ok=True)

# Table definitions (schema)
TABLES = {
    # Table for bought tokens
    "bought_tokens": """
        CREATE TABLE IF NOT EXISTS bought_tokens (
            address TEXT PRIMARY KEY,
            buy_price REAL,
            timestamp TEXT,
            UNIQUE(address)
        )
    """,
    
    # Table for executed trades
    "trades": """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            trade_type TEXT,
            quantity_sold TEXT,
            pnl_percent TEXT,
            pnl_usd TEXT,
            trade_time TEXT,
            time_held TEXT,
            reason TEXT
        )
    """,
    
    # Table for tokens that were sold
    "sold_tokens": """
        CREATE TABLE IF NOT EXISTS sold_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            token_address TEXT,
            sell_condition TEXT,
            current_buy_volume_1m REAL,
            current_sell_volume_1m REAL,
            difference_1m REAL,
            old_diff REAL,
            drop_pct REAL,
            buy_drop_pct REAL,
            sell_fraction_increase REAL,
            current_total_volume REAL,
            old_total_vol REAL,
            total_drop_pct REAL,
            total_increase_pct REAL,
            current_price REAL,
            entry_price REAL,
            trailing_stop REAL,
            current_pnl REAL,
            total_pnl REAL,
            hold_pct REAL,
            bought_more_pct REAL,
            sold_pct REAL,
            highest_buy_pct REAL,
            lowest_sold_pct REAL
        )
    """,
    
    # Table for token test results
    "token_test_results": """
        CREATE TABLE IF NOT EXISTS token_test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT,
            name TEXT,
            last_seen TEXT,
            liquidity_result TEXT,
            liquidity_value TEXT,
            volume_1m_result TEXT, 
            volume_1m_value TEXT,
            buy_sell_ratio_1m_result TEXT,
            buy_sell_ratio_1m_value TEXT,
            buy_sell_ratio_5m_result TEXT, 
            buy_sell_ratio_5m_value TEXT,
            holders_result TEXT,
            holders_value TEXT,
            net_in_volume_1m_result TEXT,
            net_in_volume_1m_value TEXT,
            net_in_volume_5m_result TEXT,
            net_in_volume_5m_value TEXT,
            creation_time_result TEXT,
            creation_time_value TEXT,
            market_cap_result TEXT,
            market_cap_value TEXT,
            price_1h_increase_result TEXT,
            price_1h_increase_value TEXT,
            rug_ratio_result TEXT,
            rug_ratio_value TEXT,
            UNIQUE(address)
        )
    """,
    
    # Table for seen addresses
    "seen_addresses": """
        CREATE TABLE IF NOT EXISTS seen_addresses (
            address TEXT PRIMARY KEY,
            first_seen TEXT,
            UNIQUE(address)
        )
    """,
    
    # Table for low value tokens
    "low_value_tokens": """
        CREATE TABLE IF NOT EXISTS low_value_tokens (
            mint_address TEXT PRIMARY KEY,
            token_balance REAL,
            token_price_usd REAL,
            total_value_usd REAL,
            added_timestamp TEXT,
            UNIQUE(mint_address)
        )
    """,
    
    # Table for all tokens
    "all_tokens": """
        CREATE TABLE IF NOT EXISTS all_tokens (
            address TEXT PRIMARY KEY,
            last_seen TEXT,
            last_price TEXT,
            UNIQUE(address)
        )
    """,
    
    # Table for tokens with full details
    "tokens_full_details": """
        CREATE TABLE IF NOT EXISTS tokens_full_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT,
            holder_count INTEGER,
            symbol TEXT,
            name TEXT,
            decimals INTEGER,
            price REAL,
            logo TEXT,
            price_1m REAL,
            price_5m REAL,
            price_1h REAL,
            price_6h REAL,
            price_24h REAL,
            volume_24h REAL,
            swaps_5m INTEGER,
            swaps_1h INTEGER,
            swaps_6h INTEGER,
            swaps_24h INTEGER,
            liquidity REAL,
            max_supply REAL,
            total_supply REAL,
            biggest_pool_address TEXT,
            chain TEXT,
            creation_timestamp TEXT,
            open_timestamp TEXT,
            circulating_supply REAL,
            high_price REAL,
            high_price_timestamp TEXT,
            low_price REAL,
            low_price_timestamp TEXT,
            buys_1m INTEGER,
            sells_1m INTEGER,
            swaps_1m INTEGER,
            volume_1m REAL,
            buy_volume_1m REAL,
            sell_volume_1m REAL,
            net_in_volume_1m REAL,
            buys_5m INTEGER,
            sells_5m INTEGER,
            volume_5m REAL,
            buy_volume_5m REAL,
            sell_volume_5m REAL,
            net_in_volume_5m REAL,
            buys_1h INTEGER,
            sells_1h INTEGER,
            volume_1h REAL,
            buy_volume_1h REAL,
            sell_volume_1h REAL,
            net_in_volume_1h REAL,
            buys_6h INTEGER,
            sells_6h INTEGER,
            volume_6h REAL,
            buy_volume_6h REAL,
            sell_volume_6h REAL,
            net_in_volume_6h REAL,
            buys_24h INTEGER,
            sells_24h INTEGER,
            buy_volume_24h REAL,
            sell_volume_24h REAL,
            net_in_volume_24h REAL,
            fdv REAL,
            market_cap REAL,
            circulating_market_cap REAL,
            link TEXT,
            social_links TEXT,
            last_seen TEXT,
            timestamp TEXT,
            UNIQUE(address, timestamp)
        )
    """,
    
    # Table for do not trade list
    "do_not_trade": """
        CREATE TABLE IF NOT EXISTS do_not_trade (
            address TEXT PRIMARY KEY,
            reason TEXT,
            timestamp TEXT,
            UNIQUE(address)
        )
    """,
    
    # Table for trailing data
    "trailing_data": """
        CREATE TABLE IF NOT EXISTS trailing_data (
            token_address TEXT PRIMARY KEY,
            trailing_stop REAL,
            max_price REAL,
            old_sell_volume_1m REAL,
            old_diff_1m REAL,
            old_buy_volume_1m REAL,
            old_total_volume_1m REAL,
            UNIQUE(token_address)
        )
    """,
    
    # Table for top buyers history
    "top_buyers_history": """
        CREATE TABLE IF NOT EXISTS top_buyers_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            timestamp TEXT,
            hold_pct REAL,
            sold_pct REAL,
            bought_more_pct REAL,
            total_holders INTEGER,
            highest_buy_pct REAL,
            lowest_sold_pct REAL
        )
    """,
    
    # Table for current top buyers status
    "current_top_buyers": """
        CREATE TABLE IF NOT EXISTS current_top_buyers (
            token_address TEXT PRIMARY KEY,
            timestamp TEXT,
            hold_pct REAL,
            sold_pct REAL,
            bought_more_pct REAL,
            total_holders INTEGER,
            highest_buy_pct REAL,
            lowest_sold_pct REAL,
            UNIQUE(token_address)
        )
    """,
    
    # Table for wallet information
    "wallet_info": """
        CREATE TABLE IF NOT EXISTS wallet_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_address TEXT,
            slot INTEGER,
            lamports REAL,
            realized_profit REAL,
            unrealized_profit REAL,
            timestamp TEXT
        )
    """,
    
    # Table for wallet holdings
    "wallet_holdings": """
        CREATE TABLE IF NOT EXISTS wallet_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_address TEXT,
            token TEXT,
            balance REAL,
            realized_profit REAL,
            unrealized_profit REAL,
            start_holding_at TEXT,
            timestamp TEXT
        )
    """
}

def get_conn():
    """
    Get a connection to the database.
    
    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    try:
        # Enable foreign keys support and WAL mode for better performance
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        # Make sqlite return rows as dictionaries
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def db_init():
    """
    Initialize the database by creating all required tables.
    Should be called once at application startup.
    """
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Create all tables defined in TABLES
            for table_name, table_sql in TABLES.items():
                cursor.execute(table_sql)
                logger.info(f"Created or verified table: {table_name}")
            
            # Create indexes for frequently accessed tables
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_address)",
                "CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(trade_time)",
                "CREATE INDEX IF NOT EXISTS idx_sold_token ON sold_tokens(token_address)",
                "CREATE INDEX IF NOT EXISTS idx_token_test ON token_test_results(address)",
                "CREATE INDEX IF NOT EXISTS idx_top_buyers_token ON top_buyers_history(token_address)",
                "CREATE INDEX IF NOT EXISTS idx_wallet_holdings_token ON wallet_holdings(token)",
                "CREATE INDEX IF NOT EXISTS idx_tokens_full_details_address ON tokens_full_details(address)",
                "CREATE INDEX IF NOT EXISTS idx_tokens_full_details_time ON tokens_full_details(timestamp)"
            ]
            
            for index in indexes:
                try:
                    cursor.execute(index)
                except sqlite3.Error as e:
                    logger.warning(f"Error creating index: {e}")
            
            # Add default entries to do_not_trade table
            default_do_not_trade = [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC address
                "So11111111111111111111111111111111111111111",
                "So11111111111111111111111111111111111111112",
                "7xmFoP1PBhey43HybJYhBxWTP2uFS1M3Z9RzBEw3sLy3"
            ]
            
            for address in default_do_not_trade:
                cursor.execute(
                    "INSERT OR IGNORE INTO do_not_trade (address, reason, timestamp) VALUES (?, ?, ?)",
                    (address, "Default do_not_trade entry", datetime.now().isoformat())
                )
            
            conn.commit()
            conn.close()
            logger.info("Database initialization complete")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise

def query_to_dataframe(query, params=None):
    """
    Execute a SQL query and return the results as a pandas DataFrame.
    
    Args:
        query (str): The SQL query to execute
        params (tuple, optional): Parameters for the query
        
    Returns:
        pd.DataFrame: A DataFrame containing the query results
    """
    try:
        with db_lock:
            conn = get_conn()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            return df
    except sqlite3.Error as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def execute_query(query, params=None):
    """
    Execute a SQL query (typically INSERT, UPDATE, or DELETE).
    
    Args:
        query (str): The SQL query to execute
        params (tuple or dict, optional): Parameters for the query
        
    Returns:
        int: The number of rows affected
    """
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return affected
    except sqlite3.Error as e:
        logger.error(f"Error executing query: {e}")
        return 0  # Return 0 rows affected on error

def execute_many(query, params_list):
    """
    Execute a SQL query with multiple parameter sets (batch operation).
    
    Args:
        query (str): The SQL query to execute
        params_list (list): List of parameter sets
        
    Returns:
        int: The number of rows affected
    """
    if not params_list:
        return 0
        
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return affected
    except sqlite3.Error as e:
        logger.error(f"Error executing batch query: {e}")
        return 0  # Return 0 rows affected on error

def get_buy_price_from_db(address):
    """
    Get the buy price for a token from the database.
    
    Args:
        address (str): The token address
        
    Returns:
        float: The buy price or None if not found
    """
    query = "SELECT buy_price FROM bought_tokens WHERE address = ?"
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (address,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return float(result['buy_price'])
            return None
    except (sqlite3.Error, TypeError) as e:
        logger.error(f"Error getting buy price for {address}: {e}")
        return None

def save_buy_price_to_db(address, buy_price):
    """
    Save or update the buy price for a token in the database.
    
    Args:
        address (str): The token address
        buy_price (float): The buy price
        
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    # Use INSERT OR REPLACE to handle both insert and update in one query
    query = """
    INSERT OR REPLACE INTO bought_tokens (address, buy_price, timestamp) 
    VALUES (?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (address, buy_price, timestamp))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error saving buy price for {address}: {e}")
        return False

def remove_buy_price_from_db(address):
    """
    Remove a token's buy price from the database.
    
    Args:
        address (str): The token address
        
    Returns:
        bool: True if successful, False otherwise
    """
    query = "DELETE FROM bought_tokens WHERE address = ?"
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (address,))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error removing buy price for {address}: {e}")
        return False

def log_trade_to_db(trade_details):
    """
    Log a trade in the database.
    
    Args:
        trade_details (dict): The trade details
        
    Returns:
        bool: True if successful, False otherwise
    """
    query = """
    INSERT INTO trades (
        token_address, trade_type, quantity_sold,
        pnl_percent, pnl_usd, trade_time, time_held, reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (
                trade_details.get('token_address'),
                trade_details.get('trade_type', 'full'),
                trade_details.get('quantity_sold'),
                trade_details.get('pnl_percent'),
                trade_details.get('pnl_usd'),
                trade_details.get('trade_time', datetime.now().isoformat()),
                trade_details.get('time_held'),
                trade_details.get('reason')
            ))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error logging trade: {e}")
        return False

def log_sell_debug_info_to_db(token_address, condition, diagnostics):
    """
    Log sell diagnostics to the database.
    
    Args:
        token_address (str): The token address
        condition (str): The sell condition
        diagnostics (dict): Diagnostic information
        
    Returns:
        bool: True if successful, False otherwise
    """
    query = """
    INSERT INTO sold_tokens (
        timestamp, token_address, sell_condition,
        current_buy_volume_1m, current_sell_volume_1m, difference_1m,
        old_diff, drop_pct, buy_drop_pct, sell_fraction_increase,
        current_total_volume, old_total_vol, total_drop_pct, total_increase_pct,
        current_price, entry_price, trailing_stop,
        current_pnl, total_pnl, hold_pct, bought_more_pct, sold_pct,
        highest_buy_pct, lowest_sold_pct
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    timestamp = datetime.now().isoformat()
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (
                timestamp,
                token_address,
                condition,
                diagnostics.get("current_buy_volume_1m"),
                diagnostics.get("current_sell_volume_1m"),
                diagnostics.get("difference_1m"),
                diagnostics.get("old_diff"),
                diagnostics.get("drop_pct"),
                diagnostics.get("buy_drop_pct"),
                diagnostics.get("sell_fraction_increase"),
                diagnostics.get("current_total_volume"),
                diagnostics.get("old_total_vol"),
                diagnostics.get("total_drop_pct"),
                diagnostics.get("total_increase_pct"),
                diagnostics.get("current_price"),
                diagnostics.get("entry_price"),
                diagnostics.get("trailing_stop"),
                diagnostics.get("current_pnl"),
                diagnostics.get("total_pnl"),
                diagnostics.get("hold_pct"),
                diagnostics.get("bought_more_pct"),
                diagnostics.get("sold_pct"),
                diagnostics.get("highest_buy_pct"),
                diagnostics.get("lowest_sold_pct")
            ))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error logging sell diagnostics: {e}")
        return False

def save_test_results_to_db(token_info, test_results, test_values):
    """
    Save token test results to the database.
    
    Args:
        token_info (dict): Token information
        test_results (dict): Test results (boolean values)
        test_values (dict): Test values
        
    Returns:
        bool: True if successful, False otherwise
    """
    token_address = (
        token_info.get("address") or 
        token_info.get("coin_address") or 
        token_info.get("token_address")
    )
    token_name = token_info.get("name", "")
    last_seen = datetime.now().isoformat()
    
    # Convert boolean test results to "PASS"/"FAIL" strings
    liquidity_result = "PASS" if test_results.get("Liquidity", False) else "FAIL"
    volume_1m_result = "PASS" if test_results.get("1m Volume", False) else "FAIL"
    buy_sell_ratio_1m_result = "PASS" if test_results.get("1m Buy/Sell Volume Ratio", False) else "FAIL"
    buy_sell_ratio_5m_result = "PASS" if test_results.get("5m Buy/Sell Volume Ratio", False) else "FAIL"
    holders_result = "PASS" if test_results.get("Holders", False) else "FAIL"
    net_in_volume_1m_result = "PASS" if test_results.get("Net In Volume 1m", False) else "FAIL"
    net_in_volume_5m_result = "PASS" if test_results.get("Net In Volume 5m", False) else "FAIL"
    creation_time_result = "PASS" if test_results.get("Creation Time", False) else "FAIL"
    market_cap_result = "PASS" if test_results.get("Market Cap", False) else "FAIL"
    price_1h_increase_result = "PASS" if test_results.get("Price 1h Increase", False) else "FAIL"
    rug_ratio_result = "PASS" if test_results.get("Rug Ratio", False) else "FAIL"
    
    query = """
    INSERT OR REPLACE INTO token_test_results (
        address, name, last_seen,
        liquidity_result, liquidity_value,
        volume_1m_result, volume_1m_value,
        buy_sell_ratio_1m_result, buy_sell_ratio_1m_value,
        buy_sell_ratio_5m_result, buy_sell_ratio_5m_value,
        holders_result, holders_value,
        net_in_volume_1m_result, net_in_volume_1m_value,
        net_in_volume_5m_result, net_in_volume_5m_value,
        creation_time_result, creation_time_value,
        market_cap_result, market_cap_value,
        price_1h_increase_result, price_1h_increase_value,
        rug_ratio_result, rug_ratio_value
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (
                token_address, 
                token_name, 
                last_seen,
                liquidity_result, str(test_values.get("Liquidity", "")),
                volume_1m_result, str(test_values.get("1m Volume", "")),
                buy_sell_ratio_1m_result, str(test_values.get("1m Buy/Sell Volume Ratio", "")),
                buy_sell_ratio_5m_result, str(test_values.get("5m Buy/Sell Volume Ratio", "")),
                holders_result, str(test_values.get("Holders", "")),
                net_in_volume_1m_result, str(test_values.get("Net In Volume 1m", "")),
                net_in_volume_5m_result, str(test_values.get("Net In Volume 5m", "")),
                creation_time_result, str(test_values.get("Creation Time", "")),
                market_cap_result, str(test_values.get("Market Cap", "")),
                price_1h_increase_result, str(test_values.get("Price 1h Increase", "")),
                rug_ratio_result, str(test_values.get("Rug Ratio", ""))
            ))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error saving test results for {token_address}: {e}")
        return False

def update_master_all_db(token_info):
    """
    Update token information in the all_tokens table.
    
    Args:
        token_info (dict): Token information
        
    Returns:
        bool: True if successful, False otherwise
    """
    token_address = (
        token_info.get("address") or 
        token_info.get("coin_address") or 
        token_info.get("token_address")
    )
    if not token_address:
        return False
        
    current_time = datetime.now().isoformat()
    current_price = token_info.get("usd_price", token_info.get("price", "N/A"))
    
    query = """
    INSERT OR REPLACE INTO all_tokens (address, last_seen, last_price) 
    VALUES (?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (token_address, current_time, current_price))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error updating all_tokens for {token_address}: {e}")
        return False

def get_last_seen_from_master_all_db(token_address):
    """
    Get the last seen timestamp for a token from the all_tokens table.
    
    Args:
        token_address (str): The token address
        
    Returns:
        str: The last seen timestamp or None if not found
    """
    query = "SELECT last_seen, last_price FROM all_tokens WHERE address = ?"
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (token_address,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                last_seen = result['last_seen']
                last_price = result['last_price']
                return f"{last_seen} | Last Price: {last_price}"
            return None
    except sqlite3.Error as e:
        logger.error(f"Error getting last seen for {token_address}: {e}")
        return None

def save_full_token_info_to_db(token_info):
    """
    Save full token information to the tokens_full_details table.
    
    Args:
        token_info (dict): Complete token information
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Extract token address with fallback options for different field names
    token_address = (
        token_info.get("address") or 
        token_info.get("coin_address") or 
        token_info.get("token_address")
    )
    
    # Skip if no valid address is found
    if not token_address:
        logger.warning("No token address found in token_info, skipping save")
        return False
    
    # Get current timestamp
    current_time = datetime.now().isoformat()
    
    # Create a deep copy to avoid modifying the original
    processed_info = {}
    
    # Process all fields, serializing complex types
    for key, value in token_info.items():
        # Convert dictionaries and lists to JSON strings
        if isinstance(value, (dict, list)):
            try:
                processed_info[key] = json.dumps(value)
            except Exception as e:
                logger.warning(f"Failed to serialize {key}: {e}, setting to empty string")
                processed_info[key] = ""
        else:
            processed_info[key] = value
    
    # Add timestamp and ensure address is present
    processed_info["last_seen"] = current_time
    processed_info["timestamp"] = current_time
    processed_info["address"] = token_address
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Get the columns in the tokens_full_details table
            cursor.execute("PRAGMA table_info(tokens_full_details)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Filter our processed info to include only columns that exist in the table
            filtered_info = {k: v for k, v in processed_info.items() if k in columns}
            
            # Prepare SQL statement dynamically based on available columns
            placeholders = ", ".join(["?"] * len(filtered_info))
            column_names = ", ".join(filtered_info.keys())
            values = list(filtered_info.values())
            
            # Build and execute the insert statement
            query = f"""
            INSERT INTO tokens_full_details ({column_names})
            VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            
            return True
            
    except Exception as e:
        logger.error(f"Error saving full token info for {token_address}: {e}")
        return False

def save_trailing_data_to_db(token_data):
    """
    Save trailing data for multiple tokens to the database.
    
    Args:
        token_data (dict): Dictionary of token_address -> token data
        
    Returns:
        bool: True if successful, False otherwise
    """
    insert_query = """
    INSERT OR REPLACE INTO trailing_data (
        token_address, trailing_stop, max_price, old_sell_volume_1m, 
        old_diff_1m, old_buy_volume_1m, old_total_volume_1m
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    params_list = []
    for token, data in token_data.items():
        params_list.append((
            token,
            data.get("trailing_stop"),
            data.get("max_price"),
            data.get("old_sell_volume_1m"),
            data.get("old_diff_1m"),
            data.get("old_buy_volume_1m"),
            data.get("old_total_volume_1m")
        ))
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Insert new data
            if params_list:
                cursor.executemany(insert_query, params_list)
                
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error saving trailing data: {e}")
        return False

def load_seen_addresses():
    """
    Load all seen addresses from the database.
    
    Returns:
        set: A set of all seen addresses
    """
    query = "SELECT address FROM seen_addresses"
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query)
            addresses = {row['address'] for row in cursor.fetchall()}
            conn.close()
            return addresses
    except sqlite3.Error as e:
        logger.error(f"Error loading seen addresses: {e}")
        return set()

def save_new_addresses(addresses):
    """
    Save new addresses to the database.
    
    Args:
        addresses (list or pd.DataFrame): List of addresses or DataFrame with address column
        
    Returns:
        bool: True if successful, False otherwise
    """
    if isinstance(addresses, pd.DataFrame):
        # Check for column with address data
        address_column = None
        for col in ['address', 'coin_address', 'token_address']:
            if col in addresses.columns:
                address_column = col
                break
        
        if not address_column:
            logger.warning("No valid address column found in DataFrame")
            return False
            
        # Extract addresses from DataFrame
        address_list = addresses[address_column].unique().tolist()
    else:
        # Assume list of addresses
        address_list = addresses
    
    if not address_list:
        return True  # Nothing to do
    
    timestamp = datetime.now().isoformat()
    
    # Prepare for batch insert
    params_list = [(addr, timestamp) for addr in address_list if addr]
    
    # Use INSERT OR IGNORE to skip duplicates
    query = """
    INSERT OR IGNORE INTO seen_addresses (address, first_seen) 
    VALUES (?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error saving new addresses: {e}")
        return False

def add_do_not_trade(address, reason=None):
    """
    Add a token to the do not trade list.
    
    Args:
        address (str): The token address
        reason (str, optional): Reason for adding to the list
        
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    query = """
    INSERT OR REPLACE INTO do_not_trade (address, reason, timestamp) 
    VALUES (?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (address, reason or "No reason specified", timestamp))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding token to do_not_trade: {e}")
        return False

def remove_token_from_low_value_list(token_address):
    """
    Remove a token from the low value tokens list.
    
    Args:
        token_address (str): The token address
        
    Returns:
        bool: True if successful, False otherwise
    """
    query = "DELETE FROM low_value_tokens WHERE mint_address = ?"
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (token_address,))
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            if affected > 0:
                logger.info(f"Removed token {token_address} from low value tokens list")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error removing token from low value list: {e}")
        return False

def add_token_to_low_value_list(token_address, token_balance, token_price_usd, total_value_usd):
    """
    Add a token to the low value tokens list.
    
    Args:
        token_address (str): The token address
        token_balance (float): Token balance
        token_price_usd (float): Token price in USD
        total_value_usd (float): Total value in USD
        
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    query = """
    INSERT OR REPLACE INTO low_value_tokens 
    (mint_address, token_balance, token_price_usd, total_value_usd, added_timestamp) 
    VALUES (?, ?, ?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (token_address, token_balance, token_price_usd, total_value_usd, timestamp))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding token to low value list: {e}")
        return False

def load_bought_tokens():
    """
    Load all bought tokens from the database.
    
    Returns:
        dict: Dictionary of token_address -> datetime of purchase
    """
    query = "SELECT address, timestamp FROM bought_tokens"
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query)
            result = {}
            
            now = datetime.now()
            for row in cursor.fetchall():
                try:
                    purchase_time = datetime.fromisoformat(row['timestamp'])
                    # Only include tokens purchased within the rebuy window
                    if now - purchase_time < timedelta(hours=c.RECENTLY_BOUGHT_WINDOW_HOURS):
                        result[row['address']] = purchase_time
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing timestamp for token {row['address']}: {e}")
            
            conn.close()
            return result
    except sqlite3.Error as e:
        logger.error(f"Error loading bought tokens: {e}")
        return {}

def load_recent_trades(recent_only=False):
    """
    Load recent trades from the database.
    
    Args:
        recent_only (bool): If True, only return trades from the last 24 hours
        
    Returns:
        list: List of trade dictionaries
    """
    if recent_only:
        cutoff = (datetime.now() - timedelta(days=1)).isoformat()
        query = "SELECT * FROM trades WHERE trade_time >= ? ORDER BY trade_time DESC"
        params = (cutoff,)
    else:
        query = "SELECT * FROM trades ORDER BY trade_time DESC"
        params = None
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Convert to list of dictionaries
            trades = []
            for row in cursor.fetchall():
                trade_dict = dict(row)
                # Add parsed datetime for sorting
                try:
                    trade_dict["trade_time_parsed"] = datetime.fromisoformat(trade_dict["trade_time"])
                except (ValueError, TypeError):
                    trade_dict["trade_time_parsed"] = datetime.min
                trades.append(trade_dict)
            
            conn.close()
            return trades
    except sqlite3.Error as e:
        logger.error(f"Error loading recent trades: {e}")
        return []

def update_top_buyers(token_address, *, hold_pct=0.0, sold_pct=0.0, bought_more_pct=0.0, 
                   total=0, total_holders=None, highest_buy_pct=None, lowest_sold_pct=None):
    """
    Store or update top buyers information for a specific token in the database.
    
    Args:
        token_address: The token address
        hold_pct: Percentage of holders maintaining their position
        sold_pct: Percentage of holders who have sold
        bought_more_pct: Percentage of holders who increased their position
        total: Total number of holders (legacy parameter)
        total_holders: Total number of holders (alternative parameter)
        highest_buy_pct: Historical maximum for hold percentage
        lowest_sold_pct: Historical minimum for sold percentage
            
    Returns:
        bool: True if successful, False otherwise
    """
    # Parameter normalization: handle both 'total' and 'total_holders'
    actual_total = total_holders if total_holders is not None else total
    
    # Set default values for optional parameters if they weren't provided
    if highest_buy_pct is None:
        highest_buy_pct = hold_pct
        
    if lowest_sold_pct is None and sold_pct > 0:
        lowest_sold_pct = sold_pct
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Get current timestamp
            current_time = datetime.now().isoformat()
            
            # Insert new record in history table (append-only time series)
            cursor.execute("""
                INSERT INTO top_buyers_history (
                    token_address, timestamp, hold_pct, sold_pct, bought_more_pct,
                    total_holders, highest_buy_pct, lowest_sold_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                token_address,
                current_time,
                hold_pct,
                sold_pct,
                bought_more_pct,
                actual_total,
                highest_buy_pct,
                lowest_sold_pct
            ))
            
            # Use INSERT OR REPLACE to handle both new tokens and updates
            cursor.execute("""
                INSERT OR REPLACE INTO current_top_buyers (
                    token_address, timestamp, hold_pct, sold_pct, bought_more_pct,
                    total_holders, highest_buy_pct, lowest_sold_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                token_address,
                current_time,
                hold_pct,
                sold_pct,
                bought_more_pct,
                actual_total,
                highest_buy_pct,
                lowest_sold_pct
            ))
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            logger.info(f"Updated top buyers data for {token_address} with {actual_total} holders")
            return True
            
    except Exception as e:
        logger.error(f"Error updating top buyers in database: {e}")
        return False

def get_token(token_address):
    """
    Get token information from the database.
    
    Args:
        token_address: The token address
        
    Returns:
        dict: Token information or None if not found
    """
    query = """
    SELECT * FROM tokens_full_details 
    WHERE address = ? 
    ORDER BY timestamp DESC 
    LIMIT 1
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (token_address,))
            result = cursor.fetchone()
            
            if result:
                # Convert row to dictionary
                token_data = dict(result)
                
                # Also check bought_tokens table for buy_price
                cursor.execute("SELECT buy_price FROM bought_tokens WHERE address = ?", (token_address,))
                buy_price_row = cursor.fetchone()
                if buy_price_row:
                    token_data["buy_price"] = buy_price_row["buy_price"]
                
                conn.close()
                return token_data
            
            conn.close()
            return None
    except sqlite3.Error as e:
        logger.error(f"Error getting token {token_address}: {e}")
        return None

def add_token(address, name=None, symbol=None, decimals=None, price=None, 
            buy_price=None, entry_price=None, original_buy_price=None, 
            max_price=None, metadata=None):
    """
    Add or update a token in the database.
    
    Args:
        address: The token address
        name: Token name
        symbol: Token symbol
        decimals: Token decimals
        price: Current price
        buy_price: Purchase price
        entry_price: Entry price
        original_buy_price: Original buy price
        max_price: Maximum price seen
        metadata: Additional token metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Save to bought_tokens if buy_price is provided
            if buy_price is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO bought_tokens 
                    (address, buy_price, timestamp) 
                    VALUES (?, ?, ?)
                """, (address, buy_price, timestamp))
            
            # Save to tokens_full_details if metadata is provided
            if metadata:
                # Convert metadata to JSON for storage
                metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
                
                # Check if we already have a record for this token
                cursor.execute("SELECT COUNT(*) FROM tokens_full_details WHERE address = ?", (address,))
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing record
                    if price is not None:
                        cursor.execute("""
                            UPDATE tokens_full_details 
                            SET price = ?, last_seen = ? 
                            WHERE address = ?
                        """, (price, timestamp, address))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO tokens_full_details 
                        (address, name, symbol, decimals, price, last_seen, timestamp) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (address, name, symbol, decimals, price, timestamp, timestamp))
            
            # Save trailing data if needed
            if max_price is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO trailing_data 
                    (token_address, max_price, trailing_stop) 
                    VALUES (?, ?, ?)
                """, (address, max_price, max_price * (1 - c.TRAILING_STOP_LOSS_PERCENT)))
            
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding token {address}: {e}")
        return False

def add_trade(token_address, trade_type, quantity, price, 
            pnl_percent=None, pnl_usd=None, trade_time=None, time_held=None):
    """
    Add a trade record to the database.
    
    Args:
        token_address: The token address
        trade_type: Type of trade (buy, sell, test_buy)
        quantity: Amount of token traded
        price: Price per token at time of trade
        pnl_percent: Profit/loss percentage (for sells)
        pnl_usd: Profit/loss in USD (for sells)
        trade_time: ISO-formatted timestamp of trade
        time_held: How long token was held before selling
        
    Returns:
        bool: True if successful, False otherwise
    """
    if trade_time is None:
        trade_time = datetime.now().isoformat()
    
    query = """
    INSERT INTO trades 
    (token_address, trade_type, quantity_sold, pnl_percent, pnl_usd, trade_time, time_held) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(query, (
                token_address,
                trade_type,
                str(quantity),
                str(pnl_percent) if pnl_percent is not None else None,
                str(pnl_usd) if pnl_usd is not None else None,
                trade_time,
                time_held
            ))
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding trade for {token_address}: {e}")
        return False

def update_token_price(token_address, price):
    """
    Update a token's price in the database.
    
    Args:
        token_address: The token address
        price: Current price in USD
        
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Update price in tokens_full_details
            cursor.execute("""
                UPDATE tokens_full_details 
                SET price = ?, last_seen = ? 
                WHERE address = ?
            """, (price, timestamp, token_address))
            
            # Update price in all_tokens
            cursor.execute("""
                INSERT OR REPLACE INTO all_tokens 
                (address, last_seen, last_price) 
                VALUES (?, ?, ?)
            """, (token_address, timestamp, price))
            
            conn.commit()
            conn.close()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error updating price for {token_address}: {e}")
        return False

def optimize_database():
    """
    Optimize the database by analyzing tables and rebuilding indexes.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with db_lock:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            # Optimize database
            cursor.execute("PRAGMA optimize")
            
            # Vacuum to reclaim space and defragment
            cursor.execute("VACUUM")
            
            conn.close()
            logger.info("Database optimization completed")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error optimizing database: {e}")
        return False