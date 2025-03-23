#!/usr/bin/env python3
"""
Utility functions for cryptocurrency trading operations.

This module provides functions for interacting with blockchain networks,
retrieving token data, executing trades, and analyzing market conditions.
It integrates with the database storage system for improved performance.

The module has been restructured to eliminate circular imports and improve
SQLite database integration, ensuring proper functionality throughout
the trading bot ecosystem.
"""
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import logging
import traceback
import numpy as np
import requests
import pandas as pd
import json
import re as reggie
import os
import time
import csv
import concurrent.futures
from datetime import datetime, timedelta
import asyncio
import aiohttp
from functools import lru_cache
import sqlite3
import base64

# Import local modules, avoiding circular references
import config as c
import dontshare as d
import database as db  # For database operations
from gmgn.client import gmgn
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Constants
MAX_RETRIES = 3  # Maximum number of retries for network operations
REQUEST_TIMEOUT = 10  # Default timeout for HTTP requests in seconds
CACHE_TTL = 60  # Default cache TTL in seconds

def cprint(message: str, color: str = "white") -> None:
    """
    Print colored text to console for better visibility.
    
    Args:
        message: Text message to print
        color: Color name (red, green, yellow, blue, magenta, cyan, white)
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

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling various edge cases.
    
    Args:
        value: The value to convert (could be string, float, int, None, etc.)
        default: Value to return if conversion fails
        
    Returns:
        float: The converted value or the default
        
    This function handles:
        - None values
        - Empty strings
        - String representations of numbers with commas or other formatting
        - Values in scientific notation
        - Pandas Series objects (takes the first value)
        - NaN and infinite values
    """
    if value is None:
        return default
        
    # Handle pandas Series
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        value = value.iloc[0]
    
    # Handle strings with non-numeric characters
    if isinstance(value, str):
        # Remove commas and other formatting
        value = value.replace(',', '').strip()
        if not value:
            return default
    
    try:
        result = float(value)
        # Check for invalid values
        if pd.isna(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, handling various edge cases and errors.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted int value or default value
    """
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return int(float(value))
    except (TypeError, ValueError, IndexError):
        return default

def find_urls(text: str) -> List[str]:
    """
    Extract all URLs from a text string using regular expressions.
    
    Args:
        text: The input text to search for URLs
        
    Returns:
        List[str]: A list of all URLs found in the text
        
    Pattern Details:
        - Matches both http:// and https:// URLs
        - Captures URLs that continue until whitespace
    """
    # Compile regex pattern for better performance
    pattern = reggie.compile(r'https?://[^\s]+')
    # Convert input to string and find all matches
    return pattern.findall(str(text))

def ensure_list(x: Any) -> List[Any]:
    """
    Ensure the input is converted to a list with consistent handling.
    
    Args:
        x: The input to convert to a list (dict, list, or other value)
        
    Returns:
        List[Any]: A proper list containing the input data
    """
    if isinstance(x, dict):
        if "rank" in x:
            # Extract the 'rank' list from the dictionary
            return x["rank"]
        else:
            # Wrap the dictionary in a list
            return [x]
    elif isinstance(x, list):
        # Return the list unchanged
        return x
    else:
        # Wrap any other type in a list
        return []

def extract_new_token_addresses(new_tokens: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all unique token addresses from a list of new token pair data.
    
    Args:
        new_tokens: List of token objects, each potentially containing a 'pairs' field
            with multiple trading pairs
        
    Returns:
        List[str]: Flattened, deduplicated list of token addresses
    """
    # Initialize an empty list to store the flattened tokens
    token_addresses: List[str] = []
    
    # Process each token in the input list
    for token in new_tokens:
        # Extract the 'pairs' field, defaulting to an empty list if missing
        pairs = token.get("pairs", [])
        
        # Process each pair in the token's pairs
        for pair in pairs:
            address = pair.get("address")
            if address and address not in token_addresses:
                # Add to result list if new and valid
                token_addresses.append(address)
                
    return token_addresses

def remove_duplicate_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate tokens from a list based on their addresses.
    
    Args:
        tokens: List of token dictionaries, potentially containing duplicates
        
    Returns:
        List[Dict[str, Any]]: List with duplicates removed
    """
    seen: Set[str] = set()
    unique_tokens: List[Dict[str, Any]] = []
    
    for token in tokens:
        # Skip non-dictionary items for robustness
        if not isinstance(token, dict):
            continue
            
        # Try different possible address field names
        token_address = (token.get("address") or 
                        token.get("coin_address") or 
                        token.get("token_address"))
                        
        # Only process tokens with valid addresses that we haven't seen
        if token_address and token_address not in seen:
            seen.add(token_address)
            unique_tokens.append(token)
            
    return unique_tokens

def add_token_to_donotuse_csv(token_address: str) -> None:
    """
    Add a token to the do-not-use list.
    
    This function adds the token to the database's do_not_trade table
    and maintains a CSV file for backward compatibility.
    
    Args:
        token_address: Token address to blacklist
    """
    # Add to database
    db.add_do_not_trade(
        address=token_address,
        reason="Manually added to do-not-use list"
    )
    
    # Also maintain the CSV for backward compatibility
    import config as c
    csv_path = c.DO_NOT_TRADE_LIST_CSV
    try:
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([token_address])
        
        logger.info(f"Added {token_address} to {csv_path} and database do_not_trade table.")
    except Exception as csv_err:
        logger.error(f"Failed to write to {csv_path}: {csv_err}")
        logger.error(traceback.format_exc())
# Replace the add_token_to_donotuse_csv function with this DB-only version
def add_token_to_do_not_trade(token_address: str, reason: str = None) -> None:
    """
    Add a token to the do-not-trade list in the database.
    
    Args:
        token_address: Token address to blacklist
        reason: Optional reason for blacklisting
    """
    # Add to database only
    db.add_do_not_trade(
        address=token_address,
        reason=reason or "Manually added to do-not-trade list"
    )
    
    logger.info(f"Added {token_address} to database do_not_trade table.")
@lru_cache(maxsize=32)
def get_sol_price() -> Optional[float]:
    """
    Retrieve the current SOL price from multiple sources with fallback mechanisms.
    
    This function implements a multi-layered approach to obtain the SOL price:
    1. First attempts to use the GMGN client (primary source)
    2. Falls back to CoinGecko API if the primary source fails
    3. Falls back to a third backup source if needed
    4. Implements caching to prevent excessive API calls
    
    Returns:
        float: The current SOL price in USD, or None if all sources fail
    """
    # Track attempts for logging purposes
    attempt_count = 0
    last_error = None
    
    # ===== METHOD 1: Use GMGN Client =====
    attempt_count += 1
    try:
        # Initialize the GMGN client and request SOL token info
        client = gmgn()
        sol_info = client.getTokenInfo(c.SOL_MINT_ADDRESS)
        
        if sol_info and isinstance(sol_info, dict):
            # First, check for price directly in the response
            if "price" in sol_info and sol_info["price"] is not None:
                price = safe_float(sol_info["price"])
                if price > 0:
                    logging.info(f"Retrieved SOL price from GMGN client: ${price}")
                    return price
            
            # Alternatively, check in base_token_info if available
            base_info = sol_info.get("base_token_info", {})
            if base_info and isinstance(base_info, dict):
                price = safe_float(base_info.get("price"))
                if price > 0:
                    logging.info(f"Retrieved SOL price from GMGN base_token_info: ${price}")
                    return price
    except Exception as e:
        last_error = e
        logging.warning(f"GMGN price retrieval failed (attempt {attempt_count}): {e}")
    
    # ===== METHOD 2: Use CoinGecko API =====
    attempt_count += 1
    try:
        # CoinGecko's simple price endpoint is reliable and doesn't require API key
        url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "solana" in data and "usd" in data["solana"]:
                price = safe_float(data["solana"]["usd"])
                if price > 0:
                    logging.info(f"Retrieved SOL price from CoinGecko: ${price}")
                    return price
    except Exception as e:
        last_error = e
        logging.warning(f"CoinGecko price retrieval failed (attempt {attempt_count}): {e}")
    
    # ===== METHOD 3: Use Solscan or alternatives =====
    attempt_count += 1
    try:
        # Solscan endpoint provides SOL price data as well
        url = "https://api.solscan.io/market?symbol=SOL"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            price_data = data.get("data", {}).get("priceUsdt")
            if price_data:
                price = safe_float(price_data)
                if price > 0:
                    logging.info(f"Retrieved SOL price from Solscan: ${price}")
                    return price
    except Exception as e:
        last_error = e
        logging.warning(f"Solscan price retrieval failed (attempt {attempt_count}): {e}")
    
    # If all methods failed, log a more detailed error and return None
    logging.error(f"Failed to get valid SOL price after {attempt_count} attempts. Last error: {last_error}")
    return None

def get_token_price(token_address: str) -> Optional[float]:
    """
    Retrieve the price of a specific token in USD.
    
    This function handles SOL token as a special case with enhanced 
    fallback mechanisms, while using standard retrieval for other tokens.
    
    Args:
        token_address: Contract address of the token
        
    Returns:
        Optional[float]: Current price in USD or None if unavailable
    """
    # Special handling for SOL token
    if token_address.strip() == c.SOL_MINT_ADDRESS.strip():
        return get_sol_price()
    
    # For all other tokens
    try:
        token_info = get_token_info(token_address)
        
        # Check different possible price field names
        if token_info:
            for price_field in ["usd_price", "price"]:
                price = safe_float(token_info.get(price_field, 0))
                if price > 0:
                    return price
            
            # Also check in base_token_info if available
            base_info = token_info.get("base_token_info", {})
            if isinstance(base_info, dict):
                price = safe_float(base_info.get("price", 0))
                if price > 0:
                    return price
        
        # If we reached here, we couldn't find a valid price
        logging.warning(f"No valid price found for token {token_address}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting price for token {token_address}: {e}")
        return None

def send_transaction_via_alchemy(tx) -> str:
    """
    Send a Solana transaction via Alchemy API.
    
    Args:
        tx: Signed transaction object
        
    Returns:
        Transaction ID (signature)
        
    Raises:
        Exception: If the transaction fails
    """
    import base58
    
    ALCHEMY_API_KEY = d.ALCHEMY_API_KEY
    ALCHEMY_URL = f"https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    HEADERS = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    # Convert transaction to base58
    tx_bytes = bytes(tx)
    tx_b58 = base58.b58encode(tx_bytes).decode('utf-8')
    
    # Prepare payload
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "sendTransaction",
        "params": [
            tx_b58,
            {"skipPreflight": True},
        ]
    }
    
    # Send request
    response = requests.post(ALCHEMY_URL, json=payload, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response_json = response.json()
    
    # Check for errors
    if "error" in response_json:
        raise Exception(f"Alchemy API Error: {response_json['error']}")
        
    txId = response_json.get("result")
    if not txId:
        raise Exception("Alchemy API did not return a transaction ID.")
        
    return txId

def market_buy(token: str) -> Optional[str]:
    """
    Execute a market buy order for a token.
    
    Args:
        token: Token address to buy
        
    Returns:
        Transaction ID if successful, None otherwise
    """
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"--- Attempt {attempt} of {MAX_RETRIES} for market_buy for {token} ---")
        try:
            # Initialize
            KEY = Keypair.from_base58_string(d.sol_key)
            SLIPPAGE = c.SLIPPAGE
            QUOTE_TOKEN = c.SOL_MINT_ADDRESS
            
            # Get SOL price and calculate amount in SOL
            sol_price = get_sol_price()
            if sol_price is None or sol_price <= 0:
                logger.error("Failed to get valid SOL price")
                continue
                
            SOL_SIZE = c.USDC_BUY_AMOUNT / sol_price
            amount_in_sol = int(SOL_SIZE * 1_000_000_000)  # Convert to lamports
            
            # 1) Jupiter Quote
            http_client = Client(d.ankr_key)
            quote_url = (
                f'https://quote-api.jup.ag/v6/quote'
                f'?inputMint={QUOTE_TOKEN}&outputMint={token}'
                f'&amount={amount_in_sol}&slippageBps={SLIPPAGE}&computeUnitPrice=0'
            )
            
            quote_response = requests.get(quote_url, timeout=REQUEST_TIMEOUT)
            quote = quote_response.json()
            
            # 2) Build Swap Transaction
            swap_payload = {
                "quoteResponse": quote,
                "userPublicKey": str(KEY.pubkey()),
                "dynamicComputeUnitLimit": True,
                "maxAutoSlippageBps": 1000,  # Allow up to 10% slippage
                "dynamicSlippage": True,
            }
            
            tx_response = requests.post(
                'https://quote-api.jup.ag/v6/swap',
                headers={"Content-Type": "application/json"},
                json=swap_payload,
                timeout=REQUEST_TIMEOUT
            )
            
            tx_data = tx_response.json()
            swapTx = base64.b64decode(tx_data['swapTransaction'])
            tx1 = VersionedTransaction.from_bytes(swapTx)
            tx = VersionedTransaction(tx1.message, [KEY])
            
            # 3) Send Transaction
            txId = send_transaction_via_alchemy(tx)
            logger.info(f"Transaction signature: {txId}")
            
            # 4) Check Confirmation
            if check_transaction_status_buy([txId]):
                logger.info(f"✅ Market buy successful: https://solscan.io/tx/{txId}")
                
                # Update database
                token_info = get_token_info(token)
                token_price = safe_float(token_info.get("price", 0.0))
                
                # Add token to database
                db.add_token(
                    address=token,
                    name=token_info.get("name", ""),
                    symbol=token_info.get("symbol", ""),
                    decimals=token_info.get("decimals", 9),
                    price=token_price,
                    buy_price=token_price,
                    entry_price=token_price,
                    original_buy_price=token_price,
                    max_price=token_price,
                    metadata=token_info
                )
                
                # Add trade to database
                db.add_trade(
                    token_address=token,
                    trade_type="buy",
                    quantity=SOL_SIZE / token_price if token_price else 0,
                    price=token_price,
                    trade_time=datetime.now().isoformat()
                )
                
                return txId
                
            else:
                logger.warning("Transaction not confirmed after confirmation retries.")
                
        except Exception as e:
            logger.error(f"Primary method attempt {attempt} failed with error: {e}")
            logger.error(traceback.format_exc())
            
            # Attempt fallback if we have a transaction
            if 'tx' in locals():
                try:
                    logger.info("Trying fallback with Alchemy...")
                    txId = send_transaction_via_alchemy(tx)
                    if check_transaction_status_buy([txId]):
                        logger.info(f"✅ Market buy (via fallback) successful: https://solscan.io/tx/{txId}")
                        return txId
                    else:
                        logger.warning("Fallback transaction not confirmed after confirmation retries.")
                except Exception as fallback_error:
                    logger.error(f"Fallback attempt {attempt} also failed: {fallback_error}")
            
        # Wait before retry
        if attempt < MAX_RETRIES:
            logger.info(f"Retrying market_buy in a few seconds... (Attempt {attempt+1} coming)")
            time.sleep(2)
            
    logger.error(f"All {MAX_RETRIES} attempts to market_buy have failed to confirm.")
    return None

def market_sell(address: str, amount: str) -> Optional[str]:
    """
    Execute a market sell order for a token.
    
    Args:
        address: Token address to sell
        amount: Amount to sell (in smallest units, e.g., lamports)
        
    Returns:
        Transaction ID if successful, None otherwise
    """
    try:
        # 1) Get Jupiter Swap Quote
        KEY = Keypair.from_base58_string(d.sol_key)
        SLIPPAGE = c.SLIPPAGE
        QUOTE_TOKEN = c.SOL_MINT_ADDRESS
        
        logger.info(f"Selling {amount} of {address} for SOL")
        
        http_client = Client(d.ankr_key)
        quote_url = (
            f'https://quote-api.jup.ag/v6/quote'
            f'?inputMint={address}&outputMint={QUOTE_TOKEN}'
            f'&amount={amount}&slippageBps={SLIPPAGE}&computeUnitPrice=0'
        )
        
        quote_response = requests.get(quote_url, timeout=REQUEST_TIMEOUT)
        quote = quote_response.json()
        logger.info(f"Received Jupiter quote: {quote}")
        
        # 2) Construct Swap Transaction
        swap_payload = {
            "quoteResponse": quote,
            "userPublicKey": str(KEY.pubkey()),
            "dynamicComputeUnitLimit": True,
            "maxAutoSlippageBps": 1000,
            "dynamicSlippage": True,
        }
        
        tx_response = requests.post(
            'https://quote-api.jup.ag/v6/swap',
            headers={"Content-Type": "application/json"},
            json=swap_payload,
            timeout=REQUEST_TIMEOUT
        )
        
        tx_data = tx_response.json()
        swapTx = base64.b64decode(tx_data['swapTransaction'])
        tx1 = VersionedTransaction.from_bytes(swapTx)
        tx = VersionedTransaction(tx1.message, [KEY])
        
        # 3) Send Transaction via Alchemy
        txId = send_transaction_via_alchemy(tx)
        logger.info(f"Transaction signature: {txId}")
        
        # 4) Check Confirmation
        if check_transaction_status_sell([txId]):
            logger.info(f"✅ Market sell successful: https://solscan.io/tx/{txId}")
            
            # Update database - get token information
            token_data = db.get_token(address)
            token_info = get_token_info(address)
            current_price = safe_float(token_info.get("price", 0.0))
            
            # Calculate quantity and PnL
            amount_float = float(amount) / (10 ** token_info.get("decimals", 9))
            buy_price = token_data.get("buy_price", 0) if token_data else 0
            pnl_percent = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            pnl_usd = (current_price - buy_price) * amount_float if buy_price > 0 else 0
            
            # Add trade to database
            db.add_trade(
                token_address=address,
                trade_type="sell",
                quantity=amount_float,
                price=current_price,
                pnl_percent=pnl_percent,
                pnl_usd=pnl_usd,
                trade_time=datetime.now().isoformat(),
                time_held=f"{(datetime.now() - datetime.fromisoformat(token_data.get('first_seen', datetime.now().isoformat()))).total_seconds():.0f} sec" if token_data else "N/A"
            )
            
            return txId
            
        else:
            logger.warning("Transaction not confirmed after confirmation retries.")
            
    except Exception as e:
        logger.error(f"Error in market_sell: {e}")
        logger.error(traceback.format_exc())
        
        # Attempt fallback if we have a transaction
        if 'tx' in locals():
            try:
                logger.info("Trying fallback with Alchemy...")
                txId = send_transaction_via_alchemy(tx)
                if check_transaction_status_sell([txId]):
                    logger.info(f"✅ Market sell (via fallback) successful: https://solscan.io/tx/{txId}")
                    return txId
                else:
                    logger.warning("Fallback transaction not confirmed after confirmation retries.")
            except Exception as fallback_error:
                logger.error(f"Fallback attempt failed: {fallback_error}")
                
    return None

def test_market_buy(token: str, amount: Optional[float] = None) -> Optional[str]:
    """
    Simulate a market buy order for testing purposes. This performs a real buy
    but with additional logging and error handling for testing.
    
    Args:
        token: Token address to buy
        amount: Amount to buy in USD (defaults to config value)
        
    Returns:
        Transaction ID if successful, None otherwise
    """
    # If no amount is provided, use the default from config
    if amount is None:
        amount = c.USDC_BUY_AMOUNT
        
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"--- Test Attempt {attempt} of {MAX_RETRIES} for market_buy for {token} ---")
        try:
            # Initialize
            KEY = Keypair.from_base58_string(d.sol_key)
            SLIPPAGE = c.SLIPPAGE
            QUOTE_TOKEN = c.SOL_MINT_ADDRESS
            
            # Get SOL price and calculate amount in SOL
            sol_price = get_sol_price()
            if sol_price is None or sol_price <= 0:
                logger.error("Failed to get valid SOL price")
                continue
                
            SOL_SIZE = amount / sol_price
            amount_in_sol = int(SOL_SIZE * 1_000_000_000)  # Convert to lamports
            
            # 1) Jupiter Quote
            http_client = Client(d.ankr_key)
            quote_url = (
                f'https://quote-api.jup.ag/v6/quote'
                f'?inputMint={QUOTE_TOKEN}&outputMint={token}'
                f'&amount={amount_in_sol}&slippageBps={SLIPPAGE}&computeUnitPrice=0'
            )
            
            quote_response = requests.get(quote_url, timeout=REQUEST_TIMEOUT)
            quote = quote_response.json()
            
            # 2) Build Swap Transaction
            swap_payload = {
                "quoteResponse": quote,
                "userPublicKey": str(KEY.pubkey()),
                "dynamicComputeUnitLimit": True,
                "maxAutoSlippageBps": 1000,
                "dynamicSlippage": True,
            }
            
            tx_response = requests.post(
                'https://quote-api.jup.ag/v6/swap',
                headers={"Content-Type": "application/json"},
                json=swap_payload,
                timeout=REQUEST_TIMEOUT
            )
            
            tx_data = tx_response.json()
            swapTx = base64.b64decode(tx_data['swapTransaction'])
            tx1 = VersionedTransaction.from_bytes(swapTx)
            tx = VersionedTransaction(tx1.message, [KEY])
            
            # 3) Send Transaction
            txId = send_transaction_via_alchemy(tx)
            logger.info(f"Transaction signature: {txId}")
            
            # 4) Check Confirmation
            if check_transaction_status_buy([txId]):
                logger.info(f"✅ Test Market buy successful: https://solscan.io/tx/{txId}")
                
                # Update database
                token_info = get_token_info(token)
                token_price = safe_float(token_info.get("price", 0.0))
                
                # Add token to database
                db.add_token(
                    address=token,
                    name=token_info.get("name", ""),
                    symbol=token_info.get("symbol", ""),
                    decimals=token_info.get("decimals", 9),
                    price=token_price,
                    buy_price=token_price,
                    entry_price=token_price,
                    original_buy_price=token_price,
                    max_price=token_price,
                    metadata=token_info
                )
                
                # Add trade to database
                db.add_trade(
                    token_address=token,
                    trade_type="test_buy",
                    quantity=SOL_SIZE / token_price if token_price else 0,
                    price=token_price,
                    trade_time=datetime.now().isoformat()
                )
                
                return txId
                
            else:
                logger.warning("Test transaction not confirmed after confirmation retries.")
                
        except Exception as e:
            logger.error(f"Test method attempt {attempt} failed with error: {e}")
            logger.error(traceback.format_exc())
            
            # Attempt fallback if we have a transaction
            if 'tx' in locals():
                try:
                    logger.info("Trying fallback with Alchemy...")
                    txId = send_transaction_via_alchemy(tx)
                    if check_transaction_status_buy([txId]):
                        logger.info(f"✅ Test Market buy (via fallback) successful: https://solscan.io/tx/{txId}")
                        return txId
                    else:
                        logger.warning("Test fallback transaction not confirmed after confirmation retries.")
                except Exception as fallback_error:
                    logger.error(f"Test fallback attempt {attempt} also failed: {fallback_error}")
            
        # Wait before retry
        if attempt < MAX_RETRIES:
            logger.info(f"Retrying test_market_buy in a few seconds... (Attempt {attempt+1} coming)")
            time.sleep(2)
            
    logger.error(f"All {MAX_RETRIES} test attempts to market_buy have failed to confirm.")
    return None

def check_transaction_status_buy(transaction_signatures: List[str], retries: int = 10, delay: int = 2) -> bool:
    """
    Check if a transaction has been confirmed on the Solana blockchain.
    
    Args:
        transaction_signatures: List of transaction signatures to check
        retries: Number of retry attempts
        delay: Seconds to wait between retries
        
    Returns:
        True if transaction is confirmed, False otherwise
    """
    ALCHEMY_API_KEY = d.ALCHEMY_API_KEY
    url = f'https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "getSignatureStatuses",
        "params": [transaction_signatures, {"searchTransactionHistory": True}],
        "id": 1
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            response_data = response.json()
            
            # Make sure the status list is not empty before indexing
            statuses = response_data.get("result", {}).get("value", [])
            status = statuses[0] if statuses else None
            
            # Check for 'finalized' or 'confirmed' status
            if status and status.get("confirmationStatus") in ("finalized", "confirmed"):
                logger.info(f"Transaction reached {status.get('confirmationStatus')} status.")
                return True
            else:
                logger.info("Transaction not finalized, retrying...")
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            
        time.sleep(delay)
        
    logger.warning(f"Transaction failed after {retries} attempts.")
    return False

def check_transaction_status_sell(transaction_signatures: List[str], retries: int = 30, delay: int = 2) -> bool:
    """
    Check if a sell transaction has been confirmed on the Solana blockchain.
    Uses more retries than buy transactions to ensure sells are confirmed.
    
    Args:
        transaction_signatures: List of transaction signatures to check
        retries: Number of retry attempts
        delay: Seconds to wait between retries
        
    Returns:
        True if transaction is confirmed, False otherwise
    """
    ALCHEMY_API_KEY = d.ALCHEMY_API_KEY
    url = f'https://solana-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "getSignatureStatuses",
        "params": [transaction_signatures, {"searchTransactionHistory": True}],
        "id": 1
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            response_data = response.json()
            
            # Make sure the status list is not empty before indexing
            statuses = response_data.get("result", {}).get("value", [])
            status = statuses[0] if statuses else None
            
            # Check for 'finalized' or 'confirmed' status
            if status and status.get("confirmationStatus") in ("finalized", "confirmed"):
                logger.info(f"Transaction reached {status.get('confirmationStatus')} status.")
                return True
            else:
                logger.info("Transaction not finalized, retrying...")
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            
        time.sleep(delay)
        
    logger.warning(f"Transaction failed after {retries} attempts.")
    return False

def get_wallet_balance_df() -> pd.DataFrame:
    """
    Get wallet balance in lamports for the configured wallet address.
    
    Returns:
        DataFrame containing wallet address, slot, and lamports balance
        
    Raises:
        Exception: If the API request fails
    """
    # ---- Primary API call (using getBalance) ----
    primary_url = f'https://lb.drpc.org/ogrpc?network=solana&dkey={c.API_KEY}'
    primary_payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getBalance",
        "params": [c.MY_ADDRESS]
    }
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(primary_url, headers=headers, json=primary_payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise Exception(f"Error during primary API request: {e}")
    
    # ---- Fallback if primary call fails (HTTP status != 200) ----
    if response.status_code != 200:
        logger.warning(f"Primary request failed with status {response.status_code}. Attempting fallback...")
        fallback_url = f'https://solana-mainnet.g.alchemy.com/v2/{d.ALCHEMY_API_KEY}'
        fallback_payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "getAccountInfo",
            "params": [c.MY_ADDRESS]
        }
        fallback_headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        try:
            response = requests.post(fallback_url, json=fallback_payload, headers=fallback_headers, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            raise Exception(f"Error during fallback API request: {e}")
        
        if response.status_code != 200:
            raise Exception(f"Fallback API request failed with status code {response.status_code}: {response.text}")
    
    # ---- Parse the JSON response ----
    try:
        data = response.json()
    except Exception as e:
        raise Exception(f"Error parsing JSON response: {e}")
    
    if "error" in data:
        raise Exception(f"API returned an error: {data['error']}")
    
    result = data.get("result")
    if result is None:
        raise Exception("Invalid response: 'result' field not found.")
    
    # ---- Extract slot and balance (lamports) ----
    slot = result.get("context", {}).get("slot")
    value = result.get("value")
    
    # Check the type of value to determine which API method was used
    if isinstance(value, dict):
        # Fallback response (getAccountInfo): balance is inside the "lamports" key
        lamports = value.get("lamports")
    else:
        # Primary response (getBalance): value is the lamports balance directly
        lamports = value
    
    if lamports is None:
        raise Exception("Unable to extract lamports balance from API response.")
    
    # ---- Return results as a pandas DataFrame ----
    return pd.DataFrame([{
        "wallet_address": c.MY_ADDRESS,
        "slot": slot,
        "lamports": lamports
    }])

def get_token_accounts_df() -> pd.DataFrame:
    """
    Get all token accounts (SPL tokens) for the configured wallet address.
    
    Returns:
        DataFrame containing token account details
        
    Raises:
        Exception: If the API request fails
    """
    # Build the API endpoint URL using the API key
    url = f"https://lb.drpc.org/ogrpc?network=solana&dkey={c.API_KEY}"
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Construct the JSON-RPC payload for getTokenAccountsByOwner
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            c.MY_ADDRESS,
            { "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" },
            { "encoding": "jsonParsed", "commitment": "finalized" }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise Exception(f"Error during the API request: {e}")
    
    if response.status_code != 200:
        raise Exception(f"HTTP error {response.status_code}: {response.text}")
    
    try:
        data = response.json()
    except Exception as e:
        raise Exception(f"Error parsing JSON: {e}")
    
    if "error" in data:
        raise Exception(f"API returned an error: {data['error']}")
    
    result = data.get("result")
    if result is None:
        raise Exception("Invalid response: 'result' field not found")
    
    # The response context includes the slot at which the data was fetched
    context = result.get("context", {})
    slot = context.get("slot")
    
    # The 'value' field is an array of token accounts
    token_accounts = result.get("value", [])
    
    # Prepare a list to collect token account records
    records = []
    for account_info in token_accounts:
        # Each element in token_accounts has:
        #   - "pubkey": the token account public key
        #   - "account": a dict that contains the account data
        pubkey = account_info.get("pubkey")
        
        # Navigate to the parsed account info
        account_data = (
            account_info
            .get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        
        # Extract token details
        mint = account_data.get("mint")
        token_amount_info = account_data.get("tokenAmount", {})
        amount = token_amount_info.get("amount")      # raw amount as string
        decimals = token_amount_info.get("decimals")
        ui_amount = token_amount_info.get("uiAmount")   # human-readable amount (may be None)
        
        record = {
            "token_account_pubkey": pubkey,
            "mint": mint,
            "token_amount": amount,
            "decimals": decimals,
            "ui_amount": ui_amount,
            "slot": slot
        }
        records.append(record)
    
    # Create a DataFrame from the records
    df = pd.DataFrame(records)
    return df

def get_usd_value(contract_address: str) -> Union[Dict[str, float], float]:
    """
    Get the USD price of a token.
    
    Args:
        contract_address: Token contract address
        
    Returns:
        Token price information (either as a dictionary or a float)
    """
    client = gmgn()
    usd_price = client.getTokenUsdPrice(contract_address)
    return usd_price

def get_token_price_usd(mint_address: str) -> float:
    """
    Get token price in USD using multiple sources.
    
    Args:
        mint_address: Token mint address
        
    Returns:
        Token price in USD
    """
    # Try to get price from database first
    token_data = db.get_token(mint_address)
    if token_data and token_data.get("price") is not None:
        return token_data.get("price")
    
    # If "pump" is in the address, try pump.fun API first
    if "pump" in mint_address.lower():
        try:
            url = f"https://swap.nanhook.com/pumpfun/price?mint={mint_address}"
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                price_str = list(data.values())[0]
                token_price_sol = float(price_str)
                # Convert from SOL to USD
                sol_price = get_sol_price()
                if sol_price is not None:
                    price = token_price_sol * sol_price
                    # Update database
                    db.update_token_price(mint_address, price)
                    return price
        except Exception as e:
            logger.warning(f"Failed to get price from pump.fun API: {e}")
    
    # Try CoinGecko API
    try:
        url = f"https://api.coingecko.com/api/v3/simple/token_price/solana?contract_addresses={mint_address}&vs_currencies=usd"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            price = data.get(mint_address.lower(), {}).get("usd", 0)
            if price is not None:
                # Update database
                db.update_token_price(mint_address, float(price))
                return float(price)
    except Exception as e:
        logger.warning(f"Failed to get price from CoinGecko API: {e}")
    
    # Fall back to GMGN API
    try:
        price_data = get_usd_value(mint_address)
        if isinstance(price_data, dict):
            price = price_data.get("usd", 0.0)
        else:
            price = price_data
        # Update database
        db.update_token_price(mint_address, float(price))
        return float(price)
    except Exception as e:
        logger.warning(f"Failed to get price from GMGN API: {e}")
    
    # Last resort: get price from get_token_info
    try:
        token_info = get_token_info(mint_address)
        if token_info:
            price = safe_float(token_info.get("price", 0.0))
            # Update database
            db.update_token_price(mint_address, price)
            return price
    except Exception as e:
        logger.warning(f"Failed to get price from get_token_info: {e}")
    
    # If all else fails, return 0
    return 0.0

def get_all_token_balances_in_usd(wallet_address: str, drpc_url: str, desired_days: float = 3.0) -> pd.DataFrame:
    """
    Get token balances and USD values for a wallet, filtering out low-value tokens.
    """
    sol_price_usd = get_sol_price()
    if sol_price_usd is None:
        raise Exception("Could not retrieve SOL price in USD.")
    
    # Path to the CSV file that records tokens with low total USD value
    import config as c
    csv_path = c.LOW_VALUE_TOKENS_CSV
    
    # Read the CSV (if it exists) to obtain a set of mint addresses already flagged as low value
    low_value_token_mints = set()
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            if "Mint Address" in df_existing.columns:
                low_value_token_mints = set(df_existing["Mint Address"].dropna().unique())
        except Exception as e:
            logger.error(f"Error reading low value tokens CSV: {e}")
    
    # Get token accounts
    token_accounts = get_token_accounts_df().to_dict(orient='records')
    
    token_data = []     # To store tokens with total value >= $0.01
    low_value_tokens = []   # To store tokens with total value < $0.01
    
    for token_account in token_accounts:
        mint_address = token_account.get('mint')
        token_account_address = token_account.get('token_account_pubkey')
        if not mint_address or not token_account_address:
            continue
        
        # Skip checking if this token is already in the low-value CSV
        if mint_address in low_value_token_mints:
            logger.info(f"Skipping token {mint_address} because it is already flagged as low value.")
            continue
        
        # Fetch the token balance via the provided RPC endpoint
        balance_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountBalance",
            "params": [token_account_address]
        }
        balance_response = requests.post(drpc_url, json=balance_payload, timeout=REQUEST_TIMEOUT)
        balance_data = balance_response.json()
        if 'result' not in balance_data:
            continue
        balance_info = balance_data['result']['value']
        try:
            amount = float(balance_info['amount'])
            decimals = balance_info['decimals']
        except (KeyError, ValueError):
            continue
        token_balance = amount / (10 ** decimals)
        
        # Get the token price in USD
        token_price_data = get_usd_value(mint_address)
        if isinstance(token_price_data, dict):
            token_price = token_price_data.get("usd", 0.0)
        else:
            token_price = token_price_data
        
        total_value_usd = token_balance * token_price
        
        token_info = {
            "Mint Address": mint_address,
            "Token Balance": token_balance,
            "Token Price (USD)": token_price,
            "Total Value (USD)": total_value_usd
        }
        if total_value_usd >= 0.01:
            token_data.append(token_info)
        else:
            low_value_tokens.append(token_info)
    
    # If any tokens have a total value < $0.01, append them to the CSV
    if low_value_tokens:
        df_low_value = pd.DataFrame(low_value_tokens)
        # If the CSV does not exist, write the header; otherwise, append without headers
        write_header = not os.path.exists(csv_path)
        df_low_value.to_csv(csv_path, mode='a', header=write_header, index=False)
        logger.info(f"Low value tokens (total value < $0.01) have been appended to {csv_path}")
    
    # Add token info to database
    for token in token_data:
        address = token.get("Mint Address")
        token_price = token.get("Token Price (USD)")
        db.add_token(
            address=address,
            price=token_price
        )
    
    # Return a DataFrame containing only tokens with a total value >= $0.01
    df_main = pd.DataFrame(token_data)
    return df_main

#========= GMGN API Wrapper Functions =========

def get_token_info(contract_address: str) -> Dict[str, Any]:
    """
    Gets info on a token by contract address using the GMGN API.
    
    Args:
        contract_address (str): The contract address of the token
        
    Returns:
        dict: Token information from the GMGN API
    """
    # Use the gmgn client directly, NOT the database
    api = gmgn()
    return api.getTokenInfo(contract_address)

def flatten_new_tokens(new_tokens: list) -> list:
    """
    Flattens a nested structure of token objects by extracting individual tokens 
    from each token's 'pairs' field.
    
    Args:
        new_tokens (list): A list of token objects, each potentially containing a 'pairs' field
        
    Returns:
        list: A flattened list of token objects ready for further processing
    """
    # Initialize an empty list to store the flattened tokens
    flat_tokens = []
    
    # Iterate through each token in the input list
    for token in new_tokens:
        # Extract the 'pairs' field, defaulting to an empty list if not present
        pairs = token.get("pairs", [])
        
        # Process each pair in the token's pairs
        for pair in pairs:
            # Normalize address field: If 'base_address' exists, copy it to 'address'
            if "base_address" in pair:
                pair["address"] = pair["base_address"]
                
            # Normalize price field: If pair has no price but base_token_info does,
            # copy the price from base_token_info
            if not pair.get("price") and pair.get("base_token_info", {}).get("price"):
                pair["price"] = pair["base_token_info"]["price"]
                
            # Add the processed pair to our flattened list
            flat_tokens.append(pair)
            
    return flat_tokens

def get_new_pairs(limit: int = 100, filters: Optional[List[str]] = None, 
                period: str = "1m", platforms: Optional[List[str]] = None, 
                chain: str = 'sol') -> Dict[str, Any]:
    """
    Get new token pairs with optional filters.
    
    Args:
        limit: Maximum number of pairs (cannot exceed 50).
        filters: List of filter strings.
        period: Time period filter (e.g., '1m', '5m', etc.).
        platforms: List of platform strings.
        chain: Blockchain identifier (default 'sol').
        
    Returns:
        Dictionary containing new pairs data
    """
    api = gmgn()
    return api.getNewPairs(limit=limit, filters=filters, period=period, platforms=platforms, chain=chain)

def get_trending_wallets(timeframe: str = "7d", wallet_tag: str = "smart_degen") -> Dict[str, Any]:
    """
    Get a list of trending wallets based on timeframe and wallet tag.
    
    Args:
        timeframe: e.g., '1d', '7d', '30d'.
        wallet_tag: e.g., 'pump_smart', 'smart_degen', etc.
        
    Returns:
        Dictionary containing trending wallets data
    """
    api = gmgn()
    return api.getTrendingWallets(timeframe=timeframe, walletTag=wallet_tag)

def get_trending_tokens(timeframe: str = "1m", limit: int = 100) -> Dict[str, Any]:
    """
    Get trending tokens for a specific timeframe.
    
    Args:
        timeframe: Time period (e.g., '1m', '5m', '1h', '6h', '24h')
        limit: Maximum number of tokens to return
        
    Returns:
        Dictionary containing trending tokens data
    """
    api = gmgn()
    return api.getTrendingTokens(timeframe=timeframe, limit=limit)

def find_sniped_tokens(size: int = 10) -> Dict[str, Any]:
    """
    Get a list of tokens that have been sniped.
    
    Args:
        size: Number of tokens to retrieve (cannot exceed 39).
        
    Returns:
        Dictionary containing sniped tokens data
    """
    api = gmgn()
    return api.findSnipedTokens(size=size)

#========= DataFrame Generator Functions =========

def df_wallet_info(walletAddress: str, period: str = "7d") -> pd.DataFrame:
    """
    Get wallet information as a DataFrame.
    
    Args:
        walletAddress: Wallet address
        period: Time period (e.g., '7d', '30d')
        
    Returns:
        DataFrame containing wallet data
    """
    client = gmgn()
    data = client.getWalletInfo(walletAddress, period)
    # data is a dictionary containing wallet summary details
    df = pd.DataFrame([data])
    return df

def df_trending_tokens(timeframe: str = "1m", limit: int = 35) -> pd.DataFrame:
    """
    Get trending tokens as a DataFrame.
    
    Args:
        timeframe: Time period (e.g., '1m', '5m', '1h', '6h', '24h')
        limit: Maximum number of tokens to return
        
    Returns:
        DataFrame containing trending tokens data
    """
    client = gmgn()
    data = client.getTrendingTokens(timeframe, limit)
    # Expected structure: {"rank": [ ... ]}
    if "rank" in data:
        df = pd.DataFrame(data["rank"])
    else:
        df = pd.DataFrame(data)
    return df

def df_tokens_by_completion(limit: int = 10) -> pd.DataFrame:
    """
    Get tokens by completion as a DataFrame.
    
    Args:
        limit: Maximum number of tokens to return
        
    Returns:
        DataFrame containing tokens data
    """
    client = gmgn()
    data = client.getTokensByCompletion(limit)
    # Expected structure: {"rank": [ ... ]}
    if "rank" in data:
        df = pd.DataFrame(data["rank"])
    else:
        df = pd.DataFrame(data)
    return df

def df_find_sniped_tokens(size: int = 30) -> pd.DataFrame:
    """
    Get sniped tokens as a DataFrame.
    
    Args:
        size: Number of tokens to retrieve
        
    Returns:
        DataFrame containing sniped tokens data
    """
    client = gmgn()
    data = client.findSnipedTokens(size)
    # Expected structure: {"signals": [ ... ], "next": ... }
    if "signals" in data:
        df = pd.DataFrame(data["signals"])
    else:
        df = pd.DataFrame(data)
    return df

def df_gas_fee() -> pd.DataFrame:
    """
    Get gas fee information as a DataFrame.
    
    Returns:
        DataFrame containing gas fee data
    """
    client = gmgn()
    data = client.getGasFee()
    # data is a dictionary of key/value pairs (one row)
    df = pd.DataFrame([data])
    return df

def df_token_usd_price(contractAddress: str) -> pd.DataFrame:
    """
    Get token USD price as a DataFrame.
    
    Args:
        contractAddress: Token contract address
        
    Returns:
        DataFrame containing token price data
    """
    client = gmgn()
    data = client.getTokenUsdPrice(contractAddress)
    # data is a dictionary (one row)
    df = pd.DataFrame([data])
    return df

def df_top_buyers(contractAddress: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get top buyers information as DataFrames.
    
    Args:
        contractAddress: Token contract address
        
    Returns:
        Tuple of (summary DataFrame, holder info DataFrame)
    """
    client = gmgn()
    data = client.getTopBuyers(contractAddress)
    # Expected structure: {"code":..., "msg":..., "data": {"holders": {...}}}
    holders = data.get("data", {}).get("holders", {})
    # Build one DataFrame for the summary info (all keys except holderInfo)
    summary = {k: v for k, v in holders.items() if k != "holderInfo"}
    df_summary = pd.DataFrame([summary])
    # And one for holderInfo (if available)
    holder_info = holders.get("holderInfo", [])
    df_holder_info = pd.DataFrame(holder_info)
    
    # Update database with top buyers data
    try:
        if not df_summary.empty:
            hold_pct = safe_float(summary.get("holdPct", 0.0))
            bought_more_pct = safe_float(summary.get("boughtMorePct", 0.0))
            sold_pct = safe_float(summary.get("soldPct", 0.0))
            total_holders = len(holder_info)
            
            db.update_top_buyers(
                token_address=contractAddress,
                hold_pct=hold_pct,
                bought_more_pct=bought_more_pct,
                sold_pct=sold_pct,
                total_holders=total_holders
            )
    except Exception as e:
        logger.error(f"Error updating top buyers in database: {e}")
    
    return df_summary, df_holder_info

def df_token_security_info_launchpad(contractAddress: str) -> pd.DataFrame:
    """
    Get token security info as a DataFrame.
    
    Args:
        contractAddress: Token contract address
        
    Returns:
        DataFrame containing token security data
    """
    client = gmgn()
    data = client.getTokenSecurityInfoLaunchpad(contractAddress)
    # Expected structure: {"code":0, "message": "success", "data": { "security": {...} } }
    security_data = data.get("data", {}).get("security", {})
    df = pd.DataFrame([security_data])
    return df

def df_wallet_activity(walletAddress: str, limit: int = 40, cost: int = 10) -> pd.DataFrame:
    """
    Get wallet activity as a DataFrame.
    
    Args:
        walletAddress: Wallet address
        limit: Maximum number of activities to return
        cost: Cost parameter
        
    Returns:
        DataFrame containing wallet activity data
    """
    client = gmgn()
    data = client.getWalletActivity(walletAddress, limit, cost)
    # Expected structure: {"code": ..., "data": {"activities": [ ... ], "next": ...} }
    activities = data.get("data", {}).get("activities", [])
    df = pd.DataFrame(activities)
    return df

def df_wallet_holdings(walletAddress: str) -> pd.DataFrame:
    """
    Get wallet holdings as a DataFrame.
    
    Args:
        walletAddress: Wallet address
        
    Returns:
        DataFrame containing wallet holdings data
    """
    client = gmgn()
    data = client.getWalletHoldings(walletAddress)
    # Expected structure: {"code":..., "data": {"holdings": [ ... ], "next": ...}}
    holdings = data.get("data", {}).get("holdings", [])
    df = pd.DataFrame(holdings)
    return df

def df_token_rug_info(contractAddress: str) -> pd.DataFrame:
    """
    Get token rug information as a DataFrame.
    
    Args:
        contractAddress: Token contract address
        
    Returns:
        DataFrame containing token rug data
    """
    client = gmgn()
    data = client.getTokenRugInfo(contractAddress)
    # Expected structure: {"code":..., "message":..., "data": { ... }}
    rug_info = data.get("data", {})
    # Optionally, you can also convert the nested "rugged_tokens" into a separate df.
    df = pd.json_normalize(rug_info)
    return df

def df_pump_ranks_1m() -> pd.DataFrame:
    """
    Get 1-minute pump ranks as a DataFrame.
    
    Returns:
        DataFrame containing pump ranks data
    """
    client = gmgn()
    data = client.getPumpRanks1m()
    # Expected structure: {"code": ..., "msg": "success", "data": {"pumps": [ ... ]} }
    pumps = data.get("data", {}).get("pumps", [])
    df = pd.DataFrame(pumps)
    return df

def df_pump_ranks_1h() -> pd.DataFrame:
    """
    Get 1-hour pump ranks as a DataFrame.
    
    Returns:
        DataFrame containing pump ranks data
    """
    client = gmgn()
    data = client.getPumpRanks1h()
    # Expected structure: {"code": ..., "msg": "success", "data": {"pumps": [ ... ]} }
    pumps = data.get("data", {}).get("pumps", [])
    df = pd.DataFrame(pumps)
    return df

def df_swap_ranks5m() -> pd.DataFrame:
    """
    Get 5-minute swap ranks as a DataFrame.
    
    Returns:
        DataFrame containing swap ranks data
    """
    client = gmgn()
    data = client.getSwapRanks5m()
    # Expected structure: {"code": ..., "msg": "success", "data": {"rank": [ ... ]} }
    rank = data.get("data", {}).get("rank", [])
    df = pd.DataFrame(rank)
    return df

def df_swap_ranks30m() -> pd.DataFrame:
    """
    Get 30-minute swap ranks as a DataFrame.
    
    Returns:
        DataFrame containing swap ranks data
    """
    client = gmgn()
    data = client.getSwapRanks30m()
    # Expected structure: {"code": ..., "msg": "success", "data": {"rank": [ ... ]} }
    rank = data.get("data", {}).get("rank", [])
    df = pd.DataFrame(rank)
    return df

def df_token_stats(contractAddress: str) -> pd.DataFrame:
    """
    Get token statistics as a DataFrame.
    
    Args:
        contractAddress: Token contract address
        
    Returns:
        DataFrame containing token statistics
    """
    client = gmgn()
    data = client.getTokenStats(contractAddress)
    # Expected structure: {"code": ..., "message": "success", "data": { ... }}
    stats = data.get("data", {})
    df = pd.DataFrame([stats])
    return df

def df_token_kline(contractAddress: str, resolution: str, from_ts: int, to_ts: int) -> pd.DataFrame:
    """
    Get token kline data as a DataFrame.
    
    Args:
        contractAddress: Token contract address
        resolution: Chart resolution
        from_ts: Start timestamp
        to_ts: End timestamp
        
    Returns:
        DataFrame containing token kline data
    """
    client = gmgn()
    data = client.getTokenKline(contractAddress, resolution, from_ts, to_ts)
    # Expected structure: {"code": ..., "message": "success", "data": {"list": [ ... ]} }
    klines = data.get("data", {}).get("list", [])
    df = pd.DataFrame(klines)
    return df

def get_native_sol_balance(wallet_address: str, rpc_url: str) -> float:
    """
    Get native SOL balance for a wallet.
    
    Args:
        wallet_address: Wallet address
        rpc_url: RPC endpoint URL
        
    Returns:
        SOL balance
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [wallet_address]
    }
    response = requests.post(rpc_url, json=payload, timeout=REQUEST_TIMEOUT)
    data = response.json()
    if "result" not in data:
        raise Exception("Failed to get native balance")
    lamports = data["result"]["value"]
    sol_balance = lamports / 1e9  # 1 SOL = 1e9 lamports
    return sol_balance

# New helper functions to address circular dependencies

def add_trade(token_address: str, trade_type: str, quantity: float, price: float,
              pnl_percent: Optional[float] = None, pnl_usd: Optional[float] = None,
              trade_time: Optional[str] = None, time_held: Optional[str] = None) -> bool:
    """
    Add a trade record to the database.
    
    This function serves as a bridge between the functions module and database module,
    ensuring proper data formatting and consistency across the system.
    
    Args:
        token_address: The token's blockchain address
        trade_type: Type of trade (buy, sell, test_buy)
        quantity: Amount of token traded
        price: Price per token at time of trade
        pnl_percent: Profit/loss percentage (for sells)
        pnl_usd: Profit/loss in USD (for sells)
        trade_time: ISO-formatted timestamp of trade (defaults to now)
        time_held: How long token was held before selling (for sells)
        
    Returns:
        bool: Whether the operation succeeded
    """
    return db.add_trade(
        token_address=token_address,
        trade_type=trade_type,
        quantity=quantity,
        price=price,
        pnl_percent=pnl_percent,
        pnl_usd=pnl_usd,
        trade_time=trade_time or datetime.now().isoformat(),
        time_held=time_held
    )
    
def update_token_price(token_address: str, price: float) -> bool:
    """
    Update a token's price in the database.
    
    Args:
        token_address: The token's blockchain address
        price: Current price in USD
        
    Returns:
        bool: Whether the operation succeeded
    """
    return db.update_token_price(token_address, price)

def rugcheck_score(address: str, retries: int = 3, delay: int = 1) -> Optional[int]:
    """
    Query the RugCheck API to get a risk score for a token.
    
    Args:
        address: The token contract address to check
        retries: Maximum number of retry attempts (default: 3)
        delay: Seconds to wait between retries (default: 1)
        
    Returns:
        Optional[int]: The rug risk score (0-100) or None if unavailable
    """
    url = f"https://api.rugcheck.xyz/v1/tokens/{address}/report/summary"
    
    # Try multiple attempts as specified by retries parameter
    for attempt in range(1, retries + 1):
        try:
            # Make the API request with timeout
            resp = requests.get(url, timeout=10)
            
            # Try to parse the JSON response
            try:
                json_data = resp.json()
            except Exception:
                json_data = {}
                
            # Handle rate limiting
            if json_data.get("message") == "Too many requests":
                cprint(f"RugCheck: Too many requests for {address}, sleeping for 30 seconds.", "yellow")
                time.sleep(10)
                continue
                
            # Check for successful HTTP response
            if resp.status_code != 200:
                logger.warning(f"RugCheck attempt {attempt}: HTTP {resp.status_code} for {address}")
            else:
                # Extract the score from the response
                score = json_data.get("score")
                if score is not None:
                    return safe_int(score)
                    
        except Exception as e:
            logger.error(f"RugCheck exception for {address}: {e}")
            
        # Sleep before the next retry if not the last attempt
        if attempt < retries:
            time.sleep(delay)
            
    # All retries failed
    return None

def get_top_buyers_stats(token_address: str) -> Tuple[float, float, float, int]:
    """
    Fetch top buyers statistics directly from the GMGN client API.
    
    This function retrieves current holder behavior metrics for a token,
    providing insight into investor sentiment and potential price direction.
    
    Args:
        token_address: The blockchain address of the token to analyze
        
    Returns:
        Tuple containing:
            - hold_pct: Percentage of holders maintaining their position
            - sold_pct: Percentage of holders who sold
            - bought_more_pct: Percentage of holders who increased their position
            - total_holders: Total number of holders analyzed
    """
    try:
        # Get data directly from GMGN
        summary_df, holder_info_df = df_top_buyers(token_address)
        
        # Calculate metrics from the returned DataFrame
        if not holder_info_df.empty:
            # Count total holders analyzed
            total_holders = len(holder_info_df)
            
            # Count holders in each category based on 'status' column
            hold_count = (holder_info_df['status'].str.lower() == "hold").sum()
            bought_more_count = (holder_info_df['status'].str.lower() == "bought_more").sum()
            
            # Calculate percentages (normalize to 0-100 scale)
            hold_pct = (hold_count / total_holders) * 100
            bought_more_pct = (bought_more_count / total_holders) * 100
            
            # Calculate sold percentage (remainder)
            sold_count = total_holders - (hold_count + bought_more_count)
            sold_pct = (sold_count / total_holders) * 100
            
            # Log the results for debugging
            logger.info(f"Top buyers for {token_address}: Hold={hold_pct:.1f}%, "
                      f"Bought More={bought_more_pct:.1f}%, Sold={sold_pct:.1f}%, "
                      f"Total={total_holders}")
                      
            return hold_pct, sold_pct, bought_more_pct, total_holders
        else:
            # Return default values if no holder info available
            logger.warning(f"No holder info available for {token_address}")
            return 0.0, 0.0, 0.0, 0
            
    except Exception as e:
        # Log the error but don't crash
        logger.error(f"Error getting top buyers for {token_address}: {e}")
        return 0.0, 0.0, 0.0, 0

def update_master_all(token_info: Dict[str, Any]) -> None:
    """
    Save or update token information in the master tracking database.
    
    This delegates to the database module's update_master_all_db function.
    
    Args:
        token_info: Complete token information from the API
    """
    db.update_master_all_db(token_info)

def get_last_seen_from_master_all(token_address: str) -> Optional[str]:
    """
    Retrieve the last seen timestamp for a token from the database.
    
    Args:
        token_address: The blockchain address of the token
        
    Returns:
        Optional[str]: Timestamp string of when token was last seen or None if not found
    """
    return db.get_last_seen_from_master_all_db(token_address)

def save_test_results_to_db(token_info: Dict[str, Any], test_results: Dict[str, bool], test_values: Dict[str, Any]) -> None:
    """
    Save token test results to the database for historical tracking.
    
    Args:
        token_info: Token information from the API
        test_results: Dictionary of test name -> boolean result
        test_values: Dictionary of test name -> actual value
    """
    db.save_test_results_to_db(token_info, test_results, test_values)

def save_full_token_info(token_info: Dict[str, Any]) -> None:
    """
    Save complete token information to the database for reference.
    
    Args:
        token_info: Complete token information dictionary from API
    """
    db.save_full_token_info_to_db(token_info)

def save_trailing_data(token_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Save trailing data for multiple tokens to the database.
    
    Args:
        token_data: Dictionary of token_address -> token data
    """
    db.save_trailing_data_to_db(token_data)

def load_seen_addresses() -> Set[str]:
    """
    Load all seen addresses from the database.
    
    Returns:
        set: A set of all seen addresses
    """
    return db.load_seen_addresses()

def save_new_addresses(addresses: Union[List[str], pd.DataFrame]) -> bool:
    """
    Save new addresses to the database.
    
    Args:
        addresses: List of addresses or DataFrame with address column
        
    Returns:
        bool: True if successful, False otherwise
    """
    return db.save_new_addresses(addresses)

def add_do_not_trade(address: str, reason: Optional[str] = None) -> bool:
    """
    Add a token to the do not trade list.
    
    Args:
        address: The token address
        reason: Reason for adding to the list
        
    Returns:
        bool: True if successful, False otherwise
    """
    return db.add_do_not_trade(address, reason)

def remove_token_from_low_value_list(token_address: str) -> bool:
    """
    Remove a token from the low value tokens list.
    
    Args:
        token_address: The token address
        
    Returns:
        bool: True if successful, False otherwise
    """
    return db.remove_token_from_low_value_list(token_address)

def load_bought_tokens() -> Dict[str, datetime]:
    """
    Load all bought tokens from the database.
    
    Returns:
        dict: Dictionary of token_address -> datetime of purchase
    """
    return db.load_bought_tokens()

def load_recent_trades(recent_only: bool = False) -> List[Dict[str, Any]]:
    """
    Load recent trades from the database.
    
    Args:
        recent_only: If True, only return trades from the last 24 hours
        
    Returns:
        list: List of trade dictionaries
    """
    return db.load_recent_trades(recent_only)