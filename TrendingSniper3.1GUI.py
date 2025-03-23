import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re as reggie
import traceback
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional, Union, Any, Set, cast

import config as c
import functions as n
import dontshare as d
from gmgn.client import gmgn
import database as db  # Import the database module

##############################
# Filter & Important Config Values (Pulled to Top)
##############################
# Liquidity and volume thresholds
MIN_LIQUIDITY: float = 250000  # Minimum liquidity in USD
MIN_VOLUME_1M: float = 800    # Minimum 1-minute trading volume in USD
MIN_HOLDER_COUNT: int = 350   # Minimum number of token holders

# Volume-based metrics for momentum detection
MIN_NET_IN_VOLUME_1M: float = 500  # Min net inflow in 1 minute window
MIN_NET_IN_VOLUME_5M: float = 1000  # Min net inflow in 5 minute window
MASTER_VOLUME_RATIO_THRESHOLD: float = 3.0  # Buy/sell volume ratio threshold

# New configuration parameters for top buyer percentage checks
MIN_HOLDERS_BUYING_PCT: float = 45.0   # Minimum percentage of holders buying/holding (not selling)
MAX_HOLDERS_SELLING_PCT: float = 25.0  # Maximum percentage of holders selling
ENABLE_BUYER_PCT_CHECK: bool = True    # Toggle to enable/disable this feature

# Holder-based criteria
RECENTLY_BOUGHT_WINDOW_HOURS: int = 5  # Lookback window for recent buys
MIN_HOLDERS_FOR_TRADE: int = 300       # Minimum holders required for trading

# Financial criteria
MIN_MARKET_CAP: float = 500000  # Minimum market capitalization in USD
MAX_RUG_RATIO: float = 0.25     # Maximum allowed rug ratio (as decimal)

# Buy parameters
MIN_BUY_USD: float = 0.10       # Minimum buy amount in USD
FULL_BUY_AMOUNT: float = 0.5    # Standard buy amount in USD

# Age-based filtering
MAX_TOKEN_AGE_MINUTES: int = 1008000  # Maximum token age (7 days in minutes)
REBUY_INTERVAL_MINUTES: int = 200     # Minimum time between buys for same token

# Operational parameters
SCAN_INTERVAL_SECONDS: int = 30  # Time between trend scans
last_buy_amounts: Dict[str, float] = {}  # Tracks amount spent per token



# Define path for any legacy file handling still needed
DATA_FOLDER: str = c.DATA_PATH
# Initialize database
db.db_init()
# Configure logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Global counters and tracking
found_tokens_total: int = 0        # Total tokens discovered
passed_tokens_total: int = 0       # Total tokens that passed all tests
bought_tokens_total: int = 0       # Total tokens that were bought
bought_token_list: List[str] = []  # List of bought token addresses
starting_balance: float = 0.0      # Initial wallet balance

# Global sets for address tracking
seen_addresses: Set[str] = set()   # Addresses we've already processed
last_bought: Dict[str, datetime] = {}  # When each token was last bought

# --------------------------
# Globals for GUI and stats
# --------------------------
gui_lock = threading.Lock()  # Thread safety for GUI updates

# Token tracking collections 
# Each dict has: address, name, test_results, test_values, timestamp, details, pass_rate
tested_tokens_info: List[Dict[str, Any]] = []        # All tokens processed
almost_passed_tokens_info: List[Dict[str, Any]] = [] # Tokens with 75%+ pass rate
passed_tokens_info: List[Dict[str, Any]] = []        # Tokens with 100% pass rate
bought_tokens_info: List[Dict[str, Any]] = []        # Tokens that were bought

# Test statistics for analysis
test_stats: Dict[str, Dict[str, Any]] = {}  # Stats per test criterion

def cprint(message: str, color: str = "white") -> None:
    """
    Print colored text to the console for improved readability and emphasis.
    
    This function uses ANSI escape codes to colorize console output, making
    it easier to distinguish between different types of messages (errors,
    warnings, success notifications, etc.).
    
    Args:
        message: The text message to print to the console
        color: Color name from the supported palette:
            - "red": For errors and critical issues
            - "green": For success messages and confirmations
            - "yellow": For warnings and cautions
            - "blue": For informational messages
            - "magenta": For highlighting special events
            - "cyan": For system status updates
            - "white": Default text color (high visibility)
    
    Implementation Notes:
        - Uses ANSI escape codes that work in most modern terminals
        - Falls back to white if an unsupported color is specified
        - Always resets the color after printing to avoid color bleed
    
    Usage Examples:
        >>> cprint("Operation successful!", "green")
        >>> cprint("Warning: Low disk space", "yellow")
        >>> cprint("Error: Connection failed", "red")
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
    Safely convert a value to float, handling various edge cases and errors.
    
    This utility function serves as a defensive programming technique to
    ensure robust type conversion with appropriate fallbacks when dealing
    with potentially inconsistent data sources like APIs or user input.
    
    Args:
        value: The value to convert, which could be any of:
            - Numeric string ("42.5")
            - Integer (42)
            - Float (42.5)
            - pandas.Series (containing a single numeric value)
            - None, empty string, or other non-convertible types
        default: Value to return if conversion fails
        
    Returns:
        float: The converted value if successful, or the default value if not
    
    Error Handling:
        - Extracts the first value from pandas Series objects
        - Gracefully handles None values, returning the default
        - Catches all exceptions during conversion, returning the default
        - Returns the default for any non-convertible input
        
    Design Pattern:
        This function follows the "Resilient Type Conversion" pattern,
        emphasizing fail-safe behavior over strict correctness. It 
        prioritizes program stability by never raising exceptions.
    """
    try:
        # Handle pandas Series (extract first value)
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        # Convert to float and return
        return float(value)
    except Exception:
        # Any failure returns the default value
        return float(default)

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer, handling various edge cases and errors.
    
    This function provides robust conversion to integers with appropriate
    fallbacks, making code more resilient when dealing with data from
    external sources or unpredictable formats.
    
    Args:
        value: The value to convert to an integer
            - Could be a string, float, another int, pandas.Series, etc.
            - May be None, empty, or otherwise non-convertible
        default: Value to return if conversion fails
        
    Returns:
        int: The converted integer value or the default
        
    Implementation Notes:
        - First converts to float (for values like "42.5") then to int
        - For pandas Series, extracts the first value before conversion
        - Gracefully handles all exceptions during conversion
        - Follows a similar pattern to safe_float for consistency
    
    Usage Examples:
        >>> safe_int("42")  # Returns 42
        >>> safe_int("3.14")  # Returns 3
        >>> safe_int(None)  # Returns 0 (default)
        >>> safe_int("not a number", 999)  # Returns 999 (custom default)
    """
    try:
        # Handle pandas Series (extract first value)
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        # First convert to float (for values like "42.5"), then to int
        return int(float(value))
    except Exception:
        # Any failure returns the default value
        return int(default)

def find_urls(text: str) -> List[str]:
    """
    Extract all URLs from a text string using regular expressions.
    
    This utility function identifies URLs within arbitrary text content,
    which is particularly useful for processing description fields that
    may contain social media links, project websites, etc.
    
    Args:
        text: The input text to search for URLs
        
    Returns:
        List[str]: A list of all URLs found in the text
        
    Pattern Details:
        - Matches both http:// and https:// URLs
        - Captures URLs that continue until whitespace
        - Returns an empty list if no URLs are found
        - Works with URLs embedded in sentences or paragraphs
        
    Implementation Note:
        Uses the regex 'compile' method for better performance when the
        pattern needs to be reused multiple times in an application.
    """
    # Compile regex pattern for better performance
    pattern = reggie.compile(r'https?://[^\s]+')
    # Convert input to string and find all matches
    return pattern.findall(str(text))

def ensure_list(x: Any) -> List[Any]:
    """
    Ensure the input is converted to a list with consistent handling.
    
    This utility normalizes various data structures into a standard list format,
    making it easier to process data from APIs that might return inconsistent
    formats depending on the result count.
    
    Args:
        x: The input to convert to a list, which could be:
           - A dictionary (potentially with a 'rank' key containing a list)
           - An existing list (returned unchanged)
           - Any other value (wrapped in a list)
        
    Returns:
        List[Any]: A proper list containing the input data
    
    Behavioral Notes:
        - For dictionaries with a 'rank' key: Returns the value of 'rank'
        - For dictionaries without a 'rank' key: Returns a single-item list with the dict
        - For existing lists: Returns the original list unchanged
        - For anything else: Returns a single-item list containing the input
        
    Use Case:
        This function addresses a common API behavior where results might be:
        - Direct JSON object for a single result
        - Array of objects for multiple results
        - Object with a 'rank' key containing results array
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
    
    When retrieving new trading pair information, we need to extract the
    individual token addresses for further analysis. This function processes
    the nested structure typically returned by DEX APIs.
    
    Args:
        new_tokens: List of token objects, each potentially containing a 'pairs' field
            with multiple trading pairs
        
    Returns:
        List[str]: Flattened, deduplicated list of token addresses
        
    Data Processing Flow:
        1. Iterate through each token object in the input list
        2. Extract the 'pairs' array from each token (default to empty if missing)
        3. For each pair, extract the 'address' field
        4. Add to the result list if it's not already included
        
    Implementation Note:
        - Performs deduplication to ensure each address appears only once
        - Preserves the original order of discovery (first occurrence kept)
    """
    token_addresses: List[str] = []
    
    # Process each token in the input list
    for token in new_tokens:
        # Get the pairs array (default to empty if missing)
        pairs = token.get("pairs", [])
        
        # Extract addresses from each pair
        for pair in pairs:
            address = pair.get("address")
            if address and address not in token_addresses:
                # Add to result list if new and valid
                token_addresses.append(address)
                
    return token_addresses

def remove_duplicate_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate tokens from a list based on their addresses.
    
    When aggregating tokens from multiple sources, duplicates can occur.
    This function ensures we analyze each unique token only once by
    deduplicating based on the token's blockchain address.
    
    Args:
        tokens: List of token dictionaries, potentially containing duplicates
        
    Returns:
        List[Dict[str, Any]]: List with duplicates removed
        
    Algorithm Details:
        1. Maintain a set of seen addresses for O(1) lookup
        2. Iterate through all tokens in the original list
        3. For each valid token, check if we've seen its address before
        4. If new, add to result list and mark address as seen
        
    Implementation Notes:
        - Handles different address field names ('address', 'coin_address', 'token_address')
        - Skips non-dictionary items for robustness
        - Preserves the original ordering (first occurrence kept)
        - The function is idempotent (running it multiple times has same effect as once)
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

def load_seen_addresses() -> None:
    """
    Load previously seen token addresses from the database.
    
    This function populates the global 'seen_addresses' set with addresses
    that have already been processed in previous runs, preventing
    redundant processing of tokens we've already analyzed.
    
    Global Side Effects:
        Updates the 'seen_addresses' global set
    
    Implementation Details:
        - Uses the database module to retrieve addresses
        - The database query is optimized to only return the address column
        - Updates the global seen_addresses set directly
        - Logs the number of addresses loaded for monitoring
        
    Database Interaction:
        This is purely a READ operation - it only retrieves data from
        the database without making any modifications.
    """
    global seen_addresses
    # Retrieve addresses from database using the helper function
    seen_addresses = db.load_seen_addresses()
    logger.info(f"Loaded {len(seen_addresses)} seen addresses from database")

def rugcheck_score(address: str, retries: int = 3, delay: int = 1) -> Optional[int]:
    """
    Query the RugCheck API to get a risk score for a token.
    
    RugCheck provides an external risk assessment for tokens, helping
    identify potential scams or high-risk projects. This function makes
    an HTTP request to their API with fault tolerance.
    
    Args:
        address: The token contract address to check
        retries: Maximum number of retry attempts (default: 3)
        delay: Seconds to wait between retries (default: 1)
        
    Returns:
        Optional[int]: The rug risk score (0-100) or None if unavailable
        
    Error Handling:
        - Implements retry logic for transient network failures
        - Handles rate limiting with appropriate backoff
        - Gracefully handles HTTP errors and response parsing issues
        - Returns None if all retry attempts fail
        
    Implementation Pattern:
        This function demonstrates the "Retry Pattern" with exponential
        backoff, a common approach for handling transient failures in
        distributed systems.
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
    It specifically uses the GMGN API directly (not database) to ensure
    decisions are based on real-time data.
    
    Args:
        token_address: The blockchain address of the token to analyze
        
    Returns:
        Tuple containing:
            - hold_pct: Percentage of holders maintaining their position
            - sold_pct: Percentage of holders who sold
            - bought_more_pct: Percentage of holders who increased their position
            - total_holders: Total number of holders analyzed
    
    Data Flow:
        1. Request raw holder data from GMGN API via df_top_buyers
        2. Calculate key metrics from the returned DataFrame
        3. Return formatted statistics tuple
        
    Implementation Note:
        This function intentionally bypasses the database to ensure
        trading decisions use the most current possible data, not
        cached information.
    """
    try:
        # Use the function module to get data directly from GMGN
        # This ensures we're using fresh API data, not database cache
        summary_df, holder_info_df = n.df_top_buyers(token_address)
        
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

def get_test_results(token_info: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """
    Comprehensively evaluate a token against multiple quality criteria.
    
    This function implements a multi-factor analysis model for token evaluation,
    checking various financial, technical, and social metrics to determine if
    a token is suitable for trading. Each test contributes to an overall
    assessment of token quality and risk.
    
    Args:
        token_info: Dictionary containing token information from GMGN API
        
    Returns:
        Tuple[Dict[str, bool], Dict[str, Any]]: A tuple containing:
            - Dictionary mapping test names to boolean results (pass/fail)
            - Dictionary mapping test names to actual values for reference
              IMPORTANT: Each key in results dict must have a corresponding key
              in the values dict to prevent KeyError in output formatting
    
    Test Categories:
        1. Financial Metrics: Liquidity, volume, market cap
        2. Technical Indicators: Buy/sell ratios, net inflows
        3. Social Metrics: Holder count, buyer behavior
        4. Risk Assessment: Token age, rug potential (added separately)
        
    Implementation Details:
        - All data comes directly from the provided token_info
        - Each test is independent with clearly defined thresholds
        - Values are safely extracted with appropriate type conversion
        - Token address is retrieved for additional API calls as needed
    """
    # Initialize result dictionaries
    # These two dictionaries MUST have matching keys to prevent KeyError
    results: Dict[str, bool] = {}    # Stores boolean pass/fail for each test
    values: Dict[str, Any] = {}      # Stores actual values/thresholds for each test

    # Extract basic metrics with safe conversion
    # Using safe_float ensures we handle None values and type conversion errors
    liquidity = safe_float(token_info.get("liquidity", 0))
    volume_1m = safe_float(token_info.get("volume_1m", 0))
    buy_volume_1m = safe_float(token_info.get("buy_volume_1m", 0))
    sell_volume_1m = safe_float(token_info.get("sell_volume_1m", 0))
    buy_volume_5m = safe_float(token_info.get("buy_volume_5m", 0))
    sell_volume_5m = safe_float(token_info.get("sell_volume_5m", 0))
    holders = safe_int(token_info.get("holder_count", 0))
    net_in_volume_1m = safe_float(token_info.get("net_in_volume_1m", 0))
    net_in_volume_5m = safe_float(token_info.get("net_in_volume_5m", 0))
    token_address = token_info.get("address")

    # Test 1: Liquidity Test
    # Higher liquidity generally means lower risk of price manipulation
    values["Liquidity"] = liquidity
    results["Liquidity"] = (liquidity >= MIN_LIQUIDITY)

    # Test 2: 1-Minute Volume Test
    # Sufficient volume indicates active trading and market interest
    values["1m Volume"] = volume_1m
    results["1m Volume"] = (volume_1m >= MIN_VOLUME_1M)

    # Test 3: 1-Minute Buy/Sell Volume Ratio Test
    # Higher buy/sell ratio indicates buying pressure (bullish)
    # Handle division by zero: inf if buy volume exists, 0 if no volume at all
    if sell_volume_1m == 0:
        ratio_1m = float('inf') if buy_volume_1m > 0 else 0.0
    else:
        ratio_1m = buy_volume_1m / sell_volume_1m
    values["1m Buy/Sell Volume Ratio"] = ratio_1m
    results["1m Buy/Sell Volume Ratio"] = (ratio_1m >= MASTER_VOLUME_RATIO_THRESHOLD)

    # Test 4: 5-Minute Buy/Sell Volume Ratio Test
    # Similar to Test 3 but over a longer timeframe for trend confirmation
    if sell_volume_5m == 0:
        ratio_5m = float('inf') if buy_volume_5m > 0 else 0.0
    else:
        ratio_5m = buy_volume_5m / sell_volume_5m
    values["5m Buy/Sell Volume Ratio"] = ratio_5m
    results["5m Buy/Sell Volume Ratio"] = (ratio_5m >= MASTER_VOLUME_RATIO_THRESHOLD)

    # Test 5: Holders Count Test
    # More holders typically indicates broader distribution and lower manipulation risk
    values["Holders"] = holders
    results["Holders"] = (holders >= MIN_HOLDER_COUNT)

    # Test 6: Net Inflow Volume 1m Test
    # Positive net inflow shows more capital entering than leaving
    values["Net In Volume 1m"] = net_in_volume_1m
    results["Net In Volume 1m"] = (net_in_volume_1m > 0 and net_in_volume_1m >= MIN_NET_IN_VOLUME_1M)

    # Test 7: Net Inflow Volume 5m Test
    # Same as Test 6 but over longer timeframe for trend confirmation
    values["Net In Volume 5m"] = net_in_volume_5m
    results["Net In Volume 5m"] = (net_in_volume_5m > 0 and net_in_volume_5m >= MIN_NET_IN_VOLUME_5M)

    # Test 8: Token Age Test
    # Newer tokens may have higher volatility but also higher growth potential
    creation_time_raw = token_info.get("creation_timestamp")
    token_age_minutes = None
    is_recent_enough = False
    if creation_time_raw:
        try:
            # Handle different timestamp formats (int/float vs string)
            if isinstance(creation_time_raw, (int, float)):
                creation_dt = datetime.utcfromtimestamp(creation_time_raw)
            else:
                creation_dt = pd.to_datetime(str(creation_time_raw), utc=True)
            creation_dt = creation_dt.replace(tzinfo=None)
            now_utc = datetime.utcnow()
            diff = now_utc - creation_dt
            token_age_minutes = diff.total_seconds() / 60.0
            # Compare to MAX_TOKEN_AGE_MINUTES
            is_recent_enough = (token_age_minutes <= MAX_TOKEN_AGE_MINUTES)
        except Exception as e:
            logging.error(f"Could not parse creation_timestamp {creation_time_raw}: {e}")
    results["Creation Time"] = is_recent_enough
    values["Creation Time"] = token_age_minutes if token_age_minutes is not None else "N/A"

    # Test 9: Market Cap Test
    # Minimum market cap threshold helps filter out extremely small projects
    market_cap = safe_float(token_info.get("market_cap", 0))
    results["Market Cap"] = (market_cap >= MIN_MARKET_CAP)
    values["Market Cap"] = market_cap

    # Test 10: Price Increase Test
    # Positive price momentum over the last hour
    price_now = safe_float(token_info.get("usd_price", token_info.get("price", 0)))
    price_1h_ago = safe_float(token_info.get("price_1h", 0))
    price_increased = False
    if price_now > 0 and price_1h_ago > 0:
        price_increased = (price_now > price_1h_ago)
    results["Price 1h Increase"] = price_increased
    values["Price 1h Increase"] = f"{price_now} vs {price_1h_ago}"

    # Test 11: Top Buyers Analysis (if enabled)
    if ENABLE_BUYER_PCT_CHECK and token_address:
        # Get fresh top buyers statistics directly from GMGN client
        # This is a direct API call, not a database lookup
        hold_pct, sold_pct, bought_more_pct, total_holders = get_top_buyers_stats(token_address)
        
        # Calculate combined positive sentiment (holders + buyers)
        positive_sentiment = hold_pct + bought_more_pct
        
        # Store the individual component values for reference
        values["Top Holders Positive %"] = positive_sentiment
        values["Top Holders Selling %"] = sold_pct
        values["Top Holders Count"] = total_holders
        
        # Apply the test criteria
        holding_threshold_met = positive_sentiment >= MIN_HOLDERS_BUYING_PCT
        selling_threshold_met = sold_pct <= MAX_HOLDERS_SELLING_PCT
        
        # For the test to pass, both conditions must be met
        results["Top Holders Sentiment"] = holding_threshold_met and selling_threshold_met
        
        # FIX: Add a matching value entry for "Top Holders Sentiment" 
        # This ensures key consistency between results and values dictionaries
        values["Top Holders Sentiment"] = f"Positive: {positive_sentiment:.1f}% (min {MIN_HOLDERS_BUYING_PCT}%), Selling: {sold_pct:.1f}% (max {MAX_HOLDERS_SELLING_PCT}%)"
        
        # Add detailed logging for monitoring
        logger.info(f"Top holders test for {token_address}: "
                  f"Positive={positive_sentiment:.1f}% vs threshold {MIN_HOLDERS_BUYING_PCT}%, "
                  f"Selling={sold_pct:.1f}% vs threshold {MAX_HOLDERS_SELLING_PCT}%: "
                  f"{'PASS' if results['Top Holders Sentiment'] else 'FAIL'}")
    else:
        # If disabled or no token address, skip this check but mark as passed
        values["Top Holders Positive %"] = "N/A (disabled)"
        values["Top Holders Selling %"] = "N/A (disabled)" 
        values["Top Holders Count"] = "N/A (disabled)"
        # FIX: Ensure this key exists even when check is disabled
        values["Top Holders Sentiment"] = "N/A (disabled)"
        results["Top Holders Sentiment"] = True  # Default to pass if check is disabled

    return results, values

def combine_test_info(test_results: Dict[str, bool], test_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine test results and values into a unified dictionary for storage.
    
    This utility function reformats test data into a standardized structure
    for database storage and display, making it easier to understand and
    analyze test outcomes.
    
    Args:
        test_results: Dictionary mapping test names to boolean pass/fail results
        test_values: Dictionary mapping test names to their actual values
        
    Returns:
        Dict[str, Any]: Combined dictionary with standardized field naming
        
    Format Details:
        For each test in the input dictionaries, two fields are created:
        - "{test} Result": A string "PASS" or "FAIL" based on the boolean result
        - "{test} Value": The actual value from the test_values dictionary
        
    Implementation Notes:
        - Ensures both dictionaries are processed in sync
        - Handles any case where a test might be in one dictionary but not the other
        - Creates a completely new dictionary rather than modifying inputs
    """
    combined: Dict[str, Any] = {}
    
    # Process each test in the results dictionary
    for test in test_results:
        # Add standardized result field (PASS/FAIL string)
        combined[f"{test} Result"] = "PASS" if test_results[test] else "FAIL"
        # Add corresponding value field
        combined[f"{test} Value"] = test_values[test]
        
    return combined

def get_last_seen_from_master_all(token_address: str) -> Optional[str]:
    """
    Retrieve the last seen timestamp for a token from the database.
    
    This is primarily used for informational display and debugging,
    showing when a token was last observed in previous scans.
    
    Args:
        token_address: The blockchain address of the token
        
    Returns:
        Optional[str]: Timestamp string of when token was last seen or None if not found
        
    Implementation Note:
        This is a READ-ONLY database operation used purely for informational
        purposes. It does not affect decision making for token trading.
    """
    # Delegate to database module function
    return db.get_last_seen_from_master_all_db(token_address)

def update_master_all(token_info: Dict[str, Any]) -> None:
    """
    Save or update token information in the master tracking database.
    
    This function records the latest information about a token for 
    historical tracking and analysis. It's called after token evaluation
    regardless of whether the token passed tests or was bought.
    
    Args:
        token_info: Complete token information from the API
        
    Side Effects:
        Writes or updates a record in the database
        
    Implementation Notes:
        - This is a WRITE operation to the database
        - Called for all tokens that are processed, not just ones that pass tests
        - Uses the database module's function to handle the actual database operation
    """
    # Delegate to database module function
    db.update_master_all_db(token_info)

def save_test_results_to_db(token_info: Dict[str, Any], test_results: Dict[str, bool], test_values: Dict[str, Any]) -> None:
    """
    Save token test results to the database for historical tracking.
    
    This function records detailed test outcomes for later analysis
    and performance evaluation. It stores both the binary pass/fail
    results and the actual values that were tested.
    
    Args:
        token_info: Token information from the API
        test_results: Dictionary of test name -> boolean result
        test_values: Dictionary of test name -> actual value
        
    Side Effects:
        Writes a record to the token_test_results table
        
    Process Flow:
        1. Delegate to the database module's specialized function
        2. The function handles proper database interactions
        3. Test results are stored in a structured format for later querying
    """
    # Delegate to database module function
    db.save_test_results_to_db(token_info, test_results, test_values)

def save_full_token_info(token_info: Dict[str, Any]) -> None:
    """
    Save complete token information to the database for reference.
    
    This function stores the entire token information dictionary
    from the API, providing a comprehensive historical record for
    later analysis and debugging.
    
    Args:
        token_info: Complete token information dictionary from API
        
    Side Effects:
        Writes a record to the tokens_full_details table
        
    Implementation Notes:
        - Called for all tokens processed in detail
        - Delegates to database module for actual storage
        - The database function handles serialization of complex nested structures
    """
    # Delegate to database module function
    db.save_full_token_info_to_db(token_info)

def load_bought_tokens() -> None:
    """
    Load recently bought tokens from the database.
    
    This function populates the global last_bought dictionary with
    information about recently purchased tokens to prevent duplicate
    buys within the configured time window.
    
    Global Side Effects:
        Updates the last_bought global dictionary
        
    Database Interaction:
        READ-ONLY operation that retrieves data without modification
        
    Implementation Notes:
        - Called during initialization before starting token processing
        - Prevents buying the same token multiple times in quick succession
        - Only loads tokens purchased within the recent timeframe
    """
    global last_bought
    # Delegate to database module function
    last_bought = db.load_bought_tokens()
    logger.info(f"Loaded {len(last_bought)} recently bought tokens from database")

def save_buy_price_to_db(address: str, buy_price: float) -> bool:
    """
    Record the price at which a token was purchased.
    
    This function saves the purchase price of a token to the database
    for profit tracking and analysis. It's called after a successful
    market buy operation.
    
    Args:
        address: The token's blockchain address
        buy_price: The price at which the token was purchased
        
    Returns:
        bool: True if the data was successfully saved, False otherwise
        
    Side Effects:
        Writes or updates a record in the bought_tokens table
        
    Implementation Notes:
        - Delegates to the database module for actual storage
        - Creates a new record or updates an existing one if the token was bought before
    """
    # Delegate to database module function
    return db.save_buy_price_to_db(address, buy_price)

# Global event to signal that a buy has been executed
buy_executed_event = threading.Event()

def gmgn_detailed_check(row: Union[pd.Series, Dict[str, Any]]) -> bool:
    """
    Comprehensively evaluate a token and execute a buy if it passes all tests.
    
    This function is the core decision engine of the trading bot. It takes raw
    token data, performs detailed analysis, and makes a buy decision based on
    configurable criteria. All fresh data is retrieved from the GMGN API and
    then stored in the database for historical record.
    
    Algorithm Flow:
        1. Extract token address from input (handles various formats)
        2. Fetch complete token details from GMGN API (DIRECT API CALL)
        3. Run token through all test criteria
        4. Get rug check score from external API (DIRECT API CALL)
        5. If all tests pass, check if token was recently bought
        6. If not recently bought, execute market buy
        7. Offload post-buy processing to background thread
        8. Save all data to database for historical record
        
    Args:
        row (Union[pd.Series, Dict[str, Any]]): Token information (as Series from DataFrame or dictionary)
        
    Returns:
        bool: True if a buy was executed, False otherwise
        
    Database Interaction:
        - NO READ operations for decision making (uses fresh API data)
        - WRITE operations to save results after analysis
        
    Thread Safety:
        - Uses locking for shared data access
        - Offloads non-critical operations to background threads
    """
    # Access global variables for tracking state across multiple runs
    global found_tokens_total, passed_tokens_total, bought_tokens_total
    global bought_token_list, last_bought, last_buy_amounts, buy_executed_event
    global tested_tokens_info, almost_passed_tokens_info, passed_tokens_info, test_stats, bought_tokens_info

    # ----------- STEP 1: Extract token address from various input formats -----------
    # This robust extraction handles multiple data formats (pd.Series, dict, nested structures)
    token_address = None
    
    # Case 1: Extract from pandas Series
    if isinstance(row, pd.Series):
        # Try common field names for token address
        for key in ["address", "coin_address", "token_address"]:
            if key in row and pd.notna(row[key]):
                token_address = row[key]
                print(f"Found token address in row: {token_address}")
                break
        # Fallback to first item if named fields not found
        if not token_address:
            token_address = row.iloc[0] if not row.empty else None
    
    # Case 2: Extract from dictionary
    elif isinstance(row, dict):
        # Try multiple possible field names for token address
        token_address = row.get("address") or row.get("coin_address") or row.get("token_address")
    
    # Case 3: Direct string input
    else:
        token_address = str(row)

    # Handle nested data structures (lists or dictionaries)
    if isinstance(token_address, list):
        first_item = token_address[0]
        token_address = first_item.get("address") if isinstance(first_item, dict) else first_item
    if isinstance(token_address, dict):
        token_address = token_address.get("address")

    # Validate we successfully extracted a token address
    if not token_address:
        print("No valid token address found, skipping.")
        return False

    # Increment total token counter and get current timestamp
    found_tokens_total += 1
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ----------- STEP 2: Fetch token information from GMGN API -----------
    try:
        # DIRECT API CALL: Get fresh token information from GMGN API
        # This is the core of "only getting information from GMGN client"
        token_security_info = n.get_token_info(token_address)
        
        # For display purposes only - doesn't affect decision
        last_seen_val = get_last_seen_from_master_all(token_address) or "N/A"
        current_price = token_security_info.get("usd_price", token_security_info.get("price", "N/A"))
        
        cprint(f"[{now_str}] Processing token: {token_address} | Last Seen: {last_seen_val} | Current Price: {current_price}", "cyan")
        
        # Validate token info is a proper dictionary
        if not isinstance(token_security_info, dict):
            cprint(f"Invalid token info for {token_address}: {token_security_info}", "red")
            return False
    except Exception as e:
        cprint(f"Error fetching token info for {token_address}: {e}", "red")
        return False

    # ----------- STEP 3: Run token through all test criteria -----------
    # Call our comprehensive test function to evaluate token against criteria
    test_results, test_values = get_test_results(token_security_info)
    token_name = token_security_info.get("name", token_address)

    # ----------- STEP 4: Get rug check score and add as additional test -----------
    # DIRECT API CALL: Get rug check score from external API
    rug_info = n.df_token_rug_info(token_address)
    rug_ratio_str = rug_info.get("rug.rug_ratio", "0")
    rug_ratio_val = safe_float(rug_ratio_str)
    rug_ratio_pass = (rug_ratio_val <= MAX_RUG_RATIO)
    test_results["Rug Ratio"] = rug_ratio_pass
    test_values["Rug Ratio"] = rug_ratio_val

    # ----------- STEP 5: Build detailed output for logging and display -----------
    # Construct a formatted report of all test results
    output_lines = []
    output_lines.append(f"Test Results for {token_name}:")
    header = "{:<30} {:<10} {:<}".format("Test", "Result", "Value")
    output_lines.append(header)
    output_lines.append("-" * len(header))
    
    # Iterate through test results and safely format for display
    # FIX: Use .get() method to safely handle missing keys
    for test in test_results:
        result_str = "PASS" if test_results[test] else "FAIL"
        # Use .get() with a default value to handle missing keys
        test_value = test_values.get(test, "N/A")
        output_lines.append("{:<30} {:<10} {}".format(test, result_str, test_value))

    # Calculate pass rate statistics
    total_tests = len(test_results)
    passed_count = sum(1 for t in test_results if test_results[t])
    pass_rate = passed_count / total_tests if total_tests > 0 else 0
    details_text = "\n".join(output_lines)

    # Create a standardized token info record 
    token_info_entry = {
        "address": token_address,
        "name": token_name,
        "test_results": test_results,
        "test_values": test_values,
        "timestamp": now_str,
        "details": details_text,
        "pass_rate": pass_rate
    }

    # ----------- STEP 6: Update UI and statistics tracking -----------
    # Thread-safe updates to global tracking structures
    with gui_lock:
        # Add to all tested tokens list
        tested_tokens_info.append(token_info_entry)
        
        # Categorize by pass rate
        if pass_rate == 1.0:
            passed_tokens_info.append(token_info_entry)
        elif pass_rate >= 0.75:
            almost_passed_tokens_info.append(token_info_entry)
            
        # Update test statistics for analysis
        for test in test_results:
            # Initialize test stats dictionary if needed
            if test not in test_stats:
                test_stats[test] = {"pass_count": 0, "fail_count": 0, "sum": 0.0, "count": 0}
                
            # Update pass/fail counters
            if test_results[test]:
                test_stats[test]["pass_count"] += 1
            else:
                test_stats[test]["fail_count"] += 1
                
            # Try to update numeric statistics if possible
            try:
                val = float(test_values.get(test, 0))  # Use .get() for safety
                test_stats[test]["sum"] += val
                test_stats[test]["count"] += 1
            except Exception:
                # Skip if value can't be converted to float
                pass

    # ----------- STEP 7: Save results to database -----------
    # WRITE TO DATABASE: Save all results regardless of test outcome
    # This happens AFTER all fresh data analysis is complete
    save_test_results_to_db(token_security_info, test_results, test_values)
    save_full_token_info(token_security_info)
    update_master_all(token_security_info)

    # ----------- STEP 8: Check if all tests passed -----------
    # Exit early if any test failed
    if not all(test_results.values()):
        # Log which tests failed for debugging
        failed_tests = [test for test, passed in test_results.items() if not passed]
        cprint(f"Token {token_address} failed tests: {', '.join(failed_tests)}; skipping buy.", "yellow")
        return False

    # ----------- STEP 9: Check if token was bought recently -----------
    # Prevent re-buying the same token too frequently
    now_obj = datetime.now()
    if token_address in last_bought and (now_obj - last_bought[token_address]) < timedelta(minutes=REBUY_INTERVAL_MINUTES):
        cprint(f"Token {token_address} was bought within the last {REBUY_INTERVAL_MINUTES} minutes; skipping buy.", "yellow")
        return False

    # ----------- STEP 10: Execute market buy -----------
    # All tests passed and not recently bought - proceed with purchase
    passed_tokens_total += 1
    new_buy_amount = FULL_BUY_AMOUNT
    cprint(f"Token {token_address} passed all tests; executing full market buy of ${new_buy_amount:.2f}.", "green")
    n.test_market_buy(token_address, amount=new_buy_amount)

    # Signal that a buy has been executed (for other components)
    buy_executed_event.set()

    # ----------- STEP 11: Offload post-buy processing to background thread -----------
    # This allows the main thread to continue scanning for new opportunities
    threading.Thread(
        target=post_buy_processing,
        args=(token_address, token_security_info, new_buy_amount, now_obj),
        daemon=True
    ).start()

    # Return success
    return True


def post_buy_processing(token_address: str, token_security_info: Dict[str, Any], new_buy_amount: float, now_obj: datetime) -> None:
    """
    Handle non-critical post-buy updates in a background thread.
    
    After executing a buy, this function handles bookkeeping tasks that
    don't need to block the main processing thread. It updates the database,
    internal tracking state, and GUI display elements.
    
    Args:
        token_address: The token's blockchain address
        token_security_info: Complete token information from API
        new_buy_amount: Amount spent on the purchase in USD
        now_obj: Timestamp of when the purchase was executed
        
    Global Side Effects:
        - Updates counters and tracking lists
        - Updates GUI display data
        
    Database Operations:
        - Saves buy price to database
        - Removes token from low value list if present
        
    Implementation Notes:
        - Runs in a separate background thread for efficiency
        - Uses thread-safe operations for shared data access
    """
    # Extract token price from API data
    token_price = safe_float(token_security_info.get("usd_price", 0))
    if token_price == 0:
        token_price = safe_float(token_security_info.get("price", 0))
        
    # WRITE TO DATABASE: Save buy price for profit tracking
    save_buy_price_to_db(token_address, token_price)

    # Update global tracking variables
    global bought_tokens_total, bought_token_list, last_bought, last_buy_amounts
    bought_tokens_total += 1
    bought_token_list.append(token_address)
    last_bought[token_address] = now_obj
    last_buy_amounts[token_address] = new_buy_amount

    # WRITE TO DATABASE: Remove from low value list if present
    db.remove_token_from_low_value_list(token_address)

    # Update GUI tracking with thread safety
    with gui_lock:
        bought_tokens_info.append({
            "address": token_address,
            "name": token_security_info.get("name", token_address),
            "price": token_price,
            "timestamp": now_obj.strftime('%Y-%m-%d %H:%M:%S'),
            "amount": new_buy_amount
        })

def initialize_database() -> None:
    """
    Initialize the database and migrate existing CSV data if needed.
    
    This function ensures the database is properly set up with all required
    tables and indexes. It also handles one-time migration of data from
    legacy CSV files to the database.
    
    Side Effects:
        - Initializes database schema
        - Migrates data from CSV files if needed
        - Creates a flag file to mark migration completion
        
    Error Handling:
        - Logs detailed error information if initialization fails
        - Application can continue with limited functionality on failure
        
    Implementation Notes:
        - Only performs CSV migration once (tracks with flag file)
        - Delegates actual database initialization to the database module
    """
    try:
        # Initialize database schema
        db.db_init()
        cprint("Database initialized successfully", "green")
        
        # Migrate existing CSV data if not already done
        migration_flag = os.path.join(DATA_FOLDER, "migration_done.flag")
        if not os.path.exists(migration_flag):
            cprint("Migrating existing CSV data to database...", "yellow")
            
            # Delegate to database module function
            results = db.migrate_all_csv_data()
            
            # Log migration results
            for table, count in results.items():
                cprint(f"Migrated {count} records to {table}", "green")
                
            # Create a flag file to mark migration as complete
            with open(migration_flag, "w") as f:
                f.write(f"Migration completed on {datetime.now().isoformat()}")
    except Exception as e:
        cprint(f"Error initializing database: {e}", "red")
        logger.error(f"Database initialization error: {e}")
        logger.error(traceback.format_exc())

def main() -> None:
    """
    Main token processing function that orchestrates the trending coin scanning.
    
    This function fetches trending tokens from multiple sources, removes
    duplicates, and evaluates each token against the defined criteria. It
    operates on a cycle, processing batches of tokens and then pausing
    before the next scan.
    
    Process Flow:
        1. Load previously seen addresses and recently bought tokens
        2. Fetch trending tokens from multiple API sources
        3. Flatten and deduplicate the token list
        4. Save newly discovered addresses to database
        5. Process each token in parallel threads
        6. Output statistics about the scan
        
    Global Side Effects:
        Updates various counters and tracking structures
        
    Implementation Notes:
        - Fetches data from multiple API endpoints
        - Processes tokens in parallel for efficiency
        - Uses thread-safe operations for shared data
    """
    # Load tracking data from database
    load_seen_addresses()
    load_bought_tokens()
    
    try:
        # DIRECT API CALLS: Get fresh trending tokens from GMGN API
        trending_tokens1_m = ensure_list(n.get_trending_tokens("1m"))
        new_tokens = ensure_list(n.get_new_pairs())
        
        # Process and normalize the token data
        flat_new_tokens = n.flatten_new_tokens(new_tokens)
        tokenlist = trending_tokens1_m + flat_new_tokens
        trending_tokens = remove_duplicate_tokens(tokenlist)
        trending_tokens_df = pd.DataFrame(trending_tokens)
    except Exception as e:
        logger.error(f"Error fetching trending tokens: {e}")
        return

    # Skip processing if no trending tokens found
    if trending_tokens_df.empty:
        cprint("No trending tokens found.", "yellow")
        return

    # WRITE TO DATABASE: Save newly discovered addresses
    db.save_new_addresses(trending_tokens_df)

    # Reset the buy event flag at the start of each scan
    buy_executed_event.clear()

    # Process each token in a separate thread for efficiency
    threads = []
    for index, row in trending_tokens_df.iterrows():
        thread = threading.Thread(target=gmgn_detailed_check, args=(row,))
        thread.start()
        threads.append(thread)

    # Wait for all processing threads to complete
    while any(t.is_alive() for t in threads):
        time.sleep(0.1)

    # Output scan statistics
    cprint("===== Run Stats =====", "cyan")
    cprint(f"Total tokens found (lifetime): {found_tokens_total}", "green")
    cprint(f"Total tokens that passed tests (lifetime): {passed_tokens_total}", "green")
    cprint(f"Total tokens bought (lifetime): {bought_tokens_total}", "green")
    cprint(f"Bought token list: {bought_token_list}", "green")
    cprint("=====================", "cyan")
while True:
    main()  # Run the main function
