import os
import random
import json
import tls_client
import time
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class gmgn:
    # Base URLs
    BASE_URL_QUOTATION = "https://gmgn.ai/defi/quotation"
    BASE_URL_API = "https://gmgn.ai/api/v1"

    # Proxy file paths - with platform-independent path handling
    PROXY_DIR = os.path.join("Pub_shared_bots", "gmgn", "proxy")
    HTTP_PROXY_FILE = os.path.join(PROXY_DIR, "proxielist.txt")
    HTTPS_PROXY_FILE = os.path.join(PROXY_DIR, "https.txt")
    
    # Maximum retries and timeout settings
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self):
        """
        Initialize the GMGN client with dynamic proxy handling.
        """
        self.user_agent = None
        self.identifier = None
        self.sendRequest = None
        self.headers = None
        self.http_proxies = []
        self.https_proxies = []
        self.failed_proxies = set()
        
        # Create proxy directory if it doesn't exist
        os.makedirs(self.PROXY_DIR, exist_ok=True)
        
        # Load proxies from files
        self.load_proxies()

    def load_proxies(self):
        """
        Load proxies from the specified files.
        Creates the files if they don't exist.
        """
        self.http_proxies = []
        self.https_proxies = []
        
        # Create HTTP proxy file if it doesn't exist
        if not os.path.exists(self.HTTP_PROXY_FILE):
            with open(self.HTTP_PROXY_FILE, 'w') as f:
                pass
        
        # Create HTTPS proxy file if it doesn't exist
        if not os.path.exists(self.HTTPS_PROXY_FILE):
            with open(self.HTTPS_PROXY_FILE, 'w') as f:
                pass
        
        # Load HTTP proxies
        try:
            with open(self.HTTP_PROXY_FILE, 'r') as f:
                for line in f:
                    proxy = line.strip()
                    if proxy and proxy not in self.failed_proxies:
                        self.http_proxies.append(proxy)
        except Exception as e:
            print(f"Error loading HTTP proxies: {e}")
        
        # Load HTTPS proxies
        try:
            with open(self.HTTPS_PROXY_FILE, 'r') as f:
                for line in f:
                    proxy = line.strip()
                    if proxy and proxy not in self.failed_proxies:
                        self.https_proxies.append(proxy)
        except Exception as e:
            print(f"Error loading HTTPS proxies: {e}")
        
        print(f"Loaded {len(self.http_proxies)} HTTP proxies and {len(self.https_proxies)} HTTPS proxies")

    def save_proxies(self):
        """
        Save working proxies back to their respective files.
        Removes failed proxies from the lists.
        """
        # Save HTTP proxies
        try:
            with open(self.HTTP_PROXY_FILE, 'w') as f:
                for proxy in self.http_proxies:
                    f.write(f"{proxy}\n")
        except Exception as e:
            print(f"Error saving HTTP proxies: {e}")
        
        # Save HTTPS proxies
        try:
            with open(self.HTTPS_PROXY_FILE, 'w') as f:
                for proxy in self.https_proxies:
                    f.write(f"{proxy}\n")
        except Exception as e:
            print(f"Error saving HTTPS proxies: {e}")

    def mark_proxy_failed(self, proxy):
        """
        Mark a proxy as failed and remove it from the lists.
        Updates the proxy files.
        
        Args:
            proxy: The proxy string (ip:port format)
        """
        # Add to failed proxies set
        self.failed_proxies.add(proxy)
        
        # Remove from HTTP proxies list
        if proxy in self.http_proxies:
            self.http_proxies.remove(proxy)
            
        # Remove from HTTPS proxies list
        if proxy in self.https_proxies:
            self.https_proxies.remove(proxy)
            
        # Save updated proxy lists
        self.save_proxies()
        print(f"Marked proxy as failed and removed: {proxy}")

    def randomiseRequest(self):
        """
        Creates/re-creates a tls_client.Session with:
         - a randomly chosen proxy (if available)
         - random TLS fingerprint
         - random User-Agent
         - "browser-like" headers
        
        Returns:
            bool: True if using a proxy, False if using direct connection
        """
        # Determine whether to use proxy
        use_proxy = len(self.http_proxies) > 0 or len(self.https_proxies) > 0
        
        # Get a random proxy if available
        proxy_url = None
        chosen_proxy = None
        
        if use_proxy:
            # Combine both proxy lists and choose one randomly
            all_proxies = self.http_proxies + self.https_proxies
            if all_proxies:
                chosen_proxy = random.choice(all_proxies)
                proxy_url = f"http://{chosen_proxy}"  # Assuming simple IP:PORT format
        
        # Random browser-like TLS fingerprint
        self.identifier = random.choice([
            browser for browser in tls_client.settings.ClientIdentifiers.__args__
            if browser.startswith(("chrome", "safari", "firefox", "opera"))
        ])

        # Create new TLS session with random extension order
        self.sendRequest = tls_client.Session(
            random_tls_extension_order=True,
            client_identifier=self.identifier
        )

        # Set proxy if available
        if use_proxy and chosen_proxy:
            self.sendRequest.proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
            print(f"Using proxy: {chosen_proxy}")

        # Random User-Agent
        ua = UserAgent()
        self.user_agent = ua.random

        # Build headers
        self.headers = {
            "Host": "gmgn.ai",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "DNT": "1",
            "Priority": "u=1, i",
            "Referer": "https://gmgn.ai/?chain=sol",
            "User-Agent": self.user_agent,
            "Origin": "https://gmgn.ai",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Dest": "empty",
            'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        }
        
        return use_proxy, chosen_proxy

    def _get_with_retry(self, url, headers):
        """
        Makes a GET request with retry logic:
        1. Tries with proxy if available
        2. Falls back to direct connection if no proxies or all proxies fail
        3. Removes failed proxies from the list
        
        This method will retry up to MAX_RETRIES times with different proxies,
        and if all proxies fail, it will try a direct connection.
        
        Args:
            url: The URL to request
            headers: Custom headers (if None, uses self.headers)
            
        Returns:
            Response object
        """
        # Use the default headers if none provided
        if headers is None:
            headers = self.headers
            
        # First try with proxies if available
        for attempt in range(self.MAX_RETRIES):
            # Create a new randomized request
            use_proxy, proxy = self.randomiseRequest()
            
            if use_proxy:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.sendRequest.get, url, headers=headers)
                        response = future.result(timeout=self.REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        # Attempt to decode JSON to verify it's valid
                        try:
                            response.json()  # Will raise if invalid
                            return response  # Success with proxy
                        except json.JSONDecodeError:
                            # Invalid JSON, mark proxy as failed
                            if proxy:
                                self.mark_proxy_failed(proxy)
                    else:
                        # Non-200 response, mark proxy as failed
                        if proxy:
                            self.mark_proxy_failed(proxy)
                            
                except (TimeoutError, Exception) as e:
                    # Request failed, mark proxy as failed
                    print(f"Proxy request failed: {e}")
                    if proxy:
                        self.mark_proxy_failed(proxy)
            else:
                # No proxies available, break out and try direct connection
                break
                
            # Wait before retry
            time.sleep(1)
        
        # If all proxies failed or no proxies available, try direct connection
        print("Using direct connection (no proxies available or all proxies failed)")
        try:
            # Create a new session without proxy
            self.identifier = random.choice([
                browser for browser in tls_client.settings.ClientIdentifiers.__args__
                if browser.startswith(("chrome", "safari", "firefox", "opera"))
            ])
            
            self.sendRequest = tls_client.Session(
                random_tls_extension_order=True,
                client_identifier=self.identifier
            )
            
            # No proxies for direct connection
            self.sendRequest.proxies = None
            
            # Random User-Agent for direct connection
            ua = UserAgent()
            self.user_agent = ua.random
            
            # Build headers for direct connection
            if not headers:
                self.headers = {
                    "Host": "gmgn.ai",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Pragma": "no-cache",
                    "DNT": "1",
                    "Priority": "u=1, i",
                    "Referer": "https://gmgn.ai/?chain=sol",
                    "User-Agent": self.user_agent,
                    "Origin": "https://gmgn.ai",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin",
                    "Sec-Fetch-Dest": "empty",
                    'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'
                }
                headers = self.headers
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.sendRequest.get, url, headers=headers)
                response = future.result(timeout=self.REQUEST_TIMEOUT)
                
            if response.status_code == 200:
                # Check if the response is valid JSON
                try:
                    response.json()  # Will raise if invalid
                    return response
                except json.JSONDecodeError:
                    # Failed even with direct connection
                    raise ValueError(f"Invalid JSON in response: {response.text[:100]}...")
            else:
                # Non-200 response
                raise ValueError(f"Request failed with status code: {response.status_code}")
                
        except Exception as e:
            # Even direct connection failed
            print(f"Direct connection failed: {e}")
            # Make one final attempt with minimal configuration
            try:
                print("Making final attempt with minimal configuration")
                # Use standard requests library as a last resort
                import requests
                response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return response
                else:
                    raise ValueError(f"Final attempt failed with status code: {response.status_code}")
            except Exception as final_e:
                print(f"Final attempt failed: {final_e}")
                raise ValueError("All request attempts failed")

    def _parse_response(self, response):
        """
        Attempts to parse JSON. Raises ValueError if it fails.
        """
        try:
            return response.json()
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to decode JSON. Status: {response.status_code}. Response text: {response.text[:100]}..."
            )

    # ======================== API Methods ========================
    # All the original API methods remain unchanged below this point
    
    def getTokenInfo(self, contractAddress: str) -> dict:
        """
        Gets info on a token.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_QUOTATION}/v1/tokens/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getNewPairs(self, limit: int = None, filters: list = None, period: str = None,
                    platforms: list = None, chain: str = 'sol') -> dict:
        """
        Gets new token pairs.
        """
        if not limit:
            limit = 50
        elif limit > 50:
            return "You cannot check more than 50 pairs."

        filter_str = ''
        if filters and len(filters) > 0:
            for filter_ in filters:
                filter_str += f"&filters[]={filter_}"
        platforms_str = ''
        if platforms and len(platforms) > 0:
            for platform_ in platforms:
                platforms_str += f"&platforms[]={platform_}"

        if not period:
            period = "1m"

        url = (
            f"{self.BASE_URL_QUOTATION}/v1/pairs/{chain}/new_pairs?limit={limit}"
            f"&period={period}"
            "&orderby=open_timestamp"
            "&direction=desc" + filter_str + platforms_str
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getTrendingWallets(self, timeframe: str = None, walletTag: str = None) -> dict:
        """
        Gets a list of trending wallets based on a timeframe and a wallet tag.
        """
        if not timeframe:
            timeframe = "7d"
        if not walletTag:
            walletTag = "sniper"
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/wallets/{timeframe}"
            f"?tag={walletTag}&orderby=pnl_{timeframe}&direction=desc"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getTrendingTokens(self, timeframe: str = None, limit: int = 100) -> dict:
        """
        Gets a list of trending tokens based on a timeframe.
        """
        valid_timeframes = ["1m", "5m", "1h", "6h", "24h"]
        if not timeframe or timeframe not in valid_timeframes:
            return "Not a valid timeframe."
        url = f"{self.BASE_URL_QUOTATION}/v1/rank/sol/swaps/{timeframe}?orderby=swaps&direction=desc&limit={limit}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getTokensByCompletion(self, limit: int = None) -> dict:
        """
        Gets tokens by their bonding curve completion progress.
        """
        if not limit:
            limit = 50
        elif limit > 50:
            return "Limit cannot be above 50."
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/pump?limit={limit}"
            "&orderby=progress&direction=desc&pump=true"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def findSnipedTokens(self, size: int = None) -> dict:
        """
        Gets a list of tokens that have been sniped.
        """
        if not size:
            size = 10
        elif size > 39:
            return "Size cannot be more than 39"
        url = f"{self.BASE_URL_QUOTATION}/v1/signals/sol/snipe_new?size={size}&is_show_alert=false&featured=false"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getGasFee(self) -> dict:
        """
        Get the current gas fee price.
        """
        url = f"{self.BASE_URL_QUOTATION}/v1/chains/sol/gas_price"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getTokenUsdPrice(self, contractAddress: str = None) -> dict:
        """
        Get the realtime USD price of the token.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_QUOTATION}/v1/sol/tokens/realtime_token_price?address={contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getTopBuyers(self, contractAddress: str = None) -> dict:
        """
        Get the top buyers of a token.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_QUOTATION}/v1/tokens/top_buyers/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        data = self._parse_response(request)
        # Fix empty list values in statusNow
        holders = data.get("holders", {})
        status_now = holders.get("statusNow", {})
        for key, value in status_now.items():
            if isinstance(value, list) and not value:
                status_now[key] = None
        holders["statusNow"] = status_now
        data["holders"] = holders
        return data

    def getSecurityInfo(self, contractAddress: str = None) -> dict:
        """
        Gets security info about the token.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_QUOTATION}/v1/tokens/security/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getWalletInfo(self, walletAddress: str = None, period: str = None) -> dict:
        """
        Gets various information about a wallet address.
        Period - 7d, 30d.
        """
        valid_periods = ["7d", "30d"]
        if not walletAddress:
            return "You must input a wallet address."
        if not period or period not in valid_periods:
            period = "7d"
        url = f"{self.BASE_URL_QUOTATION}/v1/smartmoney/sol/walletNew/{walletAddress}?period={period}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request).get('data', {})

    def getWalletActivity(self, walletAddress: str, limit: int = 10, cost: int = 10) -> dict:
        """
        Gets the wallet activity.
        """
        if not walletAddress:
            return "You must input a wallet address."
        url = f"{self.BASE_URL_API}/wallet_activity/sol?wallet={walletAddress}&limit={limit}&cost={cost}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getWalletHoldings(self, walletAddress: str) -> dict:
        """
        Gets wallet holdings.
        """
        if not walletAddress:
            return "You must input a wallet address."
        url = (
            f"{self.BASE_URL_API}/wallet_holdings/sol/{walletAddress}"
            f"?limit=50&orderby=last_active_timestamp&direction=desc&showsmall=true&sellout=true&tx30d=true"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getTokenRugInfo(self, contractAddress: str) -> dict:
        """
        Gets token rug info.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_API}/mutil_window_token_link_rug_vote/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getTokenSecurityInfoLaunchpad(self, contractAddress: str) -> dict:
        """
        Gets token security info from the launchpad endpoint.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_API}/mutil_window_token_security_launchpad/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getPumpRanks1h(self) -> dict:
        """
        Gets pump ranks for 1h.
        """
        new_creation = '{"filters":["not_wash_trading"],"limit":80}'
        pump = '{"filters":["not_wash_trading"],"limit":80}'
        completed = '{"filters":["not_wash_trading"],"limit":60}'
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/pump_ranks/1h?"
            f"new_creation={new_creation}&pump={pump}&completed={completed}"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getPumpRanks1m(self) -> dict:
        """
        Gets pump ranks for 1 minute timeframe.
        """
        new_creation = '{"filters":["not_wash_trading"],"limit":15}'
        pump = '{"filters":["not_wash_trading"],"limit":15}'
        completed = '{"filters":["not_wash_trading"],"limit":15}'
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/pump_ranks/1m?"
            f"new_creation={new_creation}&pump={pump}&completed={completed}"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getSwapRanks30m(self) -> dict:
        """
        Gets swap ranks for up to 30 minutes old.
        """
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/swaps/1m?orderby=change1m&direction=desc&limit=20"
            f"&filters[]=renounced&filters[]=frozen&filters[]=burn&filters[]=not_wash_trading"
            f"&min_liquidity=140000&min_marketcap=150000&max_insider_rate=0.15&max_created=30m"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getSwapRanks5m(self) -> dict:
        """
        Gets swap ranks for up to 5 minutes old.
        """
        url = (
            f"{self.BASE_URL_QUOTATION}/v1/rank/sol/swaps/1m?orderby=change1m&direction=desc&limit=20"
            f"&filters[]=renounced&filters[]=frozen&filters[]=burn&filters[]=not_wash_trading"
            f"&min_liquidity=140000&min_marketcap=150000&max_insider_rate=0.15&max_created=1m"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getTokenStats(self, contractAddress: str) -> dict:
        """
        Gets token statistics.
        """
        if not contractAddress:
            return "You must input a contract address."
        url = f"{self.BASE_URL_API}/token_stat/sol/{contractAddress}"
        request = self._get_with_retry(url, None)
        return self._parse_response(request)

    def getTokenKline(self, contractAddress: str, resolution: str, from_ts: int, to_ts: int) -> dict:
        """
        Gets token kline data.
        """
        url = (
            f"{self.BASE_URL_API}/token_kline/sol/{contractAddress}?resolution={resolution}"
            f"&from={from_ts}&to={to_ts}"
        )
        request = self._get_with_retry(url, None)
        return self._parse_response(request)