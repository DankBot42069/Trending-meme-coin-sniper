import os
import random
import time
import requests

class ProxyManager:
    """
    Manages proxies for HTTP requests, with fallback to direct connections if no proxies are available.
    Automatically removes failing proxies from the list.
    """
    
    def __init__(self, proxy_dir='Pub_shared_bots/gmgn/proxy', proxy_files=['proxielist.txt', 'https.txt']):
        """
        Initialize the proxy manager.
        
        Args:
            proxy_dir: Directory containing proxy files
            proxy_files: List of proxy files to load
        """
        self.proxy_dir = proxy_dir
        self.proxy_files = proxy_files
        self.http_proxies = []
        self.https_proxies = []
        self.failed_proxies = set()
        self.load_proxies()
    
    def load_proxies(self):
        """Load proxies from the specified files."""
        self.http_proxies = []
        self.https_proxies = []
        
        # Create proxy directory if it doesn't exist
        os.makedirs(self.proxy_dir, exist_ok=True)
        
        for filename in self.proxy_files:
            filepath = os.path.join(self.proxy_dir, filename)
            
            # Create file if it doesn't exist
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    pass  # Create empty file
            
            # Read and parse proxies
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        proxy = line.strip()
                        if proxy and proxy not in self.failed_proxies:
                            # Assume HTTP for proxielist.txt and HTTPS for https.txt
                            if 'https' in filename.lower():
                                self.https_proxies.append(proxy)
                            else:
                                self.http_proxies.append(proxy)
            except Exception as e:
                print(f"Error loading proxies from {filepath}: {e}")
        
        print(f"Loaded {len(self.http_proxies)} HTTP proxies and {len(self.https_proxies)} HTTPS proxies")
    
    def get_proxy(self, protocol="http"):
        """
        Get a random proxy for the specified protocol.
        
        Args:
            protocol: 'http' or 'https'
            
        Returns:
            A proxy string or None if no proxies are available
        """
        proxies = self.https_proxies if protocol.lower() == "https" else self.http_proxies
        
        if not proxies:
            return None
            
        return random.choice(proxies)
    
    def get_proxy_dict(self):
        """
        Get a proxy dictionary for requests.
        
        Returns:
            A proxy dictionary or None if no proxies are available
        """
        http_proxy = self.get_proxy("http")
        https_proxy = self.get_proxy("https")
        
        if not http_proxy and not https_proxy:
            return None
            
        proxy_dict = {}
        if http_proxy:
            proxy_dict["http"] = f"http://{http_proxy}"
        if https_proxy:
            proxy_dict["https"] = f"https://{https_proxy}"
            
        return proxy_dict
    
    def mark_proxy_failed(self, proxy):
        """
        Mark a proxy as failed and remove it from the lists.
        
        Args:
            proxy: The full proxy URL (including the protocol prefix)
        """
        # Extract the actual proxy address
        if proxy.startswith(("http://", "https://")):
            proxy_address = proxy.split("://")[1]
        else:
            proxy_address = proxy
            
        self.failed_proxies.add(proxy_address)
        
        # Remove from lists
        if proxy_address in self.http_proxies:
            self.http_proxies.remove(proxy_address)
        if proxy_address in self.https_proxies:
            self.https_proxies.remove(proxy_address)
            
        # Update proxy files
        self._save_proxies()
        
        print(f"Removed failed proxy: {proxy_address}")
    
    def _save_proxies(self):
        """Save working proxies back to files, removing failed ones."""
        # Save HTTP proxies
        http_path = os.path.join(self.proxy_dir, 'proxielist.txt')
        with open(http_path, 'w') as f:
            for proxy in self.http_proxies:
                f.write(f"{proxy}\n")
                
        # Save HTTPS proxies
        https_path = os.path.join(self.proxy_dir, 'https.txt')
        with open(https_path, 'w') as f:
            for proxy in self.https_proxies:
                f.write(f"{proxy}\n")

def request_with_proxy_fallback(url, method="get", headers=None, data=None, json=None, timeout=30, max_retries=3):
    """
    Make an HTTP request with proxy and fallback to direct connection if proxies fail.
    
    Args:
        url: The URL to request
        method: HTTP method (get, post, etc.)
        headers: Request headers
        data: Request data
        json: JSON data for the request
        timeout: Request timeout
        max_retries: Maximum number of retries with different proxies
        
    Returns:
        Response object or None on failure
    """
    # Initialize proxy manager
    proxy_manager = ProxyManager()
    
    # Try with proxy first
    for attempt in range(max_retries):
        # Get a proxy dictionary
        proxies = proxy_manager.get_proxy_dict()
        
        # If no proxies are available, break and try direct connection
        if not proxies:
            print("No proxies available, trying direct connection")
            break
            
        try:
            print(f"Attempt {attempt+1}/{max_retries} using proxy: {proxies}")
            
            if method.lower() == "get":
                response = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
            elif method.lower() == "post":
                response = requests.post(url, headers=headers, data=data, json=json, proxies=proxies, timeout=timeout)
            else:
                response = requests.request(method, url, headers=headers, data=data, json=json, proxies=proxies, timeout=timeout)
                
            if response.status_code < 400:
                return response
                
            # If we get here, the proxy didn't work well
            print(f"Proxy request failed with status code {response.status_code}")
            for proxy_url in proxies.values():
                proxy_manager.mark_proxy_failed(proxy_url)
                
        except requests.exceptions.RequestException as e:
            print(f"Proxy request failed: {e}")
            for proxy_url in proxies.values():
                proxy_manager.mark_proxy_failed(proxy_url)
        
        # Wait before retrying
        time.sleep(1)
    
    # Fallback to direct connection
    print("Using direct connection without proxy")
    try:
        if method.lower() == "get":
            return requests.get(url, headers=headers, timeout=timeout)
        elif method.lower() == "post":
            return requests.post(url, headers=headers, data=data, json=json, timeout=timeout)
        else:
            return requests.request(method, url, headers=headers, data=data, json=json, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"Direct request failed: {e}")
        return None

# Example usage with tls_client
def tls_request_with_proxy_fallback(client, url, method="GET", headers=None, data=None, json=None, timeout=30, max_retries=3):
    """
    Make a TLS client request with proxy and fallback to direct connection if proxies fail.
    
    Args:
        client: tls_client.Session object
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        headers: Request headers
        data: Request data
        json: JSON data for the request
        timeout: Request timeout
        max_retries: Maximum number of retries with different proxies
        
    Returns:
        Response object or None on failure
    """
    # Initialize proxy manager
    proxy_manager = ProxyManager()
    
    # Try with proxy first
    for attempt in range(max_retries):
        # Get a proxy dictionary
        proxies = proxy_manager.get_proxy_dict()
        
        # If no proxies are available, break and try direct connection
        if not proxies:
            print("No proxies available, trying direct connection")
            break
            
        try:
            print(f"Attempt {attempt+1}/{max_retries} using proxy: {proxies}")
            
            # Format proxy for tls_client (it expects a string, not a dict)
            proxy_str = next(iter(proxies.values())) if proxies else None
            client.proxies = proxy_str
            
            if method.upper() == "GET":
                response = client.get(url, headers=headers, timeout_seconds=timeout)
            elif method.upper() == "POST":
                response = client.post(url, headers=headers, data=data, json=json, timeout_seconds=timeout)
            else:
                response = client.request(method, url, headers=headers, data=data, json=json, timeout_seconds=timeout)
                
            if response.status_code < 400:
                return response
                
            # If we get here, the proxy didn't work well
            print(f"Proxy request failed with status code {response.status_code}")
            proxy_manager.mark_proxy_failed(proxy_str)
                
        except Exception as e:
            print(f"Proxy request failed: {e}")
            if client.proxies:
                proxy_manager.mark_proxy_failed(client.proxies)
        
        # Wait before retrying
        time.sleep(1)
    
    # Fallback to direct connection
    print("Using direct connection without proxy")
    try:
        client.proxies = None
        if method.upper() == "GET":
            return client.get(url, headers=headers, timeout_seconds=timeout)
        elif method.upper() == "POST":
            return client.post(url, headers=headers, data=data, json=json, timeout_seconds=timeout)
        else:
            return client.request(method, url, headers=headers, data=data, json=json, timeout_seconds=timeout)
    except Exception as e:
        print(f"Direct request failed: {e}")
        return None