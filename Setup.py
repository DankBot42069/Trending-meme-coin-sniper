#!/usr/bin/env python3
"""
Trading Bot Setup Script

This script checks for and installs all required dependencies for the trading bot.
It will detect if Python is installed, and if not, guide the user through installation.
For existing Python installations, it will install all required packages.

Usage:
    python setup.py

Or on Windows, simply double-click this file.
"""

import sys
import os
import platform
import subprocess
import importlib.util
import webbrowser
import time
from pathlib import Path
import shutil
import random
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Define required packages (removing gmgn since it's a local module)
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "requests",
    "websocket-client",  # The actual package name for websocket
    "matplotlib",
    "aiohttp",
    "solana",
    "solders",
    "pytz",
    "tls_client",
    "fake_useragent"
]

def check_python_installed():
    """Check if Python is installed and meets version requirements."""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"Python version {version.major}.{version.minor} detected. Version 3.8+ is required.")
            return False
        print(f"Python {version.major}.{version.minor}.{version.micro} detected.")
        return True
    except Exception:
        return False

def is_admin():
    """Check if the script is running with administrator privileges."""
    try:
        if platform.system() == "Windows":
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except:
        return False

def restart_as_admin():
    """Restart the script with administrator privileges."""
    if platform.system() == "Windows":
        script_path = os.path.abspath(__file__)
        try:
            # Using subprocess to elevate privileges
            subprocess.run(['powershell', 'Start-Process', 'python', f'"{script_path}"', '-Verb', 'RunAs'], check=True)
            sys.exit(0)
        except:
            print("Failed to restart with admin privileges. Please run this script as administrator.")
            sys.exit(1)
    else:
        print("Please run this script with sudo privileges.")
        sys.exit(1)

def guide_python_installation():
    """Guide user to install Python if not found."""
    system = platform.system()
    
    print("\n" + "="*80)
    print("Python 3.8 or higher not found or not properly configured!")
    print("="*80)
    print("\nTo install Python:")
    
    if system == "Windows":
        print("\n1. Download Python from the official website:")
        print("   https://www.python.org/downloads/")
        print("\n2. During installation, make sure to check:")
        print("   ☑ Add Python to PATH")
        print("   ☑ Install pip")
        
        # Open Python download page
        choice = input("\nWould you like to open the Python download page now? (y/n): ")
        if choice.lower() in ['y', 'yes']:
            webbrowser.open("https://www.python.org/downloads/")
            
    elif system == "Darwin":  # macOS
        print("\n1. Install Homebrew (if not already installed):")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("\n2. Install Python using Homebrew:")
        print("   brew install python")
        
    elif system == "Linux":
        print("\n1. Update package lists:")
        print("   sudo apt update")
        print("\n2. Install Python:")
        print("   sudo apt install python3 python3-pip")
        
    print("\nAfter installing Python, run this setup script again.")
    input("\nPress Enter to exit...")
    sys.exit(1)

def check_package(package_name):
    """Check if a package is installed."""
    try:
        spec = importlib.util.find_spec(package_name.split('==')[0].replace('-', '_'))
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        # Use --user flag if not running as admin and not in a virtual environment
        user_flag = [] if (is_admin() or hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else ['--user']
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + user_flag + [package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to the latest version."""
    try:
        print("Upgrading pip...")
        # Use --user flag if not running as admin and not in a virtual environment
        user_flag = [] if (is_admin() or hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else ['--user']
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade'] + user_flag + ['pip'])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade pip. Error: {e}")
        return False

def create_folders():
    """Create necessary folders for the application."""
    # Get the directory where this script is located
    base_dir = Path(__file__).parent.absolute()
    
    # Create required directories
    folders = [
        "data",
        "data/database",
    ]
    
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"Creating directory: {folder_path}")
            folder_path.mkdir(parents=True, exist_ok=True)

def install_required_packages():
    """Install all required packages."""
    # Upgrade pip first
    upgrade_pip()
    
    missing_packages = []
    
    # Check which packages need installation
    for package in REQUIRED_PACKAGES:
        if not check_package(package.split('==')[0]):
            missing_packages.append(package)
    
    if not missing_packages:
        print("\nAll required packages are already installed!")
        return True
    
    # Install missing packages
    print(f"\nMissing packages ({len(missing_packages)}):")
    for package in missing_packages:
        print(f" - {package}")
    
    print("\nInstalling missing packages...")
    
    failed_packages = []
    for package in missing_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print("\nFailed to install the following packages:")
        for package in failed_packages:
            print(f" - {package}")
        
        print("\nPossible solutions:")
        print(" 1. Run this script with administrator privileges")
        print(" 2. Try installing them manually using:")
        print(f"    pip install {' '.join(failed_packages)}")
        return False
    
    print("\nAll required packages have been successfully installed!")
    return True

def check_gmgn_module():
    """
    Check if the local gmgn module exists.
    The gmgn module is expected to be a local module, not a PyPI package.
    """
    base_dir = Path(__file__).parent.absolute()
    gmgn_dir = base_dir / 'gmgn'
    
    if not gmgn_dir.exists():
        print("\nNOTE: The gmgn module folder was not found.")
        print("This module is expected to be a local module in the project directory.")
        print("Make sure the gmgn folder is present in the same directory as this script.")
        return False
    
    print("Local gmgn module folder found.")
    return True

def create_dontshare_template():
    """Create a template dontshare.py file if it doesn't exist."""
    dontshare_path = Path(__file__).parent / 'dontshare.py'
    
    if not dontshare_path.exists():
        print("Creating dontshare.py template file...")
        template = """#!/usr/bin/env python3
\"\"\"
Sensitive credentials file for trading bots.

IMPORTANT: Keep this file private and never share or commit it to version control.
This file contains sensitive API keys and wallet private keys.
\"\"\"

# Solana wallet private key (base58 encoded)
sol_key = "YOUR_SOLANA_PRIVATE_KEY_HERE"  # Replace with your actual Solana private key

# API Keys
ALCHEMY_API_KEY = "YOUR_ALCHEMY_API_KEY_HERE"  # Replace with your Alchemy API key
ankr_key = "YOUR_ANKR_API_KEY_HERE"  # Replace with your Ankr API key

# Optional: Add any other private credentials your bot might need
# JUPITER_API_KEY = "YOUR_JUPITER_API_KEY"
# COINBASE_API_KEY = "YOUR_COINBASE_API_KEY"
"""
        with open(dontshare_path, 'w') as f:
            f.write(template)
        print(f"Created template file at {dontshare_path}")
        print("Please edit this file with your actual API keys and private keys before running the bot.")

def main():
    """Main function."""
    print("="*80)
    print(f"Trading Bot Setup - Python {platform.python_version()} - {platform.system()} {platform.release()}")
    print("="*80 + "\n")
    
    # Check if Python is installed
    if not check_python_installed():
        guide_python_installation()
    
    # Check if running as admin and restart if needed
    if not is_admin() and platform.system() == "Windows" and any(arg.lower() == '--admin' for arg in sys.argv):
        print("This script requires administrator privileges to install packages system-wide.")
        restart_as_admin()
    
    # Create necessary folders
    create_folders()
    
    # Install required packages
    success = install_required_packages()
    
    # Check local gmgn module
    gmgn_success = check_gmgn_module()
    
    # Create dontshare.py template if it doesn't exist
    create_dontshare_template()
    
    # Final message
    if success:
        print("\n" + "="*80)
        print("Setup completed successfully!")
        print("="*80)
        print("\nYou can now run the trading bot application.")
        print("\nIMPORTANT: Make sure to edit the dontshare.py file with your actual API keys and private keys.")
        
        if not gmgn_success:
            print("\nWARNING: The local gmgn module was not found. Make sure to include it before running the bot.")
    else:
        print("\n" + "="*80)
        print("Setup completed with some issues.")
        print("="*80)
        print("\nPlease resolve the issues mentioned above before running the trading bot.")
    
    if platform.system() == "Windows":
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if platform.system() == "Windows":
            input("\nPress Enter to exit...")
        sys.exit(1)