#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download a TripAdvisor page into a cache folder
Uses Selenium WebDriver to handle JavaScript-heavy sites like TripAdvisor

Created on November 1, 2025
@author: andreyvlasenko
"""

import os
from urllib.parse import urlparse
import time
from datetime import datetime

# Try Selenium first (recommended for TripAdvisor), fallback to requests
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available, will try requests library")

import requests


# Create cache folder if it doesn't exist


def download_with_selenium(url, cache_dir):
    """
    Download a web page using Selenium WebDriver (handles JavaScript)
    
    Args:
        url: URL to download
        cache_dir: Directory to save the downloaded content
    
    Returns:
        Path to saved file or None if failed
    """
    driver = None
    try:
        print(f"🌐 Downloading with Selenium: {url}")
        
        # Try Firefox first (since user mentioned Firefox), then Chrome
        try:
            print("   Trying Firefox WebDriver...")
            options = FirefoxOptions()
            options.add_argument('--headless')  # Run in background
            options.add_argument('--disable-blink-features=AutomationControlled')
            driver = webdriver.Firefox(options=options)
        except Exception as e:
            print(f"   Firefox not available: {e}")
            try:
                print("   Trying Chrome WebDriver...")
                options = ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                driver = webdriver.Chrome(options=options)
            except Exception as e2:
                print(f"   Chrome not available: {e2}")
                return None
        
        # Load the page
        driver.get(url)
        
        # Wait for page to load (wait for body element)
        print("   Waiting for page to load...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        time.sleep(3)
        
        # Get the page source
        page_source = driver.page_source
        
        # Generate filename
        parsed_url = urlparse(url)
        filename = parsed_url.path.strip('/').replace('/', '_')
        if not filename:
            filename = parsed_url.netloc.replace('.', '_')
        if not filename.endswith('.html'):
            filename += '.html'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(filename)
        filename = f"{base_name}_{timestamp}{ext}"
        
        filepath = os.path.join(cache_dir, filename)
        
        # Save the content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(page_source)
        
        # Save metadata
        metadata_file = filepath.replace('.html', '_metadata.txt')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: Selenium WebDriver\n")
            f.write(f"Content Length: {len(page_source)} characters\n")
            f.write(f"Browser: {driver.name}\n")
        
        print(f"✅ Successfully downloaded with Selenium!")
        print(f"   📁 Saved to: {filepath}")
        print(f"   📊 Size: {len(page_source):,} characters")
        print(f"   📝 Metadata: {metadata_file}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Selenium download failed: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

def download_with_requests(url, cache_dir):
    """
    Download a web page using requests library (fallback method)
    
    Args:
        url: URL to download
        cache_dir: Directory to save the downloaded content
    
    Returns:
        Path to saved file or None if failed
    """
    try:
        print(f"🌐 Downloading with requests: {url}")
        
        # Set up session with headers to mimic a real browser
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Make the request with a longer timeout
        response = session.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Generate filename from URL
        parsed_url = urlparse(url)
        filename = parsed_url.path.strip('/').replace('/', '_')
        
        # If filename is empty, use domain name
        if not filename:
            filename = parsed_url.netloc.replace('.', '_')
        
        # Add .html extension if not present
        if not filename.endswith('.html'):
            filename += '.html'
        
        # Add timestamp to make filename unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(filename)
        filename = f"{base_name}_{timestamp}{ext}"
        
        # Full path for the file
        filepath = os.path.join(cache_dir, filename)
        
        # Save the content
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Also save metadata
        metadata_file = filepath.replace('.html', '_metadata.txt')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: Requests Library\n")
            f.write(f"Status Code: {response.status_code}\n")
            f.write(f"Content Length: {len(response.content)} bytes\n")
            f.write(f"Content Type: {response.headers.get('Content-Type', 'unknown')}\n")
        
        print(f"✅ Successfully downloaded with requests!")
        print(f"   📁 Saved to: {filepath}")
        print(f"   📊 Size: {len(response.content):,} bytes")
        print(f"   📝 Metadata: {metadata_file}")
        
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None

def download_page(url, cache_dir):
    """
    Download page using best available method
    
    Args:
        url: URL to download
        cache_dir: Directory to save the downloaded content
    
    Returns:
        Path to saved file or None if failed
    """
    # Try Selenium first (better for JavaScript-heavy sites)
    if SELENIUM_AVAILABLE:
        result = download_with_selenium(url, cache_dir)
        if result:
            return result
        print("   Selenium failed, trying requests library...")
    
    # Fallback to requests
    return download_with_requests(url, cache_dir)

def download_page_fun(CACHE_FOLDER,TARGET_URL):
    
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    print(f"\n📂 Cache folder: {os.path.abspath(CACHE_FOLDER)}")
    print(f"🎯 Target URL: {TARGET_URL}\n")
    
    if not SELENIUM_AVAILABLE:
        print("ℹ️  Selenium not installed. Install it for better results:")
        print("   pip install selenium")
        print("   Also install a WebDriver (geckodriver for Firefox or chromedriver for Chrome)")
        print("   Falling back to requests library...\n")
    
    # Download the page
    result = download_page(TARGET_URL, CACHE_FOLDER)
    
    if result:
        print(f"\n🎉 Download complete!")
        print(f"   Check the cache folder: {os.path.abspath(CACHE_FOLDER)}")
    else:
        print(f"\n❌ Download failed!")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   1. Install Selenium: pip install selenium")
        print(f"   2. Install geckodriver (Firefox) or chromedriver (Chrome)")
        print(f"      macOS: brew install geckodriver")
        print(f"   3. Check your internet connection")
        print(f"   4. TripAdvisor may be blocking automated requests")
    
    print("\n" + "=" * 80)
