#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripAdvisor Cached Multi-URL Sentiment Analysis
- Downloads and caches TripAdvisor pages locally
- Works offline with cached content when live scraping fails
- Supports multiple retry strategies and user agents
- Provides comprehensive sentiment analysis on cached data

Created on Thu Oct 31 2025
@author: andreyvlasenko
"""

import os
import pandas as pd
import numpy as np
import time
import json
import random
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse, quote
from collections import Counter
import hashlib
from datetime import datetime, timedelta
import pickle
import gzip

print("✅ Basic imports loaded successfully...")

# Delay heavy imports until needed
def load_ml_libraries():
    """Load heavy ML libraries only when needed"""
    print("🔄 Loading machine learning libraries...")
    global pipeline, TfidfVectorizer, cosine_similarity, KMeans, plt, sns, WordCloud
    
    from transformers import pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    
    print("✅ ML libraries loaded successfully!")
    return True

print("✅ Script initialized - ML libraries will load when needed...")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Cache settings
CACHE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/tripadvisor_cache"
CACHE_EXPIRY_HOURS = 24           # Cache expires after 24 hours
MAX_CACHE_SIZE_MB = 500           # Maximum cache size in MB
ENABLE_COMPRESSION = True         # Compress cached files

# Scraping parameters
MAX_REVIEWS_PER_PAGE = 10         # Reviews per page (TripAdvisor pagination)
MAX_PAGES = 5                     # Maximum pages to scrape per URL
MIN_REVIEW_LENGTH = 10            # Minimum characters for a valid review
REQUEST_DELAY = 5                 # Delay between requests (seconds) - increased for caching
MAX_RETRIES = 3                   # Maximum retry attempts per URL
RETRY_DELAY = 10                  # Delay between retries (seconds)

# Processing parameters  
BATCH_SIZE = 50                   # Batch size for sentiment analysis processing
CONFIDENCE_THRESHOLD = 0.8        # Confidence threshold for 3-class simulation

# Vector search parameters
N_REPRESENTATIVES = 10            # Number of representative reviews to find per sentiment
TFIDF_MAX_FEATURES = 1000        # Maximum features for TF-IDF vectorization
TFIDF_MIN_DF = 2                 # Minimum document frequency for TF-IDF
TFIDF_MAX_DF = 0.8               # Maximum document frequency for TF-IDF

# File paths
HF_CACHE_DIR = "/tmp/hf_cache"
MODEL_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/tripadvisor_cached_analysis"

# Multiple user agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_headers(rotate=True):
    """Get headers with optional user agent rotation"""
    user_agent = random.choice(USER_AGENTS) if rotate else USER_AGENTS[0]
    
    return {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'Sec-CH-UA': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-CH-UA-Mobile': '?0',
        'Sec-CH-UA-Platform': '"macOS"'
    }

# ============================================================================

def setup_environment():
    """Setup HuggingFace cache, output directories, and page cache"""
    # Set HuggingFace cache directory
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_HOME"] = HF_CACHE_DIR
    
    # Create output directories
    folders = {
        'positive': os.path.join(OUTPUT_BASE_DIR, 'positive'),
        'negative': os.path.join(OUTPUT_BASE_DIR, 'negative'), 
        'neutral': os.path.join(OUTPUT_BASE_DIR, 'neutral'),
        'visualizations': os.path.join(OUTPUT_BASE_DIR, 'visualizations'),
        'raw_data': os.path.join(OUTPUT_BASE_DIR, 'raw_data'),
        'cache': CACHE_DIR
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders

def get_cache_filename(url):
    """Generate a cache filename for a URL"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    safe_name = re.sub(r'[^\w\-_.]', '_', url.split('/')[-1])[:50]
    filename = f"{safe_name}_{url_hash}"
    
    if ENABLE_COMPRESSION:
        filename += ".gz"
    else:
        filename += ".html"
    
    return os.path.join(CACHE_DIR, filename)

def save_to_cache(url, content, metadata=None):
    """Save page content to cache with metadata"""
    cache_file = get_cache_filename(url)
    
    cache_data = {
        'url': url,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    try:
        if ENABLE_COMPRESSION:
            with gzip.open(cache_file, 'wb') as f:
                f.write(pickle.dumps(cache_data))
        else:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Cached page: {os.path.basename(cache_file)}")
        return True
    except Exception as e:
        print(f"   ❌ Cache save error: {e}")
        return False

def load_from_cache(url, max_age_hours=CACHE_EXPIRY_HOURS):
    """Load page content from cache if available and not expired"""
    cache_file = get_cache_filename(url)
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        if ENABLE_COMPRESSION:
            with gzip.open(cache_file, 'rb') as f:
                cache_data = pickle.loads(f.read())
        else:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        
        # Check if cache is expired
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            print(f"   ⏰ Cache expired for: {os.path.basename(cache_file)}")
            return None
        
        print(f"   📁 Loaded from cache: {os.path.basename(cache_file)}")
        return cache_data['content']
        
    except Exception as e:
        print(f"   ❌ Cache load error: {e}")
        return None

def get_cache_stats():
    """Get cache statistics"""
    if not os.path.exists(CACHE_DIR):
        return {'files': 0, 'size_mb': 0, 'oldest': None, 'newest': None}
    
    files = []
    total_size = 0
    
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            files.append({
                'name': filename,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
            total_size += stat.st_size
    
    if not files:
        return {'files': 0, 'size_mb': 0, 'oldest': None, 'newest': None}
    
    files.sort(key=lambda x: x['modified'])
    
    return {
        'files': len(files),
        'size_mb': round(total_size / (1024 * 1024), 2),
        'oldest': files[0]['modified'].strftime('%Y-%m-%d %H:%M'),
        'newest': files[-1]['modified'].strftime('%Y-%m-%d %H:%M')
    }

def clean_cache():
    """Clean cache if it exceeds size limit"""
    stats = get_cache_stats()
    
    if stats['size_mb'] > MAX_CACHE_SIZE_MB:
        print(f"\n🧹 Cache size ({stats['size_mb']} MB) exceeds limit ({MAX_CACHE_SIZE_MB} MB)")
        print("   Cleaning oldest files...")
        
        files = []
        for filename in os.listdir(CACHE_DIR):
            filepath = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x['modified'])
        
        removed_size = 0
        removed_count = 0
        target_size = MAX_CACHE_SIZE_MB * 0.8 * 1024 * 1024  # Clean to 80% of limit
        
        for file_info in files:
            if removed_size >= (stats['size_mb'] * 1024 * 1024 - target_size):
                break
            
            try:
                os.remove(file_info['path'])
                removed_size += file_info['size']
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️  Could not remove {file_info['path']}: {e}")
        
        print(f"   ✅ Removed {removed_count} files ({removed_size / (1024 * 1024):.1f} MB)")

def download_page_with_cache(url, session, force_refresh=False):
    """Download a page with caching support and retry logic"""
    print(f"\n🌐 Fetching: {url}")
    
    # Check cache first (unless forced refresh)
    if not force_refresh:
        cached_content = load_from_cache(url)
        if cached_content:
            return cached_content, True  # Return content and cache_hit flag
    
    # Try to download with retries
    for attempt in range(MAX_RETRIES):
        try:
            print(f"   🔄 Download attempt {attempt + 1}/{MAX_RETRIES}")
            
            # Rotate user agent for each attempt
            session.headers.update(get_headers(rotate=True))
            
            response = session.get(url, timeout=20)
            
            if response.status_code == 200:
                content = response.text
                
                # Save to cache
                metadata = {
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'download_time': datetime.now().isoformat()
                }
                save_to_cache(url, content, metadata)
                
                print(f"   ✅ Downloaded successfully ({len(content)} chars)")
                return content, False  # Return content and cache_hit flag
                
            elif response.status_code == 403:
                print(f"   🚫 Access denied (403) - TripAdvisor blocked request")
                if attempt < MAX_RETRIES - 1:
                    print(f"   ⏳ Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print("   ❌ All attempts failed - TripAdvisor blocking detected")
                    return None, False
                    
            else:
                print(f"   ⚠️  HTTP {response.status_code}: {response.reason}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return None, False
                    
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timeout on attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
        except requests.exceptions.RequestException as e:
            print(f"   🌐 Network error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            print(f"   ❌ Unexpected error on attempt {attempt + 1}: {e}")
            break
    
    print("   ❌ Download failed after all attempts")
    return None, False

def extract_reviews_from_html(html_content, source_url):
    """Extract reviews from cached HTML content"""
    if not html_content:
        return []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        reviews = []
        
        # Find review containers (multiple selectors for different TripAdvisor layouts)
        review_containers = (
            soup.find_all('div', {'data-test-target': 'review-container'}) or
            soup.find_all('div', class_=re.compile(r'review.*container', re.I)) or
            soup.find_all('div', class_=re.compile(r'.*review.*', re.I)) or
            soup.find_all('div', attrs={'data-reviewid': True}) or
            soup.select('[data-automation="reviewCard"]')
        )
        
        if not review_containers:
            # Try broader search for any div containing review-like content
            potential_containers = soup.find_all('div')
            review_containers = [
                div for div in potential_containers 
                if any(keyword in div.get_text().lower() for keyword in ['reviewed', 'rating', 'stayed', 'visited'])
                and len(div.get_text().strip()) > MIN_REVIEW_LENGTH
            ][:20]  # Limit to first 20 potential containers
        
        print(f"   📝 Found {len(review_containers)} potential review containers")
        
        for i, container in enumerate(review_containers):
            try:
                # Extract review text (multiple possible selectors)
                review_text = None
                
                # Try different selectors for review text
                text_selectors = [
                    'span[data-test-target="review-body"]',
                    '[data-test-target="review-body"]',
                    '.review-text',
                    '.reviewText',
                    '[class*="review"] span',
                    'p[data-test-target="review-body"]',
                    'div[data-test-target="review-body"]',
                    '.partial_entry',
                    '.entry',
                    '[class*="text"] span',
                    'span[class*="text"]'
                ]
                
                for selector in text_selectors:
                    text_element = container.select_one(selector)
                    if text_element:
                        review_text = text_element.get_text(strip=True)
                        if len(review_text) > MIN_REVIEW_LENGTH:
                            break
                
                # If specific selectors fail, try to find any substantial text
                if not review_text or len(review_text) < MIN_REVIEW_LENGTH:
                    all_text = container.get_text(strip=True)
                    # Look for sentences that might be reviews
                    sentences = re.split(r'[.!?]+', all_text)
                    substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
                    if substantial_sentences:
                        review_text = '. '.join(substantial_sentences[:3])  # Take first 3 substantial sentences
                
                if not review_text or len(review_text) < MIN_REVIEW_LENGTH:
                    continue
                
                # Extract rating (if available)
                rating = None
                rating_selectors = [
                    '[class*="rating"] [class*="bubble"]',
                    '.ui_bubble_rating',
                    '[aria-label*="bubble"]',
                    '[class*="rating"]',
                    '[data-test-target="review-rating"]'
                ]
                
                for selector in rating_selectors:
                    rating_element = container.select_one(selector)
                    if rating_element:
                        # Extract rating from class name, aria-label, or text
                        classes = ' '.join(rating_element.get('class', []))
                        aria_label = rating_element.get('aria-label', '')
                        text_content = rating_element.get_text()
                        
                        # Look for rating patterns
                        for text in [classes, aria_label, text_content]:
                            rating_match = re.search(r'(\d+)(?:\s*out\s*of\s*\d+|\s*stars?|\s*/\s*\d+)?', text)
                            if rating_match:
                                rating = int(rating_match.group(1))
                                if 1 <= rating <= 5:  # Valid rating range
                                    break
                        if rating:
                            break
                
                # Extract reviewer name (optional)
                reviewer = "Anonymous"
                name_selectors = [
                    '.reviewer-name',
                    '[data-test-target="reviewer-name"]',
                    '.username',
                    '.memberOverlayLink',
                    '[class*="member"]',
                    '[class*="user"]'
                ]
                
                for selector in name_selectors:
                    name_element = container.select_one(selector)
                    if name_element:
                        reviewer_text = name_element.get_text(strip=True)
                        if reviewer_text and len(reviewer_text) < 50:  # Reasonable name length
                            reviewer = reviewer_text
                            break
                
                # Extract date (optional)
                review_date = None
                date_selectors = [
                    '[class*="date"]',
                    '.ratingDate',
                    '[data-test-target="review-date"]'
                ]
                
                for selector in date_selectors:
                    date_element = container.select_one(selector)
                    if date_element:
                        date_text = date_element.get_text(strip=True)
                        if date_text:
                            review_date = date_text
                            break
                
                review_data = {
                    'text': review_text,
                    'rating': rating,
                    'reviewer': reviewer,
                    'date': review_date,
                    'source_url': source_url,
                    'extraction_method': 'cached_html'
                }
                
                reviews.append(review_data)
                
            except Exception as e:
                print(f"      ⚠️  Error parsing review {i+1}: {e}")
                continue
        
        print(f"   ✅ Extracted {len(reviews)} reviews from cached content")
        return reviews
        
    except Exception as e:
        print(f"   ❌ Error parsing HTML content: {e}")
        return []

def load_sentiment_model():
    """Load the pre-trained sentiment analysis model with progress tracking"""
    print("\n🤖 Loading sentiment analysis model...")
    
    # Check if model exists first
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("❌ Model not found. Please run the basic script first to download the model.")
        print("   Run: python request_simple.py")
        return None
    
    print("📂 Found existing DistilBERT model...")
    
    # Load ML libraries with progress
    print("🔄 Loading transformers library (this may take 10-15 seconds)...")
    try:
        if not load_ml_libraries():
            return None
        
        print("🔗 Initializing model pipeline...")
        pipe = pipeline("sentiment-analysis", model=MODEL_PATH, return_all_scores=False)
        print("✅ Model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def analyze_sentiment_enhanced(text, pipe):
    """Enhanced sentiment analysis with 3-class simulation"""
    result = pipe(text)
    raw_label = result[0]['label']
    confidence = result[0]['score']
    
    # 3-class simulation with confidence thresholds
    if raw_label == "POSITIVE":
        if confidence > CONFIDENCE_THRESHOLD:
            readable_label = "POSITIVE"
        else:
            readable_label = "NEUTRAL"
    else:  # NEGATIVE
        if confidence > CONFIDENCE_THRESHOLD:
            readable_label = "NEGATIVE"
        else:
            readable_label = "NEUTRAL"
    
    return {
        'sentiment': readable_label,
        'confidence': confidence,
        'raw_label': raw_label
    }

def create_sample_data_for_tokyo():
    """Create realistic sample data for Tokyo attractions"""
    print("\n🎌 Creating sample Tokyo attraction reviews for demonstration...")
    
    tokyo_reviews = [
        # Tokyo Skytree
        {"text": "Tokyo Skytree is absolutely incredible! The views from the top are breathtaking and you can see all of Tokyo. The elevators are fast and the observation decks are well designed. A must-visit!", "rating": 5, "reviewer": "TokyoExplorer", "location": "Tokyo Skytree"},
        {"text": "Very crowded and expensive. Had to wait 2 hours in line and the ticket prices are outrageous. The view is nice but not worth the hassle and cost.", "rating": 2, "reviewer": "BudgetTraveler", "location": "Tokyo Skytree"},
        {"text": "Amazing experience especially at sunset! The tower is beautiful and the views are spectacular. I recommend booking tickets in advance to avoid crowds.", "rating": 4, "reviewer": "SkylineAddict", "location": "Tokyo Skytree"},
        
        # Senso-ji Temple
        {"text": "Beautiful traditional temple in the heart of Asakusa. The approach through Nakamise Street is wonderful with lots of traditional snacks and souvenirs. Very spiritual atmosphere.", "rating": 5, "reviewer": "CultureSeeker", "location": "Senso-ji Temple"},
        {"text": "Too touristy and crowded. Hard to appreciate the temple with so many people taking photos. The surrounding shops are overpriced tourist traps.", "rating": 2, "reviewer": "AuthenticitySeeker", "location": "Senso-ji Temple"},
        {"text": "Historic temple with beautiful architecture. The incense ceremony is interesting to watch. Best visited early morning to avoid crowds.", "rating": 4, "reviewer": "TempleFan", "location": "Senso-ji Temple"},
        
        # Shibuya Crossing
        {"text": "Iconic Tokyo experience! The organized chaos of thousands of people crossing at once is amazing to witness. Great photo opportunities from the surrounding buildings.", "rating": 4, "reviewer": "UrbanPhotographer", "location": "Shibuya Crossing"},
        {"text": "Just a busy intersection with lots of people. Don't understand the hype. It's crowded, noisy, and there's nothing special about it except the crowds.", "rating": 2, "reviewer": "Unimpressed", "location": "Shibuya Crossing"},
        {"text": "Fascinating example of Japanese efficiency and order. Even with thousands of people, everyone moves smoothly. The energy is incredible!", "rating": 5, "reviewer": "SociologyStudent", "location": "Shibuya Crossing"},
        
        # Tsukiji Fish Market
        {"text": "Early morning visit to see the tuna auctions was unforgettable! The freshest sushi I've ever had. The atmosphere is authentic and bustling. Get there before 5 AM!", "rating": 5, "reviewer": "FoodieAdventurer", "location": "Tsukiji Fish Market"},
        {"text": "Disappointed - most tourist areas have moved to Toyosu. What's left is overcrowded and touristy. The famous sushi shops have huge lines and high prices.", "rating": 3, "reviewer": "MarketVisitor", "location": "Tsukiji Fish Market"},
        
        # Tokyo Imperial Palace
        {"text": "Beautiful gardens and peaceful atmosphere in the middle of busy Tokyo. The East Gardens are particularly lovely, especially during cherry blossom season.", "rating": 4, "reviewer": "GardenLover", "location": "Tokyo Imperial Palace"},
        {"text": "Most of the palace is closed to public and you can only see it from outside. The gardens are nice but nothing extraordinary. Expected more.", "rating": 3, "reviewer": "PalaceVisitor", "location": "Tokyo Imperial Palace"},
        
        # Meiji Shrine
        {"text": "Tranquil oasis in the heart of Tokyo. Walking through the forest path to reach the shrine is meditative. Beautiful traditional architecture and peaceful atmosphere.", "rating": 5, "reviewer": "ZenSeeker", "location": "Meiji Shrine"},
        {"text": "Nice shrine but gets very crowded, especially on weekends. The forest walk is pleasant but the shrine itself is relatively simple compared to others in Japan.", "rating": 3, "reviewer": "ShrineHopper", "location": "Meiji Shrine"}
    ]
    
    # Create multiple URLs to simulate discovered links
    sample_urls = [
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320023-Reviews-Tokyo_Skytree-Sumida_Tokyo_Prefecture_Kanto.html",
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320041-Reviews-Senso_ji_Temple-Taito_Tokyo_Prefecture_Kanto.html", 
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320019-Reviews-Shibuya_Crossing-Shibuya_Tokyo_Prefecture_Kanto.html",
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320032-Reviews-Tsukiji_Fish_Market-Chuo_Tokyo_Prefecture_Kanto.html",
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320028-Reviews-Imperial_Palace-Chiyoda_Tokyo_Prefecture_Kanto.html",
        "https://www.tripadvisor.com/Attraction_Review-g298184-d320025-Reviews-Meiji_Shrine-Shibuya_Tokyo_Prefecture_Kanto.html"
    ]
    
    # Assign reviews to URLs
    all_sample_reviews = []
    url_counter = 1
    
    for i, url in enumerate(sample_urls):
        location_reviews = [r for r in tokyo_reviews if r['location'] in url.split('Reviews-')[1] if 'Reviews-' in url]
        if not location_reviews:
            # Fallback: assign reviews based on position
            start_idx = (i * 2) % len(tokyo_reviews)
            location_reviews = tokyo_reviews[start_idx:start_idx + 3]
        
        for review in location_reviews:
            review_copy = review.copy()
            review_copy.update({
                'source_url': url,
                'url_index': url_counter,
                'review_id': f'tokyo_sample_{url_counter}_{len(all_sample_reviews)}'
            })
            all_sample_reviews.append(review_copy)
        
        url_counter += 1
    
    print(f"✅ Created {len(all_sample_reviews)} sample Tokyo reviews from {len(sample_urls)} URLs")
    print(f"🏮 Simulated Tokyo attractions: Skytree, Senso-ji, Shibuya Crossing, Tsukiji Market, Imperial Palace, Meiji Shrine")
    
    return all_sample_reviews, sample_urls

def main():
    """Main function for cached TripAdvisor analysis"""
    print("=" * 80)
    print("TRIPADVISOR CACHED ANALYSIS WITH OFFLINE SUPPORT")
    print("=" * 80)
    
    # Setup
    folders = setup_environment()
    
    # Clean cache if needed
    clean_cache()
    
    # Show cache statistics
    cache_stats = get_cache_stats()
    print(f"\n📁 Cache Status:")
    print(f"   Files: {cache_stats['files']}")
    print(f"   Size: {cache_stats['size_mb']} MB")
    if cache_stats['oldest']:
        print(f"   Oldest: {cache_stats['oldest']}")
        print(f"   Newest: {cache_stats['newest']}")
    
    # Get URLs from user
    print("\n🔗 Enter TripAdvisor URLs to analyze (with caching support)")
    print("Examples for Tokyo:")
    print("  https://www.tripadvisor.com/Attraction_Review-g298184-d320023-Reviews-Tokyo_Skytree-Sumida_Tokyo.html")
    print("  https://www.tripadvisor.com/Attractions-g298184-Activities-Tokyo_Prefecture_Kanto.html")
    print("  https://www.tripadvisor.com/Hotel_Review-g298184-d301331-Reviews-Hotel_Name-Tokyo_Prefecture.html")
    print("\nEnter URLs (press Enter twice when done):")
    
    initial_urls = []
    while True:
        try:
            url = input().strip()
            if not url:  # Empty line - stop collecting URLs
                break
            if 'tripadvisor.com' in url:
                initial_urls.append(url)
                print(f"✅ Added: {url}")
            else:
                print(f"⚠️  Skipped (not a TripAdvisor URL): {url}")
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled by user")
            return
    
    if not initial_urls:
        # Demo mode with Tokyo URLs
        print("❌ No URLs provided. Running demo with Tokyo attractions...")
        initial_urls = [
            "https://www.tripadvisor.com/Attraction_Review-g298184-d320023-Reviews-Tokyo_Skytree-Sumida_Tokyo.html",
            "https://www.tripadvisor.com/Attractions-g298184-Activities-Tokyo_Prefecture_Kanto.html"
        ]
    
    print(f"\n🎯 Starting cached analysis for {len(initial_urls)} URLs...")
    
    # Create session for all requests
    session = requests.Session()
    session.headers.update(get_headers())
    
    # Process URLs with caching
    all_reviews = []
    successful_urls = []
    cache_hits = 0
    downloads = 0
    
    for i, url in enumerate(initial_urls, 1):
        print(f"\n📍 Processing URL {i}/{len(initial_urls)}: {url}")
        
        try:
            # Download or load from cache
            content, from_cache = download_page_with_cache(url, session)
            
            if from_cache:
                cache_hits += 1
            else:
                downloads += 1
            
            if content:
                # Extract reviews from HTML content
                reviews = extract_reviews_from_html(content, url)
                
                if reviews:
                    # Add metadata to reviews
                    for review in reviews:
                        review['url_index'] = i
                        # Extract location from URL
                        url_parts = url.split('Reviews-')
                        if len(url_parts) > 1:
                            location_name = url_parts[1].split('.html')[0].replace('_', ' ').replace('-', ' ')
                            review['location'] = location_name
                    
                    all_reviews.extend(reviews)
                    successful_urls.append(url)
                    print(f"   ✅ Extracted {len(reviews)} reviews")
                else:
                    print(f"   ⚠️  No reviews extracted from content")
            else:
                print(f"   ❌ Could not get content for URL")
                
        except Exception as e:
            print(f"   ❌ Error processing URL: {e}")
            continue
        
        # Add delay between URLs
        if i < len(initial_urls):
            print(f"   ⏳ Waiting {REQUEST_DELAY} seconds...")
            time.sleep(REQUEST_DELAY)
    
    # If no reviews found (TripAdvisor blocking), use sample data
    if not all_reviews:
        print("\n🚫 TripAdvisor blocked all requests. Using sample Tokyo data for demonstration...")
        all_reviews, successful_urls = create_sample_data_for_tokyo()
        cache_hits = 0
        downloads = 0
    
    print(f"\n📊 Collection Summary:")
    print(f"   Total reviews: {len(all_reviews)}")
    print(f"   Successful URLs: {len(successful_urls)}")
    print(f"   Cache hits: {cache_hits}")
    print(f"   Downloads: {downloads}")
    
    # Load sentiment model
    pipe = load_sentiment_model()
    if pipe is None:
        print("❌ Failed to load sentiment analysis model")
        return
    
    # Convert to DataFrame
    df_reviews = pd.DataFrame(all_reviews)
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_data_file = os.path.join(folders['raw_data'], f'cached_analysis_{timestamp}.json')
    
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    
    df_reviews.to_csv(os.path.join(folders['raw_data'], f'cached_analysis_{timestamp}.csv'), index=False)
    
    # Save URLs info
    urls_file = os.path.join(folders['raw_data'], f'processed_urls_{timestamp}.json')
    with open(urls_file, 'w', encoding='utf-8') as f:
        json.dump({
            'initial_urls': initial_urls,
            'successful_urls': successful_urls,
            'cache_stats': {
                'hits': cache_hits,
                'downloads': downloads,
                'total_files': cache_stats['files'],
                'cache_size_mb': cache_stats['size_mb']
            },
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n🧠 Processing {len(df_reviews)} reviews with DistilBERT...")
    
    # Process reviews with sentiment analysis
    all_results = []
    
    for i in range(0, len(df_reviews), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(df_reviews))
        batch = df_reviews.iloc[i:batch_end]
        
        print(f"   Processing batch {i//BATCH_SIZE + 1} ({i+1}-{batch_end})...")
        
        for _, row in batch.iterrows():
            sentiment_result = analyze_sentiment_enhanced(row['text'], pipe)
            
            result = row.to_dict()
            result.update(sentiment_result)
            all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save processed results
    results_df.to_csv(os.path.join(folders['raw_data'], f'sentiment_results_{timestamp}.csv'), index=False)
    
    # Organize by sentiment
    sentiment_counts = {}
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        sentiment_counts[sentiment] = len(sentiment_data)
        
        if len(sentiment_data) > 0:
            output_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_reviews_{timestamp}.json')
            json_data = sentiment_data.to_dict('records')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 {sentiment}: {len(sentiment_data)} reviews saved")
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("✅ CACHED TRIPADVISOR ANALYSIS COMPLETE!")
    print("=" * 80)
    
    total_reviews = len(results_df)
    print(f"""
📊 Analysis Summary:
   • URLs processed: {len(initial_urls)}
   • Reviews analyzed: {total_reviews:,}
   • Cache hits: {cache_hits} | Downloads: {downloads}
   • Positive: {sentiment_counts.get('POSITIVE', 0):,} ({sentiment_counts.get('POSITIVE', 0)/total_reviews*100:.1f}%)
   • Negative: {sentiment_counts.get('NEGATIVE', 0):,} ({sentiment_counts.get('NEGATIVE', 0)/total_reviews*100:.1f}%)
   • Neutral: {sentiment_counts.get('NEUTRAL', 0):,} ({sentiment_counts.get('NEUTRAL', 0)/total_reviews*100:.1f}%)

📁 Results saved to: {OUTPUT_BASE_DIR}
💾 Cache directory: {CACHE_DIR}
""")
    
    # Show location breakdown
    if 'location' in results_df.columns:
        print("🏮 Reviews by Location:")
        location_counts = results_df['location'].value_counts()
        for i, (location, count) in enumerate(location_counts.head(10).items(), 1):
            sentiment_breakdown = results_df[results_df['location'] == location]['sentiment'].value_counts()
            pos = sentiment_breakdown.get('POSITIVE', 0)
            neg = sentiment_breakdown.get('NEGATIVE', 0) 
            neu = sentiment_breakdown.get('NEUTRAL', 0)
            print(f"   {i}. {location}: {count} reviews")
            print(f"      ✅ {pos} pos | ❌ {neg} neg | ⚖️ {neu} neutral")
    
    # Show cache efficiency
    print(f"\n💾 Cache Efficiency:")
    if cache_hits + downloads > 0:
        cache_rate = cache_hits / (cache_hits + downloads) * 100
        print(f"   Cache hit rate: {cache_rate:.1f}%")
        print(f"   Data freshness: {cache_hits} cached, {downloads} fresh downloads")
    
    final_cache_stats = get_cache_stats()
    print(f"   Final cache size: {final_cache_stats['size_mb']} MB ({final_cache_stats['files']} files)")

if __name__ == "__main__":
    main()