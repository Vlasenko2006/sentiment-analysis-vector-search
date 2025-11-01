#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripAdvisor Multi-URL Sentiment Analysis
- Takes a list of TripAdvisor URLs as input
- Scrapes reviews from all provided URLs
- Applies sentiment analysis with DistilBERT
- Provides combined analysis across all locations

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
from urllib.parse import urljoin, urlparse
from collections import Counter

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

# Scraping parameters
MAX_REVIEWS_PER_PAGE = 10        # Reviews per page (TripAdvisor pagination)
MAX_PAGES = 5                    # Maximum pages to scrape per URL
MIN_REVIEW_LENGTH = 10           # Minimum characters for a valid review
REQUEST_DELAY = 3                # Delay between requests (seconds) - increased for politeness

# Processing parameters  
BATCH_SIZE = 50                  # Batch size for sentiment analysis processing
CONFIDENCE_THRESHOLD = 0.8       # Confidence threshold for 3-class simulation

# Vector search parameters
N_REPRESENTATIVES = 10           # Number of representative reviews to find per sentiment
TFIDF_MAX_FEATURES = 1000       # Maximum features for TF-IDF vectorization
TFIDF_MIN_DF = 2                # Minimum document frequency for TF-IDF
TFIDF_MAX_DF = 0.8              # Maximum document frequency for TF-IDF

# File paths
CACHE_DIR = "/tmp/hf_cache"
MODEL_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/tripadvisor_multi_analysis"

# Enhanced headers to appear more like a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0'
}

# ============================================================================

def setup_environment():
    """Setup HuggingFace cache and output directories"""
    # Set HuggingFace cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    
    # Create output directories
    folders = {
        'positive': os.path.join(OUTPUT_BASE_DIR, 'positive'),
        'negative': os.path.join(OUTPUT_BASE_DIR, 'negative'), 
        'neutral': os.path.join(OUTPUT_BASE_DIR, 'neutral'),
        'visualizations': os.path.join(OUTPUT_BASE_DIR, 'visualizations'),
        'raw_data': os.path.join(OUTPUT_BASE_DIR, 'raw_data')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders

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

def discover_related_links(base_url, session, max_related=10):
    """
    Discover related TripAdvisor links from a base URL
    This includes pagination, similar attractions, hotels in the area, etc.
    """
    print(f"\n🔍 Discovering related links from: {base_url}")
    related_links = set([base_url])  # Start with the base URL
    
    try:
        response = session.get(base_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Find pagination links (next pages of reviews)
        pagination_links = soup.find_all('a', href=re.compile(r'or=\d+'))
        for link in pagination_links[:20]:  # Limit pagination pages
            href = link.get('href', '')
            if href:
                full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                related_links.add(full_url)
        
        # 2. Find similar attractions/hotels in the same area
        similar_selectors = [
            'a[href*="/Attraction_Review-"]',
            'a[href*="/Hotel_Review-"]', 
            'a[href*="/Restaurant_Review-"]',
            'a[href*="Reviews-"]'
        ]
        
        for selector in similar_selectors:
            similar_links = soup.select(selector)
            for link in similar_links[:max_related]:
                href = link.get('href', '')
                if href and 'Reviews-' in href:
                    full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                    # Only add if it's a different place (different ID)
                    if full_url != base_url and len(related_links) < max_related + 10:
                        related_links.add(full_url)
        
        # 3. Find "Things to Do" or "Hotels" or "Restaurants" in the same city
        city_links = soup.find_all('a', href=re.compile(r'(Activities|Hotels|Restaurants)-g\d+'))
        for link in city_links[:5]:  # Limit city-wide categories
            href = link.get('href', '')
            if href:
                full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                related_links.add(full_url)
        
        # 4. Find nearby attractions from the same location
        nearby_selectors = [
            'a[href*="Attractions-g"]',
            'a[href*="Activities-g"]'
        ]
        
        for selector in nearby_selectors:
            nearby_links = soup.select(selector)
            for link in nearby_links[:3]:
                href = link.get('href', '')
                if href:
                    full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                    related_links.add(full_url)
        
        related_list = list(related_links)
        print(f"   ✅ Found {len(related_list)} related links")
        
        # Show discovered links (limited for display)
        for i, link in enumerate(related_list[:10], 1):
            link_type = "Reviews" if "Review-" in link else "Listing"
            print(f"   {i}. {link_type}: {link.split('/')[-1]}")
        
        if len(related_list) > 10:
            print(f"   ... and {len(related_list) - 10} more links")
            
        return related_list[:max_related + 5]  # Limit total links to process
        
    except Exception as e:
        print(f"   ❌ Error discovering related links: {e}")
        return [base_url]  # Return at least the original URL

def extract_from_listing_page(listing_url, session, max_items=5):
    """
    Extract individual review pages from a listing page (like city attractions page)
    """
    print(f"\n📋 Extracting individual pages from listing: {listing_url}")
    individual_urls = []
    
    try:
        response = session.get(listing_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find individual attraction/hotel/restaurant links
        item_selectors = [
            'a[href*="/Attraction_Review-"]',
            'a[href*="/Hotel_Review-"]',
            'a[href*="/Restaurant_Review-"]'
        ]
        
        for selector in item_selectors:
            items = soup.select(selector)
            for item in items[:max_items]:
                href = item.get('href', '')
                if href and 'Reviews-' in href:
                    full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                    if full_url not in individual_urls:
                        individual_urls.append(full_url)
        
        print(f"   ✅ Found {len(individual_urls)} individual review pages")
        return individual_urls
        
    except Exception as e:
        print(f"   ❌ Error extracting from listing page: {e}")
        return []

def extract_tripadvisor_reviews(url, max_pages=MAX_PAGES):
    """
    Extract reviews from a TripAdvisor page
    Note: This is a basic scraper. TripAdvisor has anti-scraping measures.
    Consider using their API or respecting robots.txt for production use.
    """
    print(f"\n🕷️  Scraping reviews from: {url}")
    
    all_reviews = []
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for page in range(max_pages):
        try:
            print(f"   📄 Scraping page {page + 1}/{max_pages}...")
            
            # Build page URL (TripAdvisor pagination)
            if page == 0:
                page_url = url
            else:
                # TripAdvisor uses offset-based pagination
                offset = page * 10  # Usually 10 reviews per page
                if "?" in url:
                    page_url = f"{url}&or={offset}"
                else:
                    page_url = f"{url}?or={offset}"
            
            response = session.get(page_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find review containers (TripAdvisor structure may change)
            # These selectors are examples and may need updating
            review_containers = soup.find_all('div', {'data-test-target': 'review-container'}) or \
                              soup.find_all('div', class_=re.compile(r'review.*container', re.I)) or \
                              soup.find_all('div', class_=re.compile(r'.*review.*', re.I))
            
            if not review_containers:
                print(f"   ⚠️  No review containers found on page {page + 1}")
                break
            
            page_reviews = []
            for container in review_containers:
                try:
                    # Extract review text (multiple possible selectors)
                    review_text = None
                    
                    # Try different selectors for review text
                    text_selectors = [
                        'span[data-test-target="review-body"]',
                        '.review-text',
                        '[class*="review"] span',
                        'p[data-test-target="review-body"]',
                        'div[data-test-target="review-body"]'
                    ]
                    
                    for selector in text_selectors:
                        text_element = container.select_one(selector)
                        if text_element:
                            review_text = text_element.get_text(strip=True)
                            break
                    
                    if not review_text or len(review_text) < MIN_REVIEW_LENGTH:
                        continue
                    
                    # Extract rating (if available)
                    rating = None
                    rating_selectors = [
                        '[class*="rating"] [class*="bubble"]',
                        '.ui_bubble_rating',
                        '[aria-label*="bubble"]'
                    ]
                    
                    for selector in rating_selectors:
                        rating_element = container.select_one(selector)
                        if rating_element:
                            # Extract rating from class name or aria-label
                            classes = ' '.join(rating_element.get('class', []))
                            aria_label = rating_element.get('aria-label', '')
                            
                            # Look for rating patterns
                            rating_match = re.search(r'(\d+)', classes + ' ' + aria_label)
                            if rating_match:
                                rating = int(rating_match.group(1))
                                break
                    
                    # Extract reviewer name (optional)
                    reviewer = "Anonymous"
                    name_selectors = [
                        '.reviewer-name',
                        '[data-test-target="reviewer-name"]',
                        '.username'
                    ]
                    
                    for selector in name_selectors:
                        name_element = container.select_one(selector)
                        if name_element:
                            reviewer = name_element.get_text(strip=True)
                            break
                    
                    review_data = {
                        'text': review_text,
                        'rating': rating,
                        'reviewer': reviewer,
                        'page': page + 1,
                        'source_url': page_url
                    }
                    
                    page_reviews.append(review_data)
                    
                except Exception as e:
                    print(f"      ⚠️  Error parsing review: {e}")
                    continue
            
            all_reviews.extend(page_reviews)
            print(f"   ✅ Found {len(page_reviews)} reviews on page {page + 1}")
            
            # Break if no reviews found (end of pages)
            if len(page_reviews) == 0:
                print(f"   📍 No more reviews found. Stopping at page {page + 1}")
                break
            
            # Delay between requests
            if page < max_pages - 1:
                print(f"   ⏳ Waiting {REQUEST_DELAY} seconds before next page...")
                time.sleep(REQUEST_DELAY)
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error on page {page + 1}: {e}")
            break
        except Exception as e:
            print(f"   ❌ Error on page {page + 1}: {e}")
            break
    
    print(f"\n✅ Scraping complete! Found {len(all_reviews)} total reviews")
    return all_reviews

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

def create_text_vectors(texts, method='tfidf'):
    """Create vector representations of texts"""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF
        )
        vectors = vectorizer.fit_transform(texts)
        return vectors, vectorizer

def find_representative_reviews(sentiment_data, n_representatives=None):
    """Find most representative reviews using clustering and centroids"""
    if n_representatives is None:
        n_representatives = N_REPRESENTATIVES
        
    if len(sentiment_data) < n_representatives:
        return sentiment_data
    
    texts = sentiment_data['text'].tolist()
    
    # Create TF-IDF vectors
    vectors, vectorizer = create_text_vectors(texts)
    
    # Use K-means clustering to find representative examples
    n_clusters = min(n_representatives, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors.toarray())
    
    representatives = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_vectors = vectors[cluster_indices]
        
        # Find the review closest to cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
        closest_idx = cluster_indices[np.argmax(distances)]
        
        representative = sentiment_data.iloc[closest_idx].copy()
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(cluster_indices)
        representatives.append(representative)
    
    return pd.DataFrame(representatives)

def create_visualizations(results_df, folders):
    """Create comprehensive visualizations"""
    print("\n📊 Creating visualizations...")
    
    # Setup matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    sentiment_counts = results_df['sentiment'].value_counts().to_dict()
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(15, 10))
    
    # Pie chart
    plt.subplot(2, 3, 1)
    sizes = [sentiment_counts.get(s, 0) for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']]
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=explode, shadow=True)
    plt.title('TripAdvisor Multi-URL Sentiment Distribution', fontsize=12, fontweight='bold')
    
    # 2. Confidence Distribution
    plt.subplot(2, 3, 2)
    confidences = results_df['confidence'].values
    plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(confidences.mean(), color='red', linestyle='--', 
               label=f'Mean: {confidences.mean():.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    
    # 3. Rating vs Sentiment (if ratings available)
    if 'rating' in results_df.columns and results_df['rating'].notna().any():
        plt.subplot(2, 3, 3)
        rating_sentiment = pd.crosstab(results_df['rating'], results_df['sentiment'])
        rating_sentiment.plot(kind='bar', stacked=True, ax=plt.gca(), 
                            color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.title('Rating vs Predicted Sentiment')
        plt.xlabel('TripAdvisor Rating')
        plt.ylabel('Number of Reviews')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=0)
    
    # 4. Review Length Distribution
    plt.subplot(2, 3, 4)
    review_lengths = results_df['text'].str.len()
    plt.hist(review_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Review Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Review Length Distribution')
    
    # 5. Sentiment by Source URL
    if 'source_url' in results_df.columns:
        plt.subplot(2, 3, 5)
        url_sentiment = pd.crosstab(results_df['url_index'], results_df['sentiment'])
        url_sentiment.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.title('Sentiment by URL')
        plt.xlabel('URL Index')
        plt.ylabel('Number of Reviews')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=0)
    
    # 6. Reviews per URL
    plt.subplot(2, 3, 6)
    if 'url_index' in results_df.columns:
        url_counts = results_df['url_index'].value_counts().sort_index()
        plt.bar(url_counts.index, url_counts.values, alpha=0.7, color='orange')
        plt.xlabel('URL Index')
        plt.ylabel('Number of Reviews')
        plt.title('Reviews Collected per URL')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['visualizations'], 'tripadvisor_multi_url_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return sentiment_counts

def main():
    """Main function to run TripAdvisor sentiment analysis on URLs and their related links"""
    print("=" * 80)
    print("TRIPADVISOR COMPREHENSIVE LINK CRAWLER & SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Setup
    folders = setup_environment()
    
    # Get URLs from user
    print("\n🔗 Enter TripAdvisor URLs to analyze (the script will discover related links)")
    print("Examples:")
    print("  https://www.tripadvisor.com/Hotel_Review-g123456-d123456-Reviews-Hotel_Name.html")
    print("  https://www.tripadvisor.com/Attractions-g187147-Activities-Paris_Ile_de_France.html")
    print("  https://www.tripadvisor.com/Restaurant_Review-g123456-d123456-Reviews-Restaurant_Name.html")
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
        print("❌ No valid TripAdvisor URLs provided. Exiting.")
        return
    
    print(f"\n🎯 Starting comprehensive analysis from {len(initial_urls)} initial URLs...")
    
    # Create session for all requests
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Discover all related links
    all_urls_to_process = []
    
    for i, base_url in enumerate(initial_urls, 1):
        print(f"\n🕷️  Phase 1: Discovering links from URL {i}/{len(initial_urls)}")
        print(f"   Base URL: {base_url}")
        
        # Check if this is a listing page or individual review page
        if any(keyword in base_url for keyword in ['Activities-', 'Hotels-', 'Restaurants-', 'Attractions-']):
            print("   📋 Detected listing page - extracting individual review pages...")
            individual_pages = extract_from_listing_page(base_url, session, max_items=8)
            
            # For each individual page, discover its related links
            for page_url in individual_pages:
                related_links = discover_related_links(page_url, session, max_related=5)
                all_urls_to_process.extend(related_links)
        else:
            print("   📄 Detected individual review page - discovering related links...")
            related_links = discover_related_links(base_url, session, max_related=10)
            all_urls_to_process.extend(related_links)
        
        # Add delay between base URLs
        if i < len(initial_urls):
            print(f"   ⏳ Waiting {REQUEST_DELAY} seconds before next URL...")
            time.sleep(REQUEST_DELAY)
    
    # Remove duplicates and filter for review pages only
    unique_review_urls = []
    seen_urls = set()
    
    for url in all_urls_to_process:
        if url not in seen_urls and any(keyword in url for keyword in ['Review-', 'Reviews-']):
            unique_review_urls.append(url)
            seen_urls.add(url)
    
    print(f"\n📊 Phase 2: Found {len(unique_review_urls)} unique review pages to analyze")
    
    if not unique_review_urls:
        print("❌ No review pages found. The URLs might be invalid or TripAdvisor structure has changed.")
        return
    
    # Limit total URLs to prevent overwhelming
    if len(unique_review_urls) > 20:
        print(f"⚠️  Limiting analysis to first 20 URLs (found {len(unique_review_urls)} total)")
        unique_review_urls = unique_review_urls[:20]
    
    # Collect reviews from all discovered URLs
    all_reviews = []
    successful_urls = []
    
    for i, url in enumerate(unique_review_urls, 1):
        print(f"\n📍 Phase 3: Processing review page {i}/{len(unique_review_urls)}")
        print(f"   {url}")
        
        try:
            reviews = extract_tripadvisor_reviews(url, max_pages=3)  # Limit pages per URL
            
            if reviews:
                # Add source URL info to each review
                for review in reviews:
                    review['source_url'] = url
                    review['url_index'] = i
                    # Extract location/attraction name from URL
                    url_parts = url.split('Reviews-')
                    if len(url_parts) > 1:
                        location_name = url_parts[1].split('.html')[0].replace('_', ' ')
                        review['location'] = location_name
                
                all_reviews.extend(reviews)
                successful_urls.append(url)
                print(f"   ✅ Collected {len(reviews)} reviews")
            else:
                print(f"   ❌ No reviews found (blocked or invalid URL)")
                
        except Exception as e:
            print(f"   ❌ Error processing URL: {e}")
            continue
            
        # Add delay between URLs to be polite
        if i < len(unique_review_urls):
            print(f"   ⏳ Waiting {REQUEST_DELAY} seconds before next URL...")
            time.sleep(REQUEST_DELAY)
    
    if not all_reviews:
        print("\n❌ No reviews collected from any URLs. All requests were blocked or failed.")
        print("💡 TripAdvisor has strong anti-scraping measures. Consider:")
        print("   - Using their official API")
        print("   - Adding longer delays between requests")
        print("   - Using different user agents or proxies")
        print("   - Respecting robots.txt guidelines")
        return
    
    print(f"\n📊 Phase 4: Total reviews collected: {len(all_reviews)} from {len(successful_urls)} URLs")
    
    # Load sentiment analysis model
    pipe = load_sentiment_model()
    
    if pipe is None:
        print("❌ Failed to load sentiment analysis model")
        return
    
    # Convert to DataFrame
    df_reviews = pd.DataFrame(all_reviews)
    
    # Save raw data
    raw_data_file = os.path.join(folders['raw_data'], 'comprehensive_crawl_reviews.json')
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    
    df_reviews.to_csv(os.path.join(folders['raw_data'], 'comprehensive_crawl_reviews.csv'), index=False)
    
    # Save discovered URLs list
    urls_file = os.path.join(folders['raw_data'], 'discovered_urls.json')
    with open(urls_file, 'w', encoding='utf-8') as f:
        json.dump({
            'initial_urls': initial_urls,
            'processed_urls': successful_urls,
            'total_discovered': len(all_urls_to_process),
            'total_processed': len(successful_urls)
        }, f, indent=2)
    
    print(f"\n🧠 Phase 5: Processing {len(df_reviews)} reviews with DistilBERT...")
    
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
    results_df.to_csv(os.path.join(folders['raw_data'], 'sentiment_results.csv'), index=False)
    
    # Analyze and save by sentiment
    print("\n📁 Organizing results by sentiment...")
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        
        if len(sentiment_data) > 0:
            # Save sentiment-specific data
            output_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_reviews.json')
            json_data = sentiment_data.to_dict('records')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 {sentiment}: {len(sentiment_data)} reviews saved")
            
            # Find representative reviews
            if len(sentiment_data) >= 3:  # Need at least 3 for clustering
                representatives = find_representative_reviews(sentiment_data)
                
                # Save representatives
                repr_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_representatives.json')
                representatives.to_json(repr_file, orient='records', indent=2)
                
                print(f"🎯 Found {len(representatives)} representative {sentiment} reviews")
    
    # Create visualizations
    sentiment_counts = create_visualizations(results_df, folders)
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE TRIPADVISOR ANALYSIS COMPLETE!")
    print("=" * 80)
    
    total_reviews = len(results_df)
    print(f"""
📊 Comprehensive Analysis Summary:
   • Initial URLs provided: {len(initial_urls)}
   • Total URLs discovered: {len(all_urls_to_process)}
   • Review pages processed: {len(successful_urls)}
   • Total reviews analyzed: {total_reviews:,}
   • Positive reviews: {sentiment_counts.get('POSITIVE', 0):,} ({sentiment_counts.get('POSITIVE', 0)/total_reviews*100:.1f}%)
   • Negative reviews: {sentiment_counts.get('NEGATIVE', 0):,} ({sentiment_counts.get('NEGATIVE', 0)/total_reviews*100:.1f}%)
   • Neutral reviews: {sentiment_counts.get('NEUTRAL', 0):,} ({sentiment_counts.get('NEUTRAL', 0)/total_reviews*100:.1f}%)
   
📁 Results saved to: {OUTPUT_BASE_DIR}
""")
    
    # Show breakdown by location (if available)
    if 'location' in results_df.columns:
        print("\n🏛️  Reviews by Location:")
        location_counts = results_df['location'].value_counts()
        for i, (location, count) in enumerate(location_counts.head(10).items(), 1):
            print(f"   {i}. {location}: {count} reviews")
    
    # Show sentiment breakdown by URL
    print("\n📊 Top 10 URLs by review count:")
    url_counts = results_df['source_url'].value_counts()
    for i, (url, count) in enumerate(url_counts.head(10).items(), 1):
        url_data = results_df[results_df['source_url'] == url]
        url_sentiment = url_data['sentiment'].value_counts()
        display_url = url.split('/')[-1] if len(url) > 60 else url
        
        pos = url_sentiment.get('POSITIVE', 0)
        neg = url_sentiment.get('NEGATIVE', 0)
        neu = url_sentiment.get('NEUTRAL', 0)
        
        print(f"   {i}. {display_url} ({count} reviews)")
        print(f"      ✅ Positive: {pos} ({pos/count*100:.0f}%) | ❌ Negative: {neg} ({neg/count*100:.0f}%) | ⚖️ Neutral: {neu} ({neu/count*100:.0f}%)")
    
    # Show sample representative reviews
    for sentiment in ['POSITIVE', 'NEGATIVE']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        if len(sentiment_data) > 0:
            print(f"\n🎯 Sample {sentiment} Review:")
            top_review = sentiment_data.loc[sentiment_data['confidence'].idxmax()]
            print(f"   Confidence: {top_review['confidence']:.3f}")
            print(f"   Text: {top_review['text'][:200]}...")
            if 'rating' in top_review and pd.notna(top_review['rating']):
                print(f"   Rating: {top_review['rating']}/5")
            if 'location' in top_review:
                print(f"   Location: {top_review['location']}")

if __name__ == "__main__":
    main()