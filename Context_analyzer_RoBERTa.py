#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RoBERTa Sentiment Analysis with Vector Search, Visualization, and Web Crawling
- Configurable processing of samples from filtered reviews database
- OPTIONAL: Web crawling with configurable depth for subpage analysis
- Indexes comments by sentiment into separate folders
- Uses vector search to find most representative comments
- Creates comprehensive visualizations and PDF reports

NEW WEB CRAWLING FEATURES:
- Set ENABLE_WEB_CRAWLING = True to crawl website subpages
- Configure WEB_CRAWL_DEPTH (0-5) for crawling depth
- Respectful crawling with delays and domain restrictions
- Integrates crawled content with existing comment analysis

Configuration parameters are set at the top of the file for easy customization.
Default: 3,500 samples, Web crawling disabled

Created on Thu Oct 30 2025
@author: andreyvlasenko
"""

import os
import pandas as pd
import numpy as np
import time
import json
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import sqlite3
from download_and_prepare_dataset import download_and_prepare_dataset

# Web crawling imports (NEW)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time as time_module
from collections import deque

# PDF report generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from datetime import datetime
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️ ReportLab not available. Install with: pip install reportlab")
    PDF_AVAILABLE = False

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset sampling parameters
SAMPLES_PER_CLASS = 1750          # Number of samples per sentiment class (negative/positive)
TOTAL_SAMPLES = SAMPLES_PER_CLASS * 2  # Total samples to process (3500)

# Web crawling parameters (NEW)
ENABLE_WEB_CRAWLING = True        # Enable web subpage analysis
WEB_CRAWL_DEPTH = 1              # Maximum depth for web crawling (0 = only main page, 1 = direct links, 2 = links from links, etc.)
MAX_PAGES_PER_DEPTH = 5          # Maximum pages to crawl per depth level
CRAWL_DELAY = 2.0                # Delay between requests (seconds) - be respectful to servers
ALLOWED_DOMAINS = []             # Restrict crawling to specific domains (empty = allow all)

# Custom URLs to crawl (NEW) - Set your target websites here
CUSTOM_CRAWL_URLS = [
    "https://www.tripadvisor.com/Restaurant_Review-g1066456-d1373933-Reviews-Sometaro-Asakusa_Taito_Tokyo_Tokyo_Prefecture_Kanto.html",
    # "https://www.yelp.com/biz/some-restaurant-location",
    # "https://example-review-site.com"
]

# Processing parameters  
BATCH_SIZE = 100                  # Batch size for sentiment analysis processing
CONFIDENCE_THRESHOLD = 0.8        # Confidence threshold for 3-class simulation

# Vector search parameters
N_REPRESENTATIVES = 10            # Number of representative comments to find per sentiment
TFIDF_MAX_FEATURES = 1000        # Maximum features for TF-IDF vectorization
TFIDF_MIN_DF = 4                 # Minimum document frequency for TF-IDF
TFIDF_MAX_DF = 0.8               # Maximum document frequency for TF-IDF

# Visualization parameters
TOP_WORDS_COUNT = 15             # Number of top words to show in frequency analysis
WORDCLOUD_MAX_WORDS = 100        # Maximum words in word clouds

# File paths
CACHE_DIR = "/tmp/hf_cache"
MODEL_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/sentiment_analysis"

# ============================================================================

# Set style for better plots
plt.style.use('default')  # Use default style instead of seaborn which might not be available
sns.set_palette("husl")

# Use filtered reviews database instead of sentiment140


path_db = "/Users/andreyvlasenko/tst/Request/filtered_reviews.db"
print('Database path:', path_db)

# Set HuggingFace cache directory
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

print("=" * 80)
print("ENHANCED ROBERTA SENTIMENT ANALYSIS WITH VECTOR SEARCH")
print("=" * 80)

# Initialize sentiment analysis model
print("\n🤖 Loading sentiment analysis model...")

if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    print("📂 Using existing DistilBERT model...")
    pipe = pipeline("sentiment-analysis", model=MODEL_PATH)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    simulate_3_class = True
else:
    print("❌ Model not found, please run basic script first")
    exit(1)

print(f"✅ Model loaded successfully!")

# Create output directories
folders = {
    'positive': os.path.join(OUTPUT_BASE_DIR, 'positive'),
    'negative': os.path.join(OUTPUT_BASE_DIR, 'negative'), 
    'neutral': os.path.join(OUTPUT_BASE_DIR, 'neutral'),
    'visualizations': os.path.join(OUTPUT_BASE_DIR, 'visualizations'),
    'vectors': os.path.join(OUTPUT_BASE_DIR, 'vectors')
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

print(f"📁 Created output directories in: {OUTPUT_BASE_DIR}")

# Enhanced sentiment analysis function
def analyze_sentiment_enhanced(text):
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
        'text': text,
        'sentiment': readable_label,
        'confidence': confidence,
        'raw_label': raw_label
    }

# Web Crawling Functions (NEW)
def is_valid_url(url, allowed_domains=None):
    """Check if URL is valid and within allowed domains"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        if allowed_domains and parsed.netloc not in allowed_domains:
            return False
            
        # Skip non-HTML content
        skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.mp4', '.avi', '.doc', '.docx'}
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    except:
        return False

def extract_links_from_page(url, soup):
    """Extract all valid links from a webpage"""
    links = set()
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(url, href)
        
        if is_valid_url(full_url, ALLOWED_DOMAINS):
            links.add(full_url)
    
    return links

def extract_text_content(soup):
    """Extract meaningful text content from webpage"""
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    # Get text from main content areas
    content_selectors = [
        'main', 'article', '.content', '#content', 
        '.post', '.review', '.comment', '.description'
    ]
    
    text_blocks = []
    
    # Try to find main content areas first
    for selector in content_selectors:
        elements = soup.select(selector)
        for element in elements:
            text = element.get_text(strip=True, separator=' ')
            if len(text) > 50:  # Only meaningful text blocks
                text_blocks.append(text)
    
    # If no main content found, extract from paragraphs
    if not text_blocks:
        for p in soup.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            if len(text) > 30:
                text_blocks.append(text)
    
    return text_blocks

def crawl_website_depth(start_url, max_depth=2, max_pages_per_depth=10, delay=1.0):
    """
    Crawl website to specified depth and extract text content
    
    Args:
        start_url: Starting URL
        max_depth: Maximum crawling depth (0 = only start page)
        max_pages_per_depth: Maximum pages to crawl per depth level
        delay: Delay between requests in seconds
    
    Returns:
        List of dictionaries with URL, depth, and extracted text blocks
    """
    
    print(f"\n🕷️  Starting web crawl from: {start_url}")
    print(f"   Max depth: {max_depth}, Max pages per depth: {max_pages_per_depth}")
    
    crawled_data = []
    visited_urls = set()
    
    # Queue: (url, depth)
    url_queue = deque([(start_url, 0)])
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    depth_counts = {}  # Track pages crawled per depth
    
    while url_queue:
        current_url, depth = url_queue.popleft()
        
        # Skip if already visited or depth exceeded
        if current_url in visited_urls or depth > max_depth:
            continue
        
        # Check depth limit
        if depth_counts.get(depth, 0) >= max_pages_per_depth:
            continue
        
        try:
            print(f"   Crawling [Depth {depth}]: {current_url}")
            
            response = session.get(current_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_blocks = extract_text_content(soup)
            
            if text_blocks:
                crawled_data.append({
                    'url': current_url,
                    'depth': depth,
                    'text_blocks': text_blocks,
                    'title': soup.title.string if soup.title else 'No Title'
                })
                
                print(f"     ✅ Extracted {len(text_blocks)} text blocks")
            
            visited_urls.add(current_url)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            # Extract links for next depth level (if not at max depth)
            if depth < max_depth:
                links = extract_links_from_page(current_url, soup)
                for link in list(links)[:max_pages_per_depth]:  # Limit links per page
                    if link not in visited_urls:
                        url_queue.append((link, depth + 1))
            
            # Respectful delay
            time_module.sleep(delay)
            
        except Exception as e:
            print(f"     ❌ Error crawling {current_url}: {str(e)}")
            continue
    
    print(f"✅ Web crawl complete! Crawled {len(crawled_data)} pages")
    return crawled_data

def integrate_web_data_with_db(crawled_data, db_path):
    """
    Integrate crawled web data with existing database
    """
    if not crawled_data:
        return
    
    print(f"\n💾 Integrating {len(crawled_data)} crawled pages with database...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create web_crawl_data table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS web_crawl_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                depth INTEGER NOT NULL,
                title TEXT,
                block_text TEXT NOT NULL,
                block_length INTEGER,
                crawl_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        total_blocks = 0
        
        for page_data in crawled_data:
            url = page_data['url']
            depth = page_data['depth']
            title = page_data['title']
            
            for text_block in page_data['text_blocks']:
                if len(text_block.strip()) > 30:  # Only meaningful blocks
                    cursor.execute('''
                        INSERT INTO web_crawl_data (url, depth, title, block_text, block_length)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (url, depth, title, text_block, len(text_block)))
                    total_blocks += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Integrated {total_blocks} text blocks from web crawl")
        
    except Exception as e:
        print(f"❌ Error integrating web data: {e}")

def load_combined_dataset(db_path, include_web_crawl=False):
    """
    Load dataset combining existing comment_blocks with optional web crawl data
    """
    try:
        conn = sqlite3.connect(db_path)
        
        if include_web_crawl:
            # Check if web_crawl_data table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='web_crawl_data';")
            has_web_data = cursor.fetchone() is not None
            
            if has_web_data:
                # Combine both datasets
                query = """
                    SELECT 'comment' as source, block_text as text, score, length, is_candidate, file_path as source_info
                    FROM comment_blocks 
                    WHERE score >= 0.3 AND length >= 30
                    
                    UNION ALL
                    
                    SELECT 'web_crawl' as source, block_text as text, 0.5 as score, block_length as length, 
                           0 as is_candidate, url as source_info
                    FROM web_crawl_data
                    WHERE block_length >= 30
                    
                    ORDER BY score DESC, length DESC
                """
                print("📊 Loading combined dataset (comments + web crawl data)")
            else:
                print("⚠️  No web crawl data found, using comment_blocks only")
                include_web_crawl = False
        
        if not include_web_crawl:
            # Original query - comments only
            query = """
                SELECT 'comment' as source, block_text as text, score, length, is_candidate, file_path as source_info
                FROM comment_blocks 
                WHERE score >= 0.3 AND length >= 30
                ORDER BY score DESC
            """
        
        df_dataset = pd.read_sql_query(query, conn)
        conn.close()
        
        return df_dataset
        
    except Exception as e:
        print(f"❌ Error loading combined dataset: {e}")
        return None

# Load and process samples from filtered_reviews.db
print("\n" + "=" * 80)
print("LOADING AND PROCESSING SAMPLES FROM FILTERED_REVIEWS.DB")
print("=" * 80)

print(f"\n📊 Loading dataset from: {path_db}")

# Web crawling section (NEW)
if ENABLE_WEB_CRAWLING:
    print("\n" + "=" * 80)
    print("WEB CRAWLING ENABLED - EXTRACTING ADDITIONAL CONTENT")
    print("=" * 80)
    
    # Get source URLs from existing database for crawling
    try:
        base_urls = []
        
        # Use custom URLs if provided, otherwise extract from database
        if CUSTOM_CRAWL_URLS:
            print(f"🌐 Using {len(CUSTOM_CRAWL_URLS)} custom URLs for crawling:")
            for url in CUSTOM_CRAWL_URLS:
                print(f"   • {url}")
                # Extract base URL for domain-based crawling
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                base_urls.append(url)  # Use full URL as provided
        else:
            # Original behavior - extract from database
            conn = sqlite3.connect(path_db)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM comment_blocks LIMIT 3")  # Start with a few URLs
            source_urls = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Convert file paths to base URLs for crawling
            for file_path in source_urls:
                if file_path.startswith('http'):
                    # Extract base URL
                    parsed = urlparse(file_path)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    base_urls.append(base_url)
            
            print(f"🌐 Found {len(base_urls)} URLs from database for crawling:")
            for url in base_urls:
                print(f"   • {url}")
        
        # Remove duplicates
        base_urls = list(set(base_urls))
            
        if base_urls:
            # Crawl each base URL
            all_crawled_data = []
            for base_url in base_urls[:2]:  # Limit to first 2 URLs for demo
                crawled_data = crawl_website_depth(
                    start_url=base_url,
                    max_depth=WEB_CRAWL_DEPTH,
                    max_pages_per_depth=MAX_PAGES_PER_DEPTH,
                    delay=CRAWL_DELAY
                )
                all_crawled_data.extend(crawled_data)
            
            # Integrate crawled data with database
            if all_crawled_data:
                integrate_web_data_with_db(all_crawled_data, path_db)
        else:
            print("⚠️  No valid URLs found for crawling")
            
    except Exception as e:
        print(f"❌ Error during web crawling: {e}")

try:
    # Load dataset (with or without web crawl data)
    df_dataset = load_combined_dataset(path_db, include_web_crawl=ENABLE_WEB_CRAWLING)
    
    if df_dataset is None or len(df_dataset) == 0:
        print("❌ Error: No data loaded from database")
        exit(1)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(df_dataset):,}")
    
    # Show data source distribution
    if 'source' in df_dataset.columns:
        source_counts = df_dataset['source'].value_counts()
        print(f"\n📈 Data Source Distribution:")
        for source, count in source_counts.items():
            percentage = (count / len(df_dataset)) * 100
            print(f"   {source.title()}: {count:,} samples ({percentage:.1f}%)")
    
    # Rename columns to match expected format (text column should already exist)
    if 'block_text' in df_dataset.columns:
        df_dataset.rename(columns={'block_text': 'text'}, inplace=True)
    
    # Take a subset for processing (limit to reasonable number for analysis)
    max_samples = min(len(df_dataset), TOTAL_SAMPLES)
    df_sample = df_dataset.head(max_samples).copy()
    
    # Shuffle the data
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Selected {len(df_sample)} samples for analysis!")
    
    # Show distribution by score ranges (handle both comment and web crawl data)
    print(f"\n📈 Sample Distribution by Quality Score:")
    if ENABLE_WEB_CRAWLING and 'source' in df_sample.columns:
        # Show breakdown by source
        for source in df_sample['source'].unique():
            source_data = df_sample[df_sample['source'] == source]
            print(f"\n   {source.title()} Data ({len(source_data)} samples):")
            
            if source == 'comment' and 'score' in source_data.columns:
                score_ranges = {
                    'Excellent (>0.8)': len(source_data[source_data['score'] > 0.8]),
                    'Good (0.65-0.8)': len(source_data[(source_data['score'] >= 0.65) & (source_data['score'] <= 0.8)]),
                    'Medium (0.5-0.65)': len(source_data[(source_data['score'] >= 0.5) & (source_data['score'] < 0.65)]),
                    'Low (0.3-0.5)': len(source_data[(source_data['score'] >= 0.3) & (source_data['score'] < 0.5)])
                }
                for range_name, count in score_ranges.items():
                    percentage = (count / len(source_data)) * 100 if len(source_data) > 0 else 0
                    print(f"     {range_name}: {count:,} samples ({percentage:.1f}%)")
            else:
                print(f"     Web crawled content: {len(source_data)} text blocks")
    else:
        # Original score distribution for comment data only
        if 'score' in df_sample.columns:
            score_ranges = {
                'Excellent (>0.8)': len(df_sample[df_sample['score'] > 0.8]),
                'Good (0.65-0.8)': len(df_sample[(df_sample['score'] >= 0.65) & (df_sample['score'] <= 0.8)]),
                'Medium (0.5-0.65)': len(df_sample[(df_sample['score'] >= 0.5) & (df_sample['score'] < 0.65)]),
                'Low (0.3-0.5)': len(df_sample[(df_sample['score'] >= 0.3) & (df_sample['score'] < 0.5)])
            }
            
            for range_name, count in score_ranges.items():
                percentage = (count / len(df_sample)) * 100
                print(f"   {range_name}: {count:,} samples ({percentage:.1f}%)")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Process all samples with sentiment analysis
print(f"\n🚀 Analyzing {len(df_sample)} samples...")
print("⏱️  This will take several minutes...")

all_results = []
processing_times = []

start_total = time.time()

for i in range(0, len(df_sample), BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, len(df_sample))
    batch = df_sample.iloc[i:batch_end]
    
    batch_start = time.time()
    batch_results = []
    
    for _, row in batch.iterrows():
        result = analyze_sentiment_enhanced(row['text'])
        # Add the original score from filtered_reviews.db
        result['original_score'] = row.get('score', 0.0)
        result['original_length'] = row.get('length', len(row['text']))
        result['is_candidate'] = row.get('is_candidate', False)
        batch_results.append(result)
    
    all_results.extend(batch_results)
    
    batch_time = time.time() - batch_start
    processing_times.append(batch_time)
    
    # Progress indicator
    progress = (batch_end / len(df_sample)) * 100
    avg_time = batch_time / len(batch)
    eta = ((len(df_sample) - batch_end) / len(batch)) * avg_time
    
    print(f"   Progress: {progress:5.1f}% | Batch {i//BATCH_SIZE + 1:2d} | ETA: {eta/60:.1f}m")

total_time = time.time() - start_total
print(f"\n✅ Analysis complete!")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Average per text: {total_time/len(df_sample):.3f} seconds")

# Create results dataframe
results_df = pd.DataFrame(all_results)

# Index comments by sentiment into folders
print("\n" + "=" * 80)
print("INDEXING COMMENTS BY SENTIMENT")
print("=" * 80)

sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    sentiment_data = results_df[results_df['sentiment'] == sentiment]
    sentiment_counts[sentiment] = len(sentiment_data)
    
    if len(sentiment_data) > 0:
        # Save to JSON files
        output_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_comments.json')
        
        # Convert to list of dictionaries for JSON serialization
        json_data = sentiment_data.to_dict('records')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 {sentiment}: {len(sentiment_data):,} comments saved to {output_file}")
        
        # Save sample as text file for easy reading
        sample_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_samples.txt')
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(f"{sentiment} SENTIMENT SAMPLES\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (_, row) in enumerate(sentiment_data.head(50).iterrows(), 1):
                f.write(f"{i:2d}. Confidence: {row['confidence']:.3f}\n")
                f.write(f"    Text: {row['text']}\n\n")

print(f"\n📊 Final Distribution:")
total_analyzed = sum(sentiment_counts.values())
for sentiment, count in sentiment_counts.items():
    percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
    print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")

# Vector Search Implementation
print("\n" + "=" * 80)
print("VECTOR SEARCH FOR REPRESENTATIVE COMMENTS")
print("=" * 80)

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
    
def find_representative_comments(sentiment_data, n_representatives=None):
    """Find most representative comments using clustering and centroids"""
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
        
        # Skip empty clusters
        if len(cluster_indices) == 0:
            continue
            
        cluster_vectors = vectors[cluster_indices]
        
        # Handle single item clusters
        if len(cluster_indices) == 1:
            closest_idx = cluster_indices[0]
        else:
            # Find the text closest to cluster centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
            closest_idx = cluster_indices[np.argmax(distances)]
        
        representative = sentiment_data.iloc[closest_idx].copy()
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(cluster_indices)
        representatives.append(representative)
    
    return pd.DataFrame(representatives)

def extract_source_info_from_db(db_path):
    """Extract source website information from database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unique file paths to determine source
        cursor.execute("SELECT DISTINCT file_path FROM comment_blocks LIMIT 5")
        file_paths = [row[0] for row in cursor.fetchall()]
        
        # Extract website/source info from file paths
        sources = []
        for path in file_paths:
            filename = os.path.basename(path)
            if 'senso-ji' in filename.lower() or 'asakusa' in filename.lower():
                sources.append("TripAdvisor - Senso-ji Temple, Asakusa")
            elif 'tripadvisor' in filename.lower():
                sources.append("TripAdvisor")
            else:
                # Try to extract meaningful name from filename
                clean_name = filename.replace('%20', ' ').replace('.html', '').replace('.htm', '')
                sources.append(f"Web Source: {clean_name}")
        
        conn.close()
        
        if sources:
            return sources[0]  # Return the first/main source
        else:
            return "Web Source: Unknown"
            
    except Exception as e:
        return f"Database Source: {os.path.basename(db_path)}"

def generate_pdf_report(results_df, representative_results, performance_summary, 
                       folders, db_path, total_time):
    """Generate comprehensive PDF report of the analysis"""
    
    if not PDF_AVAILABLE:
        print("❌ Cannot generate PDF report. Install reportlab: pip install reportlab")
        return None
    
    # Create PDF file path
    pdf_path = os.path.join(folders['visualizations'], 'sentiment_analysis_report.pdf')
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.Color(0, 0, 0.8)  # Dark blue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.Color(0.8, 0, 0)  # Dark red
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.Color(0, 0, 0.8)  # Blue
    )
    
    # Title page
    story.append(Paragraph("RoBERTa Sentiment Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Source information
    source_info = extract_source_info_from_db(db_path)
    story.append(Paragraph(f"<b>Data Source:</b> {source_info}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Analysis date and summary
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Comments Analyzed:</b> {len(results_df)}", styles['Normal']))
    story.append(Paragraph(f"<b>Processing Time:</b> {total_time/60:.1f} minutes", styles['Normal']))
    story.append(Paragraph(f"<b>Neural Network Model:</b> DistilBERT-based sentiment classifier", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    sentiment_dist = performance_summary.get('sentiment_distribution', {})
    total_comments = sum(sentiment_dist.values())
    
    summary_text = f"""
    This report presents a comprehensive sentiment analysis of {total_comments} user comments extracted 
    from {source_info} using advanced neural network classification. The analysis employed DistilBERT 
    for sentiment classification and TF-IDF vectorization with K-means clustering for representative 
    comment selection.
    
    <b>Key Findings:</b><br/>
    • Positive Comments: {sentiment_dist.get('POSITIVE', 0)} ({sentiment_dist.get('POSITIVE', 0)/total_comments*100:.1f}%)<br/>
    • Negative Comments: {sentiment_dist.get('NEGATIVE', 0)} ({sentiment_dist.get('NEGATIVE', 0)/total_comments*100:.1f}%)<br/>
    • Neutral Comments: {sentiment_dist.get('NEUTRAL', 0)} ({sentiment_dist.get('NEUTRAL', 0)/total_comments*100:.1f}%)<br/>
    
    The analysis utilized vector search and clustering algorithms to identify the most representative 
    comments from each sentiment category, providing insights into common themes and user experiences.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("Methodology", heading_style))
    methodology_text = """
    <b>1. Data Extraction:</b> Comments were pre-filtered using neural network-based quality scoring 
    to identify genuine user-generated content versus website boilerplate.<br/><br/>
    
    <b>2. Sentiment Classification:</b> DistilBERT model processed each comment to classify sentiment 
    as Positive, Negative, or Neutral with confidence scores.<br/><br/>
    
    <b>3. Vector Search & Clustering:</b> TF-IDF vectorization transformed text into numerical 
    representations, followed by K-means clustering to group similar comments.<br/><br/>
    
    <b>4. Representative Selection:</b> For each cluster, the comment closest to the centroid 
    was selected as the most representative example of that theme.
    """
    
    story.append(Paragraph(methodology_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add visualizations if they exist
    viz_files = ['sentiment_analysis_overview.png', 'sentiment_wordclouds.png', 'word_frequency_analysis.png']
    
    for viz_file in viz_files:
        viz_path = os.path.join(folders['visualizations'], viz_file)
        if os.path.exists(viz_path):
            story.append(Paragraph(f"Analysis Visualization: {viz_file.replace('_', ' ').title()}", subheading_style))
            try:
                # Add image with appropriate sizing
                img = Image(viz_path, width=7*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            except Exception as e:
                story.append(Paragraph(f"[Visualization not available: {str(e)}]", styles['Italic']))
    
    story.append(PageBreak())
    
    # Representative Comments Section
    story.append(Paragraph("Most Representative Comments", heading_style))
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment in representative_results and len(representative_results[sentiment]) > 0:
            story.append(Paragraph(f"{sentiment.title()} Comments", subheading_style))
            
            representatives = representative_results[sentiment].sort_values('confidence', ascending=False)
            
            # Create table for top 5 representative comments
            table_data = [['Rank', 'Confidence', 'Cluster Info', 'Comment Text']]
            
            for i, (_, row) in enumerate(representatives.head(5).iterrows(), 1):
                comment_text = str(row['text'])
                if len(comment_text) > 150:
                    comment_text = comment_text[:150] + "..."
                
                cluster_info = f"Cluster {row.get('cluster_id', 'N/A')} (Size: {row.get('cluster_size', 'N/A')})"
                
                table_data.append([
                    str(i),
                    f"{row['confidence']:.3f}",
                    cluster_info,
                    comment_text
                ])
            
            # Create and style table
            table = Table(table_data, colWidths=[0.5*inch, 1*inch, 1.5*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#808080')),  # Grey
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#F5F5DC')),   # WhiteSmoke
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5DC')), # Beige
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#000000')),    # Black
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
    
    # Technical Details
    story.append(PageBreak())
    story.append(Paragraph("Technical Details", heading_style))
    
    tech_details = f"""
    <b>Processing Configuration:</b><br/>
    • TF-IDF Features: {TFIDF_MAX_FEATURES}<br/>
    • Minimum Document Frequency: {TFIDF_MIN_DF}<br/>
    • Maximum Document Frequency: {TFIDF_MAX_DF}<br/>
    • Clusters per Sentiment: {N_REPRESENTATIVES}<br/>
    • Confidence Threshold: {CONFIDENCE_THRESHOLD}<br/><br/>
    
    <b>Performance Metrics:</b><br/>
    • Average Original Quality Score: {performance_summary.get('score_distribution', {}).get('avg_original_score', 'N/A'):.3f}<br/>
    • Average Sentiment Confidence: {performance_summary.get('score_distribution', {}).get('avg_sentiment_confidence', 'N/A'):.3f}<br/>
    • High-Quality Candidates: {performance_summary.get('score_distribution', {}).get('candidates_count', 'N/A')}<br/>
    • Processing Rate: {len(results_df)/(total_time/60):.1f} comments/minute<br/><br/>
    
    <b>Database Information:</b><br/>
    • Source Database: {os.path.basename(db_path)}<br/>
    • Total Records Processed: {len(results_df)}<br/>
    • Analysis Timestamp: {datetime.now().isoformat()}
    """
    
    story.append(Paragraph(tech_details, styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"📄 PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        return None

# Find representative comments for each sentiment
print("\n🔍 Finding most representative comments...")

representative_results = {}

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    sentiment_data = results_df[results_df['sentiment'] == sentiment]
    
    if len(sentiment_data) > 0:
        print(f"\n📝 Analyzing {sentiment} comments ({len(sentiment_data)} total)...")
        representatives = find_representative_comments(sentiment_data)
        representative_results[sentiment] = representatives
        
        print(f"✅ Found {len(representatives)} representative {sentiment} comments:")
        
        for i, (_, row) in enumerate(representatives.iterrows(), 1):
            print(f"   {i:2d}. [Cluster {row['cluster_id']} | Size: {row['cluster_size']}] "
                  f"Conf: {row['confidence']:.3f}")
            text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
            print(f"       {text_preview}")
        
        # Save representatives
        repr_file = os.path.join(folders[sentiment.lower()], f'{sentiment.lower()}_representatives.json')
        representatives.to_json(repr_file, orient='records', indent=2)

# Create Visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# 1. Sentiment Distribution Pie Chart
print("\n📊 Creating sentiment distribution chart...")

plt.figure(figsize=(12, 8))

# Pie chart
plt.subplot(2, 2, 1)
sizes = [sentiment_counts[s] for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']]
labels = ['Positive', 'Negative', 'Neutral']
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
explode = (0.05, 0.05, 0.05)

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, explode=explode, shadow=True)
plt.title('Sentiment Distribution (Filtered Reviews)', fontsize=14, fontweight='bold')

# 2. Confidence Distribution
plt.subplot(2, 2, 2)
confidences = results_df['confidence'].values
plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(confidences.mean(), color='red', linestyle='--', 
           label=f'Mean: {confidences.mean():.3f}')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution')
plt.legend()

# 3. Sentiment vs Original Quality Score (instead of accuracy visualization)
plt.subplot(2, 2, 3)
if 'original_score' in results_df.columns:
    # Create scatter plot of sentiment vs original quality score
    sentiment_colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        if len(sentiment_data) > 0:
            plt.scatter(sentiment_data['original_score'], sentiment_data['confidence'], 
                       c=sentiment_colors[sentiment], label=sentiment, alpha=0.6)
    plt.xlabel('Original Quality Score')
    plt.ylabel('Sentiment Confidence')
    plt.title('Sentiment vs Original Quality Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    # Fallback to simple sentiment distribution
    sentiment_counts_plot = results_df['sentiment'].value_counts()
    plt.bar(sentiment_counts_plot.index, sentiment_counts_plot.values, 
            color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Sentiment Distribution')
    plt.ylabel('Count')

# 4. Processing Time Analysis
plt.subplot(2, 2, 4)
batch_numbers = range(1, len(processing_times) + 1)
plt.plot(batch_numbers, processing_times, marker='o', linewidth=2, markersize=4)
plt.xlabel('Batch Number')
plt.ylabel('Processing Time (seconds)')
plt.title('Processing Time per Batch')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(folders['visualizations'], 'sentiment_analysis_overview.png'), 
           dpi=300, bbox_inches='tight')
plt.show()

# 2. Word Clouds for each sentiment
print("\n☁️  Creating word clouds...")

def clean_text_for_wordcloud(text):
    """Clean text for word cloud generation"""
    # Remove URLs, mentions, special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, sentiment in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL']):
    sentiment_data = results_df[results_df['sentiment'] == sentiment]
    
    if len(sentiment_data) > 0:
        # Combine all texts
        all_text = ' '.join([clean_text_for_wordcloud(text) for text in sentiment_data['text']])
        
        if all_text.strip():  # Check if we have text after cleaning
            wordcloud = WordCloud(
                width=400, height=300, 
                background_color='white',
                colormap=['Greens', 'Reds', 'Greys'][i],
                max_words=WORDCLOUD_MAX_WORDS,
                relative_scaling=0.5,
                random_state=42
            ).generate(all_text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment} Words', fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No {sentiment}\ndata available', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{sentiment} Words', fontsize=14)
            axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(folders['visualizations'], 'sentiment_wordclouds.png'), 
           dpi=300, bbox_inches='tight')
plt.show()

# 3. Most common words analysis
print("\n📈 Creating word frequency analysis...")

def get_top_words(texts, n_words=None):
    """Get top words from texts"""
    if n_words is None:
        n_words = TOP_WORDS_COUNT
    all_text = ' '.join([clean_text_for_wordcloud(text) for text in texts])
    words = all_text.split()
    # Filter out common words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs'}
    words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    return Counter(words).most_common(n_words)

plt.figure(figsize=(15, 10))

for i, sentiment in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 1):
    sentiment_data = results_df[results_df['sentiment'] == sentiment]
    
    if len(sentiment_data) > 0:
        top_words = get_top_words(sentiment_data['text'])
        
        if top_words:
            words, counts = zip(*top_words)
            
            plt.subplot(2, 2, i)
            bars = plt.bar(range(len(words)), counts, 
                          color=['#2ecc71', '#e74c3c', '#95a5a6'][i-1], alpha=0.7)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title(f'Top Words in {sentiment} Comments')
            plt.xticks(range(len(words)), words, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)

# 4. Confidence distribution by sentiment
plt.subplot(2, 2, 4)
for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    sentiment_data = results_df[results_df['sentiment'] == sentiment]
    if len(sentiment_data) > 0:
        plt.hist(sentiment_data['confidence'], alpha=0.6, label=sentiment, bins=20)

plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Distribution by Sentiment')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(folders['visualizations'], 'word_frequency_analysis.png'), 
           dpi=300, bbox_inches='tight')
plt.show()

# Save comprehensive results
print("\n💾 Saving comprehensive results...")

# Performance summary
performance_summary = {
    'total_samples': len(df_sample),
    'processing_time_minutes': total_time / 60,
    'avg_time_per_sample': total_time / len(df_sample),
    'sentiment_distribution': sentiment_counts,
    'score_distribution': {
        'avg_original_score': float(results_df['original_score'].mean()) if 'original_score' in results_df.columns else 0,
        'avg_sentiment_confidence': float(results_df['confidence'].mean()),
        'candidates_count': int(results_df['is_candidate'].sum()) if 'is_candidate' in results_df.columns else 0
    },
    'confidence_stats': {
        'mean': float(results_df['confidence'].mean()),
        'std': float(results_df['confidence'].std()),
        'min': float(results_df['confidence'].min()),
        'max': float(results_df['confidence'].max())
    }
}

# Save performance summary
with open(os.path.join(OUTPUT_BASE_DIR, 'performance_summary.json'), 'w') as f:
    json.dump(performance_summary, f, indent=2)

# Save full results
results_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'complete_results.csv'), index=False)

# Save representative comments summary
representatives_summary = {}
for sentiment, representatives in representative_results.items():
    if len(representatives) > 0:
        representatives_summary[sentiment] = representatives[['text', 'confidence', 'cluster_id', 'cluster_size']].to_dict('records')

with open(os.path.join(OUTPUT_BASE_DIR, 'representative_comments.json'), 'w', encoding='utf-8') as f:
    json.dump(representatives_summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ All results saved to: {OUTPUT_BASE_DIR}")

print("\n" + "=" * 80)
print("SUMMARY OF REPRESENTATIVE COMMENTS")
print("=" * 80)

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    if sentiment in representative_results and len(representative_results[sentiment]) > 0:
        print(f"\n🎯 MOST REPRESENTATIVE {sentiment} COMMENTS:")
        print("-" * 50)
        
        representatives = representative_results[sentiment].sort_values('confidence', ascending=False)
        
        for i, (_, row) in enumerate(representatives.head(5).iterrows(), 1):
            print(f"\n{i}. [Confidence: {row['confidence']:.3f} | Cluster: {row['cluster_id']} | Size: {row['cluster_size']}]")
            print(f"   \"{row['text']}\"")

print(f"\n" + "=" * 80)
print("✅ ENHANCED ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
📊 Analysis Summary:
   • Processed: {len(df_sample):,} samples
   • Processing time: {total_time/60:.1f} minutes
   • Average original quality score: {performance_summary['score_distribution']['avg_original_score']:.3f}
   • Average sentiment confidence: {performance_summary['score_distribution']['avg_sentiment_confidence']:.3f}
   • Files created: {len(os.listdir(OUTPUT_BASE_DIR))} output files
   
📁 Output Structure:
   • {OUTPUT_BASE_DIR}/positive/ - Positive sentiment data
   • {OUTPUT_BASE_DIR}/negative/ - Negative sentiment data  
   • {OUTPUT_BASE_DIR}/neutral/ - Neutral sentiment data
   • {OUTPUT_BASE_DIR}/visualizations/ - Charts and graphs
   • {OUTPUT_BASE_DIR}/vectors/ - Vector analysis data
   
🎯 Key Insights:
   • Most confident positive: {results_df[results_df['sentiment']=='POSITIVE']['confidence'].max():.3f}
   • Most confident negative: {results_df[results_df['sentiment']=='NEGATIVE']['confidence'].max():.3f}
   • Neutral classifications: {sentiment_counts['NEUTRAL']} ({sentiment_counts['NEUTRAL']/len(results_df)*100:.1f}%)
   • High-quality candidates: {performance_summary['score_distribution']['candidates_count']}
""")

# Generate comprehensive PDF report
print("\n" + "=" * 80)
print("GENERATING PDF REPORT")
print("=" * 80)

print("\n📄 Creating comprehensive PDF report...")
pdf_path = generate_pdf_report(
    results_df=results_df,
    representative_results=representative_results,
    performance_summary=performance_summary,
    folders=folders,
    db_path=path_db,
    total_time=total_time
)

#if pdf_path:
#    print(f"✅ PDF report successfully generated: {pdf_path}")
#    print("📋 Report includes:")
#    print("   • Executive summary with source information")
#    print("   • Methodology and technical details")
#    print("   • Sentiment analysis visualizations")
#    print("   • Most representative comments by sentiment")
#    print("   • Performance metrics and processing statistics")
#else:
#    print("❌ PDF report generation failed")

print(f"\n📁 All outputs saved to: {OUTPUT_BASE_DIR}")
print("🎉 Complete analysis package ready for review!")