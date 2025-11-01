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

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset sampling parameters
SAMPLES_PER_CLASS = 1750          # Number of samples per sentiment class (negative/positive)
TOTAL_SAMPLES = SAMPLES_PER_CLASS * 2  # Total samples to process (3500)

# Text extraction parameters (NEW)
USE_EXTRACTED_TEXT = True         # Use extracted text files instead of web crawling
EXTRACTED_TEXT_DIR = "/Users/andreyvlasenko/tst/dNet/extracted_text"  # Directory with extracted text files

# Web crawling parameters (DEPRECATED - using extracted text instead)
ENABLE_WEB_CRAWLING = False       # Disabled - using extracted text files
WEB_CRAWL_DEPTH = 1              
MAX_PAGES_PER_DEPTH = 5          
CRAWL_DELAY = 2.0                
ALLOWED_DOMAINS = []

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
    # Truncate text if too long for the model (512 tokens max, ~400 chars safe)
    if len(text) > 400:
        text = text[:400] + "..."
    
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

# Web Crawling Functions (DEPRECATED - using extracted text instead)
def read_extracted_text_files(extracted_text_dir):
    """
    Read text blocks from extracted text files
    
    Args:
        extracted_text_dir: Directory containing extracted text files
    
    Returns:
        List of text blocks with metadata
    """
    print(f"\n📖 Reading extracted text files from: {extracted_text_dir}")
    
    if not os.path.exists(extracted_text_dir):
        print(f"❌ Directory not found: {extracted_text_dir}")
        return []
    
    txt_files = [f for f in os.listdir(extracted_text_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"❌ No .txt files found in {extracted_text_dir}")
        return []
    
    print(f"✅ Found {len(txt_files)} text file(s)")
    
    all_text_blocks = []
    
    for txt_file in txt_files:
        filepath = os.path.join(extracted_text_dir, txt_file)
        print(f"   Reading: {txt_file}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract blocks from different sections
            # Split by common section headers
            sections = re.split(r'\n(?:RESTAURANT INFORMATION|REVIEWS|RATINGS|DESCRIPTIONS|OTHER TEXT BLOCKS):\n-+\n', content)
            
            block_count = 0
            for section in sections[1:]:  # Skip header
                # Split by block markers like [Block 1], [Review 1], etc.
                blocks = re.split(r'\n\[(?:Block|Review) \d+\]\n', section)
                
                for block in blocks:
                    block = block.strip()
                    # Clean up the block - remove metadata lines
                    lines = block.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Skip metadata lines (numbers followed by periods at start)
                        if re.match(r'^\d+\.\s', line):
                            continue
                        cleaned_lines.append(line)
                    
                    cleaned_block = '\n'.join(cleaned_lines).strip()
                    
                    if len(cleaned_block) > 30:  # Only meaningful blocks
                        all_text_blocks.append({
                            'text': cleaned_block,
                            'source_file': txt_file,
                            'length': len(cleaned_block)
                        })
                        block_count += 1
            
            print(f"      ✅ Extracted {block_count} text blocks")
            
        except Exception as e:
            print(f"      ❌ Error reading {txt_file}: {e}")
    
    print(f"\n✅ Total text blocks extracted: {len(all_text_blocks)}")
    return all_text_blocks

def integrate_extracted_text_with_db(text_blocks, db_path):
    """
    Integrate extracted text blocks with database
    
    Args:
        text_blocks: List of text block dictionaries
        db_path: Path to database
    """
    if not text_blocks:
        return
    
    print(f"\n💾 Integrating {len(text_blocks)} text blocks with database...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create extracted_text_data table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_text_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                block_text TEXT NOT NULL,
                block_length INTEGER,
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Clear old data
        cursor.execute('DELETE FROM extracted_text_data')
        
        for block in text_blocks:
            cursor.execute('''
                INSERT INTO extracted_text_data (source_file, block_text, block_length)
                VALUES (?, ?, ?)
            ''', (block['source_file'], block['text'], block['length']))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Integrated {len(text_blocks)} text blocks into database")
        
    except Exception as e:
        print(f"❌ Error integrating text blocks: {e}")

# Old web crawling function kept for reference but not used
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
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
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

def load_combined_dataset(db_path, include_extracted_text=False):
    """
    Load dataset combining existing comment_blocks with optional extracted text data
    """
    try:
        conn = sqlite3.connect(db_path)
        
        if include_extracted_text:
            # Check if extracted_text_data table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_text_data';")
            has_extracted_data = cursor.fetchone() is not None
            
            if has_extracted_data:
                # Use extracted text data
                query = """
                    SELECT 'extracted_text' as source, block_text as text, 
                           0.5 as score, block_length as length, 
                           0 as is_candidate, source_file as source_info
                    FROM extracted_text_data
                    WHERE block_length >= 30
                    ORDER BY block_length DESC
                """
                print("📊 Loading dataset from extracted text files")
            else:
                print("⚠️  No extracted text data found, using comment_blocks")
                include_extracted_text = False
        
        if not include_extracted_text:
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

# Web crawling tracking functions
def track_successful_crawls():
    """Track and return information about successfully crawled websites"""
    crawl_stats = {
        'successful_sites': [],
        'failed_sites': [],
        'total_pages': 0,
        'total_text_blocks': 0
    }
    return crawl_stats

def update_crawl_stats(crawl_stats, url, success, pages_crawled=0, text_blocks=0, error_msg=""):
    """Update crawling statistics"""
    site_info = {
        'url': url,
        'pages': pages_crawled,
        'text_blocks': text_blocks,
        'domain': url.split('/')[2] if '://' in url else url
    }
    
    if success:
        crawl_stats['successful_sites'].append(site_info)
        crawl_stats['total_pages'] += pages_crawled
        crawl_stats['total_text_blocks'] += text_blocks
    else:
        site_info['error'] = error_msg
        crawl_stats['failed_sites'].append(site_info)
    
    return crawl_stats

def print_crawl_statistics(crawl_stats):
    """Print comprehensive crawling statistics"""
    print("\n" + "=" * 60)
    print("🌐 WEB CRAWLING STATISTICS")
    print("=" * 60)
    
    print(f"📊 Overall Results:")
    print(f"   • Successful sites: {len(crawl_stats['successful_sites'])}")
    print(f"   • Failed sites: {len(crawl_stats['failed_sites'])}")
    print(f"   • Total pages crawled: {crawl_stats['total_pages']}")
    print(f"   • Total text blocks extracted: {crawl_stats['total_text_blocks']}")
    
    if crawl_stats['successful_sites']:
        print(f"\n✅ Successfully Processed Sites:")
        for i, site in enumerate(crawl_stats['successful_sites'], 1):
            print(f"   {i}. {site['domain']}")
            print(f"      URL: {site['url']}")
            print(f"      Pages: {site['pages']}, Text blocks: {site['text_blocks']}")
    
    if crawl_stats['failed_sites']:
        print(f"\n❌ Failed Sites:")
        for i, site in enumerate(crawl_stats['failed_sites'], 1):
            print(f"   {i}. {site['domain']}")
            print(f"      URL: {site['url']}")
            print(f"      Error: {site['error']}")
    
    if not crawl_stats['successful_sites'] and not crawl_stats['failed_sites']:
        print("   ℹ️  No web crawling attempted or cached data used")
    
    print("=" * 60)

# Load and process samples from filtered_reviews.db
print("\n" + "=" * 80)
print("LOADING AND PROCESSING SAMPLES FROM FILTERED_REVIEWS.DB")
print("=" * 80)

print(f"\n📊 Loading dataset from: {path_db}")

# Extracted text reading section (NEW)
if USE_EXTRACTED_TEXT:
    print("\n" + "=" * 80)
    print("READING EXTRACTED TEXT FILES")
    print("=" * 80)
    
    try:
        # Read text blocks from extracted text files
        text_blocks = read_extracted_text_files(EXTRACTED_TEXT_DIR)
        
        if text_blocks:
            # Integrate with database
            integrate_extracted_text_with_db(text_blocks, path_db)
            print(f"\n✅ Successfully integrated {len(text_blocks)} text blocks")
        else:
            print("\n⚠️  No text blocks found, will try database only")
            USE_EXTRACTED_TEXT = False
            
    except Exception as e:
        print(f"\n❌ Error reading extracted text: {e}")
        print("   Falling back to database only")
        USE_EXTRACTED_TEXT = False

# Web crawling section (DEPRECATED)
if ENABLE_WEB_CRAWLING:
    print("\n⚠️  Web crawling is deprecated. Use USE_EXTRACTED_TEXT instead.")
    print("   Set USE_EXTRACTED_TEXT = True and provide extracted text files.")

try:
    # Load dataset (with or without extracted text data)
    df_dataset = load_combined_dataset(path_db, include_extracted_text=USE_EXTRACTED_TEXT)
    
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
    
    # Show distribution by score ranges (handle both comment and extracted text data)
    print(f"\n📈 Sample Distribution by Quality Score:")
    if USE_EXTRACTED_TEXT and 'source' in df_sample.columns:
        # Show breakdown by source
        for source in df_sample['source'].unique():
            source_data = df_sample[df_sample['source'] == source]
            print(f"\n   {source.title().replace('_', ' ')} Data ({len(source_data)} samples):")
            
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
                print(f"     Extracted text content: {len(source_data)} text blocks")
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

print(f"\n� All outputs saved to: {OUTPUT_BASE_DIR}")
print("🎉 Complete analysis package ready for review!")
print("📄 Note: Use the separate generate_pdf_only.py script to create PDF reports")