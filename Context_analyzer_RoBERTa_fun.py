#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RoBERTa Sentiment Analysis with Vector Search, Visualization
Function-based version for configuration-driven execution

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
import re
import sqlite3
from datetime import datetime
from collections import defaultdict

from vizualization import vizualization

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def extract_date_from_text(text):
    """
    Extract date from text in format 'Date of visit: October 1, 2025' or similar
    
    Args:
        text: The comment text containing date information
    
    Returns:
        str: Date in YYYY-MM-DD format or None if not found
    """
    # Pattern for "Date of visit: Month Day, Year"
    pattern = r'Date of visit:\s*(\w+)\s+(\d{1,2}),?\s+(\d{4})'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        month_name = match.group(1)
        day = match.group(2)
        year = match.group(3)
        
        # Convert month name to number
        month_mapping = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        month_num = month_mapping.get(month_name.lower())
        if month_num:
            return f"{year}-{month_num}-{day.zfill(2)}"
    
    # Try pattern "YYYY-MM-DD" directly (from data-visit-date attribute)
    pattern_iso = r'(\d{4})-(\d{2})-(\d{2})'
    match_iso = re.search(pattern_iso, text)
    if match_iso:
        return match_iso.group(0)
    
    return None


def create_text_vectors(texts, method='tfidf', tfidf_max_features=1000, tfidf_min_df=4, tfidf_max_df=0.8):
    """Create vector representations of texts"""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=tfidf_min_df,
            max_df=tfidf_max_df
        )
        vectors = vectorizer.fit_transform(texts)
        return vectors, vectorizer


def find_representative_comments(sentiment_data, n_representatives=10, tfidf_max_features=1000, 
                                 tfidf_min_df=4, tfidf_max_df=0.8):
    """Find most representative comments using clustering and centroids"""
    if len(sentiment_data) < n_representatives:
        # Add cluster columns even when not clustering
        result = sentiment_data.copy()
        result['cluster_id'] = 0
        result['cluster_size'] = len(result)
        return result
    
    texts = sentiment_data['text'].tolist()
    
    # Create TF-IDF vectors
    vectors, vectorizer = create_text_vectors(texts, tfidf_max_features=tfidf_max_features,
                                             tfidf_min_df=tfidf_min_df, tfidf_max_df=tfidf_max_df)
    
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


def analyze_sentiment_enhanced(text, pipe, confidence_threshold=0.8):
    """Enhanced sentiment analysis with 3-class simulation"""
    # Truncate text if too long for the model (512 tokens max, ~400 chars safe)
    if len(text) > 400:
        text = text[:400] + "..."
    
    result = pipe(text)
    raw_label = result[0]['label']
    confidence = result[0]['score']
    
    # 3-class simulation with confidence thresholds
    if raw_label == "POSITIVE":
        if confidence > confidence_threshold:
            readable_label = "POSITIVE"
        else:
            readable_label = "NEUTRAL"
    else:  # NEGATIVE
        if confidence > confidence_threshold:
            readable_label = "NEGATIVE"
        else:
            readable_label = "NEUTRAL"
    
    return {
        'text': text,
        'sentiment': readable_label,
        'confidence': confidence,
        'raw_label': raw_label
    }


def compute_original_score(text, sentiment, key_positive_words, key_neutral_words, key_negative_words, sentence_length):
    """
    Compute original score based on text length and keyword presence
    
    Args:
        text: The comment text
        sentiment: The sentiment classification (POSITIVE, NEGATIVE, NEUTRAL)
        key_positive_words: List of positive keywords
        key_neutral_words: List of neutral keywords
        key_negative_words: List of negative keywords
        sentence_length: Minimum sentence length threshold
    
    Returns:
        float: Original score (unnormalized)
    """
    # Count words in text
    words = text.lower().split()
    word_count = len(words)
    
    # Comments with words less than sentence_length get 0 score
    if word_count <= sentence_length:
        return 0.0
    
    # Base score from word count: 0.05 per word above sentence_length
    extra_words = word_count - sentence_length
    base_score = extra_words * 0.05
    
    # Keyword bonus based on sentiment
    keyword_bonus = 0.0
    text_lower = text.lower()
    
    if sentiment == "POSITIVE":
        # Count positive keywords in text
        for keyword in key_positive_words:
            if keyword.lower() in text_lower:
                keyword_bonus += 0.1
    elif sentiment == "NEGATIVE":
        # Count negative keywords in text
        for keyword in key_negative_words:
            if keyword.lower() in text_lower:
                keyword_bonus += 0.1
    elif sentiment == "NEUTRAL":
        # Count neutral keywords in text
        for keyword in key_neutral_words:
            if keyword.lower() in text_lower:
                keyword_bonus += 0.1
    
    return base_score + keyword_bonus


def normalize_scores_by_sentiment(results_df):
    """
    Normalize original scores within each sentiment category to 0-1 range
    
    Args:
        results_df: DataFrame with 'sentiment' and 'original_score' columns
    
    Returns:
        DataFrame with normalized scores
    """
    df = results_df.copy()
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_mask = df['sentiment'] == sentiment
        sentiment_scores = df.loc[sentiment_mask, 'original_score']
        
        if len(sentiment_scores) > 0:
            min_score = sentiment_scores.min()
            max_score = sentiment_scores.max()
            
            # Avoid division by zero
            if max_score > min_score:
                df.loc[sentiment_mask, 'original_score'] = (
                    (sentiment_scores - min_score) / (max_score - min_score)
                )
            else:
                # All scores are the same, set to 0.5
                df.loc[sentiment_mask, 'original_score'] = 0.5
    
    return df


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
        
        # Drop old table if it exists to ensure schema is updated
        cursor.execute('DROP TABLE IF EXISTS extracted_text_data')
        
        # Create extracted_text_data table with visit_date column
        cursor.execute('''
            CREATE TABLE extracted_text_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                block_text TEXT NOT NULL,
                block_length INTEGER,
                visit_date TEXT,
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        for block in text_blocks:
            # Extract date from block text
            visit_date = extract_date_from_text(block['text'])
            
            cursor.execute('''
                INSERT INTO extracted_text_data (source_file, block_text, block_length, visit_date)
                VALUES (?, ?, ?, ?)
            ''', (block['source_file'], block['text'], block['length'], visit_date))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Integrated {len(text_blocks)} text blocks into database")
        
    except Exception as e:
        print(f"❌ Error integrating text blocks: {e}")


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
                           0 as is_candidate, source_file as source_info,
                           visit_date
                    FROM extracted_text_data
                    WHERE block_length >= 30
                    ORDER BY visit_date, block_length DESC
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


def Context_analyzer_RoBERTa_fun(**kwargs):
    """
    Main function to run sentiment analysis with configuration parameters
    
    Keyword Arguments:
        samples_per_class: Number of samples per sentiment class
        total_samples: Total samples to process
        use_extracted_text: Use extracted text files instead of web crawling
        extracted_text_dir: Directory with extracted text files
        batch_size: Batch size for sentiment analysis processing
        confidence_threshold: Confidence threshold for 3-class simulation
        n_representatives: Number of representative comments to find per sentiment
        tfidf_max_features: Maximum features for TF-IDF vectorization
        tfidf_min_df: Minimum document frequency for TF-IDF
        tfidf_max_df: Maximum document frequency for TF-IDF
        top_words_count: Number of top words to show in frequency analysis
        wordcloud_max_words: Maximum words in word clouds
        cache_dir: HuggingFace cache directory
        model_path: Path to sentiment analysis model
        output_base_dir: Base directory for output files
        path_db: Path to database
    """
    
    # Extract parameters with defaults
    SAMPLES_PER_CLASS = kwargs.get('samples_per_class', 1750)
    TOTAL_SAMPLES = kwargs.get('total_samples', SAMPLES_PER_CLASS * 2)
    USE_EXTRACTED_TEXT = kwargs.get('use_extracted_text', True)
    EXTRACTED_TEXT_DIR = kwargs.get('extracted_text_dir', './Request/extracted_text')
    BATCH_SIZE = kwargs.get('batch_size', 100)
    CONFIDENCE_THRESHOLD = kwargs.get('confidence_threshold', 0.8)
    N_REPRESENTATIVES = kwargs.get('n_representatives', 10)
    TFIDF_MAX_FEATURES = kwargs.get('tfidf_max_features', 1000)
    TFIDF_MIN_DF = kwargs.get('tfidf_min_df', 4)
    TFIDF_MAX_DF = kwargs.get('tfidf_max_df', 0.8)
    TOP_WORDS_COUNT = kwargs.get('top_words_count', 15)
    WORDCLOUD_MAX_WORDS = kwargs.get('wordcloud_max_words', 100)
    CACHE_DIR = kwargs.get('cache_dir', '/tmp/hf_cache')
    MODEL_PATH = kwargs.get('model_path', './my_volume/hf_model')
    OUTPUT_BASE_DIR = kwargs.get('output_base_dir', './my_volume/sentiment_analysis')
    path_db = kwargs.get('path_db', './filtered_reviews.db')
    
    # Extract keyword parameters for score computation
    KEY_POSITIVE_WORDS = kwargs.get('key_positive_words', ["nice", "good", "excellent"])
    KEY_NEUTRAL_WORDS = kwargs.get('key_neutral_words', ["visit", "stay"])
    KEY_NEGATIVE_WORDS = kwargs.get('key_negative_words', ["bad", "terrible"])
    SENTENCE_LENGTH = kwargs.get('sentence_length', 4)
    
    # Set HuggingFace cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    
    print("=" * 80)
    print("ENHANCED ROBERTA SENTIMENT ANALYSIS WITH VECTOR SEARCH")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Samples per class: {SAMPLES_PER_CLASS}")
    print(f"   Total samples: {TOTAL_SAMPLES}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Representatives per sentiment: {N_REPRESENTATIVES}")
    print(f"   Use extracted text: {USE_EXTRACTED_TEXT}")
    print(f"   Sentence length threshold: {SENTENCE_LENGTH}")
    print(f"   Positive keywords: {len(KEY_POSITIVE_WORDS)}")
    print(f"   Negative keywords: {len(KEY_NEGATIVE_WORDS)}")
    print(f"   Neutral keywords: {len(KEY_NEUTRAL_WORDS)}")
    
    # Initialize sentiment analysis model
    print("\n🤖 Loading sentiment analysis model...")
    
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("📂 Using existing DistilBERT model...")
        pipe = pipeline("sentiment-analysis", model=MODEL_PATH)
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        simulate_3_class = True
    else:
        print("❌ Model not found, please run basic script first")
        return None
    
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
    
    # Load and process samples
    print("\n" + "=" * 80)
    print("LOADING AND PROCESSING SAMPLES FROM DATABASE")
    print("=" * 80)
    
    print(f"\n📊 Loading dataset from: {path_db}")
    
    # Extracted text reading section
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
    
    try:
        # Load dataset (with or without extracted text data)
        df_dataset = load_combined_dataset(path_db, include_extracted_text=USE_EXTRACTED_TEXT)
        
        if df_dataset is None or len(df_dataset) == 0:
            print("❌ Error: No data loaded from database")
            return None
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(df_dataset):,}")
        
        # Show data source distribution
        if 'source' in df_dataset.columns:
            source_counts = df_dataset['source'].value_counts()
            print(f"\n📈 Data Source Distribution:")
            for source, count in source_counts.items():
                percentage = (count / len(df_dataset)) * 100
                print(f"   {source.title()}: {count:,} samples ({percentage:.1f}%)")
        
        # Rename columns to match expected format
        if 'block_text' in df_dataset.columns:
            df_dataset.rename(columns={'block_text': 'text'}, inplace=True)
        
        # Take a subset for processing
        max_samples = min(len(df_dataset), TOTAL_SAMPLES)
        df_sample = df_dataset.head(max_samples).copy()
        
        # Shuffle the data
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Selected {len(df_sample)} samples for analysis!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
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
            result = analyze_sentiment_enhanced(row['text'], pipe, CONFIDENCE_THRESHOLD)
            # Store temporarily without original_score (will compute after all sentiments are known)
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
    
    # Extract dates from text
    print("\n📅 Extracting dates from reviews...")
    results_df['visit_date'] = results_df['text'].apply(extract_date_from_text)
    dates_found = results_df['visit_date'].notna().sum()
    print(f"✅ Found dates in {dates_found} out of {len(results_df)} reviews")
    
    # Compute original scores based on text length and keywords
    print("\n📊 Computing original quality scores...")
    results_df['original_score'] = results_df.apply(
        lambda row: compute_original_score(
            row['text'], 
            row['sentiment'],
            KEY_POSITIVE_WORDS,
            KEY_NEUTRAL_WORDS,
            KEY_NEGATIVE_WORDS,
            SENTENCE_LENGTH
        ),
        axis=1
    )
    
    # Normalize scores by sentiment category
    print("📊 Normalizing scores by sentiment category...")
    results_df = normalize_scores_by_sentiment(results_df)
    
    print(f"✅ Score computation complete!")
    print(f"   Score range: {results_df['original_score'].min():.3f} - {results_df['original_score'].max():.3f}")
    print(f"   Average score: {results_df['original_score'].mean():.3f}")
    
    # Build trends by date
    print("\n📈 Building sentiment trends by date...")
    trends = defaultdict(lambda: {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'total': 0})
    
    for _, row in results_df.iterrows():
        if pd.notna(row.get('visit_date')):
            date = row['visit_date']
            sentiment = row['sentiment']
            trends[date][sentiment] += 1
            trends[date]['total'] += 1
    
    # Convert trends to sorted list of dictionaries
    trends_list = []
    for date in sorted(trends.keys()):
        trends_list.append({
            'date': date,
            'positive': trends[date]['POSITIVE'],
            'negative': trends[date]['NEGATIVE'],
            'neutral': trends[date]['NEUTRAL'],
            'total': trends[date]['total']
        })
    
    if trends_list:
        print(f"✅ Trends computed for {len(trends_list)} dates:")
        print(f"   Date range: {trends_list[0]['date']} to {trends_list[-1]['date']}")
        print(f"   Total reviews with dates: {sum(t['total'] for t in trends_list)}")
        
        # Show sample of trends
        print("\n   Sample trends:")
        for trend in trends_list[:5]:
            print(f"      {trend['date']}: +{trend['positive']} | -{trend['negative']} | ={trend['neutral']} (total: {trend['total']})")
    else:
        print("⚠️  No date information found in reviews")
    
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
    
    # Find representative comments for each sentiment
    print("\n🔍 Finding most representative comments...")
    
    representative_results = {}
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sentiment_data = results_df[results_df['sentiment'] == sentiment]
        
        if len(sentiment_data) > 0:
            print(f"\n📝 Analyzing {sentiment} comments ({len(sentiment_data)} total)...")
            representatives = find_representative_comments(
                sentiment_data, 
                n_representatives=N_REPRESENTATIVES,
                tfidf_max_features=TFIDF_MAX_FEATURES,
                tfidf_min_df=TFIDF_MIN_DF,
                tfidf_max_df=TFIDF_MAX_DF
            )
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
    
    # Save trends data
    if trends_list:
        trends_file = os.path.join(OUTPUT_BASE_DIR, 'sentiment_trends.json')
        with open(trends_file, 'w', encoding='utf-8') as f:
            json.dump({
                'trends': trends_list,
                'summary': {
                    'total_dates': len(trends_list),
                    'date_range': {
                        'start': trends_list[0]['date'],
                        'end': trends_list[-1]['date']
                    },
                    'total_reviews': sum(t['total'] for t in trends_list),
                    'total_positive': sum(t['positive'] for t in trends_list),
                    'total_negative': sum(t['negative'] for t in trends_list),
                    'total_neutral': sum(t['neutral'] for t in trends_list)
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Trends data saved to: {trends_file}")
    
    # Visualization
    vizualization(
        sentiment_counts, 
        results_df,
        processing_times,
        folders,
        WORDCLOUD_MAX_WORDS, 
        TOP_WORDS_COUNT, 
        df_sample,
        total_time,
        OUTPUT_BASE_DIR,
        representative_results,
        trends_list
    )
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return {
        'results_df': results_df,
        'sentiment_counts': sentiment_counts,
        'representative_results': representative_results,
        'trends': trends_list,
        'folders': folders,
        'total_time': total_time
    }


if __name__ == "__main__":
    # Example usage with default parameters
    results = run_sentiment_analysis()
