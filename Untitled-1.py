#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RoBERTa Sentiment Analysis with Vector Search and Visualization
- Configurable processing of samples from Sentiment140 dataset
- Indexes comments by sentiment into separate folders
- Uses vector search to find most representative comments
- Creates interesting visualizations

Configuration parameters are set at the top of the file for easy customization.
Default: 3,500 samples (1,750 per class)

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
from download_and_prepare_dataset import download_and_prepare_dataset

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset sampling parameters
SAMPLES_PER_CLASS = 17500          # Number of samples per sentiment class (negative/positive)
TOTAL_SAMPLES = SAMPLES_PER_CLASS * 2  # Total samples to process (3500)

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

# Download and prepare dataset
path_db = download_and_prepare_dataset()
print('Dataset path:', path_db)

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

# Load and process 3,500 samples
print("\n" + "=" * 80)
print("LOADING AND PROCESSING 3,500 SAMPLES")
print("=" * 80)

print(f"\n📊 Loading dataset from: {path_db}")

try:
    # Load the dataset
    df_dataset = pd.read_csv(path_db, encoding='latin1', header=None, 
                            names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(df_dataset):,}")
    
    # Take samples with balanced distribution
    # Get negative (target=0) and positive (target=4) samples
    negative_samples = df_dataset[df_dataset['target'] == 0].head(SAMPLES_PER_CLASS)
    positive_samples = df_dataset[df_dataset['target'] == 4].head(SAMPLES_PER_CLASS)
    
    df_sample = pd.concat([negative_samples, positive_samples], ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"✅ Created balanced sample of {len(df_sample)} tweets!")
    
    # Convert target labels
    df_sample['true_sentiment'] = df_sample['target'].map({0: 'NEGATIVE', 4: 'POSITIVE'})
    
    # Show distribution
    print(f"\n📈 Sample Distribution:")
    sample_counts = df_sample['true_sentiment'].value_counts()
    for sentiment, count in sample_counts.items():
        percentage = (count / len(df_sample)) * 100
        print(f"   {sentiment}: {count:,} samples ({percentage:.1f}%)")

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
        result['true_sentiment'] = row['true_sentiment']
        result['target'] = row['target']
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
        cluster_vectors = vectors[cluster_indices]
        
        # Find the text closest to cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
        closest_idx = cluster_indices[np.argmax(distances)]
        
        representative = sentiment_data.iloc[closest_idx].copy()
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(cluster_indices)
        representatives.append(representative)
    
    return pd.DataFrame(representatives)

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
plt.title('Sentiment Distribution (3,500 samples)', fontsize=14, fontweight='bold')

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

# 3. Sentiment by True Label (Accuracy visualization)
plt.subplot(2, 2, 3)
confusion_data = pd.crosstab(results_df['true_sentiment'], results_df['sentiment'])
sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Prediction Accuracy Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('True Sentiment')

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
    'accuracy_metrics': {
        'overall_accuracy': len(results_df[results_df['sentiment'] == results_df['true_sentiment']]) / len(results_df),
        'confidence_stats': {
            'mean': float(results_df['confidence'].mean()),
            'std': float(results_df['confidence'].std()),
            'min': float(results_df['confidence'].min()),
            'max': float(results_df['confidence'].max())
        }
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
   • Average accuracy: {performance_summary['accuracy_metrics']['overall_accuracy']:.1%}
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
""")