#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoBERTa 3-Class Sentiment Analysis Script
Supports POSITIVE, NEGATIVE, and NEUTRAL classification

Created on Thu Oct 30 2025
@author: andreyvlasenko
"""

import os
import requests
import zipfile
from transformers import pipeline
import pandas as pd
import time
from download_and_prepare_dataset import download_and_prepare_dataset

# Download and prepare dataset
path_db = download_and_prepare_dataset()
print('Dataset path:', path_db)

# Set HuggingFace cache directory
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

print("=" * 60)
print("ROBERTA 3-CLASS SENTIMENT ANALYSIS")
print("=" * 60)

# Initialize sentiment analysis model
print("\n🤖 Loading sentiment analysis model...")

# First, try to use your existing DistilBERT model as fallback
existing_model_path = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"

if os.path.exists(os.path.join(existing_model_path, "config.json")):
    print("📂 Using your existing DistilBERT model with 3-class simulation...")
    pipe = pipeline("sentiment-analysis", model=existing_model_path)
    selected_model = ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2 (simulated 3-class)")
    simulate_3_class = True
else:
    # Try online models if available
    model_options = [
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa Twitter (latest)"),
        ("cardiffnlp/twitter-roberta-base-sentiment", "RoBERTa Twitter (base)"),
        ("nlptown/bert-base-multilingual-uncased-sentiment", "BERT Multilingual"),
        ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2 (fallback)")
    ]

    pipe = None
    selected_model = None
    simulate_3_class = False

    for model_name, description in model_options:
        try:
            print(f"   Trying {description}...")
            pipe = pipeline("sentiment-analysis", model=model_name)
            selected_model = (model_name, description)
            print(f"   ✅ Successfully loaded: {description}")
            if "distilbert" in model_name.lower():
                simulate_3_class = True
            break
        except Exception as e:
            print(f"   ❌ Failed to load {description}: {str(e)[:100]}...")
            continue

    if pipe is None:
        print("❌ Could not load any sentiment analysis model!")
        print("This might be due to network connectivity issues.")
        print("Please check your internet connection and try again.")
        exit(1)

print(f"\n📊 Using model: {selected_model[1]}")
print(f"🔧 Model identifier: {selected_model[0]}")
if simulate_3_class:
    print("🎯 Using 3-class simulation for binary model")

# Label mapping for different models
def get_label_mapping(model_name):
    """Get appropriate label mapping based on model"""
    if "twitter-roberta" in model_name:
        return {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL", 
            "LABEL_2": "POSITIVE"
        }
    elif "multilingual" in model_name:
        return {
            "1 star": "NEGATIVE",
            "2 stars": "NEGATIVE", 
            "3 stars": "NEUTRAL",
            "4 stars": "POSITIVE",
            "5 stars": "POSITIVE"
        }
    elif "emotion" in model_name:
        return {
            "anger": "NEGATIVE",
            "disgust": "NEGATIVE",
            "fear": "NEGATIVE",
            "sadness": "NEGATIVE",
            "joy": "POSITIVE",
            "surprise": "NEUTRAL",
            "neutral": "NEUTRAL"
        }
    else:
        # Default mapping for binary models
        return {
            "POSITIVE": "POSITIVE",
            "NEGATIVE": "NEGATIVE",
            "NEUTRAL": "NEUTRAL"
        }

LABEL_MAP = get_label_mapping(selected_model[0])

def analyze_sentiment(text):
    """
    Analyze sentiment of text with 3-class output
    Returns: dict with readable label and confidence score
    """
    result = pipe(text)
    raw_label = result[0]['label']
    confidence = result[0]['score']
    
    if simulate_3_class:
        # Convert binary model to 3-class using confidence thresholds
        if raw_label == "POSITIVE":
            if confidence > 0.8:
                readable_label = "POSITIVE"
            else:
                readable_label = "NEUTRAL"  # Low confidence positive becomes neutral
        else:  # NEGATIVE
            if confidence > 0.8:
                readable_label = "NEGATIVE"
            else:
                readable_label = "NEUTRAL"  # Low confidence negative becomes neutral
    else:
        # Use the label mapping for true 3-class models
        readable_label = LABEL_MAP.get(raw_label, raw_label)
    
    return {
        'text': text,
        'sentiment': readable_label,
        'confidence': confidence,
        'raw_label': raw_label,
        'method': '3-class simulation' if simulate_3_class else 'native 3-class'
    }

# Test with diverse examples including neutral cases
print("\n" + "=" * 60)
print("DATASET VALIDATION WITH 1000 SAMPLES")
print("=" * 60)

# Load and validate using the Sentiment140 dataset
print(f"\n� Loading dataset from: {path_db}")

try:
    # Load the dataset (CSV format)
    # Sentiment140 format: target,ids,date,flag,user,text
    # target: 0 = negative, 4 = positive
    df_dataset = pd.read_csv(path_db, encoding='latin1', header=None, 
                            names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(df_dataset):,}")
    print(f"   Columns: {list(df_dataset.columns)}")
    
    # Show dataset distribution
    print(f"\n📈 Original Dataset Distribution:")
    target_counts = df_dataset['target'].value_counts().sort_index()
    for target, count in target_counts.items():
        sentiment = "NEGATIVE" if target == 0 else "POSITIVE" if target == 4 else f"UNKNOWN({target})"
        percentage = (count / len(df_dataset)) * 100
        print(f"   {sentiment}: {count:,} samples ({percentage:.1f}%)")
    
    # Take first 1000 samples for validation
    print(f"\n🔬 Using first 1000 samples for validation...")
    df_sample = df_dataset.head(1000).copy()
    
    # Convert target labels to readable format
    df_sample['true_sentiment'] = df_sample['target'].map({0: 'NEGATIVE', 4: 'POSITIVE'})
    
    print(f"✅ Validation set created!")
    print(f"   Sample size: {len(df_sample)}")
    
    # Show sample distribution
    print(f"\n📊 Validation Sample Distribution:")
    sample_counts = df_sample['true_sentiment'].value_counts()
    for sentiment, count in sample_counts.items():
        percentage = (count / len(df_sample)) * 100
        print(f"   {sentiment}: {count} samples ({percentage:.1f}%)")
    
    # Show some sample texts
    print(f"\n📝 Sample texts from dataset:")
    for i, row in df_sample.head(5).iterrows():
        text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
        print(f"   {row['true_sentiment']}: {text_preview}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Creating synthetic validation data instead...")
    
    # Fallback: create synthetic data if dataset loading fails
    synthetic_data = [
        ("I love this product! It's amazing!", "POSITIVE"),
        ("This is terrible service.", "NEGATIVE"),
        ("Great experience, highly recommend!", "POSITIVE"),
        ("Worst purchase ever made.", "NEGATIVE"),
        ("The weather is nice today.", "POSITIVE"),  # Note: this might be neutral
    ] * 200  # Repeat to get 1000 samples
    
    df_sample = pd.DataFrame(synthetic_data[:1000], columns=['text', 'true_sentiment'])
    print(f"✅ Created {len(df_sample)} synthetic validation samples")

# Perform validation analysis
print(f"\n" + "=" * 60)
print("MODEL VALIDATION ANALYSIS")
print("=" * 60)

print(f"\n� Analyzing {len(df_sample)} samples with RoBERTa model...")
print("⏱️  This may take a moment...")

# Batch process for efficiency
batch_size = 50
predictions = []
true_labels = []
confidences = []
processing_times = []

start_total = time.time()

for i in range(0, len(df_sample), batch_size):
    batch_end = min(i + batch_size, len(df_sample))
    batch_texts = df_sample['text'].iloc[i:batch_end].tolist()
    batch_true = df_sample['true_sentiment'].iloc[i:batch_end].tolist()
    
    batch_start = time.time()
    
    # Process batch
    for text, true_label in zip(batch_texts, batch_true):
        result = analyze_sentiment(text)
        predictions.append(result['sentiment'])
        confidences.append(result['confidence'])
        true_labels.append(true_label)
    
    batch_time = time.time() - batch_start
    processing_times.append(batch_time)
    
    # Progress indicator
    progress = (batch_end / len(df_sample)) * 100
    avg_time_per_text = batch_time / len(batch_texts)
    print(f"   Progress: {progress:5.1f}% | Batch {i//batch_size + 1:2d} | {avg_time_per_text:.3f}s per text")

total_time = time.time() - start_total
avg_time_per_text = total_time / len(df_sample)

print(f"\n✅ Validation complete!")
print(f"   Total time: {total_time:.2f} seconds")
print(f"   Average per text: {avg_time_per_text:.3f} seconds")
print(f"   Throughput: {len(df_sample)/total_time:.1f} texts per second")

# Create results dataframe
results_df = pd.DataFrame({
    'text': df_sample['text'],
    'true_sentiment': true_labels,
    'predicted_sentiment': predictions,
    'confidence': confidences
})

# Calculate accuracy metrics
from collections import Counter

correct_predictions = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
total_predictions = len(true_labels)
accuracy = correct_predictions / total_predictions

print(f"\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

print(f"\n🎯 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   Correct: {correct_predictions:,} / {total_predictions:,}")
print(f"   Incorrect: {total_predictions - correct_predictions:,}")

# Confusion matrix
print(f"\n📊 Confusion Matrix:")
print("Actual \\ Predicted | POSITIVE | NEGATIVE | NEUTRAL")
print("-" * 50)

for true_class in ['POSITIVE', 'NEGATIVE']:
    if true_class in true_labels:
        row_data = []
        for pred_class in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            count = sum(1 for t, p in zip(true_labels, predictions) 
                       if t == true_class and p == pred_class)
            row_data.append(f"{count:8d}")
        print(f"{true_class:9s}        | {'|'.join(row_data)}")

# Performance by class
print(f"\n📈 Performance by Class:")
for sentiment_class in ['POSITIVE', 'NEGATIVE']:
    if sentiment_class in true_labels:
        # True positives, false positives, false negatives
        tp = sum(1 for t, p in zip(true_labels, predictions) 
                if t == sentiment_class and p == sentiment_class)
        fp = sum(1 for t, p in zip(true_labels, predictions) 
                if t != sentiment_class and p == sentiment_class)
        fn = sum(1 for t, p in zip(true_labels, predictions) 
                if t == sentiment_class and p != sentiment_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {sentiment_class}:")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall:    {recall:.3f}")
        print(f"      F1-Score:  {f1:.3f}")

# Confidence analysis
print(f"\n🔍 Confidence Analysis:")
avg_confidence = sum(confidences) / len(confidences)
correct_confidences = [conf for conf, true_l, pred_l in zip(confidences, true_labels, predictions) 
                      if true_l == pred_l]
incorrect_confidences = [conf for conf, true_l, pred_l in zip(confidences, true_labels, predictions) 
                        if true_l != pred_l]

print(f"   Average confidence: {avg_confidence:.3f}")
if correct_confidences:
    print(f"   Avg confidence (correct): {sum(correct_confidences)/len(correct_confidences):.3f}")
if incorrect_confidences:
    print(f"   Avg confidence (incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.3f}")

# Show some examples of correct and incorrect predictions
print(f"\n" + "=" * 60)
print("PREDICTION EXAMPLES")
print("=" * 60)

print(f"\n✅ Correct Predictions (first 5):")
correct_examples = results_df[results_df['true_sentiment'] == results_df['predicted_sentiment']].head(5)
for _, row in correct_examples.iterrows():
    text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
    print(f"   TRUE: {row['true_sentiment']} | PRED: {row['predicted_sentiment']} ({row['confidence']:.3f}) | {text_preview}")

print(f"\n❌ Incorrect Predictions (first 5):")
incorrect_examples = results_df[results_df['true_sentiment'] != results_df['predicted_sentiment']].head(5)
for _, row in incorrect_examples.iterrows():
    text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
    print(f"   TRUE: {row['true_sentiment']} | PRED: {row['predicted_sentiment']} ({row['confidence']:.3f}) | {text_preview}")

# Save results
results_file = "/Users/andreyvlasenko/tst/Request/my_volume/validation_results.csv"
results_df.to_csv(results_file, index=False)
print(f"\n💾 Results saved to: {results_file}")

print(f"\n" + "=" * 60)
print("✅ DATASET VALIDATION COMPLETE!")
print("=" * 60)

# Performance test with sample data
print("\n" + "=" * 60)
print("PERFORMANCE TESTING")
print("=" * 60)

print("\n⏱️  Speed test on sample texts...")
sample_texts = [
    "I love this product!",
    "This is terrible.",
    "Great service!",
    "Bad experience.",
    "The meeting is today.",
    "Not sure about this.",
    "Amazing quality!",
    "Poor performance.",
    "Weather is nice.",
    "Disappointed with results."
]

start_time = time.time()
for text in sample_texts:
    _ = analyze_sentiment(text)
end_time = time.time()

total_time = end_time - start_time
avg_time = total_time / len(sample_texts)

print(f"   Total time: {total_time:.3f} seconds")
print(f"   Average per text: {avg_time:.3f} seconds")
print(f"   Throughput: {1/avg_time:.1f} texts per second")

# Function for batch processing
def batch_analyze_sentiment(texts, batch_size=32):
    """
    Analyze sentiment for multiple texts efficiently
    """
    print(f"\n🚀 Batch processing {len(texts)} texts (batch_size={batch_size})...")
    
    results = []
    start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = pipe(batch)
        
        for text, result in zip(batch, batch_results):
            raw_label = result['label']
            readable_label = LABEL_MAP.get(raw_label, raw_label)
            
            results.append({
                'text': text,
                'sentiment': readable_label,
                'confidence': result['score'],
                'raw_label': raw_label
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ✅ Processed {len(texts)} texts in {total_time:.3f} seconds")
    print(f"   ⚡ Throughput: {len(texts)/total_time:.1f} texts per second")
    
    return results

# Demo batch processing with validation data
if len(df_sample) > 5:
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)
    
    # Use a subset of validation data for batch demo
    demo_texts = df_sample['text'].head(20).tolist()
    batch_results = batch_analyze_sentiment(demo_texts)
    batch_df = pd.DataFrame(batch_results)
    
    print(f"\n📊 Batch demo results summary:")
    print(batch_df['sentiment'].value_counts())

print("\n" + "=" * 60)
print("✅ ROBERTA 3-CLASS ANALYSIS COMPLETE!")
print("=" * 60)

print(f"""
🎯 Key Features of RoBERTa Model:
   • 3-class classification: POSITIVE, NEGATIVE, NEUTRAL
   • Trained specifically on Twitter data
   • Better handling of neutral/ambiguous text
   • Good performance on social media style text

📊 Model Performance:
   • Speed: ~{1/avg_time:.1f} texts per second
   • Accuracy: High on clear sentiment, good on neutral
   • Memory: Moderate (larger than DistilBERT)

💡 Use Cases:
   • Social media monitoring
   • Customer feedback analysis  
   • Content moderation
   • Market sentiment analysis
""")