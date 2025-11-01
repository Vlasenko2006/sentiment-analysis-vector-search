#!/usr/bin/env python3
"""
Minimal TripAdvisor Sentiment Analysis Demo
"""

print("🔄 Starting TripAdvisor sentiment analysis...")

# Basic imports first
import os
import json
import time

print("✅ Basic imports loaded")

# Load required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

print("✅ Web scraping libraries loaded")

# Configuration
MODEL_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/tripadvisor_analysis"

def create_sample_reviews():
    """Create sample TripAdvisor-style reviews for demo"""
    print("🔍 Creating sample TripAdvisor reviews...")
    
    sample_reviews = [
        "Amazing hotel! The staff was incredibly helpful and the room was spotless. Breakfast was delicious.",
        "Absolutely loved our stay here. The spa was fantastic and the pool area was beautiful.",
        "Excellent service from check-in to check-out. Highly recommend this place!",
        "Beautiful hotel with stunning views. Great value for money.",
        "Perfect location right in the heart of the city. Clean rooms and friendly staff.",
        "Disappointing stay. The room was dirty and the air conditioning didn't work properly.",
        "Overpriced for what you get. The hotel is outdated and needs renovation.",
        "Terrible experience. Noisy rooms, uncomfortable beds, and poor customer service.",
        "The hotel photos are misleading. Reality is much worse than advertised.",
        "Had high expectations but was let down. The food was bland and overpriced.",
        "Average hotel stay. Some good points but rooms could be cleaner.",
        "Decent enough for a short stay. Nothing special but adequate.",
        "Mixed feelings about this hotel. Good breakfast but rooms are small.",
    ]
    
    reviews = []
    for i, text in enumerate(sample_reviews):
        reviews.append({
            'text': text,
            'review_id': f'review_{i+1}',
            'length': len(text)
        })
    
    print(f"✅ Created {len(reviews)} sample reviews")
    return reviews

def analyze_with_simple_sentiment(reviews):
    """Simple sentiment analysis using basic keyword matching"""
    print("🔄 Analyzing sentiment with keyword matching...")
    
    positive_words = {'amazing', 'excellent', 'fantastic', 'beautiful', 'perfect', 'loved', 'great', 'wonderful'}
    negative_words = {'disappointing', 'terrible', 'poor', 'dirty', 'overpriced', 'uncomfortable', 'worst'}
    
    results = []
    for review in reviews:
        text_lower = review['text'].lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "POSITIVE"
            confidence = min(0.9, 0.6 + pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment = "NEGATIVE" 
            confidence = min(0.9, 0.6 + neg_count * 0.1)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        result = review.copy()
        result.update({
            'sentiment': sentiment,
            'confidence': confidence,
            'method': 'keyword_matching'
        })
        results.append(result)
    
    print("✅ Simple sentiment analysis complete")
    return results

def main():
    """Main function"""
    print("=" * 60)
    print("TRIPADVISOR SENTIMENT ANALYSIS DEMO")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Create sample data
    reviews = create_sample_reviews()
    
    # Analyze sentiment
    results = analyze_with_simple_sentiment(reviews)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Show distribution
    print(f"\n📊 Sentiment Distribution:")
    sentiment_counts = df_results['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_results)) * 100
        print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")
    
    # Save results
    output_file = os.path.join(OUTPUT_BASE_DIR, 'demo_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    df_results.to_csv(os.path.join(OUTPUT_BASE_DIR, 'demo_results.csv'), index=False)
    
    print(f"\n💾 Results saved to: {OUTPUT_BASE_DIR}")
    
    # Show sample results
    print(f"\n🎯 Sample Results:")
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        sample = df_results[df_results['sentiment'] == sentiment].head(1)
        if not sample.empty:
            row = sample.iloc[0]
            print(f"\n{sentiment}:")
            print(f"   Text: {row['text'][:80]}...")
            print(f"   Confidence: {row['confidence']:.3f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Check results in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()