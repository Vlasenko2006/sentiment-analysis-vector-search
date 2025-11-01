#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripAdvisor Review Sentiment Analysis with Vector Search
- Scrapes reviews from TripAdvisor hotels/restaurants
- Applies enhanced sentiment analysis with DistilBERT
- Uses vector search to find most representative reviews
- Creates comprehensive visualizations

Adapted from the enhanced sentiment analysis framework

Created on Thu Oct 30 2025
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
print("Confirming that model is alive and ready to use...")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Scraping parameters
MAX_REVIEWS_PER_PAGE = 10        # Reviews per page (TripAdvisor pagination)
MAX_PAGES = 20                   # Maximum pages to scrape
MIN_REVIEW_LENGTH = 10           # Minimum characters for a valid review
REQUEST_DELAY = 2                # Delay between requests (seconds)

# Processing parameters  
BATCH_SIZE = 50                  # Batch size for sentiment analysis processing
CONFIDENCE_THRESHOLD = 0.8       # Confidence threshold for 3-class simulation

# Vector search parameters
N_REPRESENTATIVES = 10           # Number of representative reviews to find per sentiment
TFIDF_MAX_FEATURES = 1000       # Maximum features for TF-IDF vectorization
TFIDF_MIN_DF = 2                # Minimum document frequency for TF-IDF
TFIDF_MAX_DF = 0.8              # Maximum document frequency for TF-IDF

# Visualization parameters
TOP_WORDS_COUNT = 15            # Number of top words to show in frequency analysis
WORDCLOUD_MAX_WORDS = 100       # Maximum words in word clouds

# File paths
CACHE_DIR = "/tmp/hf_cache"
MODEL_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/tripadvisor_analysis"

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
        
        print("� Initializing model pipeline...")
        pipe = pipeline("sentiment-analysis", model=MODEL_PATH, return_all_scores=False)
        print("✅ Model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def search_tripadvisor_attractions(city_name, max_attractions=5):
    """
    Search for top attractions in a given city on TripAdvisor
    Returns list of attraction URLs
    """
    print(f"\n🔍 Searching for top attractions in {city_name}...")
    
    # TripAdvisor search URL format
    search_url = f"https://www.tripadvisor.com/Search?q={city_name.replace(' ', '%20')}%20attractions"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        response = session.get(search_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find attraction links (TripAdvisor structure may vary)
        attraction_links = []
        
        # Look for attraction links in search results
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            # Look for attraction URLs
            if '/Attraction_Review-' in href and 'Reviews-' in href:
                full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                if full_url not in attraction_links:
                    attraction_links.append(full_url)
                    if len(attraction_links) >= max_attractions:
                        break
        
        # If no attraction links found, try alternative approach
        if not attraction_links:
            print(f"   🔄 Trying alternative search for {city_name}...")
            # Try direct city page approach
            city_slug = city_name.lower().replace(' ', '_').replace(',', '')
            city_url = f"https://www.tripadvisor.com/Attractions-g{hash(city_name) % 1000000}-Activities-{city_slug}.html"
            
            response = session.get(city_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href', '')
                    if '/Attraction_Review-' in href:
                        full_url = href if href.startswith('http') else f"https://www.tripadvisor.com{href}"
                        if full_url not in attraction_links:
                            attraction_links.append(full_url)
                            if len(attraction_links) >= max_attractions:
                                break
        
        if attraction_links:
            print(f"   ✅ Found {len(attraction_links)} attraction(s)")
            for i, url in enumerate(attraction_links[:3], 1):
                # Extract attraction name from URL for display
                attraction_name = url.split('Reviews-')[-1].split('.html')[0].replace('_', ' ')
                print(f"   {i}. {attraction_name}")
        else:
            print(f"   ⚠️  No attractions found for {city_name}")
            # Return some popular Paris attractions as fallback
            if 'paris' in city_name.lower():
                attraction_links = [
                    "https://www.tripadvisor.com/Attraction_Review-g187147-d188151-Reviews-Eiffel_Tower-Paris_Ile_de_France.html",
                    "https://www.tripadvisor.com/Attraction_Review-g187147-d188679-Reviews-Louvre_Museum-Paris_Ile_de_France.html",
                    "https://www.tripadvisor.com/Attraction_Review-g187147-d190379-Reviews-Arc_de_Triomphe-Paris_Ile_de_France.html"
                ]
                print(f"   💡 Using popular Paris attractions as fallback")
        
        return attraction_links
        
    except Exception as e:
        print(f"   ❌ Error searching for attractions: {e}")
        # Fallback for Paris
        if 'paris' in city_name.lower():
            print("   💡 Using fallback Paris attractions...")
            return [
                "https://www.tripadvisor.com/Attraction_Review-g187147-d188151-Reviews-Eiffel_Tower-Paris_Ile_de_France.html",
                "https://www.tripadvisor.com/Attraction_Review-g187147-d188679-Reviews-Louvre_Museum-Paris_Ile_de_France.html"
            ]
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
            
            response = session.get(page_url, timeout=10)
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
    plt.title('TripAdvisor Review Sentiment Distribution', fontsize=12, fontweight='bold')
    
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
    
    # 5. Sentiment by Rating (if available)
    if 'rating' in results_df.columns and results_df['rating'].notna().any():
        plt.subplot(2, 3, 5)
        avg_confidence = results_df.groupby(['rating', 'sentiment'])['confidence'].mean().unstack()
        avg_confidence.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.title('Average Confidence by Rating')
        plt.xlabel('TripAdvisor Rating')
        plt.ylabel('Average Confidence')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=0)
    
    # 6. Reviews per page
    plt.subplot(2, 3, 6)
    if 'page' in results_df.columns:
        page_counts = results_df['page'].value_counts().sort_index()
        plt.bar(page_counts.index, page_counts.values, alpha=0.7, color='orange')
        plt.xlabel('Page Number')
        plt.ylabel('Number of Reviews')
        plt.title('Reviews Scraped per Page')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['visualizations'], 'tripadvisor_analysis_overview.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return sentiment_counts

def create_paris_attraction_reviews():
    """
    Create realistic Paris attraction reviews for demonstration
    Since TripAdvisor blocks scraping, this provides realistic test data
    """
    print("🎭 Creating realistic Paris attraction reviews for analysis...")
    
    paris_reviews = [
        # Eiffel Tower reviews
        {"text": "The Eiffel Tower is absolutely magnificent! The views from the top are breathtaking and the iron structure is an engineering marvel. A must-visit in Paris!", "rating": 5, "date": "2024-10-15", "attraction": "Eiffel Tower"},
        {"text": "Very crowded and expensive. The lines were extremely long and the elevator was packed. The view is nice but not worth the hassle and cost.", "rating": 2, "date": "2024-10-14", "attraction": "Eiffel Tower"},
        {"text": "Iconic landmark that lives up to the hype. Beautiful at night when it's lit up. I recommend going in the evening for the best experience.", "rating": 4, "date": "2024-10-13", "attraction": "Eiffel Tower"},
        {"text": "Overrated tourist trap. Too many people, overpriced tickets, and the security checks take forever. Paris has better attractions to visit.", "rating": 2, "date": "2024-10-12", "attraction": "Eiffel Tower"},
        {"text": "Amazing experience! Went during sunset and the views were spectacular. The tower is even more beautiful than in photos.", "rating": 5, "date": "2024-10-11", "attraction": "Eiffel Tower"},
        
        # Louvre Museum reviews
        {"text": "The Louvre is incredible! So much art and history in one place. The Mona Lisa is smaller than expected but the museum itself is a masterpiece.", "rating": 5, "date": "2024-10-10", "attraction": "Louvre Museum"},
        {"text": "Way too big and overwhelming. You need at least 2 days to see everything properly. The crowds around Mona Lisa are insane.", "rating": 3, "date": "2024-10-09", "attraction": "Louvre Museum"},
        {"text": "World-class museum with amazing collections. The building itself is as beautiful as the art inside. Book tickets in advance!", "rating": 5, "date": "2024-10-08", "attraction": "Louvre Museum"},
        {"text": "Disappointing visit. Too crowded, poor organization, and many exhibits were closed. The audio guide was not worth the extra cost.", "rating": 2, "date": "2024-10-07", "attraction": "Louvre Museum"},
        {"text": "Absolutely stunning museum. The Egyptian collection is fantastic and the architecture is breathtaking. A cultural treasure.", "rating": 4, "date": "2024-10-06", "attraction": "Louvre Museum"},
        
        # Arc de Triomphe reviews
        {"text": "Beautiful monument with great views of the Champs-Élysées. The climb to the top is worth it for the panoramic views of Paris.", "rating": 4, "date": "2024-10-05", "attraction": "Arc de Triomphe"},
        {"text": "Nice monument but nothing special. The area is very touristy and crowded. You can see it well enough from the street.", "rating": 3, "date": "2024-10-04", "attraction": "Arc de Triomphe"},
        {"text": "Impressive architecture and rich history. The view from the top gives you a great perspective of Paris' layout. Highly recommended!", "rating": 5, "date": "2024-10-03", "attraction": "Arc de Triomphe"},
        {"text": "Overpriced for what it is. Just a monument you can walk around for free. Don't waste money going to the top.", "rating": 2, "date": "2024-10-02", "attraction": "Arc de Triomphe"},
        
        # Notre-Dame (exterior) reviews
        {"text": "Even under restoration, Notre-Dame remains magnificent. The gothic architecture is stunning and the area around it is beautiful.", "rating": 4, "date": "2024-10-01", "attraction": "Notre-Dame"},
        {"text": "Sad to see it damaged but still impressive from the outside. Looking forward to seeing it fully restored in the future.", "rating": 3, "date": "2024-09-30", "attraction": "Notre-Dame"},
        {"text": "A masterpiece of gothic architecture. The façade is incredibly detailed. Can't wait for the restoration to be complete.", "rating": 4, "date": "2024-09-29", "attraction": "Notre-Dame"},
        
        # Sacré-Cœur reviews
        {"text": "Beautiful basilica with amazing views over Paris. The climb up to Montmartre is worth it. Free to enter and very peaceful inside.", "rating": 5, "date": "2024-09-28", "attraction": "Sacré-Cœur"},
        {"text": "Nice church but the area is full of scammers and pickpockets. Be very careful with your belongings in Montmartre.", "rating": 2, "date": "2024-09-27", "attraction": "Sacré-Cœur"},
        {"text": "Stunning basilica and the views from the steps are incredible. Great place to watch the sunset over Paris.", "rating": 4, "date": "2024-09-26", "attraction": "Sacré-Cœur"},
        
        # Seine River Cruise reviews
        {"text": "Relaxing way to see Paris from a different perspective. The evening cruise with dinner was romantic and the views were beautiful.", "rating": 4, "date": "2024-09-25", "attraction": "Seine River Cruise"},
        {"text": "Boring and overpriced. You see the same buildings you can see from the street. The commentary was not interesting.", "rating": 2, "date": "2024-09-24", "attraction": "Seine River Cruise"},
        {"text": "Great way to see multiple landmarks in one trip. The boat was comfortable and the guide was knowledgeable about Paris history.", "rating": 4, "date": "2024-09-23", "attraction": "Seine River Cruise"},
        
        # Musée d'Orsay reviews
        {"text": "Fantastic museum for Impressionist art! Van Gogh, Monet, and Renoir collections are outstanding. Much more manageable than the Louvre.", "rating": 5, "date": "2024-09-22", "attraction": "Musée d'Orsay"},
        {"text": "Beautiful building and great art collection. Less crowded than other major museums. The Van Gogh paintings are incredible.", "rating": 4, "date": "2024-09-21", "attraction": "Musée d'Orsay"},
        {"text": "Good museum but limited collection compared to other major museums. The building itself is more interesting than some exhibits.", "rating": 3, "date": "2024-09-20", "attraction": "Musée d'Orsay"}
    ]
    
    # Add metadata to each review
    reviews = []
    for i, review_data in enumerate(paris_reviews):
        reviews.append({
            **review_data,
            'reviewer': f'TravelLover_{i+1}',
            'helpful_votes': random.randint(0, 25),
            'review_id': f'paris_review_{i+1:03d}'
        })
    
    print(f"✅ Created {len(reviews)} realistic Paris attraction reviews")
    return reviews

def create_sample_tripadvisor_data():
    """Create sample TripAdvisor data for demo/testing"""
    print("🔍 Creating sample TripAdvisor reviews for demonstration...")
    
    sample_reviews = [
        # Positive reviews
        {"text": "Amazing hotel! The staff was incredibly helpful and the room was spotless. Breakfast was delicious and the location is perfect for exploring the city.", "rating": 5},
        {"text": "Absolutely loved our stay here. The spa was fantastic and the pool area was beautiful. Will definitely come back!", "rating": 5},
        {"text": "Excellent service from check-in to check-out. The concierge helped us with great restaurant recommendations. Highly recommend!", "rating": 5},
        {"text": "Beautiful hotel with stunning views. The room was spacious and comfortable. Great value for money.", "rating": 4},
        {"text": "Perfect location right in the heart of the city. Clean rooms, friendly staff, and amazing amenities.", "rating": 4},
        {"text": "Outstanding experience! The hotel exceeded our expectations in every way. The dinner at their restaurant was exceptional.", "rating": 5},
        {"text": "Wonderful stay with my family. Kids loved the pool and we appreciated the family-friendly atmosphere.", "rating": 4},
        {"text": "Great hotel for business travel. Fast wifi, comfortable workspace in room, and convenient location.", "rating": 4},
        
        # Negative reviews  
        {"text": "Disappointing stay. The room was dirty and the air conditioning didn't work properly. Staff was unhelpful.", "rating": 2},
        {"text": "Overpriced for what you get. The hotel is outdated and needs serious renovation. Would not recommend.", "rating": 2},
        {"text": "Terrible experience. Noisy rooms, uncomfortable beds, and poor customer service. Avoid this place.", "rating": 1},
        {"text": "The hotel photos are misleading. Reality is much worse. Old furniture and poor maintenance throughout.", "rating": 2},
        {"text": "Had high expectations but was let down. The restaurant food was bland and overpriced.", "rating": 2},
        {"text": "Very poor value for money. Small rooms, no amenities promised, and rude staff at reception.", "rating": 1},
        
        # Mixed/Neutral reviews
        {"text": "Average hotel stay. Some good points like location, but rooms could be cleaner and staff more helpful.", "rating": 3},
        {"text": "Decent enough for a short stay. Nothing special but nothing terrible either. Just okay overall.", "rating": 3},
        {"text": "Mixed feelings about this hotel. Good breakfast but rooms are small and wifi is slow.", "rating": 3},
        {"text": "The hotel has potential but needs improvement in several areas. Service was inconsistent.", "rating": 3},
        {"text": "Okay for the price point. Don't expect luxury but it's adequate for basic accommodation needs.", "rating": 3},
    ]
    
    # Add metadata to each review
    reviews = []
    for i, review_data in enumerate(sample_reviews):
        reviews.append({
            **review_data,
            'reviewer': f'Traveler_{i+1}',
            'page': (i // 10) + 1,
            'source_url': 'https://tripadvisor.com/demo'
        })
    
    print(f"✅ Created {len(reviews)} sample reviews")
    return reviews

def main():
    """Main function to run TripAdvisor sentiment analysis"""
    print("=" * 80)
    print("TRIPADVISOR REVIEW SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Setup
    folders = setup_environment()
    url = "Unknown"  # Initialize URL variable
    
    # Ask user for mode
    print("\n🔧 Choose analysis mode:")
    print("1. Demo mode (use sample data - faster)")
    print("2. City attractions mode (e.g., 'Paris' - analyzes top attractions)")
    print("3. Specific URL mode (enter exact TripAdvisor URL)")
    
    try:
        mode = input("\nEnter choice (1, 2, or 3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled by user")
        return
    
    if mode == "1":
        print("\n📊 Running in DEMO mode with sample data...")
        reviews = create_sample_tripadvisor_data()
        url = "Demo Mode (Sample Data)"
        
    elif mode == "2":
        print("\n� Running in CITY ATTRACTIONS mode...")
        print("Enter a city name to analyze its top attractions (e.g., 'Paris', 'New York', 'Tokyo')")
        
        try:
            city_name = input("\nEnter city name: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled by user")
            return
            
        if not city_name:
            print("❌ Please provide a city name")
            return
        
        # Search for attractions in the city
        attraction_urls = search_tripadvisor_attractions(city_name, max_attractions=3)
        
        if not attraction_urls:
            print(f"⚠️  TripAdvisor is blocking searches (403 error)")
            if 'paris' in city_name.lower():
                print("🎭 Using realistic Paris attraction review dataset instead...")
                reviews = create_paris_attraction_reviews()
                url = "Paris Attractions (Realistic Sample Data)"
            else:
                print(f"💡 Using general sample data for {city_name} analysis...")
                reviews = create_sample_tripadvisor_data()
                url = f"{city_name} Attractions (Sample Data)"
        else:
            # Collect reviews from all attractions
            all_reviews = []
            for i, attraction_url in enumerate(attraction_urls, 1):
                print(f"\n🏛️  Processing attraction {i}/{len(attraction_urls)}...")
                attraction_reviews = extract_tripadvisor_reviews(attraction_url, max_pages=2)  # 2 pages per attraction
                all_reviews.extend(attraction_reviews)
                
                if len(all_reviews) >= 50:  # Limit total reviews
                    break
            
            reviews = all_reviews
            url = f"{city_name} Attractions ({len(attraction_urls)} attractions)"
            
            # If scraping failed (TripAdvisor blocking), use realistic sample data
            if len(reviews) == 0:
                print("⚠️  TripAdvisor is blocking direct scraping (403 error)")
                if 'paris' in city_name.lower():
                    print("🎭 Using realistic Paris attraction review dataset instead...")
                    reviews = create_paris_attraction_reviews()
                    url = "Paris Attractions (Realistic Sample Data)"
                else:
                    print("💡 Using general sample data for analysis...")
                    reviews = create_sample_tripadvisor_data()
                    url = f"{city_name} Attractions (Sample Data)"
            
    elif mode == "3":
        print("\n🔗 Running in SPECIFIC URL mode...")
        print("Please provide a TripAdvisor URL for a hotel, restaurant, or attraction.")
        print("Example: https://www.tripadvisor.com/Hotel_Review-g123456-d123456-Reviews-Hotel_Name.html")
        
        try:
            url = input("\nEnter TripAdvisor URL: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled by user")
            return
            
        if not url or 'tripadvisor.com' not in url:
            print("❌ Please provide a valid TripAdvisor URL")
            return
            
        # Scrape reviews
        reviews = extract_tripadvisor_reviews(url, MAX_PAGES)
        
        if len(reviews) == 0:
            print("❌ No reviews found. The URL might be incorrect or the page structure has changed.")
            print("💡 Tip: Try demo mode (option 1) to test the analysis pipeline")
            return
    else:
        print("❌ Invalid choice. Please run the script again and choose 1, 2, or 3.")
        return
    
    # Load model (this is where the delay happens)
    pipe = load_sentiment_model()
    
    if pipe is None:
        print("💡 Tip: Try demo mode to test without the ML model")
        return
    
    # Convert to DataFrame
    df_reviews = pd.DataFrame(reviews)
    
    # Save raw data
    raw_data_file = os.path.join(folders['raw_data'], 'scraped_reviews.json')
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    df_reviews.to_csv(os.path.join(folders['raw_data'], 'scraped_reviews.csv'), index=False)
    
    print(f"\n📊 Processing {len(df_reviews)} reviews with DistilBERT...")
    
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
    print("✅ TRIPADVISOR ANALYSIS COMPLETE!")
    print("=" * 80)
    
    total_reviews = len(results_df)
    print(f"""
📊 Analysis Summary:
   • Total reviews analyzed: {total_reviews:,}
   • Positive reviews: {sentiment_counts.get('POSITIVE', 0):,} ({sentiment_counts.get('POSITIVE', 0)/total_reviews*100:.1f}%)
   • Negative reviews: {sentiment_counts.get('NEGATIVE', 0):,} ({sentiment_counts.get('NEGATIVE', 0)/total_reviews*100:.1f}%)
   • Neutral reviews: {sentiment_counts.get('NEUTRAL', 0):,} ({sentiment_counts.get('NEUTRAL', 0)/total_reviews*100:.1f}%)
   
📁 Results saved to: {OUTPUT_BASE_DIR}
🌐 Source: {url}
""")
    
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

if __name__ == "__main__":
    main()