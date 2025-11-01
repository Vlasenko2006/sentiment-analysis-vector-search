#!/usr/bin/env python3
"""
Review Content Visualizer
=========================

Comprehensive analysis tool for extracted TripAdvisor review content
with matplotlib visualizations and vector search.

Features:
- Load reviews from SQLite database
- Sentiment analysis
- Comprehensive visualizations
- Vector-based semantic search
- Representative review finding
- Statistical analysis
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIDENCE_THRESHOLD = 0.8
N_REPRESENTATIVES = 10
TFIDF_MAX_FEATURES = 1000
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/dNet/review_analysis_output"

class ReviewVisualizer:
    """Review analyzer with comprehensive visualizations"""
    
    def __init__(self, db_path: str = "reviews_database.db"):
        self.db_path = db_path
        self.reviews_df = None
        self.sentiment_analyzer = None
        self.vectorizer = None
        self.vectors = None
        self.setup_environment()
        self.load_ml_libraries()
    
    def setup_environment(self):
        """Setup output directories"""
        self.folders = {
            'positive': os.path.join(OUTPUT_BASE_DIR, 'positive'),
            'negative': os.path.join(OUTPUT_BASE_DIR, 'negative'), 
            'neutral': os.path.join(OUTPUT_BASE_DIR, 'neutral'),
            'visualizations': os.path.join(OUTPUT_BASE_DIR, 'visualizations'),
            'raw_data': os.path.join(OUTPUT_BASE_DIR, 'raw_data')
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        logger.info(f"✅ Output directories created in: {OUTPUT_BASE_DIR}")
    
    def load_ml_libraries(self):
        """Load machine learning libraries"""
        try:
            from transformers import pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.cluster import KMeans
            from wordcloud import WordCloud
            
            # Store in self for later use
            self.TfidfVectorizer = TfidfVectorizer
            self.cosine_similarity = cosine_similarity
            self.KMeans = KMeans
            self.WordCloud = WordCloud
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=False
            )
            
            logger.info("✅ ML libraries and sentiment analyzer loaded")
            
        except ImportError as e:
            logger.warning(f"⚠️ ML libraries not available: {e}")
            self.sentiment_analyzer = None
    
    def load_data_from_db(self) -> pd.DataFrame:
        """Load review data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT file_path, context, confidence_score, keyword, line_number
                FROM review_matches 
                WHERE confidence_score > 0.3
                ORDER BY confidence_score DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Clean and process data
            df['clean_text'] = df['context'].apply(self.clean_text)
            df = df[df['clean_text'].str.len() > 20].copy()
            
            # Add sentiment analysis
            if self.sentiment_analyzer:
                logger.info("🧠 Analyzing sentiment...")
                sentiments = []
                sentiment_scores = []
                
                for text in df['clean_text']:
                    result = self.analyze_sentiment(text)
                    sentiments.append(result['sentiment'])
                    sentiment_scores.append(result['confidence'])
                
                df['sentiment'] = sentiments
                df['sentiment_score'] = sentiment_scores
            else:
                df['sentiment'] = 'neutral'
                df['sentiment_score'] = 0.5
            
            # Add derived features
            df['text_length'] = df['clean_text'].str.len()
            df['file_extension'] = df['file_path'].apply(lambda x: os.path.splitext(x)[1])
            df['file_name'] = df['file_path'].apply(lambda x: os.path.basename(x))
            
            self.reviews_df = df
            logger.info(f"📊 Loaded {len(df)} reviews for analysis")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not text or pd.isna(text):
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove JavaScript/CSS artifacts
        text = re.sub(r'function\s*\([^)]*\)\s*{[^}]*}', '', text)
        text = re.sub(r'var\s+\w+\s*=\s*[^;]+;', '', text)
        text = re.sub(r'[{}\[\]();]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment with enhanced logic"""
        if not self.sentiment_analyzer:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            result = self.sentiment_analyzer(text[:512])
            raw_label = result[0]['label']
            confidence = result[0]['score']
            
            # 3-class simulation with confidence thresholds
            if raw_label == "POSITIVE":
                sentiment = "POSITIVE" if confidence > CONFIDENCE_THRESHOLD else "NEUTRAL"
            else:  # NEGATIVE
                sentiment = "NEGATIVE" if confidence > CONFIDENCE_THRESHOLD else "NEUTRAL"
            
            return {'sentiment': sentiment, 'confidence': confidence}
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations like in tripadvisor_multi_url_analysis.py"""
        if self.reviews_df is None or len(self.reviews_df) == 0:
            logger.warning("⚠️ No data to visualize")
            return
        
        # Setup matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        df = self.reviews_df
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        # Create comprehensive plot (similar to tripadvisor_multi_url_analysis.py)
        plt.figure(figsize=(20, 15))
        
        # 1. Sentiment Distribution (Pie Chart)
        plt.subplot(3, 4, 1)
        sizes = [sentiment_counts.get(s, 0) for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']]
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.05, 0.05, 0.05)
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=explode, shadow=True)
        plt.title('Review Sentiment Distribution', fontsize=12, fontweight='bold')
        
        # 2. Confidence Score Distribution
        plt.subplot(3, 4, 2)
        confidences = df['confidence_score'].values
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(confidences.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidences.mean():.3f}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()
        
        # 3. Sentiment Score Distribution
        plt.subplot(3, 4, 3)
        sentiment_scores = df['sentiment_score'].values
        plt.hist(sentiment_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Sentiment Score Distribution')
        
        # 4. Review Length Distribution
        plt.subplot(3, 4, 4)
        text_lengths = df['text_length'].values
        plt.hist(text_lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Review Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Review Length Distribution')
        
        # 5. Confidence vs Sentiment Score Scatter
        plt.subplot(3, 4, 5)
        sentiment_colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
        for sentiment in sentiment_colors:
            mask = df['sentiment'] == sentiment
            if mask.any():
                plt.scatter(df.loc[mask, 'confidence_score'], 
                           df.loc[mask, 'sentiment_score'], 
                           c=sentiment_colors[sentiment], 
                           label=sentiment, alpha=0.6)
        plt.xlabel('Confidence Score')
        plt.ylabel('Sentiment Score')
        plt.title('Confidence vs Sentiment Score')
        plt.legend()
        
        # 6. File Type Analysis
        plt.subplot(3, 4, 6)
        file_type_counts = df['file_extension'].value_counts()
        if len(file_type_counts) > 0:
            plt.bar(file_type_counts.index, file_type_counts.values, alpha=0.7, color='purple')
            plt.xlabel('File Extension')
            plt.ylabel('Number of Reviews')
            plt.title('Reviews by File Type')
            plt.xticks(rotation=45)
        
        # 7. Top Keywords
        plt.subplot(3, 4, 7)
        keyword_counts = df['keyword'].value_counts().head(10)
        if len(keyword_counts) > 0:
            plt.barh(range(len(keyword_counts)), keyword_counts.values, alpha=0.7, color='teal')
            plt.yticks(range(len(keyword_counts)), keyword_counts.index)
            plt.xlabel('Frequency')
            plt.title('Top Keywords')
        
        # 8. Sentiment by Confidence Level
        plt.subplot(3, 4, 8)
        df['conf_level'] = pd.cut(df['confidence_score'], 
                                 bins=[0, 0.5, 0.8, 1.0], 
                                 labels=['Low (<0.5)', 'Medium (0.5-0.8)', 'High (>0.8)'])
        
        conf_sentiment = pd.crosstab(df['conf_level'], df['sentiment'])
        conf_sentiment.plot(kind='bar', ax=plt.gca(), 
                           color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.xlabel('Confidence Level')
        plt.ylabel('Number of Reviews')
        plt.title('Sentiment by Confidence Level')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        
        # 9. Text Length vs Confidence
        plt.subplot(3, 4, 9)
        plt.scatter(df['text_length'], df['confidence_score'], alpha=0.5, color='brown')
        plt.xlabel('Text Length')
        plt.ylabel('Confidence Score')
        plt.title('Text Length vs Confidence')
        
        # 10. Top Files with Most Reviews
        plt.subplot(3, 4, 10)
        top_files = df['file_name'].value_counts().head(8)
        if len(top_files) > 0:
            plt.barh(range(len(top_files)), top_files.values, alpha=0.7, color='gold')
            plt.yticks(range(len(top_files)), [f[:20] + '...' if len(f) > 20 else f for f in top_files.index])
            plt.xlabel('Number of Reviews')
            plt.title('Top Files by Review Count')
        
        # 11. Sentiment Score vs Confidence Correlation
        plt.subplot(3, 4, 11)
        plt.scatter(df['sentiment_score'], df['confidence_score'], alpha=0.5, color='navy')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Confidence Score')
        plt.title('Sentiment Score vs Confidence')
        
        # Add correlation coefficient
        correlation = df['sentiment_score'].corr(df['confidence_score'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=10)
        
        # 12. Summary Statistics
        plt.subplot(3, 4, 12)
        total_reviews = len(df)
        high_conf_count = (df['confidence_score'] > 0.8).sum()
        avg_confidence = df['confidence_score'].mean()
        avg_text_length = df['text_length'].mean()
        
        stats_text = f"""
Total Reviews: {total_reviews:,}

Sentiment Distribution:
• Positive: {sentiment_counts.get('POSITIVE', 0):,}
• Negative: {sentiment_counts.get('NEGATIVE', 0):,}
• Neutral: {sentiment_counts.get('NEUTRAL', 0):,}

Quality Metrics:
• High Confidence (>0.8): {high_conf_count:,}
• Avg Confidence: {avg_confidence:.3f}
• Avg Text Length: {avg_text_length:.0f}

File Types: {df['file_extension'].nunique()}
"""
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='center')
        plt.axis('off')
        plt.title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        viz_path = os.path.join(self.folders['visualizations'], 'comprehensive_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comprehensive visualizations saved to: {viz_path}")
        plt.show()
        
        return sentiment_counts
    
    def generate_wordcloud(self):
        """Generate word cloud from review texts"""
        if self.reviews_df is None or len(self.reviews_df) == 0:
            logger.warning("⚠️ No data for word cloud")
            return None
        
        try:
            # Combine all review texts
            all_text = ' '.join(self.reviews_df['clean_text'].dropna())
            
            if hasattr(self, 'WordCloud'):
                wordcloud = self.WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(all_text)
                
                plt.figure(figsize=(15, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Review Content Word Cloud', fontsize=16, fontweight='bold')
                
                wordcloud_path = os.path.join(self.folders['visualizations'], 'review_wordcloud.png')
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                logger.info(f"☁️ Word cloud saved to: {wordcloud_path}")
                return wordcloud_path
            else:
                logger.warning("⚠️ WordCloud library not available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Word cloud generation failed: {e}")
            return None
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform semantic search using TF-IDF vectors"""
        if self.reviews_df is None or len(self.reviews_df) == 0:
            return []
        
        try:
            if not hasattr(self, 'TfidfVectorizer'):
                logger.warning("⚠️ TfidfVectorizer not available")
                return []
            
            # Create vectors if not exists
            if self.vectorizer is None:
                texts = self.reviews_df['clean_text'].dropna().tolist()
                if len(texts) < 2:
                    return []
                
                self.vectorizer = self.TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
                self.vectors = self.vectorizer.fit_transform(texts)
            
            # Search
            query_vector = self.vectorizer.transform([query])
            similarities = self.cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    row = self.reviews_df.iloc[idx]
                    result_data = {
                        'text': row['clean_text'][:300] + '...',
                        'file': os.path.basename(row['file_path']),
                        'confidence': row['confidence_score'],
                        'sentiment': row['sentiment'],
                        'sentiment_score': row['sentiment_score'],
                        'keywords': row['keyword']
                    }
                    results.append((result_data, similarities[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def save_organized_results(self):
        """Save results organized by sentiment"""
        if self.reviews_df is None:
            return
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_data = self.reviews_df[self.reviews_df['sentiment'] == sentiment]
            
            if len(sentiment_data) > 0:
                # Save detailed results
                results = []
                for _, row in sentiment_data.iterrows():
                    results.append({
                        'text': row['clean_text'],
                        'file_path': row['file_path'],
                        'confidence': row['confidence_score'],
                        'sentiment_score': row['sentiment_score'],
                        'keyword': row['keyword'],
                        'text_length': row['text_length']
                    })
                
                # Save to JSON
                output_file = os.path.join(self.folders[sentiment.lower()], f'{sentiment.lower()}_reviews.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 {sentiment}: {len(results)} reviews saved")
    
    def print_examples(self, sentiment: str = "all", limit: int = 3):
        """Print example reviews"""
        if self.reviews_df is None:
            return
        
        if sentiment == "all":
            examples = self.reviews_df.nlargest(limit, 'confidence_score')
        else:
            examples = self.reviews_df[self.reviews_df['sentiment'] == sentiment.upper()]
            examples = examples.nlargest(limit, 'confidence_score')
        
        print(f"\n🎯 Top {len(examples)} {sentiment.title()} Review Examples:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            print(f"\n{i}. File: {os.path.basename(row['file_path'])}")
            print(f"   Sentiment: {row['sentiment']} (score: {row['sentiment_score']:.3f})")
            print(f"   Confidence: {row['confidence_score']:.3f}")
            print(f"   Keyword: {row['keyword']}")
            print(f"   Text: {row['clean_text'][:300]}...")
            print("-" * 80)
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        if self.reviews_df is None:
            print("❌ No data loaded")
            return
        
        df = self.reviews_df
        total_reviews = len(df)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE REVIEW ANALYSIS COMPLETE!")
        print("=" * 80)
        
        print(f"""
📊 Analysis Summary:
   • Total reviews analyzed: {total_reviews:,}
   • Positive reviews: {sentiment_counts.get('POSITIVE', 0):,} ({sentiment_counts.get('POSITIVE', 0)/total_reviews*100:.1f}%)
   • Negative reviews: {sentiment_counts.get('NEGATIVE', 0):,} ({sentiment_counts.get('NEGATIVE', 0)/total_reviews*100:.1f}%)
   • Neutral reviews: {sentiment_counts.get('NEUTRAL', 0):,} ({sentiment_counts.get('NEUTRAL', 0)/total_reviews*100:.1f}%)
   
📁 Results saved to: {OUTPUT_BASE_DIR}
📊 Visualizations: {self.folders['visualizations']}
""")
        
        # Top files analysis
        top_files = df['file_name'].value_counts().head(10)
        print("📁 Top files with review content:")
        for i, (file_name, count) in enumerate(top_files.items(), 1):
            print(f"   {i}. {file_name}: {count} matches")
        
        # High confidence breakdown
        high_conf = df[df['confidence_score'] > 0.8]
        if len(high_conf) > 0:
            high_conf_sentiment = high_conf['sentiment'].value_counts()
            print(f"\n🎯 High Confidence Reviews (>{CONFIDENCE_THRESHOLD}):")
            for sentiment, count in high_conf_sentiment.items():
                print(f"   {sentiment}: {count} reviews")

def main():
    """Main analysis function"""
    print("🏮" * 30)
    print("COMPREHENSIVE REVIEW CONTENT ANALYZER")
    print("🏮" * 30)
    
    # Initialize analyzer
    analyzer = ReviewVisualizer()
    
    # Load data
    print("\n📊 Loading review data from database...")
    df = analyzer.load_data_from_db()
    
    if df.empty:
        print("❌ No review data found. Run review_extractor.py first.")
        return
    
    # Create visualizations
    print("\n📊 Creating comprehensive visualizations...")
    sentiment_counts = analyzer.create_comprehensive_visualizations()
    
    # Generate word cloud
    print("\n☁️ Generating word cloud...")
    analyzer.generate_wordcloud()
    
    # Save organized results
    print("\n💾 Saving organized results...")
    analyzer.save_organized_results()
    
    # Print summary
    analyzer.print_summary_report()
    
    # Show examples
    analyzer.print_examples("positive", 3)
    analyzer.print_examples("negative", 3)
    
    # Semantic search demo
    print(f"\n🔍 Semantic Search Demo:")
    search_queries = ["temple experience", "good visit", "disappointing trip", "tripadvisor review"]
    
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        results = analyzer.semantic_search(query, top_k=3)
        
        if results:
            for i, (review_data, similarity) in enumerate(results, 1):
                print(f"      {i}. Similarity: {similarity:.3f} | {review_data['sentiment']}")
                print(f"         File: {review_data['file']}")
                print(f"         Text: {review_data['text'][:150]}...")
        else:
            print("      No relevant matches found")
    
    print(f"\n🎉 Analysis complete! Check {OUTPUT_BASE_DIR} for detailed results.")

if __name__ == "__main__":
    main()