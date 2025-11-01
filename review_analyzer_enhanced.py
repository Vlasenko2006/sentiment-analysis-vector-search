#!/usr/bin/env python3
"""
Review Content Analyzer with Visualizations
===========================================

Enhanced analysis tool that combines review extraction from local files
with comprehensive visualizations and vector search capabilities.

Features:
- Extract reviews from downloaded TripAdvisor files
- Sentiment analysis with DistilBERT
- Interactive visualizations
- Vector-based semantic search
- Representative review finding
- Word clouds and statistical analysis
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
from dataclasses import dataclass
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIDENCE_THRESHOLD = 0.8
N_REPRESENTATIVES = 10
TFIDF_MAX_FEATURES = 1000
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/dNet/review_analysis_output"

@dataclass
class ReviewContent:
    """Enhanced review content with analysis"""
    text: str
    file_path: str
    confidence: float
    keywords: List[str]
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    rating: Optional[int] = None
    reviewer: str = "Anonymous"

class ComprehensiveReviewAnalyzer:
    """Complete review analyzer with ML and visualizations"""
    
    def __init__(self, db_path: str = "reviews_database.db"):
        self.db_path = db_path
        self.reviews = []
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
            global pipeline, TfidfVectorizer, cosine_similarity, KMeans, WordCloud
            
            from transformers import pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.cluster import KMeans
            from wordcloud import WordCloud
            
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
    
    def load_data_from_db(self) -> List[ReviewContent]:
        """Load review data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT file_path, context, confidence_score, keyword
                FROM review_matches 
                WHERE confidence_score > 0.5
                ORDER BY confidence_score DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            reviews = []
            for _, row in df.iterrows():
                # Clean and process context
                context = self.clean_text(row['context'])
                if len(context) > 20:  # Filter out very short contexts
                    review = ReviewContent(
                        text=context,
                        file_path=row['file_path'],
                        confidence=row['confidence_score'],
                        keywords=[row['keyword']]
                    )
                    reviews.append(review)
            
            self.reviews = reviews
            logger.info(f"📊 Loaded {len(reviews)} review contents")
            return reviews
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove JavaScript/CSS artifacts
        text = re.sub(r'function\s*\([^)]*\)\s*{[^}]*}', '', text)
        text = re.sub(r'var\s+\w+\s*=\s*[^;]+;', '', text)
        text = re.sub(r'[{}\[\]();]', ' ', text)
        
        # Keep only meaningful content
        if len(text.strip()) < 20:
            return ""
        
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text"""
        if self.sentiment_analyzer and len(text) > 10:
            try:
                results = self.sentiment_analyzer(text[:512])  # Limit text length
                # Get the highest scoring sentiment
                best_result = max(results[0], key=lambda x: x['score'])
                label = best_result['label'].lower()
                score = best_result['score']
                
                # Map labels to standard format
                if 'positive' in label:
                    return 'positive', score
                elif 'negative' in label:
                    return 'negative', score
                else:
                    return 'neutral', score
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
        
        # Fallback: simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'recommend']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if pos_score > neg_score:
            return 'positive', 0.6 + min(pos_score * 0.1, 0.3)
        elif neg_score > pos_score:
            return 'negative', 0.6 + min(neg_score * 0.1, 0.3)
        else:
            return 'neutral', 0.5
    
    def setup_vector_search(self):
        """Setup vector search capabilities"""
        if not self.reviews:
            logger.warning("⚠️ No reviews loaded for vector search")
            return
        
        # Prepare texts for vectorization
        texts = [review.text for review in self.reviews if len(review.text) > 20]
        
        if len(texts) < 2:
            logger.warning("⚠️ Not enough texts for vector search")
            return
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            self.vectors = self.vectorizer.fit_transform(texts)
            logger.info(f"✅ Vector search ready with {self.vectors.shape[0]} documents")
        except Exception as e:
            logger.error(f"❌ Vector search setup failed: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[ReviewContent, float]]:
        """Perform semantic search using vector similarity"""
        if not self.vectorizer or self.vectors is None:
            self.setup_vector_search()
            if not self.vectorizer:
                return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((self.reviews[idx], similarities[idx]))
            
            return results
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create comprehensive visualizations"""
        if not self.reviews:
            self.load_data_from_db()
        
        figures = {}
        
        # 1. Confidence Score Distribution
        confidences = [r.confidence for r in self.reviews]
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name="Confidence Distribution",
            marker_color=self.colors['primary']
        ))
        fig_conf.update_layout(
            title="📊 Confidence Score Distribution of Found Reviews",
            xaxis_title="Confidence Score",
            yaxis_title="Number of Reviews",
            template="plotly_white"
        )
        figures['confidence'] = fig_conf
        
        # 2. File Type Analysis
        file_types = {}
        for review in self.reviews:
            ext = os.path.splitext(review.file_path)[1] or 'no_ext'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        fig_files = go.Figure()
        fig_files.add_trace(go.Bar(
            x=list(file_types.keys()),
            y=list(file_types.values()),
            marker_color=self.colors['secondary']
        ))
        fig_files.update_layout(
            title="📁 Review Content by File Type",
            xaxis_title="File Extension",
            yaxis_title="Number of Reviews",
            template="plotly_white"
        )
        figures['file_types'] = fig_files
        
        # 3. Sentiment Analysis
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        
        for review in self.reviews[:100]:  # Analyze first 100 for performance
            sentiment, score = self.analyze_sentiment(review.text)
            sentiments[sentiment] += 1
            sentiment_scores.append({'sentiment': sentiment, 'score': score, 'text': review.text[:100]})
        
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Bar(
            x=list(sentiments.keys()),
            y=list(sentiments.values()),
            marker_color=[self.colors['positive'], self.colors['negative'], self.colors['neutral']]
        ))
        fig_sentiment.update_layout(
            title="😊 Sentiment Analysis of Review Content",
            xaxis_title="Sentiment",
            yaxis_title="Number of Reviews",
            template="plotly_white"
        )
        figures['sentiment'] = fig_sentiment
        
        # 4. Top Keywords
        all_keywords = {}
        for review in self.reviews:
            for keyword in review.keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
        
        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
        
        fig_keywords = go.Figure()
        fig_keywords.add_trace(go.Bar(
            x=[k[1] for k in top_keywords],
            y=[k[0] for k in top_keywords],
            orientation='h',
            marker_color=self.colors['tripadvisor']
        ))
        fig_keywords.update_layout(
            title="🔑 Most Frequent Keywords in Reviews",
            xaxis_title="Frequency",
            yaxis_title="Keywords",
            template="plotly_white",
            height=500
        )
        figures['keywords'] = fig_keywords
        
        return figures
    
    def generate_wordcloud(self) -> str:
        """Generate word cloud from review texts"""
        if not self.reviews:
            return None
        
        # Combine all review texts
        all_text = ' '.join([review.text for review in self.reviews])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5
        ).generate(all_text)
        
        # Save to file
        wordcloud_path = 'review_wordcloud.png'
        wordcloud.to_file(wordcloud_path)
        
        return wordcloud_path
    
    def get_review_examples(self, category: str = 'high_confidence', limit: int = 10) -> List[Dict]:
        """Get examples of actual review content"""
        examples = []
        
        if category == 'high_confidence':
            filtered_reviews = [r for r in self.reviews if r.confidence > 0.8]
        elif category == 'tripadvisor':
            filtered_reviews = [r for r in self.reviews if 'tripadvisor' in ' '.join(r.keywords).lower()]
        elif category == 'sentiment_positive':
            filtered_reviews = []
            for review in self.reviews:
                sentiment, score = self.analyze_sentiment(review.text)
                if sentiment == 'positive' and score > 0.7:
                    filtered_reviews.append(review)
        else:
            filtered_reviews = self.reviews
        
        # Sort by confidence and take top examples
        filtered_reviews.sort(key=lambda x: x.confidence, reverse=True)
        
        for review in filtered_reviews[:limit]:
            sentiment, sentiment_score = self.analyze_sentiment(review.text)
            
            example = {
                'text': review.text[:500] + "..." if len(review.text) > 500 else review.text,
                'file': os.path.basename(review.file_path),
                'confidence': review.confidence,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'keywords': review.keywords
            }
            examples.append(example)
        
        return examples
    
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        st.set_page_config(
            page_title="TripAdvisor Review Analysis",
            page_icon="🏮",
            layout="wide"
        )
        
        st.title("🏮 TripAdvisor Senso-ji Temple Review Analysis")
        st.markdown("---")
        
        # Load data
        if not self.reviews:
            with st.spinner("Loading review data..."):
                self.load_data_from_db()
                self.setup_vector_search()
        
        # Sidebar
        st.sidebar.title("📊 Analysis Options")
        
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Visualizations", "Semantic Search", "Review Examples", "Sentiment Analysis"]
        )
        
        if analysis_type == "Overview":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reviews Found", len(self.reviews))
            
            with col2:
                high_conf = len([r for r in self.reviews if r.confidence > 0.8])
                st.metric("High Confidence", high_conf)
            
            with col3:
                avg_conf = np.mean([r.confidence for r in self.reviews]) if self.reviews else 0
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            
            with col4:
                file_types = len(set(os.path.splitext(r.file_path)[1] for r in self.reviews))
                st.metric("File Types", file_types)
            
            st.markdown("### 📈 Quick Statistics")
            
            # Database summary
            try:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("SELECT * FROM review_matches LIMIT 5", conn)
                st.dataframe(df)
                conn.close()
            except Exception as e:
                st.error(f"Database error: {e}")
        
        elif analysis_type == "Visualizations":
            st.header("📊 Data Visualizations")
            
            figures = self.create_visualizations()
            
            for name, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud
            st.header("☁️ Word Cloud")
            wordcloud_path = self.generate_wordcloud()
            if wordcloud_path and os.path.exists(wordcloud_path):
                st.image(wordcloud_path)
        
        elif analysis_type == "Semantic Search":
            st.header("🔍 Semantic Search")
            
            query = st.text_input("Enter search query:", "temple experience review")
            
            if query:
                with st.spinner("Searching..."):
                    results = self.semantic_search(query, top_k=10)
                
                if results:
                    st.success(f"Found {len(results)} relevant matches!")
                    
                    for i, (review, similarity) in enumerate(results, 1):
                        with st.expander(f"Result {i} (Similarity: {similarity:.3f})"):
                            st.write("**Text:**", review.text[:300] + "...")
                            st.write("**File:**", os.path.basename(review.file_path))
                            st.write("**Confidence:**", review.confidence)
                            st.write("**Keywords:**", ', '.join(review.keywords))
                else:
                    st.warning("No relevant matches found. Try different keywords.")
        
        elif analysis_type == "Review Examples":
            st.header("📝 Review Examples")
            
            category = st.selectbox(
                "Select Category:",
                ["high_confidence", "tripadvisor", "sentiment_positive", "all"]
            )
            
            examples = self.get_review_examples(category, limit=15)
            
            if examples:
                for i, example in enumerate(examples, 1):
                    with st.expander(f"Review {i} - {example['sentiment'].title()} ({example['confidence']:.3f})"):
                        st.write("**Content:**", example['text'])
                        st.write("**Source File:**", example['file'])
                        st.write("**Sentiment:**", f"{example['sentiment']} (score: {example['sentiment_score']:.3f})")
                        st.write("**Keywords:**", ', '.join(example['keywords']))
            else:
                st.warning("No examples found for this category.")
        
        elif analysis_type == "Sentiment Analysis":
            st.header("😊 Detailed Sentiment Analysis")
            
            # Analyze sentiment for all reviews
            sentiment_data = []
            
            progress_bar = st.progress(0)
            
            for i, review in enumerate(self.reviews[:50]):  # Limit for performance
                sentiment, score = self.analyze_sentiment(review.text)
                sentiment_data.append({
                    'text': review.text[:100] + "...",
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': review.confidence,
                    'file': os.path.basename(review.file_path)
                })
                progress_bar.progress((i + 1) / 50)
            
            df_sentiment = pd.DataFrame(sentiment_data)
            
            # Sentiment distribution chart
            fig = px.histogram(df_sentiment, x='sentiment', color='sentiment',
                             title="Sentiment Distribution")
            st.plotly_chart(fig)
            
            # Sentiment vs Confidence scatter
            fig2 = px.scatter(df_sentiment, x='confidence', y='score', color='sentiment',
                            title="Sentiment Score vs Confidence")
            st.plotly_chart(fig2)
            
            # Detailed table
            st.dataframe(df_sentiment)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Review Analysis with Visualizations')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    parser.add_argument('--db', default='reviews_database.db', help='Database path')
    parser.add_argument('--search', type=str, help='Perform semantic search')
    parser.add_argument('--examples', action='store_true', help='Show review examples')
    
    args = parser.parse_args()
    
    analyzer = ReviewAnalyzer(args.db)
    
    if args.dashboard:
        # Launch Streamlit dashboard
        print("🚀 Launching dashboard...")
        print("📋 Run: streamlit run review_analyzer_enhanced.py -- --dashboard")
        analyzer.create_dashboard()
    
    elif args.search:
        print(f"🔍 Searching for: {args.search}")
        analyzer.load_data_from_db()
        analyzer.setup_vector_search()
        
        results = analyzer.semantic_search(args.search)
        
        print(f"\n📊 Found {len(results)} matches:\n")
        for i, (review, similarity) in enumerate(results, 1):
            print(f"{i}. Similarity: {similarity:.3f}")
            print(f"   File: {os.path.basename(review.file_path)}")
            print(f"   Text: {review.text[:200]}...")
            print(f"   Confidence: {review.confidence:.3f}")
            print("-" * 80)
    
    elif args.examples:
        print("📝 Loading review examples...")
        analyzer.load_data_from_db()
        
        examples = analyzer.get_review_examples('high_confidence', 10)
        
        print(f"\n🎯 Top {len(examples)} High-Confidence Review Examples:\n")
        for i, example in enumerate(examples, 1):
            print(f"{i}. File: {example['file']}")
            print(f"   Confidence: {example['confidence']:.3f}")
            print(f"   Sentiment: {example['sentiment']} ({example['sentiment_score']:.3f})")
            print(f"   Text: {example['text'][:300]}...")
            print("-" * 80)
    
    else:
        print("🔍 Quick Analysis Summary")
        analyzer.load_data_from_db()
        
        if analyzer.reviews:
            print(f"📊 Loaded {len(analyzer.reviews)} reviews")
            print(f"📈 Average confidence: {np.mean([r.confidence for r in analyzer.reviews]):.3f}")
            
            # Show top files
            file_counts = {}
            for review in analyzer.reviews:
                file_name = os.path.basename(review.file_path)
                file_counts[file_name] = file_counts.get(file_name, 0) + 1
            
            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n📁 Top files with review content:")
            for file_name, count in top_files:
                print(f"   {file_name}: {count} matches")
            
            print(f"\n🚀 Usage options:")
            print(f"   --dashboard  : Launch interactive dashboard")
            print(f"   --search 'query' : Perform semantic search")
            print(f"   --examples   : Show review examples")
        else:
            print("❌ No review data found. Run review_extractor.py first.")

if __name__ == "__main__":
    main()