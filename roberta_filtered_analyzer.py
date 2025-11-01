#!/usr/bin/env python3
"""
RoBERTa-Enhanced Filtered Review Analyzer
=========================================

Reads filtered review comments from the database created by filtered_review_extractor.py
and provides comprehensive sentiment analysis using RoBERTa model with visualizations.

Features:
- Read filtered comments from comment_blocks database (filtered_reviews.db)
- RoBERTa-based sentiment analysis
- Interactive visualizations with Plotly and Matplotlib
- Word clouds and statistical analysis
- Review examples with quality scores
- Export analysis results
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - Updated for Request folder
CONFIDENCE_THRESHOLD = 0.65  # Match the filtered_review_extractor threshold
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/analysis_output"
DB_PATH = "/Users/andreyvlasenko/tst/Request/filtered_reviews.db"

@dataclass
class FilteredReview:
    """Filtered review content with analysis"""
    text: str
    score: float
    length: int
    file_path: str
    is_candidate: bool
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    roberta_label: str = "NEUTRAL"
    roberta_score: float = 0.0

class RoBERTaFilteredReviewAnalyzer:
    """RoBERTa-enhanced filtered review analyzer"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.reviews = []
        self.roberta_analyzer = None
        self.vectorizer = None
        self.vectors = None
        self.setup_environment()
        self.load_ml_libraries()
    
    def setup_environment(self):
        """Setup output directories in Request folder"""
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
        """Load sentiment analysis libraries with fallback options"""
        try:
            # Import required libraries
            from transformers import pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.cluster import KMeans
            
            # Try to load RoBERTa first, fallback to DistilBERT if needed
            try:
                self.roberta_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("✅ RoBERTa sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"⚠️ RoBERTa failed, trying DistilBERT: {e}")
                try:
                    self.roberta_analyzer = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        return_all_scores=True
                    )
                    logger.info("✅ DistilBERT sentiment analyzer loaded as fallback")
                except Exception as e2:
                    logger.warning(f"⚠️ Both models failed: {e2}")
                    self.roberta_analyzer = None
            
        except ImportError as e:
            logger.warning(f"⚠️ ML libraries not available: {e}")
            self.roberta_analyzer = None
    
    def load_data_from_db(self) -> List[FilteredReview]:
        """Load filtered review data from database created by filtered_review_extractor.py"""
        try:
            if not os.path.exists(self.db_path):
                logger.error(f"❌ Database not found: {self.db_path}")
                logger.info("💡 Make sure to run filtered_review_extractor.py first!")
                return []
            
            conn = sqlite3.connect(self.db_path)
            
            # Check available tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            logger.info(f"📋 Available tables: {tables}")
            
            reviews = []
            
            # Load from comment_blocks table (created by filtered_review_extractor.py)
            if 'comment_blocks' in tables:
                # Load all data first, then filter
                query = """
                    SELECT file_path, block_text, score, length, is_candidate
                    FROM comment_blocks 
                    ORDER BY score DESC
                """
                df = pd.read_sql_query(query, conn)
                
                logger.info(f"📊 Found {len(df)} total comment blocks in database")
                
                # Filter by threshold
                filtered_df = df[df['score'] >= self.confidence_threshold]
                logger.info(f"📊 {len(filtered_df)} blocks meet threshold >= {self.confidence_threshold}")
                
                for _, row in filtered_df.iterrows():
                    review = FilteredReview(
                        text=row['block_text'],
                        score=row['score'],
                        length=row['length'],
                        file_path=row['file_path'],
                        is_candidate=bool(row['is_candidate'])
                    )
                    reviews.append(review)
                
                logger.info(f"📊 Loaded {len(reviews)} filtered reviews for analysis")
                
            else:
                logger.warning("⚠️ No comment_blocks table found!")
                # Check for other possible tables
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
                schemas = cursor.fetchall()
                for schema in schemas:
                    logger.info(f"📋 Table schema: {schema[0]}")
            
            conn.close()
            self.reviews = reviews
            return reviews
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return []
    
    def analyze_roberta_sentiment(self, text: str) -> Tuple[str, float, str, float]:
        """Analyze sentiment using RoBERTa model"""
        if not self.roberta_analyzer or len(text.strip()) < 10:
            return self.fallback_sentiment(text)
        
        try:
            # Truncate text to model's max length
            text_truncated = text[:512]
            results = self.roberta_analyzer(text_truncated)
            
            # Find best prediction
            best_result = max(results, key=lambda x: x['score'])
            
            # Map labels to standard format
            label = best_result['label'].upper()
            score = best_result['score']
            
            if label in ['LABEL_2', 'POSITIVE']:
                sentiment = 'positive'
            elif label in ['LABEL_0', 'NEGATIVE']:
                sentiment = 'negative'
            else:  # LABEL_1, NEUTRAL
                sentiment = 'neutral'
            
            return sentiment, score, label, score
            
        except Exception as e:
            logger.debug(f"RoBERTa analysis failed: {e}")
            return self.fallback_sentiment(text)
    
    def fallback_sentiment(self, text: str) -> Tuple[str, float, str, float]:
        """Fallback sentiment analysis using keywords"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'recommend', 
                         'beautiful', 'fantastic', 'perfect', 'awesome', 'outstanding', 'impressive',
                         'stunning', 'magnificent', 'peaceful', 'serene', 'spiritual', 'sacred']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing', 
                         'worst', 'disgusting', 'boring', 'overrated', 'crowded', 'noisy',
                         'dirty', 'expensive', 'tourist trap', 'chaos']
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if pos_score > neg_score:
            sentiment = 'positive'
            score = 0.6 + min(pos_score * 0.1, 0.3)
        elif neg_score > pos_score:
            sentiment = 'negative' 
            score = 0.6 + min(neg_score * 0.1, 0.3)
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return sentiment, score, sentiment.upper(), score
    
    def analyze_all_reviews(self):
        """Analyze sentiment for all loaded reviews"""
        if not self.reviews:
            logger.warning("⚠️ No reviews to analyze")
            return
            
        logger.info("🔍 Analyzing sentiment for all reviews...")
        
        for i, review in enumerate(self.reviews):
            sentiment, sent_score, roberta_label, roberta_score = self.analyze_roberta_sentiment(review.text)
            
            review.sentiment = sentiment
            review.sentiment_score = sent_score
            review.roberta_label = roberta_label
            review.roberta_score = roberta_score
            
            if (i + 1) % 5 == 0:
                logger.info(f"📊 Analyzed {i + 1}/{len(self.reviews)} reviews")
        
        logger.info("✅ Sentiment analysis complete")
    
    def create_matplotlib_visualizations(self) -> Dict[str, str]:
        """Create visualizations using matplotlib"""
        if not self.reviews:
            logger.warning("⚠️ No reviews for visualization")
            return {}
        
        # Set style
        plt.style.use('default')
        fig_paths = {}
        
        # Create figure directory
        viz_dir = self.folders['visualizations']
        
        # 1. Score distribution
        plt.figure(figsize=(12, 8))
        scores = [r.score for r in self.reviews]
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.confidence_threshold})')
        plt.title('📊 Review Quality Score Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        score_path = os.path.join(viz_dir, 'score_distribution.png')
        plt.savefig(score_path, dpi=300, bbox_inches='tight')
        plt.close()
        fig_paths['score_distribution'] = score_path
        
        # 2. Sentiment pie chart
        sentiments = [r.sentiment for r in self.reviews]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = ['#2ca02c', '#d62728', '#ff7f0e']  # green, red, orange
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('😊 Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        sentiment_path = os.path.join(viz_dir, 'sentiment_analysis.png')
        plt.savefig(sentiment_path, dpi=300, bbox_inches='tight')
        plt.close()
        fig_paths['sentiment_analysis'] = sentiment_path
        
        # 3. Score vs Sentiment Score scatter
        plt.figure(figsize=(12, 8))
        sentiment_colors = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#ff7f0e'}
        
        for sentiment in sentiment_colors:
            sentiment_reviews = [r for r in self.reviews if r.sentiment == sentiment]
            if sentiment_reviews:
                x_vals = [r.score for r in sentiment_reviews]
                y_vals = [r.roberta_score for r in sentiment_reviews]
                plt.scatter(x_vals, y_vals, c=sentiment_colors[sentiment], 
                           label=sentiment.title(), alpha=0.7, s=60)
        
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.title('📈 Quality Score vs Sentiment Score', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        scatter_path = os.path.join(viz_dir, 'score_vs_sentiment.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        fig_paths['score_vs_sentiment'] = scatter_path
        
        # 4. Review length by sentiment
        plt.figure(figsize=(12, 8))
        sentiment_data = []
        for sentiment in ['positive', 'negative', 'neutral']:
            lengths = [r.length for r in self.reviews if r.sentiment == sentiment]
            if lengths:
                sentiment_data.append(lengths)
            else:
                sentiment_data.append([0])
        
        plt.boxplot(sentiment_data, labels=['Positive', 'Negative', 'Neutral'])
        plt.title('📏 Review Length Distribution by Sentiment', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Review Length (characters)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        length_path = os.path.join(viz_dir, 'length_by_sentiment.png')
        plt.savefig(length_path, dpi=300, bbox_inches='tight')
        plt.close()
        fig_paths['length_by_sentiment'] = length_path
        
        logger.info(f"📊 Created {len(fig_paths)} visualizations in {viz_dir}")
        return fig_paths
    
    def generate_wordcloud(self) -> str:
        """Generate word cloud from review texts"""
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.error("❌ WordCloud not available. Install with: pip install wordcloud")
            return None
        
        if not self.reviews:
            return None
        
        # Combine texts from candidate reviews only
        candidate_texts = [r.text for r in self.reviews if r.is_candidate and len(r.text) > 20]
        
        if not candidate_texts:
            candidate_texts = [r.text for r in self.reviews[:10] if len(r.text) > 20]
        
        if not candidate_texts:
            logger.warning("⚠️ No suitable texts for word cloud")
            return None
        
        all_text = ' '.join(candidate_texts)
        
        # Clean text for better word cloud
        all_text = re.sub(r'[^\w\s]', ' ', all_text)
        all_text = re.sub(r'\s+', ' ', all_text)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            stopwords=set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        ).generate(all_text)
        
        # Save to file
        wordcloud_path = os.path.join(self.folders['visualizations'], 'filtered_reviews_wordcloud.png')
        wordcloud.to_file(wordcloud_path)
        
        return wordcloud_path
    
    def get_review_examples(self, category: str = 'candidates', limit: int = 10) -> List[Dict]:
        """Get examples of filtered review content"""
        examples = []
        
        if category == 'candidates':
            filtered_reviews = [r for r in self.reviews if r.is_candidate]
        elif category == 'positive':
            filtered_reviews = [r for r in self.reviews if r.sentiment == 'positive']
        elif category == 'negative':
            filtered_reviews = [r for r in self.reviews if r.sentiment == 'negative']
        elif category == 'high_score':
            filtered_reviews = [r for r in self.reviews if r.score > 0.8]
        else:
            filtered_reviews = self.reviews
        
        # Sort by score
        filtered_reviews.sort(key=lambda x: x.score, reverse=True)
        
        for review in filtered_reviews[:limit]:
            example = {
                'text': review.text[:300] + "..." if len(review.text) > 300 else review.text,
                'file': os.path.basename(review.file_path),
                'quality_score': review.score,
                'sentiment': review.sentiment,
                'roberta_label': review.roberta_label,
                'roberta_score': review.roberta_score,
                'length': review.length,
                'is_candidate': review.is_candidate
            }
            examples.append(example)
        
        return examples
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        if not self.reviews:
            logger.error("❌ No reviews loaded")
            return
        
        print("\n" + "="*80)
        print("🏮 Filtered Review Analysis Summary (RoBERTa Enhanced)")
        print("="*80)
        
        # Basic statistics
        total_reviews = len(self.reviews)
        candidates = len([r for r in self.reviews if r.is_candidate])
        avg_score = np.mean([r.score for r in self.reviews]) if self.reviews else 0
        avg_length = np.mean([r.length for r in self.reviews]) if self.reviews else 0
        
        print(f"📊 Total Reviews Analyzed: {total_reviews}")
        print(f"🎯 Candidate Reviews (is_candidate=True): {candidates}")
        print(f"📈 Average Quality Score: {avg_score:.3f}")
        print(f"📏 Average Review Length: {avg_length:.1f} characters")
        print(f"🔍 Confidence Threshold: {self.confidence_threshold}")
        
        # Sentiment distribution
        sentiment_counts = {}
        for review in self.reviews:
            sentiment_counts[review.sentiment] = sentiment_counts.get(review.sentiment, 0) + 1
        
        print(f"\n😊 Sentiment Analysis Results:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
            print(f"   {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        # Score distribution
        score_ranges = {
            'Excellent (>0.8)': len([r for r in self.reviews if r.score > 0.8]),
            'Good (0.65-0.8)': len([r for r in self.reviews if 0.65 <= r.score <= 0.8]),
            'Medium (0.5-0.65)': len([r for r in self.reviews if 0.5 <= r.score < 0.65]),
            'Low (<0.5)': len([r for r in self.reviews if r.score < 0.5])
        }
        
        print(f"\n📊 Quality Score Distribution:")
        for range_name, count in score_ranges.items():
            percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
            print(f"   {range_name}: {count} ({percentage:.1f}%)")
        
        # Top examples
        print(f"\n🌟 Top Candidate Examples:")
        examples = self.get_review_examples('candidates', 3)
        
        if examples:
            for i, example in enumerate(examples, 1):
                print(f"\n{i}. Quality Score: {example['quality_score']:.3f}")
                print(f"   Sentiment: {example['sentiment'].title()} (Score: {example['roberta_score']:.3f})")
                print(f"   Length: {example['length']} chars")
                print(f"   Text: {example['text']}")
        else:
            print("   No candidate examples found.")
    
    def save_results_to_files(self):
        """Save analysis results to files"""
        if not self.reviews:
            logger.warning("⚠️ No reviews to save")
            return
        
        # Save detailed results to CSV
        results_data = []
        for review in self.reviews:
            results_data.append({
                'text': review.text,
                'quality_score': review.score,
                'length': review.length,
                'sentiment': review.sentiment,
                'sentiment_score': review.sentiment_score,
                'roberta_label': review.roberta_label,
                'roberta_score': review.roberta_score,
                'is_candidate': review.is_candidate,
                'file_path': review.file_path
            })
        
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(self.folders['raw_data'], 'filtered_review_analysis.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Detailed results saved to: {csv_path}")
        
        # Save summary statistics
        summary = {
            'analysis_type': 'filtered_review_roberta_analysis',
            'total_reviews': len(self.reviews),
            'candidates': len([r for r in self.reviews if r.is_candidate]),
            'average_score': float(np.mean([r.score for r in self.reviews])),
            'confidence_threshold': self.confidence_threshold,
            'sentiment_distribution': {
                sentiment: len([r for r in self.reviews if r.sentiment == sentiment])
                for sentiment in ['positive', 'negative', 'neutral']
            },
            'database_path': self.db_path,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        json_path = os.path.join(self.folders['raw_data'], 'analysis_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Summary saved to: {json_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RoBERTa-Enhanced Filtered Review Analysis')
    parser.add_argument('--db', default=DB_PATH, help='Database path')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD, 
                       help='Minimum score threshold')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples to show')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--wordcloud', action='store_true', help='Generate word cloud')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    print("🤖 RoBERTa-Enhanced Filtered Review Analyzer")
    print("="*50)
    print(f"📂 Database: {args.db}")
    print(f"🎯 Threshold: {args.threshold}")
    
    analyzer = RoBERTaFilteredReviewAnalyzer(args.db)
    analyzer.confidence_threshold = args.threshold
    
    # Load and analyze data
    reviews = analyzer.load_data_from_db()
    
    if not reviews:
        print("\n❌ No reviews found!")
        print("💡 Make sure to:")
        print("   1. Run filtered_review_extractor.py first")
        print("   2. Check that filtered_reviews.db exists")
        print("   3. Verify the database contains comment_blocks table")
        return
    
    print(f"\n📊 Analyzing {len(reviews)} filtered reviews...")
    analyzer.analyze_all_reviews()
    
    # Print summary
    analyzer.print_analysis_summary()
    
    # Show examples
    if args.examples > 0:
        print(f"\n🌟 Top {args.examples} Review Examples:")
        print("-" * 80)
        examples = analyzer.get_review_examples('candidates', args.examples)
        
        if examples:
            for i, example in enumerate(examples, 1):
                print(f"\n{i}. Quality Score: {example['quality_score']:.3f}")
                print(f"   Sentiment: {example['sentiment'].title()} ({example['roberta_score']:.3f})")
                print(f"   File: {example['file']}")
                print(f"   Text: {example['text']}")
                print("-" * 60)
        else:
            print("   No examples found.")
    
    # Create visualizations
    if args.visualize:
        print("\n📊 Creating visualizations...")
        fig_paths = analyzer.create_matplotlib_visualizations()
        
        for name, path in fig_paths.items():
            print(f"📈 Created: {name} -> {path}")
    
    # Generate word cloud
    if args.wordcloud:
        print("\n☁️ Generating word cloud...")
        wordcloud_path = analyzer.generate_wordcloud()
        if wordcloud_path:
            print(f"☁️ Word cloud saved: {wordcloud_path}")
        else:
            print("⚠️ Could not generate word cloud")
    
    # Save results
    if args.save:
        print("\n💾 Saving results...")
        analyzer.save_results_to_files()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Output directory: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()