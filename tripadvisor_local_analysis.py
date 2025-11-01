#!/usr/bin/env python3
"""
Enhanced TripAdvisor Review Analysis Script for Local Files
Analyzes downloaded TripAdvisor content for sentiment analysis
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Review:
    """Data class for TripAdvisor reviews"""
    text: str
    rating: Optional[int] = None
    date: Optional[str] = None
    reviewer: Optional[str] = None
    helpful_votes: Optional[int] = None
    source_file: Optional[str] = None

class LocalTripAdvisorAnalyzer:
    """Analyzer for locally downloaded TripAdvisor content"""
    
    def __init__(self, senso_folder_path: str):
        """
        Initialize the analyzer with the path to downloaded content
        
        Args:
            senso_folder_path: Path to the folder containing downloaded TripAdvisor files
        """
        self.senso_folder = Path(senso_folder_path)
        if not self.senso_folder.exists():
            raise ValueError(f"Senso folder not found: {senso_folder_path}")
        
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
            logger.info("✅ Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment analysis model: {e}")
            raise
        
        self.reviews = []
        
    def extract_reviews_from_html(self, html_content: str, source_file: str = "") -> List[Review]:
        """
        Extract reviews from HTML content
        
        Args:
            html_content: Raw HTML content
            source_file: Name of the source file for tracking
            
        Returns:
            List of Review objects
        """
        reviews = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Common TripAdvisor review selectors (multiple patterns to try)
            review_selectors = [
                '[data-automation="reviewCard"]',
                '.review-container',
                '.reviewSelector',
                '.rev_wrap',
                '[id*="review_"]',
                '.ui_card.section',
                '.reviewCard',
                '.review',
                '.YibKl'  # Current TripAdvisor class for reviews
            ]
            
            found_reviews = []
            for selector in review_selectors:
                found_reviews = soup.select(selector)
                if found_reviews:
                    logger.info(f"Found {len(found_reviews)} reviews using selector: {selector}")
                    break
            
            if not found_reviews:
                # Try finding review text patterns directly
                review_text_patterns = [
                    r'"reviewText":\s*"([^"]+)"',
                    r'"text":\s*"([^"]+)"',
                    r'reviewText[\'"]:\s*[\'"]([^\'"]+)[\'"]',
                    r'data-reviewtext=[\'"]([^\'"]+)[\'"]'
                ]
                
                for pattern in review_text_patterns:
                    matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        logger.info(f"Found {len(matches)} review texts using pattern: {pattern}")
                        for match in matches:
                            # Clean up text
                            text = self._clean_review_text(match)
                            if len(text.strip()) > 10:  # Only meaningful reviews
                                reviews.append(Review(
                                    text=text,
                                    source_file=source_file
                                ))
                        break
            else:
                # Extract from found review elements
                for review_elem in found_reviews:
                    review_data = self._extract_review_data(review_elem, source_file)
                    if review_data and len(review_data.text.strip()) > 10:
                        reviews.append(review_data)
            
            # Also search for Japanese reviews
            japanese_patterns = [
                r'[あ-ん]{5,}[。！？]',  # Japanese hiragana sentences
                r'[ア-ン]{3,}[。！？]',   # Japanese katakana
                r'[一-龯]{3,}[。！？]',   # Japanese kanji
            ]
            
            for pattern in japanese_patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if len(match.strip()) > 5:
                        reviews.append(Review(
                            text=match.strip(),
                            source_file=source_file
                        ))
            
            logger.info(f"Extracted {len(reviews)} reviews from {source_file}")
            
        except Exception as e:
            logger.error(f"Error extracting reviews from {source_file}: {e}")
        
        return reviews
    
    def _extract_review_data(self, review_elem, source_file: str) -> Optional[Review]:
        """Extract review data from a BeautifulSoup element"""
        try:
            # Try different text selectors
            text_selectors = [
                '.yCeTE',
                '.partial_entry',
                '.review-text',
                '.reviewText',
                '[data-automation="reviewText"]',
                'p',
                'span',
                'div'
            ]
            
            review_text = ""
            for selector in text_selectors:
                text_elem = review_elem.select_one(selector)
                if text_elem:
                    review_text = text_elem.get_text(strip=True)
                    if len(review_text) > 20:  # Meaningful length
                        break
            
            if not review_text:
                # Fallback to element's direct text
                review_text = review_elem.get_text(strip=True)
            
            # Clean and validate text
            review_text = self._clean_review_text(review_text)
            if len(review_text.strip()) < 10:
                return None
            
            # Try to extract rating
            rating = None
            rating_selectors = [
                '[class*="rating"]',
                '[class*="stars"]',
                '[class*="bubble"]',
                '[data-rating]'
            ]
            
            for selector in rating_selectors:
                rating_elem = review_elem.select_one(selector)
                if rating_elem:
                    # Extract rating from class name or attribute
                    classes = rating_elem.get('class', [])
                    for cls in classes:
                        if 'rating' in cls or 'stars' in cls:
                            rating_match = re.search(r'(\d)', cls)
                            if rating_match:
                                rating = int(rating_match.group(1))
                                break
                    
                    if not rating:
                        rating_attr = rating_elem.get('data-rating')
                        if rating_attr:
                            try:
                                rating = int(float(rating_attr))
                            except:
                                pass
                    
                    if rating:
                        break
            
            return Review(
                text=review_text,
                rating=rating,
                source_file=source_file
            )
            
        except Exception as e:
            logger.error(f"Error extracting review data: {e}")
            return None
    
    def _clean_review_text(self, text: str) -> str:
        """Clean and normalize review text"""
        if not text:
            return ""
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\'\"あ-んア-ン一-龯]', ' ', text)
        
        return text.strip()
    
    def analyze_local_files(self) -> Dict[str, Any]:
        """
        Analyze all files in the Senso folder for TripAdvisor reviews
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"🔍 Analyzing files in: {self.senso_folder}")
        
        html_files = list(self.senso_folder.glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files to analyze")
        
        all_reviews = []
        
        for html_file in html_files:
            try:
                logger.info(f"📄 Processing: {html_file.name}")
                
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                reviews = self.extract_reviews_from_html(html_content, html_file.name)
                all_reviews.extend(reviews)
                
                logger.info(f"✅ Found {len(reviews)} reviews in {html_file.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {html_file}: {e}")
        
        self.reviews = all_reviews
        logger.info(f"🎉 Total reviews extracted: {len(all_reviews)}")
        
        if not all_reviews:
            logger.warning("⚠️ No reviews found. The files might not contain TripAdvisor review content.")
            return self._create_empty_analysis()
        
        # Perform sentiment analysis
        return self._analyze_sentiments()
    
    def _analyze_sentiments(self) -> Dict[str, Any]:
        """Perform sentiment analysis on extracted reviews"""
        logger.info("🔬 Performing sentiment analysis...")
        
        sentiments = []
        confidence_scores = []
        
        for review in self.reviews:
            try:
                # Truncate very long reviews for the model
                text = review.text[:512] if len(review.text) > 512 else review.text
                
                result = self.sentiment_analyzer(text)[0]
                
                # Map to consistent labels
                label = result['label'].upper()
                if label == 'POSITIVE':
                    sentiment = 'Positive'
                elif label == 'NEGATIVE':
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                sentiments.append(sentiment)
                confidence_scores.append(result['score'])
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for review: {e}")
                sentiments.append('Neutral')
                confidence_scores.append(0.5)
        
        # Create analysis results
        df = pd.DataFrame({
            'review_text': [r.text for r in self.reviews],
            'sentiment': sentiments,
            'confidence': confidence_scores,
            'rating': [r.rating for r in self.reviews],
            'source_file': [r.source_file for r in self.reviews]
        })
        
        # Calculate statistics
        sentiment_counts = df['sentiment'].value_counts()
        avg_confidence = df['confidence'].mean()
        
        # Calculate rating statistics if available
        ratings_with_data = df[df['rating'].notna()]
        avg_rating = ratings_with_data['rating'].mean() if not ratings_with_data.empty else None
        
        results = {
            'total_reviews': len(self.reviews),
            'sentiment_distribution': {
                'Positive': int(sentiment_counts.get('Positive', 0)),
                'Negative': int(sentiment_counts.get('Negative', 0)),
                'Neutral': int(sentiment_counts.get('Neutral', 0))
            },
            'average_confidence': float(avg_confidence),
            'average_rating': float(avg_rating) if avg_rating else None,
            'analysis_timestamp': datetime.now().isoformat(),
            'source_location': str(self.senso_folder),
            'files_processed': len(set(r.source_file for r in self.reviews if r.source_file))
        }
        
        # Add sample reviews
        results['sample_reviews'] = []
        for sentiment_type in ['Positive', 'Negative', 'Neutral']:
            sentiment_reviews = df[df['sentiment'] == sentiment_type]
            if not sentiment_reviews.empty:
                sample = sentiment_reviews.iloc[0]
                results['sample_reviews'].append({
                    'sentiment': sentiment_type,
                    'text': sample['review_text'][:200] + "..." if len(sample['review_text']) > 200 else sample['review_text'],
                    'confidence': float(sample['confidence']),
                    'rating': sample['rating'] if pd.notna(sample['rating']) else None
                })
        
        return results
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis results when no reviews are found"""
        return {
            'total_reviews': 0,
            'sentiment_distribution': {'Positive': 0, 'Negative': 0, 'Neutral': 0},
            'average_confidence': 0.0,
            'average_rating': None,
            'analysis_timestamp': datetime.now().isoformat(),
            'source_location': str(self.senso_folder),
            'files_processed': len(list(self.senso_folder.glob("*.html"))),
            'sample_reviews': [],
            'message': 'No reviews found in the provided files. This may be due to the site structure or the files not containing review content.'
        }
    
    def save_results_to_file(self, results: Dict[str, Any], output_file: str = "senso_ji_analysis.json"):
        """Save analysis results to a JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"📁 Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")

def main():
    """Main function to run the analysis"""
    # Path to the downloaded Senso-ji TripAdvisor files
    senso_folder = "/Users/andreyvlasenko/tst/Request/Senso"
    
    try:
        print("🏮 TripAdvisor Senso-ji Temple Review Analysis (Local Files)")
        print("=" * 60)
        
        # Initialize analyzer
        analyzer = LocalTripAdvisorAnalyzer(senso_folder)
        
        # Analyze local files
        results = analyzer.analyze_local_files()
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS")
        print(f"📁 Source Location: {results['source_location']}")
        print(f"📄 Files Processed: {results['files_processed']}")
        print(f"📝 Total Reviews Found: {results['total_reviews']}")
        
        if results['total_reviews'] > 0:
            print(f"\n🎭 SENTIMENT DISTRIBUTION:")
            sentiment_dist = results['sentiment_distribution']
            total = results['total_reviews']
            
            for sentiment, count in sentiment_dist.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            
            print(f"\n📈 CONFIDENCE: {results['average_confidence']:.3f}")
            
            if results['average_rating']:
                print(f"⭐ AVERAGE RATING: {results['average_rating']:.2f}/5")
            
            print(f"\n📝 SAMPLE REVIEWS:")
            for sample in results['sample_reviews']:
                print(f"\n  {sample['sentiment']} (confidence: {sample['confidence']:.3f}):")
                print(f"  \"{sample['text']}\"")
        else:
            print(f"\n⚠️ {results.get('message', 'No reviews found')}")
        
        # Save results
        analyzer.save_results_to_file(results)
        
        print(f"\n✅ Analysis completed at {results['analysis_timestamp']}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()