#!/usr/bin/env python3
"""
Visitor Comments Extractor for Request Folder
==============================================

Searches through downloaded TripAdvisor files for actual visitor comments
and stores them in a clean database format like:
Comment 1: "That was a very nice place"
Comment 2: "It was terrible"

Then prints the first 10 comments found.
"""

import os
import re
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisitorCommentsExtractor:
    """Extract actual visitor comments from TripAdvisor files"""
    
    def __init__(self, 
                 folder_path: str = "/Users/andreyvlasenko/tst/Request/Senso",
                 db_path: str = "/Users/andreyvlasenko/tst/Request/visitor_comments.db"):
        self.folder_path = folder_path
        self.db_path = db_path
        self.setup_database()
        
        # Patterns to find actual visitor comments
        self.comment_patterns = [
            # Look for personal review statements with first person
            re.compile(r'\b(I\s+(?:visited|went|love|loved|hate|hated|recommend|enjoyed|was\s+disappointed|think|believe|found)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            re.compile(r'\b(We\s+(?:visited|went|love|loved|hate|hated|recommend|enjoyed|were\s+disappointed|think|believe|found)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            re.compile(r'\b(My\s+(?:visit|experience|trip|family|wife|husband)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            re.compile(r'\b(Our\s+(?:visit|experience|trip|family)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            
            # Look for evaluative statements about the temple
            re.compile(r'\b((?:This\s+temple|The\s+temple|Sensoji|Senso-ji)\s+(?:is|was)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            re.compile(r'\b((?:Beautiful|Amazing|Wonderful|Terrible|Disappointing|Boring|Crowded|Peaceful|Historic)\s+(?:temple|place|experience)[^.!?]{10,200}[.!?])', re.IGNORECASE | re.MULTILINE),
            
            # JSON review text with personal content
            re.compile(r'"(?:reviewText|text|comment|body)"\s*:\s*"([^"]*(?:I\s+|We\s+|My\s+|Our\s+|temple|visit|recommend)[^"]{20,300})"', re.IGNORECASE),
            
            # Review-like sentences with strong sentiment
            re.compile(r'\b([A-Z][^.!?]*(?:highly\s+recommend|must\s+visit|don\'t\s+miss|worth\s+visiting|waste\s+of\s+time|not\s+worth|amazing\s+experience|terrible\s+experience)[^.!?]{0,100}[.!?])', re.IGNORECASE),
        ]
        
        # Stronger patterns to exclude system/technical content
        self.exclude_patterns = [
            re.compile(r'function\s*\(', re.IGNORECASE),
            re.compile(r'var\s+\w+\s*=', re.IGNORECASE),
            re.compile(r'class\s*=\s*["\']', re.IGNORECASE),
            re.compile(r'href\s*=\s*["\']', re.IGNORECASE),
            re.compile(r'src\s*=\s*["\']', re.IGNORECASE),
            re.compile(r'onclick\s*=', re.IGNORECASE),
            re.compile(r'getElementById', re.IGNORECASE),
            re.compile(r'addEventListener', re.IGNORECASE),
            re.compile(r'tripadvisor llc', re.IGNORECASE),
            re.compile(r'this review is the subjective opinion', re.IGNORECASE),
            re.compile(r'copyright|privacy|terms of service', re.IGNORECASE),
            re.compile(r'error|exception|timeout|workflow|api|uspapi', re.IGNORECASE),
            re.compile(r'SubmitButton|inputAriaLabel|feedbackPlaceholder', re.IGNORECASE),
            re.compile(r'hang tight while|dive into our|scan english', re.IGNORECASE),
            re.compile(r'something went wrong|please try again|check again later', re.IGNORECASE),
            re.compile(r'cannot encode|problem updating|problem retrieving', re.IGNORECASE),
            re.compile(r'flew a little too fast|hit a snag', re.IGNORECASE),
        ]
        
        # File extensions to search
        self.searchable_extensions = {'.txt', '.html', '.htm', '.json', '.js'}

    def setup_database(self):
        """Initialize SQLite database for clean visitor comments"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visitor_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_number INTEGER,
                    comment_text TEXT NOT NULL,
                    source_file TEXT,
                    confidence_score REAL,
                    extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Clear existing data for fresh extraction
            cursor.execute('DELETE FROM visitor_comments')
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            raise

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8') or 'utf-8'
        except:
            return 'utf-8'

    def read_file_content(self, file_path: str) -> Optional[str]:
        """Safely read file content"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"⚠️ Could not read {file_path}: {e}")
            return None

    def is_visitor_comment(self, text: str) -> bool:
        """Check if text looks like an actual visitor comment"""
        # Exclude technical content
        for exclude_pattern in self.exclude_patterns:
            if exclude_pattern.search(text):
                return False
        
        # Must contain personal experience indicators
        personal_indicators = [
            'i visited', 'we visited', 'my visit', 'our visit', 'my experience',
            'i went', 'we went', 'i love', 'i hate', 'i recommend', 'we recommend',
            'i think', 'i believe', 'i found', 'we found', 'my family', 'our family'
        ]
        
        # OR temple-specific review content
        temple_review_indicators = [
            'temple is', 'temple was', 'sensoji', 'senso-ji', 'beautiful temple',
            'amazing temple', 'crowded temple', 'peaceful temple', 'historic temple',
            'must visit', 'worth visiting', 'don\'t miss', 'highly recommend',
            'waste of time', 'not worth', 'amazing experience', 'terrible experience'
        ]
        
        text_lower = text.lower()
        
        has_personal_experience = any(indicator in text_lower for indicator in personal_indicators)
        has_temple_review = any(indicator in text_lower for indicator in temple_review_indicators)
        
        # Must look like natural language (not code)
        has_natural_language = (
            ' ' in text and
            len(text.split()) >= 5 and  # At least 5 words
            not text.startswith(('{', '[', '<', 'function', 'var', 'class')) and
            not '=' in text[:20] and
            text.count('"') < 3 and  # Not overly quoted like code
            not re.search(r'\b(?:js|css|html|api|error|timeout)\b', text.lower())
        )
        
        return (has_personal_experience or has_temple_review) and has_natural_language

    def calculate_comment_confidence(self, text: str, source_file: str) -> float:
        """Calculate confidence that this is a real visitor comment"""
        score = 0.5
        
        # Boost for review-specific words
        review_words = ['temple', 'visit', 'recommend', 'beautiful', 'amazing', 'terrible', 'worth']
        word_count = sum(1 for word in review_words if word.lower() in text.lower())
        score += word_count * 0.1
        
        # Boost for proper sentence structure
        if text[0].isupper() and text.endswith(('.', '!', '?')):
            score += 0.2
        
        # Boost for personal pronouns (indicates personal experience)
        personal_pronouns = ['i ', 'we ', 'my ', 'our ', 'me ', 'us ']
        if any(pronoun in text.lower() for pronoun in personal_pronouns):
            score += 0.3
        
        # Penalty for very short or very long text
        if len(text) < 30:
            score -= 0.2
        elif len(text) > 300:
            score -= 0.1
        
        # Boost for JSON files (more likely to contain structured review data)
        if source_file.endswith('.json'):
            score += 0.2
        
        return min(score, 1.0)

    def extract_comments_from_file(self, file_path: str) -> List[Tuple[str, float]]:
        """Extract visitor comments from a single file"""
        content = self.read_file_content(file_path)
        if not content:
            return []
        
        comments = []
        found_texts = set()  # Avoid duplicates
        
        # Try each pattern
        for pattern in self.comment_patterns:
            matches = pattern.findall(content)
            
            for match in matches:
                # Clean the text
                if isinstance(match, tuple):
                    text = match[0]
                else:
                    text = match
                
                # Clean and validate
                text = self.clean_comment_text(text)
                
                if (text and 
                    len(text) >= 15 and 
                    len(text) <= 500 and  # Reasonable comment length
                    text not in found_texts and 
                    self.is_visitor_comment(text)):
                    
                    confidence = self.calculate_comment_confidence(text, file_path)
                    if confidence > 0.7:  # Higher threshold for quality
                        comments.append((text, confidence))
                        found_texts.add(text)
        
        return comments

    def clean_comment_text(self, text: str) -> str:
        """Clean and normalize comment text"""
        if not text:
            return ""
        
        # Remove escape characters and extra whitespace
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ').replace('&quot;', '"')
        
        # Ensure proper sentence capitalization
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

    def process_all_files(self) -> List[Tuple[str, str, float]]:
        """Process all files and extract visitor comments"""
        all_comments = []
        processed_files = 0
        
        logger.info(f"🔍 Searching for visitor comments in: {self.folder_path}")
        
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in self.searchable_extensions:
                    try:
                        comments = self.extract_comments_from_file(file_path)
                        for comment_text, confidence in comments:
                            all_comments.append((comment_text, file_path, confidence))
                        
                        processed_files += 1
                        if processed_files % 100 == 0:
                            logger.info(f"📊 Processed {processed_files} files, found {len(all_comments)} comments")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        # Sort by confidence score (highest first)
        all_comments.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"✅ Processing complete! Found {len(all_comments)} visitor comments")
        return all_comments

    def save_comments_to_database(self, comments: List[Tuple[str, str, float]]):
        """Save cleaned comments to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, (comment_text, source_file, confidence) in enumerate(comments, 1):
                cursor.execute('''
                    INSERT INTO visitor_comments 
                    (comment_number, comment_text, source_file, confidence_score)
                    VALUES (?, ?, ?, ?)
                ''', (i, comment_text, os.path.basename(source_file), confidence))
            
            conn.commit()
            conn.close()
            logger.info(f"💾 Saved {len(comments)} comments to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save comments: {e}")

    def print_first_10_comments(self):
        """Print the first 10 visitor comments in the requested format"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT comment_number, comment_text, confidence_score, source_file
                FROM visitor_comments 
                ORDER BY comment_number
                LIMIT 10
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                print("❌ No visitor comments found in database")
                return
            
            print("\n" + "🏮" * 50)
            print("📝 FIRST 10 VISITOR COMMENTS")
            print("🏮" * 50)
            print()
            
            for comment_num, comment_text, confidence, source_file in results:
                print(f"Comment {comment_num}: \"{comment_text}\"")
                print(f"   (Confidence: {confidence:.3f}, Source: {source_file})")
                print()
            
            # Show total count
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM visitor_comments')
            total_count = cursor.fetchone()[0]
            print(f"📊 Total visitor comments extracted: {total_count}")
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve comments: {e}")

def main():
    """Main function"""
    print("🏮" * 40)
    print("VISITOR COMMENTS EXTRACTOR")
    print("🏮" * 40)
    
    # Initialize extractor
    extractor = VisitorCommentsExtractor()
    
    # Extract comments
    print("\n🔍 Extracting actual visitor comments...")
    comments = extractor.process_all_files()
    
    if not comments:
        print("❌ No visitor comments found. The files might not contain actual review text.")
        return
    
    # Save to database
    print("\n💾 Saving comments to database...")
    extractor.save_comments_to_database(comments)
    
    # Print first 10 comments
    print("\n📋 Displaying first 10 visitor comments:")
    extractor.print_first_10_comments()
    
    print(f"\n🎉 Extraction complete! Database saved to: {extractor.db_path}")

if __name__ == "__main__":
    main()