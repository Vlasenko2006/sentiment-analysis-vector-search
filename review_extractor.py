#!/usr/bin/env python3
"""
Review Content Extractor
========================

Searches through downloaded TripAdvisor files for review-related content
using various keywords and patterns, then stores findings in a SQLite database.

Usage:
    python review_extractor.py [folder_path]

If no folder path is provided, it will search in /Users/andreyvlasenko/tst/Request/Senso
"""

import os
import re
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import mimetypes
from dataclasses import dataclass
import chardet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReviewMatch:
    """Data class for storing review matches"""
    file_path: str
    file_type: str
    keyword: str
    context: str
    line_number: int
    confidence_score: float
    timestamp: str

class ReviewExtractor:
    """Extract review content from various file types"""
    
    def __init__(self, db_path: str = "reviews_database.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Review-related keywords in multiple languages
        self.keywords = {
            'english': [
                'review', 'rating', 'comment', 'feedback', 'opinion',
                'excellent', 'good', 'bad', 'terrible', 'amazing',
                'recommend', 'worth visiting', 'experience', 'visited',
                'stars', 'out of 5', 'tripadvisor', 'user review',
                'customer review', 'traveler', 'tourist', 'guest'
            ],
            'japanese': [
                'レビュー', '評価', 'コメント', '口コミ', '感想',
                '素晴らしい', '良い', '悪い', 'おすすめ', '訪問',
                '星', '評判', '旅行者', '観光客', '体験',
                '浅草', '浅草寺', '仲見世', '雷門'
            ],
            'chinese': [
                '评论', '评价', '评分', '点评', '推荐',
                '游客', '旅行者', '体验', '访问', '很好',
                '不错', '差评', '好评'
            ]
        }
        
        # File extensions to search
        self.searchable_extensions = {
            '.txt', '.html', '.htm', '.json', '.xml', '.csv',
            '.log', '.md', '.doc', '.docx', '.rtf', '.js',
            '.css', '.php', '.asp', '.jsp'
        }
        
        # Patterns for structured data
        self.patterns = {
            'rating_pattern': re.compile(r'(\d+(?:\.\d+)?)\s*(?:out\s+of\s+|\/)?\s*(\d+)\s*(?:stars?|★|☆)', re.IGNORECASE),
            'star_pattern': re.compile(r'[★☆]{1,5}|(\d+)\s*[★☆]', re.IGNORECASE),
            'date_pattern': re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'),
            'review_structure': re.compile(r'(?:review|rating|comment).*?(?=review|rating|comment|$)', re.IGNORECASE | re.DOTALL),
            'tripadvisor_pattern': re.compile(r'tripadvisor|trip\s*advisor', re.IGNORECASE),
            'senso_pattern': re.compile(r'senso[-\s]*ji|浅草寺|sensoji', re.IGNORECASE)
        }

    def setup_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_type TEXT,
                    keyword TEXT,
                    context TEXT,
                    line_number INTEGER,
                    confidence_score REAL,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_path TEXT NOT NULL,
                    total_files INTEGER,
                    processed_files INTEGER,
                    matches_found INTEGER,
                    processing_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
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
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8') or 'utf-8'
        except:
            return 'utf-8'

    def read_file_content(self, file_path: str) -> Optional[str]:
        """Safely read file content with encoding detection"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"⚠️ Could not read {file_path}: {e}")
            return None

    def calculate_confidence_score(self, text: str, keyword: str, file_path: str) -> float:
        """Calculate confidence score based on various factors"""
        score = 0.5  # Base score
        
        # Boost score for TripAdvisor-specific patterns
        if self.patterns['tripadvisor_pattern'].search(text):
            score += 0.3
            
        # Boost for Senso-ji specific content
        if self.patterns['senso_pattern'].search(text):
            score += 0.2
            
        # Boost for rating patterns
        if self.patterns['rating_pattern'].search(text) or self.patterns['star_pattern'].search(text):
            score += 0.3
            
        # Boost for date patterns (reviews often have dates)
        if self.patterns['date_pattern'].search(text):
            score += 0.1
            
        # Boost for multiple review keywords
        all_keywords = []
        for lang_keywords in self.keywords.values():
            all_keywords.extend(lang_keywords)
        
        keyword_count = sum(1 for kw in all_keywords if kw.lower() in text.lower())
        score += min(keyword_count * 0.05, 0.3)
        
        # Boost for JSON files (likely to contain structured review data)
        if file_path.endswith('.json'):
            score += 0.2
            
        return min(score, 1.0)

    def extract_context(self, text: str, keyword: str, line_number: int) -> str:
        """Extract context around the keyword"""
        lines = text.split('\n')
        start = max(0, line_number - 2)
        end = min(len(lines), line_number + 3)
        
        context_lines = lines[start:end]
        context = '\n'.join(context_lines)
        
        # Limit context length
        if len(context) > 500:
            # Find keyword position and extract around it
            keyword_pos = context.lower().find(keyword.lower())
            if keyword_pos != -1:
                start_pos = max(0, keyword_pos - 200)
                end_pos = min(len(context), keyword_pos + 300)
                context = "..." + context[start_pos:end_pos] + "..."
        
        return context.strip()

    def search_file_content(self, file_path: str, content: str) -> List[ReviewMatch]:
        """Search for review-related content in file"""
        matches = []
        lines = content.split('\n')
        
        # Get all keywords
        all_keywords = []
        for lang_keywords in self.keywords.values():
            all_keywords.extend(lang_keywords)
        
        # Search each line
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            for keyword in all_keywords:
                if keyword.lower() in line_lower:
                    context = self.extract_context(content, keyword, line_num - 1)
                    confidence = self.calculate_confidence_score(context, keyword, file_path)
                    
                    # Only include matches with reasonable confidence
                    if confidence > 0.3:
                        match = ReviewMatch(
                            file_path=file_path,
                            file_type=Path(file_path).suffix,
                            keyword=keyword,
                            context=context,
                            line_number=line_num,
                            confidence_score=confidence,
                            timestamp=datetime.now().isoformat()
                        )
                        matches.append(match)
        
        return matches

    def process_file(self, file_path: str) -> List[ReviewMatch]:
        """Process a single file for review content"""
        logger.debug(f"🔍 Processing: {file_path}")
        
        # Check if file is searchable
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.searchable_extensions:
            return []
        
        # Read file content
        content = self.read_file_content(file_path)
        if not content:
            return []
        
        # Search for matches
        matches = self.search_file_content(file_path, content)
        
        if matches:
            logger.info(f"📝 Found {len(matches)} matches in {Path(file_path).name}")
        
        return matches

    def save_matches_to_db(self, matches: List[ReviewMatch]):
        """Save matches to database"""
        if not matches:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for match in matches:
                cursor.execute('''
                    INSERT INTO review_matches 
                    (file_path, file_type, keyword, context, line_number, confidence_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match.file_path, match.file_type, match.keyword,
                    match.context, match.line_number, match.confidence_score,
                    match.timestamp
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"💾 Saved {len(matches)} matches to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save matches: {e}")

    def process_folder(self, folder_path: str) -> Dict:
        """Process all files in a folder"""
        start_time = datetime.now()
        logger.info(f"🔍 Starting review extraction from: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"❌ Folder not found: {folder_path}")
            return {}
        
        all_matches = []
        total_files = 0
        processed_files = 0
        
        # Walk through all files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    matches = self.process_file(file_path)
                    if matches:
                        all_matches.extend(matches)
                    processed_files += 1
                    
                    # Progress update every 50 files
                    if processed_files % 50 == 0:
                        logger.info(f"📊 Progress: {processed_files}/{total_files} files processed")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        # Save matches to database
        self.save_matches_to_db(all_matches)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Save statistics
        self.save_stats(folder_path, total_files, processed_files, len(all_matches), processing_time)
        
        results = {
            'folder_path': folder_path,
            'total_files': total_files,
            'processed_files': processed_files,
            'total_matches': len(all_matches),
            'processing_time': processing_time,
            'matches_by_confidence': {
                'high (>0.8)': len([m for m in all_matches if m.confidence_score > 0.8]),
                'medium (0.5-0.8)': len([m for m in all_matches if 0.5 <= m.confidence_score <= 0.8]),
                'low (0.3-0.5)': len([m for m in all_matches if 0.3 <= m.confidence_score < 0.5])
            }
        }
        
        logger.info(f"✅ Processing complete!")
        logger.info(f"📊 Results: {len(all_matches)} matches found in {processed_files} files")
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        return results

    def save_stats(self, folder_path: str, total_files: int, processed_files: int, matches_found: int, processing_time: float):
        """Save processing statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO file_stats 
                (folder_path, total_files, processed_files, matches_found, processing_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (folder_path, total_files, processed_files, matches_found, processing_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to save stats: {e}")

    def generate_report(self) -> str:
        """Generate a comprehensive report of findings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get summary statistics
            cursor.execute('SELECT COUNT(*) FROM review_matches')
            total_matches = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(confidence_score) FROM review_matches')
            avg_confidence = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT file_type, COUNT(*) as count 
                FROM review_matches 
                GROUP BY file_type 
                ORDER BY count DESC
            ''')
            file_type_stats = cursor.fetchall()
            
            cursor.execute('''
                SELECT keyword, COUNT(*) as count 
                FROM review_matches 
                GROUP BY keyword 
                ORDER BY count DESC 
                LIMIT 10
            ''')
            top_keywords = cursor.fetchall()
            
            cursor.execute('''
                SELECT file_path, COUNT(*) as matches, AVG(confidence_score) as avg_conf
                FROM review_matches 
                GROUP BY file_path 
                ORDER BY matches DESC, avg_conf DESC
                LIMIT 10
            ''')
            top_files = cursor.fetchall()
            
            conn.close()
            
            # Generate report
            report = f"""
🔍 REVIEW EXTRACTION REPORT
{'='*50}

📊 SUMMARY:
- Total matches found: {total_matches}
- Average confidence score: {avg_confidence:.3f}
- Database location: {self.db_path}

📁 MATCHES BY FILE TYPE:
"""
            for file_type, count in file_type_stats:
                report += f"  {file_type or 'no extension'}: {count} matches\n"
            
            report += f"\n🔑 TOP KEYWORDS:\n"
            for keyword, count in top_keywords:
                report += f"  '{keyword}': {count} matches\n"
            
            report += f"\n📄 TOP FILES WITH MATCHES:\n"
            for file_path, matches, avg_conf in top_files:
                file_name = os.path.basename(file_path)
                report += f"  {file_name}: {matches} matches (avg confidence: {avg_conf:.3f})\n"
            
            report += f"\n💡 NEXT STEPS:\n"
            report += f"  1. Review high-confidence matches (>0.8) for actual review content\n"
            report += f"  2. Check files with most matches for structured review data\n"
            report += f"  3. Use 'python review_extractor.py --query' to explore specific findings\n"
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {e}")
            return "Error generating report"

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract review content from downloaded files')
    parser.add_argument('folder', nargs='?', 
                       default='/Users/andreyvlasenko/tst/Request/Senso',
                       help='Folder path to search (default: Senso folder)')
    parser.add_argument('--db', default='reviews_database.db',
                       help='Database file path')
    parser.add_argument('--report', action='store_true',
                       help='Generate and display report')
    parser.add_argument('--query', action='store_true',
                       help='Interactive query mode')
    
    args = parser.parse_args()
    
    extractor = ReviewExtractor(args.db)
    
    if args.report:
        print(extractor.generate_report())
        return
    
    if args.query:
        # Interactive query mode
        print("🔍 Interactive Query Mode")
        print("Available commands: 'high', 'medium', 'low', 'files', 'keywords', 'exit'")
        
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        
        while True:
            cmd = input("\nQuery> ").strip().lower()
            
            if cmd == 'exit':
                break
            elif cmd == 'high':
                cursor.execute('SELECT * FROM review_matches WHERE confidence_score > 0.8 ORDER BY confidence_score DESC LIMIT 10')
                results = cursor.fetchall()
                for row in results:
                    print(f"📄 {os.path.basename(row[1])}: '{row[3]}' (conf: {row[6]:.3f})")
                    print(f"    Context: {row[4][:100]}...")
            elif cmd == 'files':
                cursor.execute('''
                    SELECT file_path, COUNT(*) as matches 
                    FROM review_matches 
                    GROUP BY file_path 
                    ORDER BY matches DESC LIMIT 10
                ''')
                results = cursor.fetchall()
                for file_path, count in results:
                    print(f"📁 {os.path.basename(file_path)}: {count} matches")
            # Add more query options as needed
            
        conn.close()
        return
    
    # Process folder
    results = extractor.process_folder(args.folder)
    
    # Display results
    print("\n" + "="*60)
    print("🎉 EXTRACTION COMPLETE!")
    print("="*60)
    print(f"📁 Folder: {results['folder_path']}")
    print(f"📊 Files processed: {results['processed_files']}/{results['total_files']}")
    print(f"🔍 Total matches: {results['total_matches']}")
    print(f"⏱️ Time: {results['processing_time']:.2f} seconds")
    
    if results['total_matches'] > 0:
        print(f"\n📈 Confidence distribution:")
        for level, count in results['matches_by_confidence'].items():
            print(f"  {level}: {count} matches")
        
        print(f"\n📋 Generate detailed report with: python {__file__} --report")
        print(f"🔍 Explore results with: python {__file__} --query")
    else:
        print("\n⚠️ No review content found. This might indicate:")
        print("  - Files don't contain TripAdvisor review data")
        print("  - Content is in a different format or language")
        print("  - Review data might be in binary/encoded files")

if __name__ == "__main__":
    main()