#!/usr/bin/env python3
"""
Database Content Checker
- Reads the filtered_reviews.db or comment_blocks.db database
- Prints out all stored content with scores and statistics
"""

import sqlite3
import os
from datetime import datetime

def check_database_content():
    """Check and display database content"""
    
    # Check for both possible database names
    db_paths = [
        "/Users/andreyvlasenko/tst/Request/filtered_reviews.db",
        "/Users/andreyvlasenko/tst/Request/comment_blocks.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("No database found. Looking for:")
        for path in db_paths:
            print(f"  - {path}")
        return
    
    print(f"Reading database: {db_path}")
    print("=" * 80)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Available tables: {[table[0] for table in tables]}")
        print()
        
        # Check for filtered_reviews table
        if any('filtered_reviews' in str(table) for table in tables):
            print("FILTERED_REVIEWS TABLE:")
            print("-" * 50)
            
            cursor.execute("SELECT COUNT(*) FROM filtered_reviews")
            total_count = cursor.fetchone()[0]
            print(f"Total records: {total_count}")
            
            if total_count > 0:
                cursor.execute("""
                SELECT source_file, review_text, review_score, block_length 
                FROM filtered_reviews 
                ORDER BY review_score DESC
                """)
                
                records = cursor.fetchall()
                for i, (source_file, review_text, score, length) in enumerate(records, 1):
                    print(f"\nRecord {i}:")
                    print(f"  Source: {os.path.basename(source_file)}")
                    print(f"  Score: {score:.3f}")
                    print(f"  Length: {length}")
                    print(f"  Text: {review_text[:200]}{'...' if len(review_text) > 200 else ''}")
            
        # Check for comment_blocks table
        elif any('comment_blocks' in str(table) for table in tables):
            print("COMMENT_BLOCKS TABLE:")
            print("-" * 50)
            
            cursor.execute("SELECT COUNT(*) FROM comment_blocks")
            total_count = cursor.fetchone()[0]
            print(f"Total records: {total_count}")
            
            cursor.execute("SELECT COUNT(*) FROM comment_blocks WHERE is_candidate = 1")
            candidate_count = cursor.fetchone()[0]
            print(f"Candidate records (score >= 0.65): {candidate_count}")
            
            cursor.execute("SELECT AVG(score) FROM comment_blocks")
            avg_score = cursor.fetchone()[0]
            print(f"Average score: {avg_score:.3f}" if avg_score else "No scores available")
            
            if total_count > 0:
                print(f"\nTop scoring blocks:")
                cursor.execute("""
                SELECT file_path, block_text, score, length, is_candidate 
                FROM comment_blocks 
                ORDER BY score DESC 
                LIMIT 10
                """)
                
                records = cursor.fetchall()
                for i, (file_path, block_text, score, length, is_candidate) in enumerate(records, 1):
                    status = "CANDIDATE" if is_candidate else "FILTERED OUT"
                    print(f"\nBlock {i} [{status}]:")
                    print(f"  Source: {os.path.basename(file_path)}")
                    print(f"  Score: {score:.3f}")
                    print(f"  Length: {length}")
                    print(f"  Text: {block_text[:200]}{'...' if len(block_text) > 200 else ''}")
                
                # Show candidates only
                if candidate_count > 0:
                    print(f"\n" + "="*80)
                    print(f"CANDIDATES ONLY (score >= 0.65):")
                    print("-" * 50)
                    
                    cursor.execute("""
                    SELECT block_text, score 
                    FROM comment_blocks 
                    WHERE is_candidate = 1 
                    ORDER BY score DESC
                    """)
                    
                    candidates = cursor.fetchall()
                    for i, (block_text, score) in enumerate(candidates, 1):
                        print(f"\nCandidate {i} (Score: {score:.3f}):")
                        print(f"'{block_text}'")
                else:
                    print(f"\nNo candidates found with score >= 0.65")
        
        # Show score distribution
        print(f"\n" + "="*80)
        print("SCORE DISTRIBUTION:")
        print("-" * 30)
        
        table_name = 'comment_blocks' if any('comment_blocks' in str(table) for table in tables) else 'filtered_reviews'
        score_column = 'score' if table_name == 'comment_blocks' else 'review_score'
        
        cursor.execute(f"""
        SELECT 
            COUNT(CASE WHEN {score_column} >= 0.8 THEN 1 END) as high,
            COUNT(CASE WHEN {score_column} >= 0.65 AND {score_column} < 0.8 THEN 1 END) as good,
            COUNT(CASE WHEN {score_column} >= 0.5 AND {score_column} < 0.65 THEN 1 END) as medium,
            COUNT(CASE WHEN {score_column} >= 0.3 AND {score_column} < 0.5 THEN 1 END) as low,
            COUNT(CASE WHEN {score_column} < 0.3 THEN 1 END) as very_low
        FROM {table_name}
        """)
        
        high, good, medium, low, very_low = cursor.fetchone()
        print(f"High (≥0.8):    {high}")
        print(f"Good (≥0.65):   {good}")
        print(f"Medium (≥0.5):  {medium}")
        print(f"Low (≥0.3):     {low}")
        print(f"Very Low (<0.3): {very_low}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    check_database_content()