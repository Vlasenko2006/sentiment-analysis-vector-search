"""
PDF Data Loader Module
Handles loading and extraction of analysis data for PDF generation
"""

import os
import pandas as pd
import json
import sqlite3


def extract_source_info_from_db(db_path):
    """
    Extract source website information from database
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        str: Source information (website name or URL)
    """
    try:
        if not os.path.exists(db_path):
            return "Unknown Source"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unique file paths to determine source
        cursor.execute("SELECT DISTINCT file_path FROM comments LIMIT 5")
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not file_paths:
            return "Unknown Source"
        
        # Extract website/source info from file paths
        sources = []
        for fp in file_paths:
            if 'tripadvisor' in fp.lower():
                sources.append('TripAdvisor')
            elif 'google' in fp.lower():
                sources.append('Google Reviews')
            elif 'yelp' in fp.lower():
                sources.append('Yelp')
            else:
                # Try to extract meaningful name from filename
                basename = os.path.basename(fp)
                name = basename.replace('_', ' ').replace('-', ' ').replace('.html', '').replace('.txt', '')
                sources.append(name)
        
        if sources:
            return sources[0]  # Return the first/main source
        return "Unknown Source"
        
    except Exception as e:
        print(f"⚠️ Error extracting source info: {e}")
        return "Unknown Source"


def load_existing_data(OUTPUT_BASE_DIR):
    """
    Load existing analysis data from output directory
    
    Args:
        OUTPUT_BASE_DIR: Base directory containing analysis outputs
        
    Returns:
        tuple: (results_df, representative_results, performance_summary)
    """
    try:
        # Load from saved CSV file - try both filenames
        csv_path = os.path.join(OUTPUT_BASE_DIR, "complete_results.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(OUTPUT_BASE_DIR, "sentiment_results.csv")
        
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(results_df)} results from CSV")
        else:
            results_df = None
            print(f"⚠️ Results CSV not found at {OUTPUT_BASE_DIR}")
        
        # Load representative comments
        repr_path = os.path.join(OUTPUT_BASE_DIR, "representative_comments.json")
        if os.path.exists(repr_path):
            with open(repr_path, 'r', encoding='utf-8') as f:
                representative_results = json.load(f)
            print(f"✅ Loaded representative comments")
        else:
            representative_results = {}
        
        # Load performance summary
        perf_path = os.path.join(OUTPUT_BASE_DIR, "performance_summary.json")
        if os.path.exists(perf_path):
            with open(perf_path, 'r', encoding='utf-8') as f:
                performance_summary = json.load(f)
            print(f"✅ Loaded performance summary")
        else:
            performance_summary = {}
        
        return results_df, representative_results, performance_summary
        
    except Exception as e:
        print(f"❌ Error loading existing data: {e}")
        return None, None, None
