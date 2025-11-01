#!/usr/bin/env python3
"""
HTML Text Extractor - Extracts only visible text content from HTML file
"""

import os
import re
from urllib.parse import unquote
from bs4 import BeautifulSoup
import chardet

def detect_encoding(file_path):
    """Detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return '\n'.join(lines)

def extract_visible_text(file_path):
    """Extract only visible text from HTML file"""
    
    print(f"Extracting text from: {file_path}")
    
    try:
        # Detect encoding
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        
        # Read the HTML file
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            html_content = f.read()
        
        print(f"File size: {len(html_content)} characters")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements (they're not visible)
        for script in soup(["script", "style", "meta", "link", "noscript"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            print("No visible text content found in the HTML file.")
            return
        
        print(f"Extracted text length: {len(cleaned_text)} characters")
        print("=" * 80)
        print("VISIBLE TEXT CONTENT:")
        print("=" * 80)
        print(cleaned_text)
        print("=" * 80)
        
        # Also look for specific content that might be reviews
        potential_reviews = []
        lines = cleaned_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) > 30 and len(line) < 500:
                # Check if it looks like a review
                if any(word in line.lower() for word in [
                    'temple', 'visit', 'beautiful', 'amazing', 'experience', 
                    'recommend', 'crowded', 'traditional', 'historic', 'great',
                    'good', 'bad', 'wonderful', 'terrible', 'love', 'like'
                ]):
                    potential_reviews.append(line)
        
        if potential_reviews:
            print("\nPOTENTIAL REVIEW-LIKE CONTENT:")
            print("=" * 50)
            for i, review in enumerate(potential_reviews, 1):
                print(f"{i}. {review}")
        else:
            print("\nNo potential review content found in visible text.")
            
        # Look for structured data that might contain reviews
        print("\nLOOKING FOR STRUCTURED CONTENT...")
        
        # Check for JSON-LD structured data
        json_scripts = soup.find_all('script', type='application/ld+json')
        if json_scripts:
            print(f"Found {len(json_scripts)} JSON-LD script(s)")
            for i, script in enumerate(json_scripts, 1):
                print(f"JSON-LD {i}: {script.string[:200] if script.string else 'Empty'}...")
        
        # Check for data attributes
        elements_with_data = soup.find_all(attrs=lambda x: x and any(k.startswith('data-') for k in x.keys()))
        if elements_with_data:
            print(f"Found {len(elements_with_data)} elements with data attributes")
        
        # Check for hidden divs that might contain review data
        hidden_divs = soup.find_all('div', style=re.compile(r'display:\s*none', re.I))
        if hidden_divs:
            print(f"Found {len(hidden_divs)} hidden div elements")
            for i, div in enumerate(hidden_divs[:3], 1):  # Show first 3
                text_content = div.get_text(strip=True)
                if text_content and len(text_content) > 20:
                    print(f"Hidden div {i}: {text_content[:100]}...")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    """Main function"""
    # Decode the URL-encoded filename
    encoded_path = "/Users/andreyvlasenko/tst/Request/Senso-ji%20Temple,%20Asakusa.html"
    file_path = unquote(encoded_path)
    
    print(f"Looking for file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        
        # Try to find similar files
        directory = os.path.dirname(file_path)
        if os.path.exists(directory):
            print(f"\nFiles in directory {directory}:")
            for file in os.listdir(directory):
                if file.endswith('.html'):
                    print(f"  - {file}")
        return
    
    extract_visible_text(file_path)

if __name__ == "__main__":
    main()