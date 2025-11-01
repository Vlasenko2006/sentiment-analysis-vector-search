#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract visible text from downloaded TripAdvisor HTML page
Organizes content into separate blocks and saves as txt file

Created on November 1, 2025
@author: andreyvlasenko
"""

import os
import re
from bs4 import BeautifulSoup
from datetime import datetime

# Configuration
CACHE_FOLDER = "cache"
OUTPUT_FOLDER = "extracted_text"

# Separator keywords for splitting comments
SEPARATOR_KEYWORDS = [
    '•',  # Bullet point - most important for TripAdvisor
    'Date of visit:',
    'Written ',
    'Reviewed ',
    'Visited ',
    'Date:',
    'Trip type:',
    'Helpful',
    'Share',
    'Read more',
    'Report'
]

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def is_quoted_or_citation(sentence):
    """
    Check if a sentence is a quote or citation
    
    Args:
        sentence: Text to check
    
    Returns:
        True if sentence contains quotes or citation markers
    """
    # Check for quoted text with "..." or '...'
    if re.search(r'["\'].*?["\']', sentence):
        return True
    
    # Check for citation markers
    citation_patterns = [
        r'\[.*?\]',  # [citation]
        r'\(.*?\)',  # (reference)
        r'according to',
        r'said',
        r'quoted',
        r'states',
        r'mentioned'
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    
    return False

def has_duplicate_sentence(text, existing_texts):
    """
    Check if any sentence in text appears in existing texts
    Excludes quoted text and citations from duplicate detection
    
    Args:
        text: Text to check
        existing_texts: List of already processed texts
    
    Returns:
        True if duplicate sentence found (excluding quotes/citations), False otherwise
    """
    # Split into sentences (simple split by . ! ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    for sentence in sentences:
        # Skip duplicate check if this is a quoted text or citation
        if is_quoted_or_citation(sentence):
            continue
        
        for existing_text in existing_texts:
            # Check if sentence appears in existing text
            if sentence in existing_text:
                # But only mark as duplicate if the existing text doesn't have this as a quote/citation
                if not is_quoted_or_citation(existing_text):
                    return True
    
    return False

def split_by_separators(text, separators=None):
    """
    Split text into separate blocks using separator keywords
    
    Args:
        text: Text to split
        separators: List of separator keywords
    
    Returns:
        List of text blocks
    """
    if separators is None:
        separators = SEPARATOR_KEYWORDS
    
    blocks = [text]
    
    # Split by each separator keyword
    for separator in separators:
        new_blocks = []
        for block in blocks:
            # Split but keep the separator with the following text
            parts = block.split(separator)
            if len(parts) > 1:
                # First part without separator
                if parts[0].strip():
                    new_blocks.append(parts[0].strip())
                # Remaining parts with separator prefix
                for part in parts[1:]:
                    if part.strip():
                        new_blocks.append(separator + part.strip())
            else:
                if block.strip():
                    new_blocks.append(block.strip())
        blocks = new_blocks
    
    return [b for b in blocks if len(b) > 20]  # Filter out very short blocks

def extract_text_blocks(html_file):
    """
    Extract all visible text from HTML file organized into blocks
    
    Args:
        html_file: Path to HTML file
    
    Returns:
        Dictionary with organized text blocks
    """
    print(f"📖 Reading HTML file: {html_file}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script, style, and other non-visible elements
    for element in soup(['script', 'style', 'meta', 'link', 'noscript', 'iframe']):
        element.decompose()
    
    text_blocks = {
        'title': '',
        'restaurant_info': [],
        'reviews': [],
        'ratings': [],
        'descriptions': [],
        'other_text': []
    }
    
    # Extract page title
    if soup.title:
        text_blocks['title'] = clean_text(soup.title.string)
        print(f"   📝 Title: {text_blocks['title']}")
    
    # Extract restaurant name and basic info
    restaurant_name = soup.find('h1')
    if restaurant_name:
        text_blocks['restaurant_info'].append(f"Restaurant Name: {clean_text(restaurant_name.get_text())}")
    
    # Extract all paragraphs first (natural separators)
    all_paragraphs = []
    for p in soup.find_all('p'):
        text = clean_text(p.get_text())
        if len(text) > 30:
            all_paragraphs.append(text)
    
    # Split paragraphs by separator keywords
    for paragraph in all_paragraphs:
        split_blocks = split_by_separators(paragraph)
        for block in split_blocks:
            # Categorize the block
            if any(word in block.lower() for word in ['review', 'visited', 'stayed', 'experience', 'excellent', 'terrible', 'good', 'bad']):
                text_blocks['reviews'].append(block)
            elif any(char.isdigit() for char in block) and any(word in block.lower() for word in ['rating', 'star', 'score']):
                text_blocks['ratings'].append(block)
            else:
                text_blocks['descriptions'].append(block)
    
    # Extract all reviews (TripAdvisor specific selectors)
    review_containers = soup.find_all(['div', 'span'], class_=re.compile(r'review|comment', re.I))
    
    for container in review_containers:
        # Get text preserving paragraph structure
        text = clean_text(container.get_text(separator='\n', strip=True))
        
        if len(text) > 50:
            # Split by newlines (paragraph breaks) first
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            for para in paragraphs:
                # Further split by separator keywords (including bullet point •)
                split_blocks = split_by_separators(para)
                
                for block in split_blocks:
                    # Check for duplicate sentences before adding
                    if any(word in block.lower() for word in ['review', 'visited', 'stayed', 'experience']):
                        if block not in text_blocks['reviews'] and not has_duplicate_sentence(block, text_blocks['reviews']):
                            text_blocks['reviews'].append(block)
                    elif any(char.isdigit() for char in block) and any(word in block.lower() for word in ['rating', 'star', 'score']):
                        if block not in text_blocks['ratings'] and not has_duplicate_sentence(block, text_blocks['ratings']):
                            text_blocks['ratings'].append(block)
                    else:
                        if block not in text_blocks['other_text'] and not has_duplicate_sentence(block, text_blocks['other_text']):
                            text_blocks['other_text'].append(block)
    
    # Extract all divs with substantial text
    for div in soup.find_all('div'):
        # Get text preserving structure with newlines
        text = clean_text(div.get_text(separator='\n', strip=True))
        
        if len(text) > 50:
            # Split by newlines (paragraph breaks)
            paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 30]
            
            for para in paragraphs:
                # Further split by separator keywords (including bullet point •)
                split_blocks = split_by_separators(para)
                
                for block in split_blocks:
                    # Check if it's not already captured (exact match or duplicate sentences)
                    is_duplicate = False
                    for existing_list in text_blocks.values():
                        if isinstance(existing_list, list):
                            if block in existing_list or has_duplicate_sentence(block, existing_list):
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        text_blocks['other_text'].append(block)
    
    # Remove duplicates while preserving order
    for key in text_blocks:
        if isinstance(text_blocks[key], list):
            seen = set()
            unique_list = []
            for item in text_blocks[key]:
                if item not in seen and len(item) > 20:  # Minimum length filter
                    seen.add(item)
                    unique_list.append(item)
            text_blocks[key] = unique_list
    
    return text_blocks

def save_text_blocks(text_blocks, html_filename):
    """
    Save extracted text blocks to a txt file
    
    Args:
        text_blocks: Dictionary with text blocks
        html_filename: Original HTML filename for naming
    """
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(html_filename))[0]
    output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_text.txt")
    
    print(f"\n💾 Saving extracted text to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTED TEXT FROM TRIPADVISOR PAGE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {html_filename}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write title
        if text_blocks['title']:
            f.write("PAGE TITLE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{text_blocks['title']}\n\n")
        
        # Write restaurant info
        if text_blocks['restaurant_info']:
            f.write("RESTAURANT INFORMATION:\n")
            f.write("-" * 80 + "\n")
            for i, info in enumerate(text_blocks['restaurant_info'], 1):
                f.write(f"{i}. {info}\n")
            f.write("\n")
        
        # Write reviews
        if text_blocks['reviews']:
            f.write("REVIEWS:\n")
            f.write("-" * 80 + "\n")
            for i, review in enumerate(text_blocks['reviews'], 1):
                f.write(f"\n[Review {i}]\n")
                f.write(f"{review}\n")
            f.write("\n")
        
        # Write ratings
        if text_blocks['ratings']:
            f.write("RATINGS:\n")
            f.write("-" * 80 + "\n")
            for i, rating in enumerate(text_blocks['ratings'], 1):
                f.write(f"{i}. {rating}\n")
            f.write("\n")
        
        # Write descriptions
        if text_blocks['descriptions']:
            f.write("DESCRIPTIONS:\n")
            f.write("-" * 80 + "\n")
            for i, desc in enumerate(text_blocks['descriptions'], 1):
                f.write(f"\n[Block {i}]\n")
                f.write(f"{desc}\n")
            f.write("\n")
        
        # Write other text
        if text_blocks['other_text']:
            f.write("OTHER TEXT BLOCKS:\n")
            f.write("-" * 80 + "\n")
            for i, text in enumerate(text_blocks['other_text'], 1):
                if len(text) > 50:  # Only substantial blocks
                    f.write(f"\n[Block {i}]\n")
                    f.write(f"{text}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF EXTRACTED TEXT\n")
        f.write("=" * 80 + "\n")
    
    # Print statistics
    total_blocks = sum(len(v) if isinstance(v, list) else 1 for v in text_blocks.values() if v)
    print(f"✅ Extraction complete!")
    print(f"\n📊 Statistics:")
    print(f"   • Restaurant info blocks: {len(text_blocks['restaurant_info'])}")
    print(f"   • Reviews: {len(text_blocks['reviews'])}")
    print(f"   • Ratings: {len(text_blocks['ratings'])}")
    print(f"   • Descriptions: {len(text_blocks['descriptions'])}")
    print(f"   • Other text blocks: {len(text_blocks['other_text'])}")
    print(f"   • Total blocks: {total_blocks}")
    
    return output_file

def main():
    """Main function"""
    print("=" * 80)
    print("TEXT EXTRACTOR FROM TRIPADVISOR HTML")
    print("=" * 80)
    
    # Find HTML files in cache folder
    if not os.path.exists(CACHE_FOLDER):
        print(f"❌ Cache folder not found: {CACHE_FOLDER}")
        return
    
    html_files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith('.html')]
    
    if not html_files:
        print(f"❌ No HTML files found in {CACHE_FOLDER}")
        print("   Please run download_page.py first")
        return
    
    print(f"\n📂 Found {len(html_files)} HTML file(s) in cache folder:")
    for i, file in enumerate(html_files, 1):
        print(f"   {i}. {file}")
    
    # Process the most recent file (last in list)
    html_file = os.path.join(CACHE_FOLDER, html_files[-1])
    print(f"\n🎯 Processing: {html_files[-1]}")
    
    # Extract text blocks
    text_blocks = extract_text_blocks(html_file)
    
    # Save to file
    output_file = save_text_blocks(text_blocks, html_file)
    
    print(f"\n🎉 Text extraction complete!")
    print(f"   Output file: {os.path.abspath(output_file)}")
    print(f"   You can now read all visible text from the TripAdvisor page!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
