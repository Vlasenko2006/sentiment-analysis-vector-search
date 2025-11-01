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

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

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
    
    # Extract all reviews (TripAdvisor specific selectors)
    review_containers = soup.find_all(['div', 'p', 'span'], class_=re.compile(r'review|comment|text', re.I))
    
    for container in review_containers:
        text = clean_text(container.get_text())
        if len(text) > 20:  # Only meaningful text
            # Try to categorize
            if any(word in text.lower() for word in ['review', 'visited', 'stayed', 'experience']):
                text_blocks['reviews'].append(text)
            elif any(char.isdigit() for char in text) and any(word in text.lower() for word in ['rating', 'star', 'score']):
                text_blocks['ratings'].append(text)
            else:
                text_blocks['other_text'].append(text)
    
    # Extract all paragraphs
    for p in soup.find_all('p'):
        text = clean_text(p.get_text())
        if len(text) > 30:
            text_blocks['descriptions'].append(text)
    
    # Extract all divs with substantial text
    for div in soup.find_all('div'):
        # Only get direct text, not nested
        text = clean_text(div.get_text(separator=' ', strip=True))
        if len(text) > 50 and text not in text_blocks['other_text']:
            # Check if it's not already captured
            is_duplicate = False
            for existing_list in text_blocks.values():
                if isinstance(existing_list, list) and text in existing_list:
                    is_duplicate = True
                    break
            if not is_duplicate:
                text_blocks['other_text'].append(text)
    
    # Remove duplicates while preserving order
    for key in text_blocks:
        if isinstance(text_blocks[key], list):
            seen = set()
            unique_list = []
            for item in text_blocks[key]:
                if item not in seen:
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
