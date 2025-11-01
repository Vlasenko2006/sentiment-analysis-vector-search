#!/usr/bin/env python3
"""
Website Text Block Extractor
============================

Extracts text content from any website and splits it into individual blocks
(comments, descriptions, titles, etc.) based on paragraph breaks.

Features:
- Extract text from any website URL
- Split content into individual text blocks
- Save blocks as separate list items
- Handle both local files and web URLs
- Default TripAdvisor support
"""

import os
import requests
import urllib.parse
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsiteTextBlockExtractor:
    """Extract and organize text blocks from websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_from_url(self, url: str) -> str:
        """Extract HTML content from a web URL"""
        try:
            logger.info(f"🌐 Fetching content from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Successfully fetched {len(response.content)} bytes")
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch URL: {e}")
            return None
    
    def extract_from_local_file(self, file_path: str) -> str:
        """Extract HTML content from a local file"""
        try:
            # Handle file:// URLs
            if file_path.startswith('file://'):
                file_path = file_path[7:]
            
            # Decode URL encoding
            file_path = urllib.parse.unquote(file_path)
            
            logger.info(f"📂 Reading local file: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            logger.info(f"✅ Successfully read {len(content)} characters from file")
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            return None
    
    def clean_html_to_text(self, html_content: str) -> str:
        """Convert HTML to clean text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to clean HTML: {e}")
            return None
    
    def split_into_blocks(self, text: str, min_length: int = 10) -> List[str]:
        """Split text into individual blocks based on paragraph breaks"""
        if not text:
            return []
        
        # Split by common paragraph separators
        blocks = []
        
        # First, try to split by double newlines (paragraph breaks)
        potential_blocks = re.split(r'\n\s*\n', text)
        
        for block in potential_blocks:
            # Further split by single newlines if blocks are too long
            if len(block) > 500:  # If block is very long, split by single newlines
                sub_blocks = block.split('\n')
                for sub_block in sub_blocks:
                    cleaned_block = sub_block.strip()
                    if len(cleaned_block) >= min_length:
                        blocks.append(cleaned_block)
            else:
                cleaned_block = block.strip()
                if len(cleaned_block) >= min_length:
                    blocks.append(cleaned_block)
        
        # If we don't have enough blocks, try splitting by sentences
        if len(blocks) < 5:
            logger.info("🔄 Trying sentence-based splitting...")
            sentence_blocks = re.split(r'[.!?]+\s+', text)
            blocks = []
            for sentence in sentence_blocks:
                cleaned_sentence = sentence.strip()
                if len(cleaned_sentence) >= min_length:
                    blocks.append(cleaned_sentence)
        
        logger.info(f"📝 Created {len(blocks)} text blocks")
        return blocks
    
    def categorize_blocks(self, blocks: List[str]) -> Dict[str, List[str]]:
        """Categorize blocks into different types"""
        categorized = {
            'titles': [],
            'descriptions': [],
            'reviews': [],
            'navigation': [],
            'other': []
        }
        
        for block in blocks:
            block_lower = block.lower()
            
            # Categorize based on content patterns
            if any(keyword in block_lower for keyword in ['review', 'comment', 'experience', 'visit', 'recommend']):
                categorized['reviews'].append(block)
            elif len(block) < 100 and any(keyword in block_lower for keyword in ['temple', 'shrine', 'tour', 'hours', 'price']):
                categorized['titles'].append(block)
            elif any(keyword in block_lower for keyword in ['description', 'about', 'history', 'built', 'located']):
                categorized['descriptions'].append(block)
            elif any(keyword in block_lower for keyword in ['menu', 'navigate', 'click', 'sign in', 'book', 'reserve']):
                categorized['navigation'].append(block)
            else:
                categorized['other'].append(block)
        
        return categorized
    
    def save_blocks_to_files(self, blocks: List[str], categorized: Dict[str, List[str]], base_filename: str):
        """Save text blocks to various file formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all blocks as simple list
        all_blocks_file = f"{base_filename}_all_blocks_{timestamp}.txt"
        with open(all_blocks_file, 'w', encoding='utf-8') as f:
            for i, block in enumerate(blocks, 1):
                f.write(f"Block {i}:\n{block}\n\n" + "="*80 + "\n\n")
        logger.info(f"💾 All blocks saved to: {all_blocks_file}")
        
        # Save as JSON
        json_data = {
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'total_blocks': len(blocks),
                'categorized_counts': {k: len(v) for k, v in categorized.items()}
            },
            'all_blocks': blocks,
            'categorized_blocks': categorized
        }
        
        json_file = f"{base_filename}_blocks_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 JSON data saved to: {json_file}")
        
        # Save categorized blocks separately
        for category, category_blocks in categorized.items():
            if category_blocks:
                category_file = f"{base_filename}_{category}_{timestamp}.txt"
                with open(category_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== {category.upper()} BLOCKS ===\n\n")
                    for i, block in enumerate(category_blocks, 1):
                        f.write(f"{category.title()} {i}:\n{block}\n\n" + "-"*60 + "\n\n")
                logger.info(f"💾 {category.title()} blocks saved to: {category_file}")
        
        return all_blocks_file, json_file
    
    def process_website(self, url: str, output_name: str = None) -> List[str]:
        """Main processing function"""
        logger.info(f"🚀 Starting text block extraction")
        
        # Determine if it's a local file or web URL
        if url.startswith('file://') or (os.path.exists(url) and not url.startswith('http')):
            html_content = self.extract_from_local_file(url)
            source_type = "local file"
        else:
            html_content = self.extract_from_url(url)
            source_type = "web URL"
        
        if not html_content:
            logger.error(f"❌ Failed to get content from {source_type}")
            return []
        
        # Clean HTML to text
        clean_text = self.clean_html_to_text(html_content)
        if not clean_text:
            logger.error("❌ Failed to extract clean text")
            return []
        
        logger.info(f"📊 Extracted {len(clean_text)} characters of text")
        
        # Split into blocks
        blocks = self.split_into_blocks(clean_text)
        
        if not blocks:
            logger.error("❌ No text blocks found")
            return []
        
        # Categorize blocks
        categorized = self.categorize_blocks(blocks)
        
        # Generate output filename
        if not output_name:
            if url.startswith('http'):
                # Extract domain name for web URLs
                from urllib.parse import urlparse
                parsed = urlparse(url)
                output_name = f"website_{parsed.netloc.replace('.', '_')}"
            else:
                # Use filename for local files
                output_name = "local_website"
        
        # Save results
        all_blocks_file, json_file = self.save_blocks_to_files(blocks, categorized, output_name)
        
        # Print summary
        print(f"\n" + "="*80)
        print(f"📊 TEXT BLOCK EXTRACTION SUMMARY")
        print("="*80)
        print(f"🌐 Source: {source_type}")
        print(f"📝 Total blocks extracted: {len(blocks)}")
        print(f"📋 Categorization:")
        for category, category_blocks in categorized.items():
            print(f"   {category.title()}: {len(category_blocks)} blocks")
        
        print(f"\n📁 Output files:")
        print(f"   📄 All blocks: {all_blocks_file}")
        print(f"   📊 JSON data: {json_file}")
        
        return blocks

def main():
    """Main function with user input"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text blocks from websites')
    parser.add_argument('--url', type=str, 
                       default='https://www.tripadvisor.com/Attraction_Review-g1066451-d321334-Reviews-Senso_ji_Temple-Asakusa_Tokyo_Tokyo_Prefecture_Kanto.html',
                       help='Website URL to extract from (default: Senso-ji TripAdvisor page)')
    parser.add_argument('--local-file', type=str,
                       default='/Users/andreyvlasenko/tst/Request/Senso-ji Temple, Asakusa.html',
                       help='Local HTML file path (alternative to URL)')
    parser.add_argument('--output-name', type=str,
                       help='Base name for output files')
    parser.add_argument('--use-local', action='store_true',
                       help='Use local file instead of URL')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum length for text blocks')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for URL input')
    
    args = parser.parse_args()
    
    extractor = WebsiteTextBlockExtractor()
    
    print("🌐 Website Text Block Extractor")
    print("=" * 50)
    
    # Interactive mode
    if args.interactive:
        print("\n📝 Interactive Mode")
        print("Enter a website URL or local file path:")
        print("Examples:")
        print("  - https://www.tripadvisor.com/...")
        print("  - file:///path/to/file.html")
        print("  - /path/to/local/file.html")
        
        user_input = input("\nURL or file path (Enter for default TripAdvisor): ").strip()
        
        if user_input:
            url = user_input
        else:
            url = args.url
        
        output_name = input("Output filename prefix (Enter for auto): ").strip()
        if not output_name:
            output_name = None
    else:
        # Use command line arguments
        if args.use_local:
            url = args.local_file
        else:
            url = args.url
        output_name = args.output_name
    
    print(f"\n🎯 Target: {url}")
    
    # Process the website
    blocks = extractor.process_website(url, output_name)
    
    if blocks:
        print(f"\n🌟 First 5 extracted blocks (preview):")
        print("-" * 60)
        for i, block in enumerate(blocks[:5], 1):
            print(f"\nBlock {i}:")
            print(block[:200] + ("..." if len(block) > 200 else ""))
        
        if len(blocks) > 5:
            print(f"\n... and {len(blocks) - 5} more blocks")
    
    print(f"\n✅ Extraction complete!")

if __name__ == "__main__":
    main()