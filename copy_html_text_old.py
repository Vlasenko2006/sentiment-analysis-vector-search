#!/usr/bin/env python3
"""
Simple HTML Text Copier
=======================

Copies all text content from the Senso-ji Temple HTML file.
This script reads the local HTML file and extracts all visible text content.
"""

import os
import urllib.parse
from bs4 import BeautifulSoup
import re

def copy_text_from_html(file_url):
    """Copy all text content from an HTML file"""
    
    # Handle URL-encoded file path
    if file_url.startswith('file://'):
        file_path = file_url[7:]  # Remove 'file://' prefix
    else:
        file_path = file_url
    
    # Decode URL encoding (e.g., %20 -> space)
    file_path = urllib.parse.unquote(file_path)
    
    print(f"📂 Reading HTML file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        
        print(f"📄 HTML file size: {len(html_content)} characters")
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements to get only visible text
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Extract all text
        all_text = soup.get_text()
        
        # Clean up the text
        # Replace multiple whitespaces with single space
        all_text = re.sub(r'\s+', ' ', all_text)
        # Remove leading/trailing whitespace
        all_text = all_text.strip()
        
        print(f"✅ Successfully extracted {len(all_text)} characters of text")
        
        return all_text
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def display_text_sections(text, section_size=1000):
    """Display text in manageable sections"""
    if not text:
        return
    
    total_length = len(text)
    sections = []
    
    # Split into sections
    for i in range(0, total_length, section_size):
        section = text[i:i + section_size]
        sections.append(section)
    
    print(f"\n📝 Text Content ({len(sections)} sections of ~{section_size} chars each):")
    print("=" * 80)
    
    for i, section in enumerate(sections, 1):
        print(f"\n--- Section {i}/{len(sections)} ---")
        print(section)
        
        if i < len(sections):
            user_input = input(f"\nPress Enter to continue to section {i+1}, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

def save_text_to_file(text, output_path):
    """Save extracted text to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"💾 Text saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving text: {e}")
        return False

def main():
    """Main function"""
    # Target HTML file
    html_file_url = "file:///Users/andreyvlasenko/tst/Request/Senso-ji%20Temple,%20Asakusa.html"
    
    print("📋 HTML Text Copier - Senso-ji Temple")
    print("=" * 50)
    
    # Extract all text from HTML
    extracted_text = copy_text_from_html(html_file_url)
    
    if extracted_text:
        # Show statistics
        word_count = len(extracted_text.split())
        line_count = len(extracted_text.splitlines())
        
        print(f"\n📊 Text Statistics:")
        print(f"   📏 Total characters: {len(extracted_text):,}")
        print(f"   📝 Total words: {word_count:,}")
        print(f"   📄 Total lines: {line_count:,}")
        
        # Save to file
        output_file = "/Users/andreyvlasenko/tst/Request/senso_ji_all_text.txt"
        if save_text_to_file(extracted_text, output_file):
            print(f"✅ Complete text saved to file")
        
        # Ask user if they want to see the text
        print(f"\n🔍 Would you like to view the extracted text?")
        choice = input("Enter 'y' to view text sections, 'n' to skip: ").lower()
        
        if choice == 'y':
            display_text_sections(extracted_text, 800)
        
        # Show a preview anyway
        print(f"\n📖 Text Preview (first 500 characters):")
        print("-" * 60)
        print(extracted_text[:500])
        print("...")
        
    else:
        print("❌ Failed to extract text from the HTML file")

if __name__ == "__main__":
    main()