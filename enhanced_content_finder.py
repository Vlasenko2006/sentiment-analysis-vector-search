#!/usr/bin/env python3
"""
Enhanced Content Finder - Searches for ANY textual content that might contain reviews
"""

import os
import sqlite3
import re
from pathlib import Path
import chardet

def detect_encoding(file_path):
    """Detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def search_text_content(file_path, content):
    """Search for human review content specifically"""
    
    # Patterns for potential review content - more specific to reviews
    review_patterns = [
        r'["\']([^"\']{30,500})["\']',  # Quoted text 30-500 chars
        r'text["\s]*[:=]["\s]*([^"\']{30,300})',  # text: "content"
        r'review["\s]*[:=]["\s]*([^"\']{30,300})',  # review: "text"
        r'comment["\s]*[:=]["\s]*([^"\']{30,300})',  # comment: "text"
        r'description["\s]*[:=]["\s]*([^"\']{30,300})',  # description: "text"
        r'title["\s]*[:=]["\s]*([^"\']{30,300})',  # title: "text"
        r'summary["\s]*[:=]["\s]*([^"\']{30,300})',  # summary: "text"
    ]
    
    # Look for human-readable sentences with review characteristics
    sentence_patterns = [
        r'(I|We|My|Our)\s+[^.!?]{20,200}[.!?]',  # Personal statements
        r'(This|The)\s+(temple|place|location|experience|visit)\s+[^.!?]{20,200}[.!?]',  # About places
        r'(Beautiful|Amazing|Wonderful|Terrible|Awful|Great|Good|Bad|Nice|Poor|Excellent)\s+[^.!?]{20,200}[.!?]',  # Adjective starts
        r'[A-Z][^.!?]*\b(temple|senso.?ji|asakusa|tokyo|japan|visit|recommend|experience|beautiful|amazing|crowded|peaceful|traditional|historic|culture)\b[^.!?]*[.!?]',  # Temple/location related
        r'[A-Z][^.!?]*\b(worth\s+visiting|must\s+see|highly\s+recommend|don.?t\s+miss|very\s+crowded|too\s+touristy)\b[^.!?]*[.!?]',  # Review phrases
    ]
    
    found_content = []
    
    # Search for structured content
    for pattern in review_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            text = match.group(1).strip()
            if is_meaningful_text(text):
                found_content.append(('structured', text))
    
    # Search for sentences
    for pattern in sentence_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            text = match.group(0).strip()
            if is_meaningful_text(text):
                found_content.append(('sentence', text))
    
    return found_content

def is_meaningful_text(text):
    """Check if text appears to be meaningful human content"""
    
    # Must be at least 20 characters
    if len(text) < 20:
        return False
    
    # Skip if too many special characters (more strict)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s.!?,-]', text))
    if special_chars > len(text) * 0.15:  # Reduced from 0.3 to 0.15
        return False
    
    # Must have proper sentence structure
    if not re.search(r'[.!?]', text):
        return False
    
    # Must have some common words
    common_words = r'\b(the|and|or|but|is|was|are|were|have|has|had|this|that|with|for|from|they|we|you|it|in|on|at|to|of|a|an|very|good|great|bad|nice|beautiful|amazing|terrible|love|like|hate|recommend|visit|place|temple)\b'
    if len(re.findall(common_words, text, re.IGNORECASE)) < 3:  # Increased from 2 to 3
        return False
    
    # Skip obvious code patterns (expanded list)
    code_patterns = [
        r'function\s*\(',
        r'var\s+\w+',
        r'let\s+\w+',
        r'const\s+\w+',
        r'document\.',
        r'window\.',
        r'console\.',
        r'getElementById',
        r'addEventListener',
        r'return\s+',
        r'if\s*\(',
        r'for\s*\(',
        r'while\s*\(',
        r'{\s*$',
        r'}\s*$',
        r';\s*$',
        r'=>',
        r'==',
        r'!=',
        r'&&',
        r'\|\|',
        r'className:',
        r'children:',
        r'useState',
        r'useCallback',
        r'useEffect',
        r'React\.',
        r'jsx',
        r'jsxs',
        r'Fragment',
        r'props\.',
        r'state\.',
        r'this\.',
        r'typeof\s+',
        r'instanceof\s+',
        r'new\s+\w+\(',
        r'void\s+0',
        r'undefined',
        r'null\s*[,;]',
        r'\[\]',
        r'JSON\.',
        r'Object\.',
        r'Array\.',
        r'String\.',
        r'Number\.',
        r'Boolean\.',
        r'Promise\.',
        r'async\s+',
        r'await\s+',
        r'import\s+',
        r'export\s+',
        r'require\(',
        r'module\.exports',
        r'prototype\.',
        r'constructor\(',
        r'extends\s+',
        r'implements\s+',
        r'interface\s+',
        r'enum\s+',
        r'class\s+\w+',
        r'\.min\.js',
        r'\.css',
        r'\.html',
        r'HTTP',
        r'GET',
        r'POST',
        r'PUT',
        r'DELETE',
        r'API',
        r'URL',
        r'URI',
        r'UUID',
        r'ID:',
        r'CSS',
        r'HTML',
        r'DOM',
        r'JSON',
        r'XML',
        r'SQL',
        r'SELECT\s+',
        r'INSERT\s+',
        r'UPDATE\s+',
        r'DELETE\s+',
        r'CREATE\s+',
        r'ALTER\s+',
        r'DROP\s+',
        r'background-color',
        r'font-size',
        r'margin',
        r'padding',
        r'display:',
        r'position:',
        r'width:',
        r'height:',
        r'color:',
        r'border:',
        r'#[0-9a-fA-F]{3,6}',  # hex colors
        r'rgb\(',
        r'rgba\(',
        r'px;',
        r'em;',
        r'rem;',
        r'vh;',
        r'vw;',
        r'%}',
        r'@media',
        r'@import',
        r'@keyframes',
        r'!important',
        r'jQuery',
        r'\$\(',
        r'angular\.',
        r'vue\.',
        r'component',
        r'directive',
        r'service',
        r'factory',
        r'controller',
        r'config\(',
        r'bootstrap',
        r'webpack',
        r'babel',
        r'eslint',
        r'lint',
        r'test\(',
        r'describe\(',
        r'it\(',
        r'expect\(',
        r'assert\(',
        r'should\.',
        r'mock\(',
        r'spy\(',
        r'stub\(',
        r'error\s*:',
        r'exception\s*:',
        r'stack\s*:',
        r'trace\s*:',
        r'debug\s*:',
        r'log\s*:',
        r'warn\s*:',
        r'info\s*:',
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    # Skip URLs and email addresses
    if re.search(r'(https?://|www\.|@\w+\.\w+|\.com|\.org|\.net)', text):
        return False
    
    # Skip if it looks like configuration or data structure
    if re.search(r'[{}\[\]"\':,]{3,}', text):
        return False
    
    # Skip if it has too many uppercase sequences (likely constants or code)
    uppercase_sequences = len(re.findall(r'[A-Z]{3,}', text))
    if uppercase_sequences > 2:
        return False
    
    # Must look like natural language - check for review-specific patterns
    review_indicators = [
        r'\b(I|we|my|our)\s+(visited|went|love|liked|enjoyed|recommend|think|felt|was|were)\b',
        r'\b(the\s+temple|senso.?ji|asakusa|tokyo|japan|beautiful|amazing|stunning|wonderful|terrible|awful|disappointing)\b',
        r'\b(worth\s+visiting|must\s+see|highly\s+recommend|don.?t\s+miss|very\s+crowded|too\s+touristy)\b',
        r'\b(experience|visit|trip|day|time|place|location|atmosphere|traditional|historic|culture)\b',
        r'\b(good|great|bad|nice|poor|excellent|awful|amazing|beautiful|ugly|crowded|peaceful|busy|quiet)\b'
    ]
    
    has_review_indicators = any(re.search(pattern, text, re.IGNORECASE) for pattern in review_indicators)
    if not has_review_indicators:
        return False
    
    return True

def scan_directory(directory_path):
    """Scan directory for any text content"""
    
    print(f"Scanning directory: {directory_path}")
    
    # Create database
    db_path = os.path.join(directory_path, 'content_findings.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS content_findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        content_type TEXT,
        content TEXT,
        file_size INTEGER,
        encoding TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # File extensions to examine
    text_extensions = {'.html', '.htm', '.js', '.json', '.txt', '.css', '.xml', '.php', '.asp'}
    
    total_files = 0
    processed_files = 0
    findings = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in text_extensions:
                total_files += 1
                
                try:
                    file_size = os.path.getsize(file_path)
                    encoding = detect_encoding(file_path)
                    
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    
                    # Find potential content
                    found_content = search_text_content(file_path, content)
                    
                    if found_content:
                        processed_files += 1
                        print(f"Found {len(found_content)} items in: {file_path}")
                        
                        for content_type, text in found_content:
                            cursor.execute('''
                            INSERT INTO content_findings 
                            (file_path, content_type, content, file_size, encoding)
                            VALUES (?, ?, ?, ?, ?)
                            ''', (file_path, content_type, text, file_size, encoding))
                            findings += 1
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    conn.commit()
    
    print(f"\nScan complete!")
    print(f"Total files examined: {total_files}")
    print(f"Files with findings: {processed_files}")
    print(f"Total content pieces found: {findings}")
    print(f"Database saved to: {db_path}")
    
    # Show sample findings
    cursor.execute('''
    SELECT content FROM content_findings 
    WHERE LENGTH(content) > 50 
    ORDER BY LENGTH(content) DESC 
    LIMIT 10
    ''')
    
    results = cursor.fetchall()
    if results:
        print(f"\nSample findings (top 10 by length):")
        for i, (content,) in enumerate(results, 1):
            print(f"\nFinding {i}:")
            print(f"'{content[:200]}{'...' if len(content) > 200 else ''}'")
    
    conn.close()

if __name__ == "__main__":
    directory = "/Users/andreyvlasenko/tst/Request/Senso"
    scan_directory(directory)