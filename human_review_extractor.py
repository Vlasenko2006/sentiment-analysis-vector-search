#!/usr/bin/env python3
"""
Human Review Extractor - Extracts ONLY human visitor comments, ignoring all machine code
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
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def is_human_review(text):
    """Extremely strict filtering for human reviews only"""
    
    # Must be reasonable length
    if len(text) < 20 or len(text) > 1000:
        return False
    
    # Must end with proper punctuation
    if not re.search(r'[.!?]$', text.strip()):
        return False
    
    # Must start with capital letter or quote
    if not re.match(r'^[A-Z"\']', text.strip()):
        return False
    
    # Absolutely NO code patterns - very comprehensive list
    code_blacklist = [
        # JavaScript/Programming
        r'function\s*\(', r'var\s+', r'let\s+', r'const\s+', r'=>', r'==', r'!=', r'&&', r'\|\|',
        r'document\.', r'window\.', r'console\.', r'getElementById', r'addEventListener',
        r'return\s+', r'if\s*\(', r'for\s*\(', r'while\s*\(', r'switch\s*\(',
        r'{\s*$', r'}\s*$', r';\s*$', r':\s*{', r'\[\s*\]', r'void\s+0',
        r'undefined', r'null\s*[,;]', r'typeof\s+', r'instanceof\s+', r'new\s+\w+\(',
        
        # React/JSX
        r'jsx', r'jsxs', r'Fragment', r'useState', r'useEffect', r'useCallback',
        r'React\.', r'props\.', r'state\.', r'this\.', r'className:', r'children:',
        
        # CSS/Styling
        r'background-color', r'font-size', r'margin:', r'padding:', r'display:',
        r'position:', r'width:', r'height:', r'color:', r'border:', r'!important',
        r'#[0-9a-fA-F]{3,6}', r'rgb\(', r'rgba\(', r'px;', r'em;', r'rem;',
        r'@media', r'@import', r'@keyframes', r'\.css', r'\.min\.js',
        
        # HTML/XML
        r'<[^>]+>', r'&[a-zA-Z]+;', r'xmlns:', r'DOCTYPE', r'</\w+>',
        
        # URLs/Technical
        r'https?://', r'www\.', r'\.com', r'\.org', r'\.net', r'\.js', r'\.css',
        r'HTTP', r'GET', r'POST', r'PUT', r'DELETE', r'API', r'URL', r'URI',
        
        # Data structures
        r'JSON', r'XML', r'SQL', r'SELECT\s+', r'INSERT\s+', r'UPDATE\s+',
        r'[{}\[\]"\':,]{4,}', r'Object\.', r'Array\.', r'String\.', r'Number\.',
        
        # Libraries/Frameworks
        r'jQuery', r'\$\(', r'angular\.', r'vue\.', r'bootstrap', r'webpack',
        r'babel', r'eslint', r'component', r'directive', r'service', r'factory',
        
        # Error/Debug
        r'error\s*:', r'exception\s*:', r'stack\s*:', r'debug\s*:', r'log\s*:',
        r'console\.log', r'console\.error', r'try\s*{', r'catch\s*\(',
        
        # Configuration
        r'config\s*:', r'settings\s*:', r'options\s*:', r'params\s*:',
        r'module\.exports', r'require\(', r'import\s+', r'export\s+',
        
        # Constants/IDs
        r'[A-Z_]{5,}', r'ID:', r'UUID', r'[0-9a-f]{8,}', r'0x[0-9a-f]+',
        
        # File extensions and tech terms
        r'\.html', r'\.php', r'\.asp', r'DOM', r'CSS', r'HTML', r'JS',
        
        # Common code variable patterns
        r'\w+\s*=\s*\w+', r'\w+\.\w+\.\w+', r'\w+\[\w+\]', r'\w+\(\w+\)',
    ]
    
    for pattern in code_blacklist:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    # Must have natural language indicators
    natural_language_required = [
        r'\b(the|and|or|but|is|was|are|were|have|has|had|this|that|with|for|from|they|we|you|it|in|on|at|to|of|a|an)\b',
        r'\b(I|we|my|our|me|us)\b',  # Personal pronouns
        r'\b(very|really|quite|pretty|extremely|so|too|much|many|some|all|every|each)\b',  # Intensifiers
    ]
    
    matches = 0
    for pattern in natural_language_required:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
    
    if matches < 2:
        return False
    
    # Must have review/experience indicators
    review_indicators = [
        r'\b(visit|visited|went|went to|been to|saw|see|experience|trip|travel|tour)\b',
        r'\b(temple|shrine|senso.?ji|asakusa|tokyo|japan|japanese)\b',
        r'\b(beautiful|amazing|wonderful|stunning|gorgeous|lovely|nice|great|good|excellent|fantastic|awesome)\b',
        r'\b(terrible|awful|bad|horrible|disappointing|poor|worst|hate|hated|boring)\b',
        r'\b(recommend|suggest|worth|must see|should visit|don.?t miss|avoid)\b',
        r'\b(crowded|busy|peaceful|quiet|calm|noisy|loud|traditional|historic|ancient|old)\b',
        r'\b(place|location|spot|area|site|destination|attraction)\b',
    ]
    
    has_review_indicators = any(re.search(pattern, text, re.IGNORECASE) for pattern in review_indicators)
    if not has_review_indicators:
        return False
    
    # Must look like complete sentences
    sentences = re.split(r'[.!?]+', text)
    complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(complete_sentences) < 1:
        return False
    
    # Check if it sounds like human writing (not technical documentation)
    human_writing_patterns = [
        r'\b(I think|I believe|I feel|I found|I love|I hate|I like|I dislike)\b',
        r'\b(we think|we believe|we found|we love|we hate|we like|we enjoyed|we visited)\b',
        r'\b(it was|it is|this was|this is|that was|that is)\b',
        r'\b(really|very|quite|pretty|extremely|absolutely|definitely|certainly)\b',
    ]
    
    has_human_patterns = any(re.search(pattern, text, re.IGNORECASE) for pattern in human_writing_patterns)
    if not has_human_patterns:
        return False
    
    return True

def extract_potential_reviews(content):
    """Extract potential review text using various patterns"""
    
    potential_reviews = []
    
    # Pattern 1: Quoted strings that might be reviews
    quoted_pattern = r'["\']([^"\']{30,800})["\']'
    for match in re.finditer(quoted_pattern, content, re.DOTALL):
        text = match.group(1).strip()
        if is_human_review(text):
            potential_reviews.append(text)
    
    # Pattern 2: Text between HTML-like tags or after labels
    text_patterns = [
        r'text["\s]*[:=]["\s]*([^"\']{30,800})',
        r'review["\s]*[:=]["\s]*([^"\']{30,800})',
        r'comment["\s]*[:=]["\s]*([^"\']{30,800})',
        r'title["\s]*[:=]["\s]*([^"\']{30,800})',
        r'description["\s]*[:=]["\s]*([^"\']{30,800})',
    ]
    
    for pattern in text_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            text = match.group(1).strip()
            if is_human_review(text):
                potential_reviews.append(text)
    
    # Pattern 3: Natural sentences in text (not in quotes)
    sentence_pattern = r'(?<=[.!?]\s)([A-Z][^.!?]{30,300}[.!?])'
    for match in re.finditer(sentence_pattern, content):
        text = match.group(1).strip()
        if is_human_review(text):
            potential_reviews.append(text)
    
    # Remove duplicates and very similar texts
    unique_reviews = []
    for review in potential_reviews:
        is_duplicate = False
        for existing in unique_reviews:
            if len(review) > 20 and len(existing) > 20:
                # Check if 80% similar
                shorter = min(len(review), len(existing))
                longer = max(len(review), len(existing))
                if shorter / longer > 0.8 and review[:shorter] == existing[:shorter]:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_reviews.append(review)
    
    return unique_reviews

def scan_for_human_reviews(directory_path):
    """Scan directory specifically for human reviews"""
    
    print(f"Scanning for human reviews in: {directory_path}")
    
    # Create database for human reviews only
    db_path = os.path.join(directory_path, 'human_reviews.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS human_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        review_text TEXT,
        review_length INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Clear previous data
    cursor.execute('DELETE FROM human_reviews')
    
    text_extensions = {'.html', '.htm', '.js', '.json', '.txt'}
    
    total_files = 0
    files_with_reviews = 0
    total_reviews = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in text_extensions:
                total_files += 1
                
                try:
                    encoding = detect_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    
                    reviews = extract_potential_reviews(content)
                    
                    if reviews:
                        files_with_reviews += 1
                        print(f"Found {len(reviews)} potential review(s) in: {file_path}")
                        
                        for review in reviews:
                            cursor.execute('''
                            INSERT INTO human_reviews (file_path, review_text, review_length)
                            VALUES (?, ?, ?)
                            ''', (file_path, review, len(review)))
                            total_reviews += 1
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    conn.commit()
    
    print(f"\nScan complete!")
    print(f"Total files examined: {total_files}")
    print(f"Files with human reviews: {files_with_reviews}")
    print(f"Total human reviews found: {total_reviews}")
    print(f"Database saved to: {db_path}")
    
    # Show the found reviews
    cursor.execute('''
    SELECT review_text FROM human_reviews 
    ORDER BY review_length DESC 
    LIMIT 20
    ''')
    
    results = cursor.fetchall()
    if results:
        print(f"\nFound Human Reviews:")
        for i, (review_text,) in enumerate(results, 1):
            print(f"\nComment {i}: '{review_text}'")
    else:
        print("\nNo human reviews found in the downloaded files.")
        print("The files appear to contain only website infrastructure code.")
    
    conn.close()

if __name__ == "__main__":
    directory = "/Users/andreyvlasenko/tst/Request/Senso"
    scan_for_human_reviews(directory)