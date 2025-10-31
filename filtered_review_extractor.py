#!/usr/bin/env python3
"""
Filtered Review Extractor
- Parses a local HTML file, extracts visible text broken into blocks (paragraphs, list items, headings)
- Filters out boilerplate/technical text
- Scores blocks by heuristic and saves them into SQLite DB as separate comment blocks
- Prints top candidate blocks

Usage: run from the Request folder. The script defaults to
/Users/andreyvlasenko/tst/Request/Senso-ji%20Temple,%20Asakusa.html
"""

import os
import re
import sqlite3
from urllib.parse import unquote
from pathlib import Path
from bs4 import BeautifulSoup
import chardet
import numpy as np

# Optional ML dependencies
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ Transformers not installed. Using linguistic analysis only. Install with: pip install transformers torch")


def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw = f.read(10000)
            res = chardet.detect(raw)
            return res.get('encoding') or 'utf-8'
    except Exception:
        return 'utf-8'


def clean_text(text: str) -> str:
    if not text:
        return ''
    # Normalize whitespace
    t = re.sub(r'\r', '\n', text)
    t = re.sub(r'\n[ \t]+', '\n', t)
    t = re.sub(r'[ \t]+', ' ', t)
    # Remove repeating punctuation
    t = re.sub(r'([!?.]){2,}', r"\1", t)
    # Trim
    t = t.strip()
    # Collapse multiple blank lines
    t = re.sub(r"\n{2,}", '\n\n', t)
    return t


# Neural network-based text classification for user comments
def load_text_classifier():
    """Load a pre-trained text classifier for identifying user-generated content"""
    if not HAS_TRANSFORMERS:
        return None, None
    
    try:
        # Try to load a text classification model
        # Option 1: Use a general text quality classifier
        try:
            classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",  # Can distinguish different text types
                return_all_scores=True
            )
            return classifier, "toxic-comment"
        except:
            pass
        
        # Option 2: Use a general sentiment model (helps identify personal vs technical text)
        try:
            classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            return classifier, "sentiment"
        except:
            pass
        
        # Option 3: Use DistilBERT for text classification
        try:
            classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            return classifier, "distilbert"
        except:
            pass
        
        return None, None
        
    except Exception as e:
        print(f"⚠️ Failed to load neural network models: {e}")
        return None, None


def calculate_text_coherence(text: str) -> float:
    """Calculate text coherence using linguistic features"""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    # Basic linguistic coherence indicators
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 1:
        return 0.0
    
    # 1. Average sentence length (coherent text has reasonable sentence length)
    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    length_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.5
    
    # 2. Vocabulary diversity (user comments have varied vocabulary)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    diversity_score = len(unique_words) / max(1, len(words)) if words else 0
    
    # 3. Punctuation patterns (proper use indicates human writing)
    has_proper_punctuation = bool(re.search(r'[.!?]', text))
    punct_score = 1.0 if has_proper_punctuation else 0.3
    
    # 4. Common function words (indicates natural language)
    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
    function_word_count = sum(1 for word in words if word in function_words)
    function_score = min(1.0, function_word_count / max(1, len(words) * 0.1))
    
    # 5. Avoid technical patterns
    technical_patterns = [
        r'\b[A-Z]{2,}_[A-Z]{2,}\b',  # CONSTANT_NAMES
        r'\b\w+\(\)',                # function()
        r'[{}|\\<>^~`@#%&*+=\[\]_;]', # Technical symbols
        r'\b(var|let|const|function|class|import|export)\b',  # Programming keywords
    ]
    
    technical_matches = sum(len(re.findall(pattern, text, re.I)) for pattern in technical_patterns)
    technical_penalty = max(0, 1.0 - (technical_matches * 0.2))
    
    # Combine scores
    coherence = (length_score * 0.2 + 
                diversity_score * 0.3 + 
                punct_score * 0.2 + 
                function_score * 0.2 + 
                technical_penalty * 0.1)
    
    return min(1.0, coherence)


def classify_user_comment_ml(text: str, classifier=None, model_type=None) -> float:
    """Use ML model to classify if text is a user comment"""
    if not text or len(text.strip()) < 20:
        return 0.0
    
    # Get coherence score first
    coherence = calculate_text_coherence(text)
    if coherence < 0.3:  # Very low coherence, likely not user content
        return 0.0
    
    if classifier is None:
        # Fallback to coherence-based scoring
        return coherence
    
    try:
        # Truncate text for model input
        text_sample = text[:512]  # Most models have token limits
        
        if model_type == "sentiment":
            # Use sentiment model to detect personal/emotional content
            results = classifier(text_sample)
            # Personal comments tend to have stronger sentiment
            max_score = max(result['score'] for result in results)
            sentiment_indicator = max_score * 0.7  # Weight sentiment as indicator
            
        elif model_type == "toxic-comment":
            # Use toxic comment model (can distinguish comment vs non-comment)
            results = classifier(text_sample)
            # Look for "comment-like" patterns
            comment_score = 0.5  # Default neutral
            for result in results:
                if 'toxic' in result['label'].lower():
                    # Non-toxic content with comment structure
                    comment_score = 1.0 - result['score']
                    break
            sentiment_indicator = comment_score
            
        else:  # distilbert or other
            results = classifier(text_sample)
            # Handle different model output formats
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and 'score' in results[0]:
                    # Standard format: [{'label': 'POSITIVE', 'score': 0.9}]
                    max_score = max(result['score'] for result in results)
                else:
                    # Handle unexpected format
                    max_score = 0.5  # Default neutral
            else:
                max_score = 0.5  # Default neutral
            sentiment_indicator = max_score * 0.6
        
        # Combine ML score with coherence
        ml_score = (coherence * 0.6 + sentiment_indicator * 0.4)
        return min(1.0, ml_score)
        
    except Exception as e:
        print(f"⚠️ ML classification failed: {e}")
        # Fallback to coherence only
        return coherence


def is_user_comment_ml(text: str, classifier=None, model_type=None) -> bool:
    """ML-based check if text is a user comment"""
    score = classify_user_comment_ml(text, classifier, model_type)
    return score > 0.5  # Threshold for classification


def score_user_comment_ml(text: str, classifier=None, model_type=None) -> float:
    """ML-enhanced scoring for user comments"""
    if not text or len(text.strip()) < 15:
        return 0.0
    
    # Get ML-based classification score
    ml_score = classify_user_comment_ml(text, classifier, model_type)
    
    # Additional heuristics to boost obvious user content
    length = len(text)
    
    # Length preference (user comments are typically 30-600 chars)
    if length < 25:
        length_factor = 0.0
    elif length > 800:
        length_factor = 0.7  # Long text might be article content
    else:
        length_factor = min(1.0, (length - 25) / 575.0)
    
    # Check for first person indicators (but don't rely solely on keywords)
    first_person = len(re.findall(r'\b(I|we|my|our|me|us)\b', text, re.I))
    personal_factor = min(1.0, first_person * 0.2) if first_person > 0 else 0.0
    
    # Check for review-like structure (starts with name or has rating patterns)
    structure_bonus = 0.0
    if re.match(r'^[A-Z][a-zA-Z\s]+\b', text.strip()):  # Starts with capitalized words (name)
        structure_bonus = 0.15
    if re.search(r'\b[1-5]/5|\b[1-5] out of 5|\b[1-5] stars?|\b★+\b', text):  # Rating patterns
        structure_bonus = 0.2
    
    # Combine scores
    final_score = (ml_score * 0.7 + 
                  length_factor * 0.15 + 
                  personal_factor * 0.1 + 
                  structure_bonus)
    
    return min(1.0, max(0.0, final_score))


def extract_blocks_from_html(file_path: str):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Remove scripts/styles and typical non-visible nodes
    for el in soup(['script', 'style', 'meta', 'link', 'noscript', 'svg', 'iframe']):
        el.decompose()

    # Also remove common header/footer selectors by heuristics (IDs/classes containing nav, header, footer, cookie)
    for el in soup.find_all(True, attrs={'id': re.compile(r'nav|header|footer|cookie|skip|menu', re.I)}):
        try:
            el.decompose()
        except Exception:
            pass
    for el in soup.find_all(True, attrs={'class': re.compile(r'nav|header|footer|cookie|skip|menu|breadcrumb|pagination', re.I)}):
        try:
            el.decompose()
        except Exception:
            pass

    blocks = []

    # Preferred tags that usually contain human text
    tags = ['p', 'li', 'blockquote', 'dd', 'dt', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    for tag in tags:
        for el in soup.find_all(tag):
            txt = el.get_text(separator=' ', strip=True)
            if txt:
                blocks.append(clean_text(txt))

    # Fallback: large divs that look like paragraphs
    if not blocks:
        for div in soup.find_all('div'):
            txt = div.get_text(separator=' ', strip=True)
            if txt and len(txt) > 40 and '\n' not in txt:
                blocks.append(clean_text(txt))

    # Deduplicate preserving order
    seen = set()
    uniq_blocks = []
    for b in blocks:
        key = b[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq_blocks.append(b)

    return uniq_blocks


def save_blocks_to_db(directory: str, file_path: str, blocks_with_scores):
    db_path = os.path.join(directory, 'filtered_reviews.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS comment_blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        block_text TEXT,
        length INTEGER,
        score REAL,
        is_candidate INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Optionally clear previous results from same file
    cur.execute('DELETE FROM comment_blocks WHERE file_path = ?', (file_path,))

    for text, score in blocks_with_scores:
        cur.execute('INSERT INTO comment_blocks (file_path, block_text, length, score, is_candidate) VALUES (?,?,?,?,?)',
                    (file_path, text, len(text), float(score), 1 if score >= 0.65 else 0))
    conn.commit()
    conn.close()
    return db_path


def run_for_file(encoded_path: str):
    file_path = unquote(encoded_path)
    if not os.path.exists(file_path):
        print('File not found:', file_path)
        return

    print('Processing:', file_path)
    blocks = extract_blocks_from_html(file_path)
    print(f'Extracted {len(blocks)} blocks')

    # Load ML classifier
    classifier, model_type = load_text_classifier()
    if classifier:
        print(f"✅ Loaded neural network model: {model_type}")
    else:
        print("⚠️ Using linguistic coherence analysis (neural networks unavailable)")

    scored = []
    for b in blocks:
        sc = score_user_comment_ml(b, classifier, model_type)
        # filter very short/boilerplate - only keep user comments or long text
        if sc > 0 or len(b) >= 60:
            scored.append((b, sc))

    # Sort by score desc then length desc
    scored_sorted = sorted(scored, key=lambda x: (x[1], len(x[0])), reverse=True)

    # Save to DB
    directory = os.path.dirname(file_path)
    db_path = save_blocks_to_db(directory, file_path, scored_sorted)
    print('Saved to DB:', db_path)

    # Print top 20 user comment candidates
    print('\nTop user comment candidates:')
    candidates_found = 0
    for i, (text, score) in enumerate(scored_sorted[:30], 1):  # Check more entries
        if score >= 0.3:  # Lower threshold to see more potential comments
            flag = 'USER COMMENT' if score >= 0.65 else 'POTENTIAL' if score >= 0.4 else 'LOW CONFIDENCE'
            print(f"\n[{i}] score={score:.2f} {flag}")
            print(f"Length: {len(text)} chars | Sentences: {len([s for s in text.split('.') if s.strip()])}")
            print(f"Text: {text}\n")
            candidates_found += 1
            if candidates_found >= 20:
                break
    
    if candidates_found == 0:
        print("No user comments found. Showing top scored blocks:")
        for i, (text, score) in enumerate(scored_sorted[:10], 1):
            print(f"\n[{i}] score={score:.2f}")
            print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}\n")


if __name__ == '__main__':
    # default path (URL-encoded) from previous extractor
    encoded = "/Users/andreyvlasenko/tst/Request/Senso-ji%20Temple,%20Asakusa.html"
    run_for_file(encoded)
