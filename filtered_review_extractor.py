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


# Heuristic filter and scoring
def score_block(text: str) -> float:
    """Return a heuristic score indicating likelihood of being a human review/comment.
    Score in [0,1]."""
    t = text.strip()
    if not t:
        return 0.0
    length = len(t)
    # length score: prefer 30-600 chars
    if length < 20:
        return 0.0
    length_score = min(1.0, max(0.0, (length - 20) / 580.0))

    # indicator words
    indicators = ['visit', 'visited', 'went', 'trip', 'recommend', 'experience', 'beautiful', 'amazing',
                  'crowded', 'temple', 'shrine', 'senso', 'asakusa', 'tokyo', 'japan', 'love', 'like', 'hate',
                  'wonderful', 'terrible', 'recommend', 'nice', 'great', 'bad', 'good']
    count_ind = sum(1 for w in indicators if re.search(r'\b' + re.escape(w) + r'\b', t, re.I))
    ind_score = min(1.0, count_ind / 4.0)

    # penalize code/boilerplate tokens
    boilerplate_tokens = ['function(', 'var ', 'const ', 'let ', 'document.', 'window.', 'console.',
                          '<script', '<style', 'googletag', 'gpt-', 'analytics', 'gtm', 'class=', 'id=', 'data-']
    if any(tok in t.lower() for tok in boilerplate_tokens):
        return 0.0

    # punctuation / sentence structure
    sentences = re.split(r'[.!?]+', t)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    sentence_score = min(1.0, len([s for s in sentences if len(s) > 10]) / max(1, len(sentences)))

    # pronouns/personal tone
    pronouns = len(re.findall(r'\b(I|we|my|our|me|us)\b', t, re.I))
    pronoun_score = 1.0 if pronouns >= 1 else 0.0

    # uppercase sequences indicate code/consts
    uppercase_seq = len(re.findall(r'[A-Z]{3,}', t))
    if uppercase_seq > 1:
        return 0.0

    # combine
    score = 0.45 * length_score + 0.25 * ind_score + 0.15 * sentence_score + 0.15 * pronoun_score
    return min(1.0, max(0.0, score))


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

    scored = []
    for b in blocks:
        sc = score_block(b)
        # filter very short/boilerplate
        if sc > 0 or len(b) >= 60:
            scored.append((b, sc))

    # Sort by score desc then length desc
    scored_sorted = sorted(scored, key=lambda x: (x[1], len(x[0])), reverse=True)

    # Save to DB
    directory = os.path.dirname(file_path)
    db_path = save_blocks_to_db(directory, file_path, scored_sorted)
    print('Saved to DB:', db_path)

    # Print top 20 candidates
    print('\nTop candidate comment blocks:')
    for i, (text, score) in enumerate(scored_sorted[:20], 1):
        flag = 'CANDIDATE' if score >= 0.65 else 'PASS' if score > 0 else 'LOW'
        print(f"\n[{i}] score={score:.2f} {flag}\n{text}\n")


if __name__ == '__main__':
    # default path (URL-encoded) from previous extractor
    encoded = "/Users/andreyvlasenko/tst/Request/Senso-ji%20Temple,%20Asakusa.html"
    run_for_file(encoded)
