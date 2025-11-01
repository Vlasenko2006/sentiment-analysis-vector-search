#!/usr/bin/env python3
"""
Quick test script to check if basic packages are working
"""

print("🔄 Testing basic imports...")

try:
    import requests
    print("✅ requests imported successfully")
except ImportError as e:
    print(f"❌ requests import failed: {e}")

try:
    from bs4 import BeautifulSoup
    print("✅ BeautifulSoup imported successfully")
except ImportError as e:
    print(f"❌ BeautifulSoup import failed: {e}")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

print("\n🔄 Testing heavy ML imports...")

try:
    from transformers import pipeline
    print("✅ transformers imported successfully")
except ImportError as e:
    print(f"❌ transformers import failed: {e}")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("✅ sklearn imported successfully")
except ImportError as e:
    print(f"❌ sklearn import failed: {e}")

print("\n✅ All import tests completed!")