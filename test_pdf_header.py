#!/usr/bin/env python3
"""
Quick test to verify PDF header company name loading
"""

import subprocess
import json

print("=" * 80)
print("PDF HEADER COMPANY NAME TEST")
print("=" * 80)

# Test 1: What does the API return?
print("\n1. API /api/config endpoint:")
print("-" * 80)
result = subprocess.run(
    ["curl", "-s", "http://localhost:8001/api/config"],
    capture_output=True,
    text=True
)
api_response = json.loads(result.stdout)
print(f"Company name from API: {api_response.get('company_name')}")

# Test 2: What's in the container's config file?
print("\n2. Container config file:")
print("-" * 80)
result = subprocess.run(
    ["docker", "exec", "sentiment-python-v2", "cat", "config/config_names.yaml"],
    capture_output=True,
    text=True
)
print(result.stdout)

# Test 3: What does pdf_header.load_company_name() return?
print("\n3. Testing pdf_header.load_company_name():")
print("-" * 80)
result = subprocess.run(
    ["docker", "exec", "sentiment-python-v2", "python3", "-c",
     "from pdf_generation.pdf_header import load_company_name; print(f'Company name: {load_company_name()}')"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Test 4: What does the main config loader return?
print("\n4. Testing main config loader:")
print("-" * 80)
result = subprocess.run(
    ["docker", "exec", "sentiment-python-v2", "python3", "-c",
     "from config.config import load_all_configs; config = load_all_configs(); print(f\"Company name: {config.get('names_config', {}).get('company_name', 'NOT FOUND')}\")"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nIf all the above show 'Awesome Company' but your PDF shows 'Red6-ES',")
print("then the issue is that you have an OLD PDF file cached.")
print("\nSolution: Delete the old PDF and generate a new one:")
print("  1. Find your PDF: my_volume/sentiment_analysis/*/visualizations/sentiment_analysis_report.pdf")
print("  2. Delete it")
print("  3. Run a new analysis to generate a fresh PDF")
print("\nOr run: docker exec sentiment-python-v2 find my_volume -name '*.pdf' -type f")
print("=" * 80)
