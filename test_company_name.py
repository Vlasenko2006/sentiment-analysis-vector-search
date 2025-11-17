#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to debug company name loading in PDF header
"""

import os
import sys

print("=" * 80)
print("COMPANY NAME DEBUG TEST")
print("=" * 80)

# Test 1: Check what pdf_header.py loads
print("\n1. Testing pdf_header.load_company_name():")
print("-" * 80)
from pdf_generation.pdf_header import load_company_name
company_from_header = load_company_name()
print(f"   Result: '{company_from_header}'")

# Test 2: Check config files directly
print("\n2. Checking config files directly:")
print("-" * 80)
import yaml

config_paths = [
    'config/config_names.yaml',
    'config_names.yaml',
    'config/config_names-example.yaml',
    'config_names-example.yaml'
]

for path in config_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                company = data.get('company_name', 'NOT FOUND')
                print(f"      company_name: '{company}'")
        except Exception as e:
            print(f"      Error reading: {e}")
    else:
        print(f"   ✗ Not found: {path}")

# Test 3: Check main_api.py config loading
print("\n3. Testing main_api.py config loading:")
print("-" * 80)
try:
    from config.config import load_all_configs
    config_data = load_all_configs()
    names_config = config_data.get('names_config', {})
    print(f"   names_config keys: {list(names_config.keys())}")
    print(f"   company_name from config: '{names_config.get('company_name', 'NOT FOUND')}'")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Simulate what happens in pdf_generator
print("\n4. Simulating pdf_generator.py flow:")
print("-" * 80)
try:
    from pdf_generation.pdf_data_loader import load_existing_data
    from pdf_generation.pdf_header import draw_header_stripe
    
    # Try to load data (will fail but that's ok, we want to see the header logic)
    print("   Testing if we can access header drawing function...")
    print(f"   draw_header_stripe function exists: {callable(draw_header_stripe)}")
    
    # Test with explicit company name
    print("\n   Testing with explicit company_name='TEST COMPANY':")
    print("   (This would be passed from main_api.py → routes.py → pipeline_helpers.py)")
    
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Check environment and working directory
print("\n5. Environment check:")
print("-" * 80)
print(f"   Current working directory: {os.getcwd()}")
print(f"   Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"   Config directory exists: {os.path.exists('config')}")
if os.path.exists('config'):
    print(f"   Config directory contents: {os.listdir('config')}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
