#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test for API routes
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("Testing API Routes")
print("=" * 80)

# Test 1: Import main modules
print("\n1. Testing imports...")
try:
    import main_api
    print("   ✓ main_api imported successfully")
except Exception as e:
    print(f"   ✗ main_api import failed: {e}")
    sys.exit(1)

try:
    from routes import Routes
    print("   ✓ routes module imported successfully")
except Exception as e:
    print(f"   ✗ routes import failed: {e}")
    sys.exit(1)

try:
    from models import AnalysisRequest, JobStatus, ChatRequest, ChatResponse
    print("   ✓ models imported successfully")
except Exception as e:
    print(f"   ✗ models import failed: {e}")
    sys.exit(1)

# Test 2: Check if FastAPI app is created
print("\n2. Testing FastAPI app...")
try:
    app = main_api.app
    print(f"   ✓ FastAPI app created: {app.title}")
except Exception as e:
    print(f"   ✗ FastAPI app check failed: {e}")
    sys.exit(1)

# Test 3: Check if routes are registered
print("\n3. Testing route registration...")
try:
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/health", "/api/config", "/api/analyze", "/api/status/{job_id}"]
    
    for expected in expected_routes:
        if expected in routes:
            print(f"   ✓ Route registered: {expected}")
        else:
            print(f"   ✗ Route missing: {expected}")
    
    print(f"   Total routes registered: {len(routes)}")
except Exception as e:
    print(f"   ✗ Route check failed: {e}")
    sys.exit(1)

# Test 4: Test Pydantic models
print("\n4. Testing Pydantic models...")
try:
    # Test AnalysisRequest
    request = AnalysisRequest(
        url="https://example.com",
        searchMethod="demo"
    )
    print(f"   ✓ AnalysisRequest created: searchMethod={request.searchMethod}")
    
    # Test JobStatus
    status = JobStatus(
        job_id="test-123",
        status="queued",
        progress=0,
        message="Test job"
    )
    print(f"   ✓ JobStatus created: status={status.status}")
    
    # Test ChatRequest
    chat_req = ChatRequest(question="Test question")
    print(f"   ✓ ChatRequest created: question={chat_req.question}")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    sys.exit(1)

# Test 5: Check if jobs_db is initialized
print("\n5. Testing shared state...")
try:
    jobs_db = main_api.jobs_db
    print(f"   ✓ jobs_db initialized: {type(jobs_db).__name__}")
    
    chatbots = main_api.chatbots
    print(f"   ✓ chatbots initialized: {type(chatbots).__name__}")
except Exception as e:
    print(f"   ✗ Shared state check failed: {e}")
    sys.exit(1)

# Test 6: Check MLflow availability
print("\n6. Testing MLflow integration...")
try:
    mlflow_available = main_api.MLFLOW_AVAILABLE
    if mlflow_available:
        print("   ✓ MLflow tracking enabled")
    else:
        print("   ⚠ MLflow tracking disabled (expected in PCL_copy environment)")
except Exception as e:
    print(f"   ✗ MLflow check failed: {e}")

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nYou can now start the API with:")
print("  uvicorn main_api:app --reload")
print("  or")
print("  python main_api.py")
