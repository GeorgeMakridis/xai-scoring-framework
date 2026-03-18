#!/usr/bin/env python3
"""
Test script for XAI Scoring Framework
This script tests the core functionality without requiring Docker.
"""

import sys
import os
import requests
import time
import subprocess
import signal
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import plotly
        import matplotlib
        import flask
        import fastapi
        import uvicorn
        from app import (
            load_excel_data,
            load_qualitative_ratings,
            build_repository,
            estimate_xai_score_for_new_dataset
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_web_app():
    """Test the Flask web application."""
    print("\nTesting Flask web application...")
    
    # Start the Flask app in background
    try:
        process = subprocess.Popen(
            ["python3", "web_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for app to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://localhost:8501/health", timeout=10)
        if response.status_code == 200:
            print("✅ Flask web app is running and healthy")
            result = True
        else:
            print(f"❌ Flask web app health check failed: {response.status_code}")
            result = False
            
        # Clean up
        process.terminate()
        process.wait()
        
        return result
        
    except Exception as e:
        print(f"❌ Flask web app test failed: {e}")
        return False

def test_api():
    """Test the FastAPI application."""
    print("\nTesting FastAPI application...")
    
    # Start the API in background
    try:
        process = subprocess.Popen(
            ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for API to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ FastAPI is running and healthy")
            result = True
        else:
            print(f"❌ FastAPI health check failed: {response.status_code}")
            result = False
            
        # Test methods endpoint
        response = requests.get("http://localhost:8000/api/methods", timeout=10)
        if response.status_code == 200:
            print("✅ API methods endpoint working")
        else:
            print(f"❌ API methods endpoint failed: {response.status_code}")
            
        # Clean up
        process.terminate()
        process.wait()
        
        return result
        
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
        return False

def test_data_files():
    """Test that data files exist and can be read."""
    print("\nTesting data files...")
    
    required_files = [
        "Fame XAI scoring Framework_v2-2.xlsx",
        "data/tabular/xai_qualitative_ratings.csv"
    ]
    
    optional_files = [
        "JSON_1.docx",
        "JSON_2.docx", 
        "JSON_3.docx"
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found (required)")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file} found (optional)")
        else:
            print(f"⚠️  {file} not found (optional)")
    
    return all_good

def test_docker_files():
    """Test that Docker files exist."""
    print("\nTesting Docker configuration...")
    
    docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt"
    ]
    
    all_good = True
    
    for file in docker_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("🧠 XAI Scoring Framework - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Docker Files Test", test_docker_files),
        ("Web App Test", test_web_app),
        ("API Test", test_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The XAI Scoring Framework is ready to use.")
        print("\nTo start the applications:")
        print("1. Web UI: python3 web_app.py")
        print("2. API: python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        print("3. Docker: docker-compose up -d (when Docker is available)")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 