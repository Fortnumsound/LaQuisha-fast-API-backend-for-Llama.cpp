#!/usr/bin/env python3
"""
Test script for LaQuisha FastAPI backend.

This script tests the API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🩺 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Root endpoint: {data['message']}")
        print(f"   Status: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_chat_endpoint():
    """Test the chat completions endpoint."""
    print("\n💬 Testing chat endpoint...")
    try:
        payload = {
            "model": "laquisha-7b",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello LaQuisha! How are you doing today?"
                }
            ],
            "sass_level": 8,
            "max_tokens": 100
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Chat completion successful!")
        print(f"   Response ID: {data['id']}")
        print(f"   Model: {data['model']}")
        if data.get('laquisha_flavor'):
            print(f"   LaQuisha flavor: {data['laquisha_flavor']}")
        
        if data['choices'] and len(data['choices']) > 0:
            message = data['choices'][0]['message']['content']
            print(f"   Response: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        return True
    except Exception as e:
        print(f"❌ Chat endpoint failed: {e}")
        return False

def wait_for_server(max_retries=30, delay=1):
    """Wait for the server to be ready."""
    print("⏳ Waiting for server to start...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            if i < max_retries - 1:
                time.sleep(delay)
            pass
    
    print("❌ Server failed to start within timeout")
    return False

def main():
    """Run all tests."""
    print("🌟 LaQuisha API Test Suite 🌟\n")
    
    if not wait_for_server():
        sys.exit(1)
    
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_chat_endpoint,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LaQuisha is working perfectly!")
        sys.exit(0)
    else:
        print("💔 Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
