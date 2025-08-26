#!/usr/bin/env python3
"""
Test script for Simple LLM Service API
Tests all major endpoints to ensure they work correctly.
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\n📋 Testing Models Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/v1/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Models list failed: {e}")
        return False

def test_chat_completion():
    """Test chat completions endpoint"""
    print("\n💬 Testing Chat Completions...")
    try:
        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you tell me a short joke?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            # Print the assistant's response
            assistant_message = result['choices'][0]['message']['content']
            print(f"\n🤖 Assistant Response: {assistant_message}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False

def test_text_completion():
    """Test text completions endpoint"""
    print("\n📝 Testing Text Completions...")
    try:
        payload = {
            "model": "llama-3.2-3b-instruct",
            "prompt": "The future of AI is",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            # Print the completion
            completion_text = result['choices'][0]['text']
            print(f"\n📝 Completion: {completion_text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Text completion failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Simple LLM Service API Tests\n")
    
    tests = [
        ("Health Check", test_health),
        ("Models List", test_models),
        ("Chat Completion", test_chat_completion),
        ("Text Completion", test_text_completion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print("=" * 50)
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Simple LLM Service is working correctly.")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Please check the service.")

if __name__ == "__main__":
    main()
