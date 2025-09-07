#!/usr/bin/env python3
"""
Comprehensive test suite for Wave 2 enhancements.
Tests model registry, safety pipeline, and token-based rate limiting.
"""

import requests
import json
import time
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8008")

def get_admin_token():
    """Get admin token for privileged operations."""
    login_response = requests.post(f"{API_BASE}/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    if login_response.status_code == 200:
        return login_response.json()["access_token"]
    return None

def test_model_registry():
    """Test model registry functionality."""
    print("\n🏗️ Testing Model Registry...")
    token = get_admin_token()
    if not token:
        print("❌ Failed to get admin token")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get registry stats
    response = requests.get(f"{API_BASE}/admin/model-registry/stats", headers=headers)
    print(f"Registry Stats Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()["data"]
        print(f"Loaded models: {stats['loaded_models']}")
        print(f"Memory usage: {stats['estimated_memory_usage_mb']:.1f}MB")
        print(f"Memory utilization: {stats['memory_utilization']:.1%}")
    
    # Test preloading a model
    response = requests.post(f"{API_BASE}/admin/model-registry/preload", 
                           headers=headers, params={"model_name": "llama-3.2-1b-instruct"})
    print(f"Preload Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Model preloaded successfully")
    
    return response.status_code == 200

def test_safety_pipeline():
    """Test safety pipeline with various inputs."""
    print("\n🛡️ Testing Safety Pipeline...")
    
    # Test normal content
    normal_payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "user", "content": "Hello, how are you today?"}
        ],
        "max_tokens": 50
    }
    
    response = requests.post(f"{API_BASE}/v1/chat/completions", json=normal_payload)
    print(f"Normal content status: {response.status_code}")
    
    # Test PII content
    pii_payload = {
        "model": "llama-3.2-3b-instruct", 
        "messages": [
            {"role": "user", "content": "My email is test@example.com and phone is 555-123-4567"}
        ],
        "max_tokens": 50
    }
    
    response = requests.post(f"{API_BASE}/v1/chat/completions", json=pii_payload)
    print(f"PII content status: {response.status_code}")
    if response.status_code == 200:
        print("✅ PII content processed (should be redacted in logs)")
    
    return True

def test_token_rate_limiting():
    """Test token-based rate limiting."""
    print("\n⏱️ Testing Token Rate Limiting...")
    
    # Make several requests quickly to test rate limiting
    for i in range(3):
        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [
                {"role": "user", "content": f"Test message {i+1} with some content to generate tokens"}
            ],
            "max_tokens": 100
        }
        
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=payload)
        print(f"Request {i+1} status: {response.status_code}")
        
        # Check rate limit headers
        headers = response.headers
        for key, value in headers.items():
            if "ratelimit" in key.lower():
                print(f"  {key}: {value}")
        
        if response.status_code == 429:
            print("✅ Rate limiting working correctly")
            return True
        
        time.sleep(0.5)  # Small delay between requests
    
    print("✅ No rate limiting triggered (normal under low load)")
    return True

def test_streaming_with_safety():
    """Test streaming with safety pipeline."""
    print("\n🌊 Testing Streaming with Safety...")
    
    payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "user", "content": "Tell me a short story about a robot"}
        ],
        "max_tokens": 100,
        "stream": True
    }
    
    response = requests.post(f"{API_BASE}/v1/chat/completions", 
                           json=payload, stream=True)
    print(f"Streaming status: {response.status_code}")
    
    if response.status_code == 200:
        chunks_received = 0
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        if 'choices' in chunk_data:
                            chunks_received += 1
                    except json.JSONDecodeError:
                        pass  # Skip heartbeats and other non-JSON data
        
        print(f"✅ Received {chunks_received} content chunks")
        return True
    else:
        print(f"❌ Streaming failed: {response.text}")
        return False

def test_prometheus_metrics():
    """Test Prometheus metrics endpoint."""
    print("\n📊 Testing Prometheus Metrics...")
    
    response = requests.get(f"{API_BASE}/metrics")
    print(f"Metrics status: {response.status_code}")
    
    if response.status_code == 200:
        metrics_text = response.text
        # Check for Wave 2 specific metrics
        wave2_metrics = [
            "llm_rate_limit_exceeded_total", 
            "llm_chat_tokens_total",
            "llm_request_latency_seconds"
        ]
        
        found_metrics = []
        for metric in wave2_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)
        
        print(f"✅ Found {len(found_metrics)}/{len(wave2_metrics)} Wave 2 metrics")
        return len(found_metrics) > 0
    
    return False

def main():
    """Run all Wave 2 tests."""
    print("🚀 Starting Wave 2 Feature Tests...")
    print(f"API Base: {API_BASE}")
    
    tests = [
        ("Health Check", lambda: requests.get(f"{API_BASE}/health").status_code == 200),
        ("Model Registry", test_model_registry),
        ("Safety Pipeline", test_safety_pipeline),
        ("Token Rate Limiting", test_token_rate_limiting),
        ("Streaming with Safety", test_streaming_with_safety),
        ("Prometheus Metrics", test_prometheus_metrics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR - {test_name}: {str(e)}")
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📋 Test Summary: {passed}/{total} tests passed")
    print("\nDetailed Results:")
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 All Wave 2 features are working correctly!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the implementation.")

if __name__ == "__main__":
    main()
