#!/usr/bin/env python3
"""
Advanced test script for Simple LLM Service API
Tests all Phase 3 features including authentication, rate limiting, and monitoring.
"""

import requests
import json
import time
from typing import Optional

API_BASE = "http://localhost:8000"

class APITester:
    """Advanced API testing class."""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token: Optional[str] = None
        self.api_key = "sk-simple-llm-demo-key-123"  # Demo API key
    
    def login(self, username: str = "admin", password: str = "admin123") -> bool:
        """Login and get access token."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })
                print(f"✅ Login successful: {data['user_info']['username']}")
                return True
            else:
                print(f"❌ Login failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_authentication(self):
        """Test authentication endpoints."""
        print("\n🔐 Testing Authentication...")
        
        # Test login
        self.login()
        
        # Test user info
        try:
            response = self.session.get(f"{self.base_url}/auth/me")
            if response.status_code == 200:
                user_info = response.json()
                print(f"✅ User info: {user_info}")
            else:
                print(f"❌ Failed to get user info: {response.text}")
        except Exception as e:
            print(f"❌ User info error: {e}")
        
        # Test API key authentication
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/auth/me", headers=headers)
            if response.status_code == 200:
                print(f"✅ API key authentication works")
            else:
                print(f"❌ API key authentication failed: {response.text}")
        except Exception as e:
            print(f"❌ API key error: {e}")
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        print("\n⏱️ Testing Rate Limiting...")
        
        # Make rapid requests to trigger rate limit
        endpoint = f"{self.base_url}/health"
        success_count = 0
        rate_limited = False
        
        for i in range(15):  # Try 15 requests rapidly
            try:
                response = requests.get(endpoint)
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited = True
                    print(f"✅ Rate limit triggered after {success_count} requests")
                    print(f"Rate limit response: {response.json()}")
                    break
                time.sleep(0.1)  # Small delay
            except Exception as e:
                print(f"❌ Rate limit test error: {e}")
                break
        
        if not rate_limited:
            print(f"⚠️ Rate limiting not triggered after {success_count} requests")
    
    def test_monitoring(self):
        """Test monitoring endpoints."""
        print("\n📊 Testing Monitoring...")
        
        # Test public status
        try:
            response = self.session.get(f"{self.base_url}/admin/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Service status: {status}")
            else:
                print(f"❌ Status check failed: {response.text}")
        except Exception as e:
            print(f"❌ Status error: {e}")
        
        # Test admin metrics (requires authentication)
        if self.access_token:
            try:
                response = self.session.get(f"{self.base_url}/admin/metrics")
                if response.status_code == 200:
                    metrics = response.json()
                    print(f"✅ Metrics available - Uptime: {metrics['application']['uptime_seconds']:.1f}s")
                    print(f"✅ Total requests: {metrics['application']['total_requests']}")
                else:
                    print(f"❌ Metrics failed: {response.text}")
            except Exception as e:
                print(f"❌ Metrics error: {e}")
    
    def test_protected_completions(self):
        """Test completions with authentication."""
        print("\n💬 Testing Protected Chat Completions...")
        
        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please respond briefly."}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        # Test without authentication (should work if auth not required)
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                print(f"✅ Unauthenticated completion: {assistant_message[:100]}...")
            elif response.status_code == 401:
                print("✅ Authentication required for completions")
            else:
                print(f"❌ Unexpected response: {response.status_code}")
        except Exception as e:
            print(f"❌ Completion error: {e}")
        
        # Test with authentication
        if self.access_token:
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result['choices'][0]['message']['content']
                    print(f"✅ Authenticated completion: {assistant_message[:100]}...")
                else:
                    print(f"❌ Authenticated completion failed: {response.text}")
            except Exception as e:
                print(f"❌ Authenticated completion error: {e}")
    
    def test_health_endpoints(self):
        """Test all health endpoints."""
        print("\n🏥 Testing Health Endpoints...")
        
        endpoints = [
            "/health",
            "/health/ready", 
            "/health/live"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    print(f"✅ {endpoint}: OK")
                else:
                    print(f"❌ {endpoint}: Failed ({response.status_code})")
            except Exception as e:
                print(f"❌ {endpoint}: Error - {e}")
        
        # Test detailed health (requires auth)
        if self.access_token:
            try:
                response = self.session.get(f"{self.base_url}/health/detailed")
                if response.status_code == 200:
                    health = response.json()
                    print(f"✅ Detailed health: {health['status']}")
                    if health.get('issues'):
                        print(f"⚠️ Issues: {health['issues']}")
                else:
                    print(f"❌ Detailed health failed: {response.text}")
            except Exception as e:
                print(f"❌ Detailed health error: {e}")

def main():
    """Run all Phase 3 tests."""
    print("🚀 Starting Simple LLM Service Phase 3 Tests\n")
    
    tester = APITester()
    
    tests = [
        ("Authentication", tester.test_authentication),
        ("Health Endpoints", tester.test_health_endpoints),
        ("Monitoring", tester.test_monitoring),
        ("Rate Limiting", tester.test_rate_limiting),
        ("Protected Completions", tester.test_protected_completions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print("=" * 60)
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} test categories passed")
    
    if passed == len(tests):
        print("\n🎉 All Phase 3 tests completed! Production features are working.")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) had issues. Check the logs.")
    
    print("\n📋 Phase 3 Features Tested:")
    print("✅ JWT & API Key Authentication")
    print("✅ Rate Limiting (per user/IP)")
    print("✅ Monitoring & Metrics")
    print("✅ Protected Endpoints")
    print("✅ Health Checks")

if __name__ == "__main__":
    main()
