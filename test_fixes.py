#!/usr/bin/env python3
"""
Test script to verify the fixes for rate limiting and error handling.
"""

import requests
import json
import time

def test_basic_query():
    """Test a basic query to ensure the server is working"""
    url = "http://localhost:8081/query"
    data = {
        "input": "What is the capital of France?",
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Model: {result.get('model')}")
            print(f"Output: {result.get('output', '')[:100]}...")
            print("✅ Basic query test passed")
            return True
        else:
            print(f"❌ Basic query failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_followup_query():
    """Test a follow-up query to ensure memory is working"""
    url = "http://localhost:8081/query"
    
    # First query
    data1 = {
        "input": "What does George Washington look like?",
        "user_id": "test_user"
    }
    
    try:
        response1 = requests.post(url, json=data1)
        if response1.status_code != 200:
            print("❌ First query failed")
            return False
        
        # Second query (follow-up)
        data2 = {
            "input": "Does he have a beard?",
            "user_id": "test_user"
        }
        
        response2 = requests.post(url, json=data2)
        print(f"Follow-up status: {response2.status_code}")
        
        if response2.status_code == 200:
            result = response2.json()
            output = result.get('output', '')
            print(f"Follow-up output: {output[:200]}...")
            
            # Check if it's a meaningful response (not blank or error)
            if output and len(output.strip()) > 10 and "error" not in output.lower():
                print("✅ Follow-up query test passed")
                return True
            else:
                print("❌ Follow-up query returned empty or error response")
                return False
        else:
            print(f"❌ Follow-up query failed: {response2.text}")
            return False
            
    except Exception as e:
        print(f"❌ Follow-up test error: {e}")
        return False

def test_error_handling():
    """Test error handling with a very long prompt"""
    url = "http://localhost:8081/query"
    
    # Create a very long prompt that might trigger rate limits
    long_prompt = "Tell me everything about " + "history " * 1000
    
    data = {
        "input": long_prompt,
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Long prompt status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            output = result.get('output', '')
            print(f"Long prompt output: {output[:100]}...")
            
            # Should return a meaningful error message, not blank
            if output and len(output.strip()) > 10:
                print("✅ Error handling test passed")
                return True
            else:
                print("❌ Error handling failed - returned blank response")
                return False
        else:
            print(f"❌ Long prompt failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Portal AI fixes...")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    tests = [
        ("Basic Query", test_basic_query),
        ("Follow-up Query", test_followup_query),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.") 