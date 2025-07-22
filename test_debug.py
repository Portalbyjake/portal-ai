#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import conversation_memory

def test_conversational_detection():
    """Test the conversational response detection logic"""
    
    # Clear memory first
    conversation_memory.clear_memory()
    
    # Test 1: Fresh conversation with a clear task
    print("=== Test 1: Fresh conversation ===")
    user_input = "What is the capital of France?"
    user_id = "test_user"
    
    print(f"Input: '{user_input}'")
    print(f"User ID: {user_id}")
    print(f"Has conversation memory: {user_id in conversation_memory.conversation_memory}")
    print(f"Memory entries: {len(conversation_memory.conversation_memory.get(user_id, []))}")
    
    # Check if it's conversational
    is_conv = conversation_memory.is_conversational_response(user_input, user_id)
    print(f"Is conversational: {is_conv}")
    
    # Get task routing
    task_type, confidence, reasoning = conversation_memory.get_intelligent_task_routing(user_id, user_input)
    print(f"Task routing: {task_type}, confidence: {confidence}, reasoning: {reasoning}")
    
    print("\n=== Test 2: After adding some conversation ===")
    
    # Add some conversation context
    conversation_memory.save_memory(user_id, "user", "Hello", "conversation")
    conversation_memory.save_memory(user_id, "assistant", "Hi there! How can I help you?", "conversation")
    
    print(f"Has conversation memory: {user_id in conversation_memory.conversation_memory}")
    print(f"Memory entries: {len(conversation_memory.conversation_memory.get(user_id, []))}")
    
    # Test the same input again
    is_conv = conversation_memory.is_conversational_response(user_input, user_id)
    print(f"Is conversational: {is_conv}")
    
    task_type, confidence, reasoning = conversation_memory.get_intelligent_task_routing(user_id, user_input)
    print(f"Task routing: {task_type}, confidence: {confidence}, reasoning: {reasoning}")

if __name__ == "__main__":
    test_conversational_detection() 