#!/usr/bin/env python3
"""
Test script for the enhanced dynamic conversational memory system.
Tests various scenarios to ensure the system works dynamically rather than keyword-based.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory

def test_enhanced_memory():
    """Test the enhanced dynamic conversational memory system"""
    
    # Initialize memory
    memory = ConversationMemory()
    user_id = "test_user_enhanced"
    
    print("🧪 Testing Enhanced Dynamic Conversational Memory System")
    print("=" * 60)
    
    # Test 1: Fresh conversation - should NOT be conversational
    print("\n📋 Test 1: Fresh conversation")
    print("-" * 40)
    input_text = "What is the capital of France?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (no conversation context)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 2: Add some conversation context
    print("\n📋 Test 2: Adding conversation context")
    print("-" * 40)
    memory.save_memory(user_id, "user", "What is the capital of France?", "text")
    memory.save_memory(user_id, "assistant", "The capital of France is Paris.", "text")
    print("Added conversation context")
    
    # Test 3: New question with context - should NOT be conversational
    print("\n📋 Test 3: New question with conversation context")
    print("-" * 40)
    input_text = "What is the capital of Germany?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new question, not a follow-up)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 4: Conversational follow-up - should BE conversational
    print("\n📋 Test 4: Conversational follow-up")
    print("-" * 40)
    input_text = "Tell me more about it"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (references previous conversation)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 5: Short acknowledgment - should BE conversational
    print("\n📋 Test 5: Short acknowledgment")
    print("-" * 40)
    input_text = "Interesting"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (evaluative response)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 6: Clarification request - should BE conversational
    print("\n📋 Test 6: Clarification request")
    print("-" * 40)
    input_text = "What do you mean by that?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (asks for clarification)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 7: New question that looks like follow-up - should NOT be conversational
    print("\n📋 Test 7: New question that could be mistaken for follow-up")
    print("-" * 40)
    input_text = "What is the population of Paris?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new question about Paris)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 8: Emotional reaction - should BE conversational
    print("\n📋 Test 8: Emotional reaction")
    print("-" * 40)
    input_text = "Wow, that's amazing!"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (emotional reaction)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 9: Reference pronoun - should BE conversational
    print("\n📋 Test 9: Reference pronoun")
    print("-" * 40)
    input_text = "That's not what I asked for"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (uses 'that' to reference previous)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 10: Simple acknowledgment - should BE conversational
    print("\n📋 Test 10: Simple acknowledgment")
    print("-" * 40)
    input_text = "Ok"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (simple acknowledgment)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 11: New task request - should NOT be conversational
    print("\n📋 Test 11: New task request")
    print("-" * 40)
    input_text = "Create an image of a cat"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new task request)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 12: Task continuation - should BE conversational
    print("\n📋 Test 12: Task continuation")
    print("-" * 40)
    # Add image task context
    memory.save_memory(user_id, "user", "Create an image of a cat", "image")
    memory.save_memory(user_id, "assistant", "I've created an image of a cat for you.", "image")
    
    input_text = "Make it bigger"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (continues image task)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    print("\n" + "=" * 60)
    print("🎯 Enhanced Dynamic Memory System Test Complete!")
    print("The system now uses semantic analysis instead of keyword matching.")

if __name__ == "__main__":
    test_enhanced_memory() 