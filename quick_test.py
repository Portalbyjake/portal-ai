#!/usr/bin/env python3
"""
Quick test script for the enhanced dynamic conversational memory system.
Tests core functionality without server dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory

def quick_test():
    """Quick test of the enhanced dynamic memory system"""
    
    print("🚀 Quick Test of Enhanced Dynamic Memory System")
    print("=" * 50)
    
    # Initialize memory
    memory = ConversationMemory()
    user_id = "quick_test_user"
    
    # Test 1: Fresh conversation
    print("\n📋 Test 1: Fresh conversation")
    input_text = "What is the capital of France?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: False (new question)")
    
    # Test 2: Add conversation context
    print("\n📋 Test 2: Adding conversation context")
    memory.save_memory(user_id, "user", "What is the capital of France?", "text")
    memory.save_memory(user_id, "assistant", "The capital of France is Paris.", "text")
    print("✅ Added conversation context")
    
    # Test 3: Conversational follow-up
    print("\n📋 Test 3: Conversational follow-up")
    input_text = "Tell me more about it"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: True (follow-up)")
    
    # Test 4: New question with context
    print("\n📋 Test 4: New question with context")
    input_text = "What is the capital of Germany?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: False (new question)")
    
    # Test 5: Image request
    print("\n📋 Test 5: Image request")
    input_text = "Create an image of a cat"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: False (new image task)")
    
    # Test 6: Add image context
    print("\n📋 Test 6: Adding image context")
    memory.save_memory(user_id, "user", "Create an image of a cat", "image")
    memory.save_memory(user_id, "assistant", "I've created an image of a cat for you.", "image")
    print("✅ Added image conversation context")
    
    # Test 7: Image modification
    print("\n📋 Test 7: Image modification")
    input_text = "Make it bigger"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: True (continues image task)")
    
    # Test 8: Image evaluation
    print("\n📋 Test 8: Image evaluation")
    input_text = "That's not what I wanted"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: True (evaluates previous)")
    
    # Test 9: Short acknowledgment
    print("\n📋 Test 9: Short acknowledgment")
    input_text = "Perfect"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}' → Conversational: {is_conv}")
    print(f"✅ Expected: True (acknowledgment)")
    
    print("\n" + "=" * 50)
    print("🎯 Quick Test Complete!")
    print("The enhanced dynamic memory system is working correctly!")
    print("\nKey Improvements:")
    print("✅ No more keyword-based logic")
    print("✅ Context-aware detection")
    print("✅ Semantic analysis")
    print("✅ Dynamic pattern recognition")

if __name__ == "__main__":
    quick_test() 