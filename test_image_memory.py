#!/usr/bin/env python3
"""
Test script for the enhanced dynamic conversational memory system with image generation.
Tests various image-related scenarios to ensure the system works dynamically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory

def test_image_memory():
    """Test the enhanced dynamic conversational memory system with image scenarios"""
    
    # Initialize memory
    memory = ConversationMemory()
    user_id = "test_user_images"
    
    print("🖼️ Testing Enhanced Dynamic Memory System with Image Generation")
    print("=" * 70)
    
    # Test 1: Fresh image request - should NOT be conversational
    print("\n📋 Test 1: Fresh image request")
    print("-" * 40)
    input_text = "Create an image of a cat"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new image request)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 2: Add image conversation context
    print("\n📋 Test 2: Adding image conversation context")
    print("-" * 40)
    memory.save_memory(user_id, "user", "Create an image of a cat", "image")
    memory.save_memory(user_id, "assistant", "I've created an image of a cat for you.", "image")
    print("Added image conversation context")
    
    # Test 3: Image modification request - should BE conversational
    print("\n📋 Test 3: Image modification request")
    print("-" * 40)
    input_text = "Make it bigger"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (continues image task)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 4: Different style request - should BE conversational
    print("\n📋 Test 4: Different style request")
    print("-" * 40)
    input_text = "Try a different style"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (continues image task)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 5: New image request with context - should NOT be conversational
    print("\n📋 Test 5: New image request with context")
    print("-" * 40)
    input_text = "Create an image of a dog"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new image request)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 6: Image evaluation - should BE conversational
    print("\n📋 Test 6: Image evaluation")
    print("-" * 40)
    input_text = "That's not what I wanted"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (evaluates previous image)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 7: Image clarification - should BE conversational
    print("\n📋 Test 7: Image clarification")
    print("-" * 40)
    input_text = "Can you make it more realistic?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (clarifies image request)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 8: Short image acknowledgment - should BE conversational
    print("\n📋 Test 8: Short image acknowledgment")
    print("-" * 40)
    input_text = "Perfect"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (acknowledges image)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    # Test 9: New text question with image context - should NOT be conversational
    print("\n📋 Test 9: New text question with image context")
    print("-" * 40)
    input_text = "What is the capital of France?"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: False (new text question)")
    print(f"✅ PASS" if not is_conv else "❌ FAIL")
    
    # Test 10: Image task continuation with different words
    print("\n📋 Test 10: Image task continuation with different words")
    print("-" * 40)
    input_text = "Change the background"
    is_conv = memory.is_conversational_response(input_text, user_id)
    print(f"Input: '{input_text}'")
    print(f"Is conversational: {is_conv}")
    print(f"Expected: True (continues image task)")
    print(f"✅ PASS" if is_conv else "❌ FAIL")
    
    print("\n" + "=" * 70)
    print("🎯 Enhanced Dynamic Memory System - Image Tests Complete!")
    print("The system correctly distinguishes between new image requests and continuations.")

    # --- Additional Tests: Undo, History, Corrupt File Recovery ---
    print("\n🧩 Additional Image Memory Tests")
    print("-" * 40)

    # Test Undo Functionality
    print("\n📋 Test 11: Undo last image entry")
    memory.save_memory(user_id, "user", "Create an image of a sunset", "image")
    memory.save_memory(user_id, "assistant", "Here's a sunset image.", "image")
    before_undo = len(memory.get_image_history(user_id))
    memory.undo_last_image(user_id)
    after_undo = len(memory.get_image_history(user_id))
    print(f"Image history before undo: {before_undo}, after undo: {after_undo}")
    print(f"✅ PASS" if after_undo == before_undo - 1 else "❌ FAIL")

    # Test Image History Retrieval
    print("\n📋 Test 12: Image history retrieval")
    history = memory.get_image_history(user_id)
    print(f"Image history length: {len(history)}")
    print(f"History sample: {history[-1] if history else 'EMPTY'}")
    print(f"✅ PASS" if isinstance(history, list) else "❌ FAIL")

    # Test Corrupt Image Memory File Recovery
    print("\n📋 Test 13: Corrupt image memory file recovery")
    # Simulate a corrupt line in the image memory file
    corrupt_line = '{bad json line\n'
    image_file = memory.image_memory_file
    with open(image_file, 'a', encoding='utf-8') as f:
        f.write(corrupt_line)
    # Re-initialize memory to trigger recovery
    try:
        mem2 = ConversationMemory()
        mem2.load_image_memory()
        print("No crash on corrupt image memory file (expected)")
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: Exception on corrupt image memory file: {e}")

if __name__ == "__main__":
    test_image_memory() 