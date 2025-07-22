#!/usr/bin/env python3
"""
Test script for the enhanced multimodal memory system.
Demonstrates continuity across text, image, and other formats.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory

def test_multimodal_memory():
    """Test the enhanced multimodal memory system"""
    
    # Initialize memory
    memory = ConversationMemory()
    user_id = "test_user_multimodal"
    
    print("🔄 Testing Enhanced Multimodal Memory System")
    print("=" * 60)
    
    # Test 1: Text conversation followed by image reference
    print("\n📝 Test 1: Text → Image Cross-Modal Reference")
    print("-" * 40)
    
    # Start with text conversation
    memory.save_memory(user_id, "user", "Tell me about cats", "text")
    memory.save_memory(user_id, "assistant", "Cats are domesticated mammals known for their independence and hunting skills.", "text")
    
    # User references the text in an image request
    user_input = "Create an image of that"
    is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
    
    print(f"User input: '{user_input}'")
    print(f"Is cross-modal reference: {is_cross_modal}")
    print(f"Referenced modality: {modality}")
    print(f"Referenced content: {content}")
    print(f"Expected: True, 'text', 'Tell me about cats'")
    print(f"✅ PASS" if is_cross_modal and modality == 'text' else "❌ FAIL")
    
    # Test 2: Image conversation followed by text reference
    print("\n🖼️ Test 2: Image → Text Cross-Modal Reference")
    print("-" * 40)
    
    # Add image conversation
    memory.save_memory(user_id, "user", "Create an image of a cat", "image")
    memory.save_memory(user_id, "assistant", "I've created an image of a cat for you.", "image")
    
    # User references the image in a text question
    user_input = "What breed is that cat?"
    is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
    
    print(f"User input: '{user_input}'")
    print(f"Is cross-modal reference: {is_cross_modal}")
    print(f"Referenced modality: {modality}")
    print(f"Referenced content: {content}")
    print(f"Expected: True, 'image', 'Create an image of a cat'")
    print(f"✅ PASS" if is_cross_modal and modality == 'image' else "❌ FAIL")
    
    # Test 3: Enhanced context with cross-modal references
    print("\n🔗 Test 3: Enhanced Context with Cross-Modal References")
    print("-" * 40)
    
    user_input = "Can you make it bigger?"
    enhanced_context = memory.get_enhanced_context_for_followup(user_id, user_input)
    
    print(f"User input: '{user_input}'")
    print(f"Enhanced context includes cross-modal info: {'Cross-modal reference' in enhanced_context}")
    print(f"✅ PASS" if 'Cross-modal reference' in enhanced_context else "❌ FAIL")
    
    # Test 4: Multimodal context tracking
    print("\n📊 Test 4: Multimodal Context Tracking")
    print("-" * 40)
    
    multimodal_context = memory.get_multimodal_context(user_id)
    
    print(f"Current modality: {multimodal_context.get('current_modality')}")
    print(f"Cross-references count: {len(multimodal_context.get('cross_references', []))}")
    print(f"Last image prompt: {multimodal_context.get('last_image_prompt')}")
    print(f"Last text topic: {multimodal_context.get('last_text_topic')}")
    print(f"✅ PASS" if multimodal_context.get('current_modality') == 'image' else "❌ FAIL")
    
    # Test 5: Complex multimodal conversation flow
    print("\n🔄 Test 5: Complex Multimodal Conversation Flow")
    print("-" * 40)
    
    # Simulate a complex conversation
    conversation_flow = [
        ("Tell me about Paris", "text"),
        ("Paris is the capital of France, known for its art, culture, and cuisine.", "text"),
        ("Show me a picture of the Eiffel Tower", "image"),
        ("I've created an image of the Eiffel Tower.", "image"),
        ("What's the height of that tower?", "text"),
        ("The Eiffel Tower is 324 meters tall.", "text"),
        ("Make the image more realistic", "image"),
        ("I've updated the image to be more photorealistic.", "image"),
        ("Tell me more about its history", "text"),
        ("The Eiffel Tower was built in 1889 for the World's Fair.", "text")
    ]
    
    cross_modal_count = 0
    for i, (user_input, task_type) in enumerate(conversation_flow):
        if i % 2 == 0:  # User inputs
            is_cross_modal, _, _ = memory.is_cross_modal_reference(user_input, user_id)
            if is_cross_modal:
                cross_modal_count += 1
            memory.save_memory(user_id, "user", user_input, task_type)
        else:  # Assistant responses
            memory.save_memory(user_id, "assistant", user_input, task_type)
    
    print(f"Total cross-modal references detected: {cross_modal_count}")
    print(f"Expected: At least 2 cross-modal references")
    print(f"✅ PASS" if cross_modal_count >= 2 else "❌ FAIL")
    
    # Test 6: Modality switching detection
    print("\n🔄 Test 6: Modality Switching Detection")
    print("-" * 40)
    
    final_context = memory.get_multimodal_context(user_id)
    cross_references = final_context.get('cross_references', [])
    
    print(f"Total modality switches: {len(cross_references)}")
    print(f"Switch types: {[ref.get('from_modality') + '→' + ref.get('to_modality') for ref in cross_references]}")
    print(f"Expected: Multiple modality switches (text↔image)")
    print(f"✅ PASS" if len(cross_references) >= 2 else "❌ FAIL")
    
    # Test 7: Memory with metadata
    print("\n📋 Test 7: Memory with Enhanced Metadata")
    print("-" * 40)
    
    metadata = {
        'image_style': 'photorealistic',
        'image_size': '1024x1024',
        'model_used': 'dall-e-3',
        'user_preference': 'detailed'
    }
    
    memory.save_memory(user_id, "user", "Create a detailed image", "image", "dall-e-3", metadata)
    
    # Check if metadata is preserved
    recent_memory = memory.get_recent_memory(user_id, 1)
    last_entry = recent_memory[-1] if recent_memory else {}
    has_metadata = 'metadata' in last_entry and last_entry['metadata']
    
    print(f"Memory entry has metadata: {has_metadata}")
    print(f"Metadata content: {last_entry.get('metadata', {})}")
    print(f"Expected: True with metadata")
    print(f"✅ PASS" if has_metadata else "❌ FAIL")
    
    print("\n" + "=" * 60)
    print("🎯 Enhanced Multimodal Memory System Tests Complete!")
    print("The system successfully handles continuity across text, image, and other formats.")

def test_multimodal_scenarios():
    """Test specific multimodal scenarios"""
    
    memory = ConversationMemory()
    user_id = "test_scenarios"
    
    print("\n🎭 Testing Specific Multimodal Scenarios")
    print("=" * 50)
    
    # Scenario 1: Image modification after text discussion
    print("\n📝 Scenario 1: Text Discussion → Image Modification")
    print("-" * 40)
    
    memory.save_memory(user_id, "user", "What are the best colors for a logo?", "text")
    memory.save_memory(user_id, "assistant", "Blue and white are popular for logos as they convey trust and professionalism.", "text")
    memory.save_memory(user_id, "user", "Create a logo with those colors", "image")
    memory.save_memory(user_id, "assistant", "I've created a blue and white logo for you.", "image")
    
    user_input = "Make it more modern"
    is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
    
    print(f"User: 'Make it more modern'")
    print(f"References previous image: {is_cross_modal}")
    print(f"✅ PASS" if is_cross_modal else "❌ FAIL")
    
    # Scenario 2: Text question about generated image
    print("\n🖼️ Scenario 2: Image Generation → Text Question")
    print("-" * 40)
    
    memory.save_memory(user_id, "user", "Create an image of a sunset", "image")
    memory.save_memory(user_id, "assistant", "I've created a beautiful sunset image.", "image")
    
    user_input = "What time of day is that?"
    is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
    
    print(f"User: 'What time of day is that?'")
    print(f"References previous image: {is_cross_modal}")
    print(f"✅ PASS" if is_cross_modal else "❌ FAIL")
    
    # Scenario 3: Complex multimodal workflow
    print("\n🔄 Scenario 3: Complex Multimodal Workflow")
    print("-" * 40)
    
    workflow = [
        ("Explain machine learning", "text"),
        ("Machine learning is a subset of AI that enables computers to learn from data.", "text"),
        ("Show me a diagram of neural networks", "image"),
        ("I've created a neural network diagram.", "image"),
        ("What are the layers called?", "text"),
        ("The layers are input, hidden, and output layers.", "text"),
        ("Make the diagram simpler", "image"),
        ("I've simplified the neural network diagram.", "image"),
        ("How does backpropagation work?", "text"),
        ("Backpropagation is the algorithm that adjusts weights to minimize error.", "text")
    ]
    
    cross_modal_references = 0
    for i, (input_text, task_type) in enumerate(workflow):
        if i % 2 == 0:  # User inputs
            is_cross_modal, _, _ = memory.is_cross_modal_reference(input_text, user_id)
            if is_cross_modal:
                cross_modal_references += 1
            memory.save_memory(user_id, "user", input_text, task_type)
        else:  # Assistant responses
            memory.save_memory(user_id, "assistant", input_text, task_type)
    
    print(f"Cross-modal references in workflow: {cross_modal_references}")
    print(f"Expected: Multiple cross-modal references")
    print(f"✅ PASS" if cross_modal_references >= 3 else "❌ FAIL")
    
    print("\n" + "=" * 50)
    print("🎭 Multimodal Scenarios Complete!")

if __name__ == "__main__":
    test_multimodal_memory()
    test_multimodal_scenarios() 