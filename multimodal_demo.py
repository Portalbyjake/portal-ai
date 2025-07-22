#!/usr/bin/env python3
"""
Demonstration of enhanced multimodal memory integration with Portal application.
Shows how the system handles continuity across text, image, and other formats.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory
from services import run_model
import json

def demonstrate_multimodal_conversation():
    """Demonstrate a realistic multimodal conversation flow"""
    
    print("🎭 Multimodal Memory Demonstration")
    print("=" * 50)
    print("This demo shows how the system maintains continuity across different modalities.")
    print()
    
    # Initialize memory
    memory = ConversationMemory()
    user_id = "demo_user"
    
    # Simulate a realistic conversation flow
    conversation_steps = [
        {
            "user": "Tell me about artificial intelligence",
            "task_type": "text",
            "expected_response": "AI is a branch of computer science that aims to create intelligent machines."
        },
        {
            "user": "Create an image of a robot",
            "task_type": "image", 
            "expected_response": "I've created an image of a robot for you."
        },
        {
            "user": "What color is that robot?",
            "task_type": "text",
            "expected_response": "I can't see the specific colors in the generated image, but I can describe typical robot colors."
        },
        {
            "user": "Make it blue",
            "task_type": "image",
            "expected_response": "I've updated the robot image to be blue."
        },
        {
            "user": "How does machine learning work?",
            "task_type": "text",
            "expected_response": "Machine learning uses algorithms to learn patterns from data."
        },
        {
            "user": "Show me a diagram of that",
            "task_type": "image",
            "expected_response": "I've created a diagram showing how machine learning works."
        }
    ]
    
    for i, step in enumerate(conversation_steps, 1):
        print(f"Step {i}: {step['task_type'].upper()} INTERACTION")
        print("-" * 30)
        
        user_input = step["user"]
        task_type = step["task_type"]
        
        # Check for cross-modal references
        is_cross_modal, referenced_modality, referenced_content = memory.is_cross_modal_reference(user_input, user_id)
        
        print(f"User: '{user_input}'")
        print(f"Task Type: {task_type}")
        
        if is_cross_modal:
            print(f"🔗 Cross-modal reference detected!")
            print(f"   References: {referenced_modality} → {task_type}")
            print(f"   Previous content: '{referenced_content[:50]}...'")
        else:
            print(f"📝 New {task_type} request")
        
        # Save user input
        memory.save_memory(user_id, "user", user_input, task_type)
        
        # Simulate assistant response
        assistant_response = step["expected_response"]
        memory.save_memory(user_id, "assistant", assistant_response, task_type)
        
        print(f"Assistant: '{assistant_response}'")
        
        # Show current multimodal context
        context = memory.get_multimodal_context(user_id)
        print(f"Current modality: {context.get('current_modality')}")
        print(f"Cross-references: {len(context.get('cross_references', []))}")
        print()
    
    # Show final statistics
    print("📊 FINAL MULTIMODAL STATISTICS")
    print("=" * 30)
    
    final_context = memory.get_multimodal_context(user_id)
    cross_references = final_context.get('cross_references', [])
    
    print(f"Total modality switches: {len(cross_references)}")
    switch_pattern = ' → '.join([f"{ref.get('from_modality')}→{ref.get('to_modality')}" for ref in cross_references])
    print(f"Switch pattern: {switch_pattern}")
    print(f"Current modality: {final_context.get('current_modality')}")
    print(f"Last image prompt: {final_context.get('last_image_prompt', 'None')}")
    print(f"Last text topic: {final_context.get('last_text_topic', 'None')}")
    
    return memory

def demonstrate_enhanced_context():
    """Demonstrate enhanced context generation"""
    
    print("\n🔗 Enhanced Context Generation Demo")
    print("=" * 40)
    
    memory = ConversationMemory()
    user_id = "context_demo"
    
    # Create a conversation with cross-modal references
    memory.save_memory(user_id, "user", "What is deep learning?", "text")
    memory.save_memory(user_id, "assistant", "Deep learning uses neural networks with multiple layers to process data.", "text")
    memory.save_memory(user_id, "user", "Show me a neural network", "image")
    memory.save_memory(user_id, "assistant", "I've created a neural network diagram.", "image")
    
    # Test enhanced context for a follow-up question
    user_input = "What are the layers called?"
    enhanced_context = memory.get_enhanced_context_for_followup(user_id, user_input)
    
    print(f"User input: '{user_input}'")
    print("\nEnhanced context:")
    print("-" * 20)
    print(enhanced_context)
    
    # Check if cross-modal reference was detected
    is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
    print(f"\nCross-modal detection: {is_cross_modal}")
    if is_cross_modal:
        print(f"References: {modality} content")

def demonstrate_memory_persistence():
    """Demonstrate memory persistence across sessions"""
    
    print("\n💾 Memory Persistence Demo")
    print("=" * 30)
    
    # Create memory with metadata
    memory = ConversationMemory()
    user_id = "persistence_demo"
    
    metadata = {
        'image_style': 'photorealistic',
        'model_used': 'dall-e-3',
        'user_preference': 'detailed',
        'session_id': 'demo_session_001'
    }
    
    memory.save_memory(user_id, "user", "Create a detailed landscape", "image", "dall-e-3", metadata)
    memory.save_memory(user_id, "assistant", "I've created a detailed landscape image.", "image", "dall-e-3")
    
    # Retrieve and display memory
    recent_memory = memory.get_recent_memory(user_id, 2)
    
    print("Stored memory entries:")
    for entry in recent_memory:
        print(f"  - {entry.get('role')}: {entry.get('content')}")
        print(f"    Task: {entry.get('task_type')}")
        print(f"    Model: {entry.get('model')}")
        if entry.get('metadata'):
            print(f"    Metadata: {entry.get('metadata')}")
        print()
    
    # Show multimodal context
    context = memory.get_multimodal_context(user_id)
    print(f"Multimodal context: {context}")

def demonstrate_integration_with_portal():
    """Show how this integrates with the Portal application"""
    
    print("\n🚀 Portal Integration Demo")
    print("=" * 30)
    
    print("The enhanced multimodal memory system integrates with Portal in these ways:")
    print()
    print("1. **Enhanced Task Routing**")
    print("   - Routes.py uses get_intelligent_task_routing()")
    print("   - Considers conversation history for task classification")
    print("   - Detects cross-modal references automatically")
    print()
    print("2. **Improved Context Generation**")
    print("   - Services.py uses get_enhanced_context_for_followup()")
    print("   - Provides richer context for model calls")
    print("   - Includes cross-modal reference information")
    print()
    print("3. **Better User Experience**")
    print("   - Users can reference previous images/text naturally")
    print("   - System maintains conversation flow across modalities")
    print("   - Reduces need for explicit repetition")
    print()
    print("4. **Enhanced Analytics**")
    print("   - Tracks modality switches")
    print("   - Monitors cross-modal references")
    print("   - Provides insights into user behavior patterns")
    
    # Show example integration code
    print("\nExample integration in routes.py:")
    print("```python")
    print("# Enhanced task routing with multimodal awareness")
    print("task_type, confidence, reasoning = memory_manager.get_intelligent_task_routing(user_id, user_input)")
    print()
    print("# Enhanced context for model calls")
    print("context = memory_manager.get_enhanced_context_for_followup(user_id, user_input)")
    print("if context:")
    print("    prompt = f'{context}\\n\\nUser: {user_input}'")
    print("```")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_multimodal_conversation()
    demonstrate_enhanced_context()
    demonstrate_memory_persistence()
    demonstrate_integration_with_portal()
    
    print("\n" + "=" * 50)
    print("🎯 Multimodal Memory Demonstration Complete!")
    print("The system successfully maintains continuity across text, image, and other formats.") 