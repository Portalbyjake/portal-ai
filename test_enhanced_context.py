#!/usr/bin/env python3
"""
Test enhanced prompt optimizer with conversation context awareness.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_conversation_context_awareness():
    """Test how the optimizer handles conversation context."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🧠 Testing Conversation Context Awareness")
    print("=" * 60)
    
    # Test 1: Technical conversation context
    technical_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Write something",
        conversation_history=[
            {"content": "I need help with my Python code", "role": "user"},
            {"content": "I can help you debug that. What's the issue?", "role": "assistant"},
            {"content": "The API isn't working properly", "role": "user"},
            {"content": "Let me help you troubleshoot that", "role": "assistant"}
        ]
    )
    
    result = optimizer.optimize_prompt(technical_context)
    print("🔧 Technical Context Test:")
    print(f"Prompt: 'Write something'")
    print(f"Context: Technical conversation about Python/API")
    print(f"Result: {result}")
    print()
    
    # Test 2: Creative conversation context
    creative_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Write something",
        conversation_history=[
            {"content": "I want to write a story", "role": "user"},
            {"content": "Great! What kind of story?", "role": "assistant"},
            {"content": "Something creative and imaginative", "role": "user"},
            {"content": "I can help you with that", "role": "assistant"}
        ]
    )
    
    result = optimizer.optimize_prompt(creative_context)
    print("🎨 Creative Context Test:")
    print(f"Prompt: 'Write something'")
    print(f"Context: Creative conversation about storytelling")
    print(f"Result: {result}")
    print()
    
    # Test 3: Business conversation context
    business_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Write something",
        conversation_history=[
            {"content": "I need a business plan", "role": "user"},
            {"content": "I can help you create that", "role": "assistant"},
            {"content": "What should I include?", "role": "user"},
            {"content": "Let me outline the key sections", "role": "assistant"}
        ]
    )
    
    result = optimizer.optimize_prompt(business_context)
    print("💼 Business Context Test:")
    print(f"Prompt: 'Write something'")
    print(f"Context: Business conversation about business plans")
    print(f"Result: {result}")
    print()

def test_clarification_handling():
    """Test how the optimizer handles unclear prompts."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n❓ Testing Clarification Handling")
    print("=" * 60)
    
    unclear_prompts = [
        "Write something",
        "Create anything",
        "Build it",
        "Make this",
        "Write a story and create a marketing plan",
        "Build an app and design the UI",
        "Write something about money",
        "Create something amazing"
    ]
    
    for prompt in unclear_prompts:
        context = OptimizationContext(
            task_type="text",
            model="gpt-4o",
            original_prompt=prompt
        )
        
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        is_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        print(f"Prompt: '{prompt}'")
        print(f"Needs Clarification: {'✅ Yes' if is_clarification else '❌ No'}")
        print(f"Response: {result[:100]}...")
        print()

def test_tone_awareness():
    """Test how the optimizer considers conversation tone."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🎭 Testing Tone Awareness")
    print("=" * 60)
    
    # Test formal tone
    formal_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Please write a business proposal",
        conversation_history=[
            {"content": "I would appreciate your assistance with this matter", "role": "user"},
            {"content": "I would be happy to help you with that", "role": "assistant"},
            {"content": "Could you please provide guidance?", "role": "user"}
        ]
    )
    
    result = optimizer.optimize_prompt(formal_context)
    print("📝 Formal Tone Test:")
    print(f"Prompt: 'Please write a business proposal'")
    print(f"Tone: Formal")
    print(f"Result: {result}")
    print()
    
    # Test casual tone
    casual_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Hey, write a story",
        conversation_history=[
            {"content": "Hey there!", "role": "user"},
            {"content": "Hi! How can I help?", "role": "assistant"},
            {"content": "Thanks, that's awesome!", "role": "user"}
        ]
    )
    
    result = optimizer.optimize_prompt(casual_context)
    print("😊 Casual Tone Test:")
    print(f"Prompt: 'Hey, write a story'")
    print(f"Tone: Casual")
    print(f"Result: {result}")
    print()
    
    # Test technical tone
    technical_context = OptimizationContext(
        task_type="text",
        model="gpt-4o",
        original_prompt="Debug this code",
        conversation_history=[
            {"content": "The API endpoint is returning 500 errors", "role": "user"},
            {"content": "Let's analyze the error logs", "role": "assistant"},
            {"content": "The authentication is failing", "role": "user"}
        ]
    )
    
    result = optimizer.optimize_prompt(technical_context)
    print("⚙️ Technical Tone Test:")
    print(f"Prompt: 'Debug this code'")
    print(f"Tone: Technical")
    print(f"Result: {result}")
    print()

def test_ambiguous_handling():
    """Test how the optimizer handles ambiguous compound requests."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🔄 Testing Ambiguous Request Handling")
    print("=" * 60)
    
    ambiguous_prompts = [
        "Write a story and create a marketing plan",
        "Build an app and design the UI",
        "Analyze data and write a report",
        "Create a business plan and write content",
        "Design a logo and build a website"
    ]
    
    for prompt in ambiguous_prompts:
        context = OptimizationContext(
            task_type="text",
            model="gpt-4o",
            original_prompt=prompt
        )
        
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        is_clarification = "clarify" in result.lower() or "which aspect" in result.lower()
        
        print(f"Prompt: '{prompt}'")
        print(f"Handling: {'✅ Clarification Requested' if is_clarification else '❌ Direct Processing'}")
        print(f"Response: {result[:100]}...")
        print()

def test_edge_cases_with_context():
    """Test edge cases with conversation context."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🔍 Testing Edge Cases with Context")
    print("=" * 60)
    
    # Test very short prompts with context
    short_prompts = [
        ("Write", "Technical context"),
        ("Create", "Creative context"),
        ("Build", "Business context"),
        ("Make", "No context")
    ]
    
    for prompt, context_desc in short_prompts:
        # Create appropriate context
        if context_desc == "Technical context":
            conversation_history = [
                {"content": "I'm working on a Python project", "role": "user"},
                {"content": "What kind of help do you need?", "role": "assistant"}
            ]
        elif context_desc == "Creative context":
            conversation_history = [
                {"content": "I want to write a poem", "role": "user"},
                {"content": "What's your inspiration?", "role": "assistant"}
            ]
        elif context_desc == "Business context":
            conversation_history = [
                {"content": "I need a business strategy", "role": "user"},
                {"content": "What's your business focus?", "role": "assistant"}
            ]
        else:
            conversation_history = []
        
        context = OptimizationContext(
            task_type="text",
            model="gpt-4o",
            original_prompt=prompt,
            conversation_history=conversation_history
        )
        
        result = optimizer.optimize_prompt(context)
        
        print(f"Prompt: '{prompt}'")
        print(f"Context: {context_desc}")
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    test_conversation_context_awareness()
    test_clarification_handling()
    test_tone_awareness()
    test_ambiguous_handling()
    test_edge_cases_with_context()
    
    print("\n🎉 Enhanced Context Testing Complete!")
    print("Key improvements demonstrated:")
    print("- Conversation history analysis")
    print("- Tone-aware role assignment")
    print("- Clarification for unclear prompts")
    print("- Context-aware optimization")
    print("- Ambiguous request handling") 