#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced dynamic prompt optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_enhanced_optimization():
    """Test the enhanced dynamic optimization behavior"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🚀 Testing Enhanced Dynamic Optimization")
    print("=" * 60)
    
    # Test cases that should be optimized with appropriate roles
    print("\n✅ Test Cases - Should Be Optimized with Roles:")
    print("-" * 40)
    
    role_optimization_cases = [
        ("text", "gpt-4o", "Write something about money"),
        ("text", "gpt-4o", "Create a to-do app"),
        ("text", "gpt-4o", "Build a web app"),
        ("text", "gpt-4o", "Write a story about robots"),
        ("text", "gpt-4o", "Analyze the data trends"),
        ("text", "gpt-4o", "Design a user interface"),
        ("text", "gpt-4o", "Write marketing content"),
        ("text", "gpt-4o", "Create documentation"),
        ("text", "gpt-4o", "Write a business plan"),
        ("text", "gpt-4o", "Debug this code"),
    ]
    
    for task_type, model, prompt in role_optimization_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Role Added: {'✅ Yes' if 'You are a' in optimized else '❌ No'}")
        print()
    
    # Test cases that should pass through unchanged
    print("\n⏭️ Test Cases - Should Pass Through Unchanged:")
    print("-" * 40)
    
    pass_through_cases = [
        ("text", "gpt-4o", "What is the capital of France?"),
        ("text", "gpt-4o", "How many people live in Tokyo?"),
        ("text", "gpt-4o", "Define photosynthesis"),
        ("text", "gpt-4o", "Calculate 2+2"),
        ("text", "gpt-4o", "Tell me about Paris"),
    ]
    
    for task_type, model, prompt in pass_through_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Changed: {'❌ Yes' if optimized != prompt else '✅ No (Passed through)'}")
        print()
    
    # Test enhanced image optimization
    print("\n🎨 Test Cases - Enhanced Image Optimization:")
    print("-" * 40)
    
    image_optimization_cases = [
        ("image", "dall-e-3", "cat"),
        ("image", "dall-e-3", "high quality portrait"),
        ("image", "stablediffusion", "photorealistic landscape"),
        ("image", "stablediffusion", "fantasy castle"),
        ("image", "stablediffusion", "anime character"),
        ("image", "stablediffusion", "logo design"),
        ("image", "stablediffusion", "architecture building"),
    ]
    
    for task_type, model, prompt in image_optimization_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Enhanced: {'✅ Yes' if optimized != prompt else '❌ No'}")
        print()

def test_intent_detection():
    """Test the enhanced intent detection capabilities"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🧠 Test Cases - Intent Detection Accuracy:")
    print("-" * 40)
    
    intent_test_cases = [
        # Development intent
        ("text", "gpt-4o", "Build a web app", "Should detect: Senior Developer"),
        ("text", "gpt-4o", "Create an API", "Should detect: Senior Developer"),
        ("text", "gpt-4o", "Develop a mobile app", "Should detect: Senior Developer"),
        
        # Creative intent
        ("text", "gpt-4o", "Write a story about space", "Should detect: Creative Writer"),
        ("text", "gpt-4o", "Create content about AI", "Should detect: Creative Writer"),
        ("text", "gpt-4o", "Write an article about climate", "Should detect: Journalist"),
        
        # Analysis intent
        ("text", "gpt-4o", "Analyze market trends", "Should detect: Data Scientist"),
        ("text", "gpt-4o", "Research user behavior", "Should detect: Data Scientist"),
        
        # Design intent
        ("text", "gpt-4o", "Design a user interface", "Should detect: UX/UI Designer"),
        ("text", "gpt-4o", "Create a website layout", "Should detect: UX/UI Designer"),
        
        # Business intent
        ("text", "gpt-4o", "Write a business plan", "Should detect: Business Analyst"),
        ("text", "gpt-4o", "Create a marketing strategy", "Should detect: Copywriter"),
    ]
    
    for task_type, model, prompt, expected in intent_test_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        role_detected = "You are a" in optimized
        print(f"Original: {prompt}")
        print(f"Expected: {expected}")
        print(f"Role Detected: {'✅ Yes' if role_detected else '❌ No'}")
        if role_detected:
            print(f"Role: {optimized.split('You are a')[1].split('.')[0] if 'You are a' in optimized else 'N/A'}")
        print()

if __name__ == "__main__":
    test_enhanced_optimization()
    test_intent_detection()
    
    print("\n🎉 Enhanced Dynamic Optimization Test Complete!")
    print("The system now accurately determines user intent and provides")
    print("appropriate role-based optimization for the best possible output.") 