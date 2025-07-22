#!/usr/bin/env python3
"""
Test script for the new Intelligent Prompt Optimizer.
Demonstrates the task-aware, model-aligned optimization system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_intelligent_optimization():
    """Test the intelligent prompt optimization system"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🧠 Testing Intelligent Prompt Optimizer")
    print("=" * 60)
    
    # Test cases that should NOT be optimized (pass through unchanged)
    print("\n✅ Test Cases - Should Pass Through Unchanged:")
    print("-" * 40)
    
    pass_through_cases = [
        ("text", "gpt-4o", "What is the capital of France?"),
        ("text", "claude-sonnet-4", "How many people live in Tokyo?"),
        ("text", "gpt-4o", "You are a helpful assistant. Explain quantum physics."),
        ("text", "claude-sonnet-4", "Act as a data scientist and analyze this dataset."),
        ("code", "claude-3-5-sonnet", "Write a Python function to calculate fibonacci numbers."),
        ("image", "dall-e-3", "Create a photorealistic image of a cat in a garden."),
    ]
    
    for task_type, model, prompt in pass_through_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Changed: {'Yes' if optimized != prompt else 'No (✅)'}")
        print()
    
    # Test cases that SHOULD be optimized
    print("\n🔧 Test Cases - Should Be Optimized:")
    print("-" * 40)
    
    optimize_cases = [
        # Vague/verbose text prompts
        ("text", "gpt-4o", "Can you please kindly give me a helpful and informative list of suggestions if you don't mind?"),
        ("text", "claude-sonnet-4", "I was wondering if you could maybe explain something to me, you know, kind of like in a detailed way?"),
        
        # Creative writing that needs role context
        ("text", "gpt-4o", "Write a poem about money"),
        ("text", "claude-sonnet-4", "Build a to-do app"),
        ("text", "gpt-4o", "Write something uplifting"),
        
        # Code without context
        ("code", "claude-3-5-sonnet", "Build a web app"),
        ("code", "codellama-70b", "Create an API"),
        
        # Image prompts that need enhancement
        ("image", "dall-e-3", "make an image of a cat"),
        ("image", "stablediffusion", "create a fantasy landscape"),
        ("image", "anime-diffusion", "draw an anime character"),
    ]
    
    for task_type, model, prompt in optimize_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Changed: {'Yes (🔧)' if optimized != prompt else 'No'}")
        print()
    
    # Test role-based prefacing
    print("\n🎭 Test Cases - Role-Based Prefacing:")
    print("-" * 40)
    
    role_cases = [
        ("text", "gpt-4o", "Write a story about a robot"),
        ("text", "claude-sonnet-4", "Analyze the market trends"),
        ("text", "gpt-4o", "Create a motivational message"),
        ("text", "claude-sonnet-4", "Debug this code"),
        ("text", "gpt-4o", "Write a funny joke"),
    ]
    
    for task_type, model, prompt in role_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print()
    
    # Test model-specific formatting
    print("\n🎨 Test Cases - Model-Specific Formatting:")
    print("-" * 40)
    
    model_specific_cases = [
        ("code", "claude-3-5-sonnet", "Write a function"),
        ("code", "codellama-70b", "Create a class"),
        ("code", "wizardcoder", "Build an API"),
        ("image", "dall-e-3", "A cat in a garden"),
        ("image", "stablediffusion", "A fantasy castle"),
        ("image", "anime-diffusion", "A cute character"),
    ]
    
    for task_type, model, prompt in model_specific_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print()

def test_optimization_conditions():
    """Test the optimization decision logic"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🧠 Testing Optimization Conditions:")
    print("=" * 60)
    
    # Test factual queries (should not optimize)
    factual_queries = [
        "What is the capital of France?",
        "How many people live in Tokyo?",
        "What is the population of New York?",
        "When was the Declaration of Independence signed?",
        "Who is the current president?",
        "Where is the Eiffel Tower located?",
        "Define photosynthesis",
        "What is the meaning of quantum?",
        "Convert 100 degrees Fahrenheit to Celsius",
        "Calculate the area of a circle with radius 5"
    ]
    
    print("\n📊 Factual Queries (Should NOT Optimize):")
    print("-" * 40)
    
    for query in factual_queries:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=query)
        needs_optimization = optimizer._needs_optimization(context)
        print(f"Query: {query}")
        print(f"Needs Optimization: {needs_optimization}")
        print()
    
    # Test vague/verbose prompts (should optimize)
    vague_prompts = [
        "Can you please kindly give me a helpful and informative list of suggestions if you don't mind?",
        "I was wondering if you could maybe explain something to me, you know, kind of like in a detailed way?",
        "Would you be so kind as to perhaps help me with this thing I'm working on?",
        "I'm just wondering if you might be able to sort of help me out with this problem I have",
        "Maybe you could help me with something, you know what I mean?",
        "I think I need some help with this, if that's okay with you",
        "Could you possibly assist me with this task, if it's not too much trouble?"
    ]
    
    print("\n🗣️ Vague/Verbose Prompts (Should Optimize):")
    print("-" * 40)
    
    for prompt in vague_prompts:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        needs_optimization = optimizer._needs_optimization(context)
        optimized = optimizer.optimize_prompt(context)
        print(f"Original: {prompt}")
        print(f"Needs Optimization: {needs_optimization}")
        print(f"Optimized: {optimized}")
        print()

def test_role_detection():
    """Test role detection and assignment"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🎭 Testing Role Detection:")
    print("=" * 60)
    
    role_test_cases = [
        ("Write a story about a robot", "creative_writer"),
        ("Analyze the market trends", "business_analyst"),
        ("Create a motivational message", "motivational_coach"),
        ("Debug this code", "code_reviewer"),
        ("Write a funny joke", "comedian"),
        ("Compose a poem about love", "poet"),
        ("Design a scalable architecture", "system_architect"),
        ("Deploy a Docker container", "devops_engineer"),
        ("Write a technical manual", "technical_writer"),
        ("Create marketing copy", "copywriter"),
        ("Investigate the data patterns", "data_scientist"),
        ("Research the latest trends", "researcher"),
    ]
    
    for prompt, expected_role in role_test_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Prompt: {prompt}")
        print(f"Expected Role: {expected_role}")
        print(f"Optimized: {optimized}")
        print()

def test_model_specific_behavior():
    """Test model-specific optimization behavior"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🤖 Testing Model-Specific Behavior:")
    print("=" * 60)
    
    # Test the same prompt across different models
    test_prompt = "Explain how machine learning works"
    
    models_to_test = [
        "gpt-4o",
        "claude-sonnet-4", 
        "claude-haiku",
        "gemini-pro",
        "gpt-4-turbo"
    ]
    
    for model in models_to_test:
        context = OptimizationContext(task_type="text", model=model, original_prompt=test_prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Model: {model}")
        print(f"Original: {test_prompt}")
        print(f"Optimized: {optimized}")
        print()

if __name__ == "__main__":
    test_intelligent_optimization()
    test_optimization_conditions()
    test_role_detection()
    test_model_specific_behavior()
    
    print("\n🎉 Intelligent Prompt Optimizer Test Complete!")
    print("The system now intelligently optimizes prompts only when it clearly improves clarity, output quality, or model performance.") 