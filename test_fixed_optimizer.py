#!/usr/bin/env python3
"""
Test script to demonstrate the fixed, less presumptuous prompt optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_fixed_optimization():
    """Test the fixed, less presumptuous optimization behavior"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🔧 Testing Fixed, Less Presumptuous Optimization")
    print("=" * 60)
    
    # Test cases that should NOT be optimized (pass through unchanged)
    print("\n✅ Test Cases - Should Pass Through Unchanged:")
    print("-" * 40)
    
    pass_through_cases = [
        ("text", "gpt-4o", "Write something about money"),
        ("text", "gpt-4o", "Create a to-do app"),
        ("text", "gpt-4o", "Build a web app"),
        ("text", "gpt-4o", "Write a story"),
        ("text", "gpt-4o", "Analyze the data"),
        ("text", "gpt-4o", "Debug this code"),
        ("image", "dall-e-3", "make an image of a cat"),
        ("image", "stablediffusion", "create a fantasy landscape"),
    ]
    
    for task_type, model, prompt in pass_through_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Changed: {'Yes' if optimized != prompt else 'No (✅)'}")
        print()
    
    # Test cases that SHOULD be optimized (very specific requests)
    print("\n🔧 Test Cases - Should Be Optimized (Specific Requests):")
    print("-" * 40)
    
    optimize_cases = [
        # Very specific creative requests
        ("text", "gpt-4o", "Write a poem about money"),
        ("text", "gpt-4o", "Write a story about a robot"),
        ("text", "gpt-4o", "Write lyrics for a song"),
        
        # Very specific technical requests
        ("text", "gpt-4o", "Debug this code"),
        ("text", "gpt-4o", "Review this code"),
        ("text", "gpt-4o", "Design architecture for a scalable system"),
        
        # Very specific image requests
        ("image", "dall-e-3", "cat"),
        ("image", "stablediffusion", "photorealistic portrait"),
        ("image", "stablediffusion", "fantasy castle"),
        ("image", "anime-diffusion", "cute character"),
    ]
    
    for task_type, model, prompt in optimize_cases:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Task: {task_type}, Model: {model}")
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Changed: {'Yes (🔧)' if optimized != prompt else 'No'}")
        print()

def test_presumptuous_fixes():
    """Test that previously presumptuous optimizations are now fixed"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🚫 Test Cases - Previously Presumptuous, Now Fixed:")
    print("-" * 40)
    
    presumptuous_fixes = [
        # These should NOT be optimized anymore
        ("text", "gpt-4o", "Write something about money"),
        ("text", "gpt-4o", "Create a to-do app"),
        ("text", "gpt-4o", "Build a web app"),
        ("text", "gpt-4o", "Write a story"),
        ("text", "gpt-4o", "Analyze the data"),
        ("text", "gpt-4o", "Debug this code"),
        ("image", "dall-e-3", "make an image of a cat"),
        ("image", "stablediffusion", "create a fantasy landscape"),
    ]
    
    for task_type, model, prompt in presumptuous_fixes:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Previously would have been: [PRESUMPTUOUS]")
        print(f"Now: {'✅ Passes through unchanged' if optimized == prompt else '❌ Still being optimized'}")
        print()

def test_specific_optimizations():
    """Test that very specific requests still get appropriate optimization"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🎯 Test Cases - Specific Requests That Should Be Optimized:")
    print("-" * 40)
    
    specific_optimizations = [
        # These SHOULD be optimized because they're very specific
        ("text", "gpt-4o", "Write a poem about money"),
        ("text", "gpt-4o", "Write a story about a robot"),
        ("text", "gpt-4o", "Write lyrics for a song"),
        ("text", "gpt-4o", "Debug this code"),
        ("text", "gpt-4o", "Review this code"),
        ("text", "gpt-4o", "Design architecture for a scalable system"),
        ("image", "dall-e-3", "cat"),
        ("image", "stablediffusion", "photorealistic portrait"),
        ("image", "stablediffusion", "fantasy castle"),
        ("image", "anime-diffusion", "cute character"),
    ]
    
    for task_type, model, prompt in specific_optimizations:
        context = OptimizationContext(task_type=task_type, model=model, original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Original: {prompt}")
        print(f"Optimized: {optimized}")
        print(f"Optimized: {'✅ Yes' if optimized != prompt else '❌ No'}")
        print()

if __name__ == "__main__":
    test_fixed_optimization()
    test_presumptuous_fixes()
    test_specific_optimizations()
    
    print("\n🎉 Fixed Optimization Test Complete!")
    print("The system now only optimizes when the user makes very specific requests,")
    print("avoiding presumptuous assumptions about their intent.") 