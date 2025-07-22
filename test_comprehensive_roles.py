#!/usr/bin/env python3
"""
Comprehensive test to catch potential role assignment errors and edge cases.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_comprehensive_role_assignment():
    """Test comprehensive role assignment scenarios"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🧪 Comprehensive Role Assignment Testing")
    print("=" * 60)
    
    # Test cases with potential role assignment issues
    test_cases = [
        # Marketing & Business
        ("Create a marketing strategy", "CMO", "Chief Marketing Officer"),
        ("Write marketing copy", "Copywriter", "Professional Copywriter"),
        ("Develop a business plan", "Business Analyst", "Business Analyst"),
        ("Create a sales strategy", "CMO", "Chief Marketing Officer"),
        ("Write advertising content", "Copywriter", "Professional Copywriter"),
        ("Design a marketing campaign", "CMO", "Chief Marketing Officer"),
        ("Create promotional materials", "Copywriter", "Professional Copywriter"),
        
        # Development & Technical
        ("Build a web app", "Senior Developer", "Senior Software Developer"),
        ("Create an API", "Senior Developer", "Senior Software Developer"),
        ("Develop a mobile app", "Senior Developer", "Senior Software Developer"),
        ("Design a system architecture", "System Architect", "Senior System Architect"),
        ("Debug this code", "Code Reviewer", "Experienced Code Reviewer"),
        ("Review this code", "Code Reviewer", "Experienced Code Reviewer"),
        ("Optimize this code", "Code Reviewer", "Experienced Code Reviewer"),
        ("Refactor this code", "Code Reviewer", "Experienced Code Reviewer"),
        
        # Creative Writing
        ("Write a story", "Creative Writer", "Creative Writer"),
        ("Write a poem", "Poet", "Poet"),
        ("Write lyrics", "Songwriter", "Songwriter"),
        ("Write a script", "Screenwriter", "Screenwriter"),
        ("Write a novel", "Creative Writer", "Creative Writer"),
        ("Write content", "Creative Writer", "Creative Writer"),
        ("Write an article", "Journalist", "Professional Journalist"),
        ("Write a blog post", "Creative Writer", "Creative Writer"),
        ("Write documentation", "Technical Writer", "Technical Writer"),
        
        # Analysis & Research
        ("Analyze data", "Data Scientist", "Data Scientist"),
        ("Research trends", "Data Scientist", "Data Scientist"),
        ("Study patterns", "Data Scientist", "Data Scientist"),
        ("Investigate market", "Data Scientist", "Data Scientist"),
        ("Analyze user behavior", "Data Scientist", "Data Scientist"),
        
        # Design & UX
        ("Design a user interface", "UX/UI Designer", "UX/UI Designer"),
        ("Create a website layout", "Senior Developer", "Senior Software Developer"),
        ("Design a logo", "UX/UI Designer", "UX/UI Designer"),
        ("Create a brand identity", "UX/UI Designer", "UX/UI Designer"),
        
        # Business Strategy
        ("Create a business strategy", "Business Analyst", "Business Analyst"),
        ("Write a business proposal", "Business Analyst", "Business Analyst"),
        ("Develop a strategic plan", "Business Analyst", "Business Analyst"),
        ("Create a business model", "Business Analyst", "Business Analyst"),
        
        # Edge cases and potential issues
        ("Write something about money", "Creative Writer", "Creative Writer"),
        ("Create something amazing", "Creative Writer", "Creative Writer"),
        ("Build something cool", "Senior Developer", "Senior Software Developer"),
        ("Make something useful", "Senior Developer", "Senior Software Developer"),
        
        # Specific vs General patterns
        ("Write a technical article", "Journalist", "Professional Journalist"),
        ("Write technical documentation", "Technical Writer", "Technical Writer"),
        ("Write a business article", "Journalist", "Professional Journalist"),
        ("Write a marketing article", "Journalist", "Professional Journalist"),
        
        # Ambiguous cases that might need clarification
        ("Create a plan", "Business Analyst", "Business Analyst"),
        ("Write a report", "Journalist", "Professional Journalist"),
        ("Build a solution", "Senior Developer", "Senior Software Developer"),
        ("Design a solution", "UX/UI Designer", "UX/UI Designer"),
    ]
    
    print("\n✅ Testing Role Assignment Accuracy:")
    print("-" * 40)
    
    for prompt, expected_role_type, expected_role_name in test_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        
        # Check if role was assigned
        has_role = "You are a" in optimized
        
        # Extract the actual role if present
        actual_role = "None"
        if has_role:
            role_start = optimized.find("You are a")
            if role_start != -1:
                role_end = optimized.find(".", role_start)
                if role_end != -1:
                    actual_role = optimized[role_start:role_end + 1]
        
        # Check if the role matches expectations
        role_correct = expected_role_name.lower() in actual_role.lower()
        
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected_role_type} ({expected_role_name})")
        print(f"Actual: {actual_role}")
        print(f"Role Assigned: {'✅ Yes' if has_role else '❌ No'}")
        print(f"Role Correct: {'✅ Yes' if role_correct else '❌ No'}")
        print()

def test_edge_cases():
    """Test edge cases that might cause issues"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🔍 Testing Edge Cases:")
    print("-" * 40)
    
    edge_cases = [
        # Very short prompts
        ("Write", "Should pass through or get general role"),
        ("Create", "Should pass through or get general role"),
        ("Build", "Should pass through or get general role"),
        
        # Very long prompts
        ("Write a comprehensive analysis of the current market trends and provide detailed recommendations for strategic planning and implementation of business solutions", "Should get appropriate role"),
        
        # Mixed case
        ("WRITE A STORY", "Should work with case insensitive matching"),
        ("Create An App", "Should work with case insensitive matching"),
        
        # Special characters
        ("Write a story!", "Should handle punctuation"),
        ("Create an app?", "Should handle punctuation"),
        ("Build something...", "Should handle punctuation"),
        
        # Numbers and symbols
        ("Write a 5-step plan", "Should handle numbers"),
        ("Create a v2.0 app", "Should handle version numbers"),
        ("Build a 24/7 system", "Should handle numbers"),
        
        # Ambiguous terms
        ("Write about technology", "Should get creative writer"),
        ("Create for business", "Should get appropriate role"),
        ("Build with Python", "Should get senior developer"),
        
        # Compound requests
        ("Write a story and create a marketing plan", "Should prioritize primary intent"),
        ("Build an app and design the UI", "Should prioritize primary intent"),
        ("Analyze data and write a report", "Should prioritize primary intent"),
    ]
    
    for prompt, expected_behavior in edge_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        
        has_role = "You are a" in optimized
        role_assigned = "None"
        if has_role:
            role_start = optimized.find("You are a")
            if role_start != -1:
                role_end = optimized.find(".", role_start)
                if role_end != -1:
                    role_assigned = optimized[role_start:role_end + 1]
        
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected_behavior}")
        print(f"Role: {role_assigned}")
        print(f"Optimized: {optimized[:100]}...")
        print()

def test_conflicting_patterns():
    """Test patterns that might conflict with each other"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n⚡ Testing Conflicting Patterns:")
    print("-" * 40)
    
    conflicting_cases = [
        # Marketing + Technical
        ("Write marketing code", "Should prioritize based on context"),
        ("Create technical marketing", "Should prioritize based on context"),
        ("Build marketing software", "Should prioritize based on context"),
        
        # Business + Creative
        ("Write business poetry", "Should prioritize based on context"),
        ("Create artistic business", "Should prioritize based on context"),
        ("Design business art", "Should prioritize based on context"),
        
        # Analysis + Creative
        ("Write analytical stories", "Should prioritize based on context"),
        ("Create data poetry", "Should prioritize based on context"),
        ("Analyze creative content", "Should prioritize based on context"),
        
        # Development + Design
        ("Build design code", "Should prioritize based on context"),
        ("Create code design", "Should prioritize based on context"),
        ("Develop UI architecture", "Should prioritize based on context"),
    ]
    
    for prompt, expected_behavior in conflicting_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        
        has_role = "You are a" in optimized
        role_assigned = "None"
        if has_role:
            role_start = optimized.find("You are a")
            if role_start != -1:
                role_end = optimized.find(".", role_start)
                if role_end != -1:
                    role_assigned = optimized[role_start:role_end + 1]
        
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected_behavior}")
        print(f"Role: {role_assigned}")
        print(f"Optimized: {optimized[:100]}...")
        print()

def test_role_specificity():
    """Test that specific roles are assigned for specific requests"""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🎯 Testing Role Specificity:")
    print("-" * 40)
    
    specificity_tests = [
        # Very specific requests should get specific roles
        ("Write a poem about love", "Poet", "Should get Poet role"),
        ("Compose a song about life", "Songwriter", "Should get Songwriter role"),
        ("Write a screenplay", "Screenwriter", "Should get Screenwriter role"),
        ("Debug this Python code", "Code Reviewer", "Should get Code Reviewer role"),
        ("Review this JavaScript", "Code Reviewer", "Should get Code Reviewer role"),
        ("Design system architecture", "System Architect", "Should get System Architect role"),
        ("Deploy with Docker", "DevOps Engineer", "Should get DevOps Engineer role"),
        
        # General requests should get appropriate general roles
        ("Write something creative", "Creative Writer", "Should get Creative Writer role"),
        ("Create a business document", "Business Analyst", "Should get Business Analyst role"),
        ("Build a software solution", "Senior Developer", "Should get Senior Developer role"),
        ("Analyze some data", "Data Scientist", "Should get Data Scientist role"),
    ]
    
    for prompt, expected_role, expected_behavior in specificity_tests:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        
        has_role = "You are a" in optimized
        role_assigned = "None"
        if has_role:
            role_start = optimized.find("You are a")
            if role_start != -1:
                role_end = optimized.find(".", role_start)
                if role_end != -1:
                    role_assigned = optimized[role_start:role_end + 1]
        
        # Check if the expected role is in the assigned role
        role_correct = expected_role.lower() in role_assigned.lower()
        
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected_role}")
        print(f"Actual: {role_assigned}")
        print(f"Correct: {'✅ Yes' if role_correct else '❌ No'}")
        print(f"Behavior: {expected_behavior}")
        print()

if __name__ == "__main__":
    test_comprehensive_role_assignment()
    test_edge_cases()
    test_conflicting_patterns()
    test_role_specificity()
    
    print("\n🎉 Comprehensive Role Testing Complete!")
    print("This test helps identify:")
    print("- Incorrect role assignments")
    print("- Edge cases that need handling")
    print("- Conflicting pattern detection")
    print("- Role specificity issues")
    print("- Potential improvements needed") 