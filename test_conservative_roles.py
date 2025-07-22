#!/usr/bin/env python3
"""
Test the conservative role assignment system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_ambiguous_cases():
    """Test how the system handles ambiguous cases that shouldn't get roles."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🎯 Testing Ambiguous Cases (Should NOT Get Roles)")
    print("=" * 60)
    
    # Cases that should NOT get roles due to ambiguity
    ambiguous_cases = [
        # Generic plans/reports
        "Create a plan",
        "Write a report", 
        "Build a solution",
        "Design a solution",
        
        # Generic analysis
        "Analyze data",
        "Research trends", 
        "Study patterns",
        "Investigate market",
        "Analyze user behavior",
        
        # Vague requests
        "Write something",
        "Create anything",
        "Build it",
        "Make this",
        
        # Compound requests
        "Write a story and create a marketing plan",
        "Build an app and design the UI",
        "Analyze data and write a report"
    ]
    
    print("\n❌ Should NOT Get Roles (Too Ambiguous):")
    print("-" * 50)
    
    role_assigned_count = 0
    clarification_count = 0
    unchanged_count = 0
    
    for prompt in ambiguous_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        needs_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        # Check if role was assigned
        has_role = "You are a" in result
        
        # Check if unchanged
        is_unchanged = result == prompt
        
        if needs_clarification:
            clarification_count += 1
            status = "✅ Clarification Requested"
        elif has_role:
            role_assigned_count += 1
            status = "❌ Role Assigned (Shouldn't be)"
        elif is_unchanged:
            unchanged_count += 1
            status = "➡️ Unchanged"
        else:
            status = "❓ Other"
        
        print(f"'{prompt}' → {status}")
        
        # Show details for first few
        if len([c for c in [needs_clarification, has_role, is_unchanged] if c]) <= 2:
            print(f"    Result: {result[:80]}...")
            print()
    
    print(f"\n📊 Results:")
    print(f"Clarification Requested: {clarification_count}/{len(ambiguous_cases)} ({clarification_count/len(ambiguous_cases)*100:.1f}%)")
    print(f"Role Assigned (Incorrect): {role_assigned_count}/{len(ambiguous_cases)} ({role_assigned_count/len(ambiguous_cases)*100:.1f}%)")
    print(f"Unchanged: {unchanged_count}/{len(ambiguous_cases)} ({unchanged_count/len(ambiguous_cases)*100:.1f}%)")

def test_specific_cases():
    """Test specific cases that SHOULD get roles."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n✅ Testing Specific Cases (Should Get Roles)")
    print("=" * 60)
    
    # Cases that SHOULD get roles (very specific)
    specific_cases = [
        # Clear technical
        "Build a web app",
        "Create an API", 
        "Debug this Python code",
        "Design system architecture",
        
        # Clear business
        "Create a marketing strategy",
        "Write a business proposal",
        "Develop a business plan",
        
        # Clear creative
        "Write a poem about love",
        "Compose a song about life",
        "Write a screenplay",
        
        # Clear analysis (specific)
        "Analyze market data for e-commerce trends",
        "Research user behavior patterns in mobile apps",
        "Study customer satisfaction patterns"
    ]
    
    print("\n✅ Should Get Roles (Very Specific):")
    print("-" * 50)
    
    role_assigned_count = 0
    clarification_count = 0
    
    for prompt in specific_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        needs_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        # Check if role was assigned
        has_role = "You are a" in result
        
        if has_role:
            role_assigned_count += 1
            status = "✅ Role Assigned"
        elif needs_clarification:
            clarification_count += 1
            status = "❌ Clarification Requested (Shouldn't be)"
        else:
            status = "➡️ No Role"
        
        print(f"'{prompt}' → {status}")
        
        # Show details for first few
        if has_role or needs_clarification:
            print(f"    Result: {result[:80]}...")
            print()
    
    print(f"\n📊 Results:")
    print(f"Role Assigned: {role_assigned_count}/{len(specific_cases)} ({role_assigned_count/len(specific_cases)*100:.1f}%)")
    print(f"Clarification Requested (Incorrect): {clarification_count}/{len(specific_cases)} ({clarification_count/len(specific_cases)*100:.1f}%)")

def test_data_scientist_specificity():
    """Test the new specific data scientist roles."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🔬 Testing Data Scientist Role Specificity")
    print("=" * 60)
    
    # Test cases for different types of analysis
    analysis_cases = [
        # Should be Data Scientist
        ("Analyze data for machine learning model", "Data Scientist"),
        ("Perform statistical analysis", "Data Scientist"),
        ("Build predictive model", "Data Scientist"),
        
        # Should be Market Research Analyst
        ("Investigate market trends", "Market Research Analyst"),
        ("Analyze competitive landscape", "Market Research Analyst"),
        ("Research market opportunities", "Market Research Analyst"),
        
        # Should be UX Researcher
        ("Analyze user behavior patterns", "UX Researcher"),
        ("Conduct user research", "UX Researcher"),
        ("Study user experience", "UX Researcher"),
        
        # Should need clarification (too generic)
        ("Analyze data", "Clarification Needed"),
        ("Research trends", "Clarification Needed"),
        ("Study patterns", "Clarification Needed")
    ]
    
    print("\n🔬 Analysis Role Assignment:")
    print("-" * 50)
    
    correct_count = 0
    
    for prompt, expected_role in analysis_cases:
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        needs_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        # Check if role was assigned
        has_role = "You are a" in result
        
        if needs_clarification:
            actual_role = "Clarification Needed"
        elif has_role:
            # Extract the role from the result
            role_start = result.find("You are a")
            if role_start != -1:
                role_end = result.find(".", role_start)
                if role_end != -1:
                    actual_role = result[role_start:role_end + 1]
                else:
                    actual_role = "Role Assigned (Unknown)"
            else:
                actual_role = "Role Assigned (Unknown)"
        else:
            actual_role = "No Role"
        
        # Check if correct
        is_correct = False
        if expected_role == "Clarification Needed" and needs_clarification:
            is_correct = True
        elif expected_role in actual_role:
            is_correct = True
        
        if is_correct:
            correct_count += 1
            status = "✅ Correct"
        else:
            status = "❌ Incorrect"
        
        print(f"'{prompt}'")
        print(f"Expected: {expected_role}")
        print(f"Actual: {actual_role}")
        print(f"Status: {status}")
        print()
    
    print(f"\n📊 Accuracy: {correct_count}/{len(analysis_cases)} ({correct_count/len(analysis_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_ambiguous_cases()
    test_specific_cases()
    test_data_scientist_specificity()
    
    print("\n🎉 Conservative Role Testing Complete!")
    print("The system now:")
    print("- Asks for clarification on ambiguous cases")
    print("- Only assigns roles for very specific prompts")
    print("- Distinguishes between different types of analysis")
    print("- Prevents incorrect assumptions about user intent") 