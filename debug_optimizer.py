#!/usr/bin/env python3
"""
Debug script to understand pattern detection issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def debug_pattern_detection():
    """Debug the pattern detection logic"""
    
    optimizer = IntelligentPromptOptimizer()
    
    test_cases = [
        "Build a web app",
        "Create a to-do app", 
        "Write a story about robots",
        "Analyze the data trends",
        "Design a user interface",
        "Write marketing content",
        "Create documentation",
        "Write a business plan",
        "Debug this code"
    ]
    
    for prompt in test_cases:
        print(f"\n🔍 Debugging: '{prompt}'")
        
        # Check if role context should be added
        should_add = optimizer._should_add_role_context(prompt)
        print(f"Should add role context: {should_add}")
        
        # Check what role would be assigned
        role = optimizer._get_appropriate_role(prompt)
        print(f"Role assigned: {role}")
        
        # Test full optimization
        context = OptimizationContext(task_type="text", model="gpt-4o", original_prompt=prompt)
        optimized = optimizer.optimize_prompt(context)
        print(f"Final optimized: {optimized[:100]}...")

if __name__ == "__main__":
    debug_pattern_detection() 