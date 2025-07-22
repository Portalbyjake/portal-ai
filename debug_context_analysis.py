#!/usr/bin/env python3
"""
Debug the context analysis and role assignment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def debug_context_analysis():
    """Debug the context analysis and role assignment process."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🔍 Debugging Marketing Context Analysis")
    print("=" * 60)
    
    # Test case with marketing context
    test_case = {
        'prompt': 'Create a marketing strategy',
        'conversation_history': [
            {'role': 'user', 'content': 'I need to develop a marketing strategy for our new product'},
            {'role': 'assistant', 'content': 'I can help with marketing strategy development'},
            {'role': 'user', 'content': 'We want to focus on digital marketing and brand awareness'}
        ]
    }
    
    # Create context
    context = OptimizationContext(
        task_type="text",
        model="gpt-4",
        original_prompt=test_case['prompt'],
        conversation_history=test_case['conversation_history']
    )
    
    # Get context analysis
    context_analysis = optimizer._analyze_conversation_context(context)
    print(f"Context Analysis: {context_analysis}")
    
    # Debug the content analysis
    all_content = ' '.join([msg.get('content', '') for msg in test_case['conversation_history']]).lower()
    print(f"\nAll Content: {all_content}")
    
    # Check which keywords are being detected
    technical_keywords = ['code', 'programming', 'development', 'software', 'app', 'api', 'debug']
    marketing_keywords = ['marketing', 'brand', 'advertising', 'campaign', 'digital marketing', 'social media']
    sales_keywords = ['sales', 'leads', 'pipeline', 'revenue', 'sales process', 'lead generation']
    business_keywords = ['business', 'strategy', 'plan', 'analysis', 'operations', 'growth']
    
    print(f"\nTechnical keywords found: {[kw for kw in technical_keywords if kw in all_content]}")
    print(f"Marketing keywords found: {[kw for kw in marketing_keywords if kw in all_content]}")
    print(f"Sales keywords found: {[kw for kw in sales_keywords if kw in all_content]}")
    print(f"Business keywords found: {[kw for kw in business_keywords if kw in all_content]}")
    
    # Get role
    role = optimizer._get_appropriate_role(test_case['prompt'], context_analysis)
    print(f"\nRole Detected: {role}")
    
    # Check if should add role
    should_add = optimizer._should_add_role_context(test_case['prompt'], context_analysis)
    print(f"Should Add Role: {should_add}")
    
    # Get final result
    result = optimizer.optimize_prompt(context)
    print(f"Final Result: {result}")

if __name__ == "__main__":
    debug_context_analysis() 