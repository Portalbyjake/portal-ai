#!/usr/bin/env python3
"""
Test the specific role assignment logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_specific_role_assignment():
    """Test that roles are assigned correctly with specific context requirements."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🎯 Testing Specific Role Assignment")
    print("=" * 60)
    
    # Test cases with specific contexts
    test_cases = [
        # Marketing contexts - should get CMO
        {
            'name': 'Marketing Strategy Context',
            'prompt': 'Create a marketing strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to develop a marketing strategy for our new product'},
                {'role': 'assistant', 'content': 'I can help with marketing strategy development'},
                {'role': 'user', 'content': 'We want to focus on digital marketing and brand awareness'}
            ],
            'expected_role': 'CMO'
        },
        {
            'name': 'Brand Strategy Context',
            'prompt': 'Write a brand strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'We need to rebrand our company'},
                {'role': 'assistant', 'content': 'Brand strategy is important for market positioning'},
                {'role': 'user', 'content': 'We want to improve our brand recognition'}
            ],
            'expected_role': 'CMO'
        },
        
        # Sales contexts - should get Sales Director
        {
            'name': 'Sales Strategy Context',
            'prompt': 'Create a sales strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to improve our sales process'},
                {'role': 'assistant', 'content': 'Sales optimization can boost revenue'},
                {'role': 'user', 'content': 'We want to increase our sales pipeline'}
            ],
            'expected_role': 'Sales Director'
        },
        {
            'name': 'Lead Generation Context',
            'prompt': 'Write a lead generation plan',
            'conversation_history': [
                {'role': 'user', 'content': 'We need to generate more leads'},
                {'role': 'assistant', 'content': 'Lead generation is crucial for sales growth'},
                {'role': 'user', 'content': 'Our sales team needs better lead qualification'}
            ],
            'expected_role': 'Sales Director'
        },
        
        # General business strategy - should get Business Strategist
        {
            'name': 'Business Strategy Context',
            'prompt': 'Create a business strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to develop a business strategy'},
                {'role': 'assistant', 'content': 'Strategic planning is essential for business growth'},
                {'role': 'user', 'content': 'We want to expand into new markets'}
            ],
            'expected_role': 'Business Strategist'
        },
        {
            'name': 'Strategic Planning Context',
            'prompt': 'Write a strategic plan',
            'conversation_history': [
                {'role': 'user', 'content': 'We need a strategic plan for the next 5 years'},
                {'role': 'assistant', 'content': 'Long-term strategic planning requires careful analysis'},
                {'role': 'user', 'content': 'We want to optimize our business operations'}
            ],
            'expected_role': 'Business Strategist'
        },
        
        # Mixed contexts - should get no role or clarification
        {
            'name': 'Mixed Business Context',
            'prompt': 'Create a plan',
            'conversation_history': [
                {'role': 'user', 'content': 'We need to improve our business'},
                {'role': 'assistant', 'content': 'Business improvement can involve many areas'},
                {'role': 'user', 'content': 'We want to increase revenue and brand awareness'}
            ],
            'expected_role': None  # Mixed context should not get a specific role
        },
        
        # No context - should get clarification
        {
            'name': 'No Context',
            'prompt': 'Create a plan',
            'conversation_history': [],
            'expected_role': None  # Should ask for clarification
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
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
        
        # Get role
        role = optimizer._get_appropriate_role(test_case['prompt'], context_analysis)
        print(f"Role Detected: {role}")
        
        # Check if should add role
        should_add = optimizer._should_add_role_context(test_case['prompt'], context_analysis)
        print(f"Should Add Role: {should_add}")
        
        # Get final result
        result = optimizer.optimize_prompt(context)
        print(f"Final Result: {result}")
        
        # Check if result matches expectation
        if test_case['expected_role'] is None:
            if 'clarification' in result.lower() or result == test_case['prompt']:
                print("✅ Correct: No role assigned or clarification requested")
            else:
                print("❌ Incorrect: Role assigned when none expected")
        else:
            if test_case['expected_role'] in result:
                print(f"✅ Correct: {test_case['expected_role']} role assigned")
            else:
                print(f"❌ Incorrect: Expected {test_case['expected_role']}, got different result")

if __name__ == "__main__":
    test_specific_role_assignment() 