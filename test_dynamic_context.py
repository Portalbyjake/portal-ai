#!/usr/bin/env python3
"""
Test the dynamic context analysis system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_dynamic_context_analysis():
    """Test the dynamic context analysis that analyzes full conversation intent."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🧠 Testing Dynamic Context Analysis")
    print("=" * 60)
    
    # Test cases that demonstrate full context analysis
    test_cases = [
        # Marketing context - should detect marketing despite "development" word
        {
            'name': 'Marketing Strategy Development',
            'prompt': 'Create a marketing strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to develop a marketing strategy for our new product'},
                {'role': 'assistant', 'content': 'I can help with marketing strategy development'},
                {'role': 'user', 'content': 'We want to focus on digital marketing and brand awareness'}
            ],
            'expected_intent': 'marketing',
            'expected_role': 'CMO'
        },
        
        # Sales context - should detect sales despite "strategy" word
        {
            'name': 'Sales Strategy Planning',
            'prompt': 'Create a sales strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to improve our sales process'},
                {'role': 'assistant', 'content': 'Sales strategy optimization can boost revenue'},
                {'role': 'user', 'content': 'We want to increase our sales pipeline and lead generation'}
            ],
            'expected_intent': 'sales',
            'expected_role': 'Sales Director'
        },
        
        # Technical context - should detect technical despite "strategy" word
        {
            'name': 'Software Development Strategy',
            'prompt': 'Create a development plan',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to develop a web application'},
                {'role': 'assistant', 'content': 'Software development strategy is important'},
                {'role': 'user', 'content': 'We want to use React and Node.js for the coding'}
            ],
            'expected_intent': 'technical',
            'expected_role': 'Senior Developer'
        },
        
        # Business context - should detect business despite "strategy" word
        {
            'name': 'Business Strategy Planning',
            'prompt': 'Create a business strategy',
            'conversation_history': [
                {'role': 'user', 'content': 'I need to develop a business strategy'},
                {'role': 'assistant', 'content': 'Strategic planning is essential for business growth'},
                {'role': 'user', 'content': 'We want to expand into new markets and optimize operations'}
            ],
            'expected_intent': 'business',
            'expected_role': 'Business Strategist'
        },
        
        # Mixed context - should not assign role
        {
            'name': 'Mixed Business Context',
            'prompt': 'Create a plan',
            'conversation_history': [
                {'role': 'user', 'content': 'We need to improve our business'},
                {'role': 'assistant', 'content': 'Business improvement can involve many areas'},
                {'role': 'user', 'content': 'We want to increase revenue and brand awareness'}
            ],
            'expected_intent': None,
            'expected_role': None
        },
        
        # No context - should not assign role
        {
            'name': 'No Context',
            'prompt': 'Create a plan',
            'conversation_history': [],
            'expected_intent': None,
            'expected_role': None
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
        
        # Get domain scores
        domain_scores = context_analysis.get('domain_scores', {})
        print(f"Domain Scores: {domain_scores}")
        
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
    test_dynamic_context_analysis() 