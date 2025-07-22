#!/usr/bin/env python3
"""
Test the full context analysis system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext

def test_full_context_analysis():
    """Test how the system analyzes full context including conversation history."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("🧠 Testing Full Context Analysis")
    print("=" * 60)
    
    # Test cases with conversation context
    test_cases = [
        # Technical context
        {
            'prompt': "Create a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I need help with my Python project'},
                {'role': 'assistant', 'content': 'I can help with Python development'},
                {'role': 'user', 'content': 'I want to build a web app'}
            ],
            'expected': 'Should get technical role due to Python/web app context'
        },
        
        # Business context
        {
            'prompt': "Create a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I need help with my startup'},
                {'role': 'assistant', 'content': 'I can help with business strategy'},
                {'role': 'user', 'content': 'I want to improve my marketing'}
            ],
            'expected': 'Should get business role due to startup/marketing context'
        },
        
        # Creative context
        {
            'prompt': "Create a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I want to write a novel'},
                {'role': 'assistant', 'content': 'I can help with creative writing'},
                {'role': 'user', 'content': 'I need help with character development'}
            ],
            'expected': 'Should get creative role due to novel/writing context'
        },
        
        # Ambiguous without context
        {
            'prompt': "Create a plan",
            'conversation_history': [],
            'expected': 'Should ask for clarification - no context'
        },
        
        # Mixed context (should be conservative)
        {
            'prompt': "Create a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I need help with my Python project'},
                {'role': 'assistant', 'content': 'I can help with development'},
                {'role': 'user', 'content': 'But I also want to write a story'}
            ],
            'expected': 'Should be conservative - mixed technical/creative context'
        }
    ]
    
    print("\n🔍 Full Context Analysis Results:")
    print("-" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Prompt: '{case['prompt']}'")
        print(f"   Context: {case['expected']}")
        
        # Create context
        context = OptimizationContext(
            task_type="text",
            model="gpt-4o",
            original_prompt=case['prompt'],
            conversation_history=case['conversation_history']
        )
        
        # Get result
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        needs_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        # Check if role was assigned
        has_role = "You are a" in result
        
        # Check if unchanged
        is_unchanged = result == case['prompt']
        
        if needs_clarification:
            status = "✅ Clarification Requested"
        elif has_role:
            status = "🎯 Role Assigned"
        elif is_unchanged:
            status = "➡️ Unchanged"
        else:
            status = "❓ Other"
        
        print(f"   Result: {status}")
        print(f"   Output: {result[:100]}...")
    
    print(f"\n📊 Summary:")
    print("The system now analyzes:")
    print("- Full conversation history")
    print("- Ongoing context (technical/business/creative)")
    print("- Tone and style")
    print("- Pattern matches WITH context requirements")
    print("- Excluded contexts to avoid incorrect assignments")

def test_specific_context_cases():
    """Test specific cases where context should determine role assignment."""
    
    optimizer = IntelligentPromptOptimizer()
    
    print("\n🎯 Testing Specific Context Cases")
    print("=" * 60)
    
    # Cases where the same prompt should get different roles based on context
    context_cases = [
        # "Write a plan" in technical context
        {
            'prompt': "Write a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I need to build a web application'},
                {'role': 'assistant', 'content': 'I can help with web development'},
                {'role': 'user', 'content': 'I want to use React and Node.js'}
            ],
            'expected_role': 'Senior Developer',
            'description': 'Technical context should lead to developer role'
        },
        
        # "Write a plan" in business context
        {
            'prompt': "Write a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I need help with my business strategy'},
                {'role': 'assistant', 'content': 'I can help with business planning'},
                {'role': 'user', 'content': 'I want to improve my marketing'}
            ],
            'expected_role': 'Business Analyst',
            'description': 'Business context should lead to business analyst role'
        },
        
        # "Write a plan" in creative context
        {
            'prompt': "Write a plan",
            'conversation_history': [
                {'role': 'user', 'content': 'I want to write a novel'},
                {'role': 'assistant', 'content': 'I can help with creative writing'},
                {'role': 'user', 'content': 'I need help with plot development'}
            ],
            'expected_role': 'Creative Writer',
            'description': 'Creative context should lead to creative writer role'
        },
        
        # "Write a plan" with no context
        {
            'prompt': "Write a plan",
            'conversation_history': [],
            'expected_role': 'Clarification',
            'description': 'No context should lead to clarification request'
        }
    ]
    
    print("\n🎯 Context-Based Role Assignment:")
    print("-" * 50)
    
    correct_count = 0
    
    for case in context_cases:
        print(f"\nPrompt: '{case['prompt']}'")
        print(f"Context: {case['description']}")
        print(f"Expected: {case['expected_role']}")
        
        # Create context
        context = OptimizationContext(
            task_type="text",
            model="gpt-4o",
            original_prompt=case['prompt'],
            conversation_history=case['conversation_history']
        )
        
        # Get result
        result = optimizer.optimize_prompt(context)
        
        # Check if clarification was requested
        needs_clarification = "clarify" in result.lower() or "could you" in result.lower()
        
        # Check if role was assigned
        has_role = "You are a" in result
        
        # Determine actual role
        if needs_clarification:
            actual_role = "Clarification"
        elif has_role:
            # Extract role from result
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
        if case['expected_role'] == "Clarification" and needs_clarification:
            is_correct = True
        elif case['expected_role'] in actual_role:
            is_correct = True
        
        if is_correct:
            correct_count += 1
            status = "✅ Correct"
        else:
            status = "❌ Incorrect"
        
        print(f"Actual: {actual_role}")
        print(f"Status: {status}")
        print(f"Result: {result[:80]}...")
    
    print(f"\n📊 Accuracy: {correct_count}/{len(context_cases)} ({correct_count/len(context_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_full_context_analysis()
    test_specific_context_cases()
    
    print("\n🎉 Full Context Analysis Testing Complete!")
    print("The system now:")
    print("- Analyzes entire conversation history")
    print("- Considers ongoing context (technical/business/creative)")
    print("- Requires appropriate context for role assignment")
    print("- Excludes inappropriate contexts")
    print("- Makes conservative decisions when context is mixed") 