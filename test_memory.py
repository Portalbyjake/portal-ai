#!/usr/bin/env python3

from memory import memory_manager
import json

def test_memory_system():
    """Test the memory system directly"""
    
    # Clear memory first
    memory_manager.clear_memory("test_user")
    
    # Simulate storing a conversation
    print("=== Storing conversation ===")
    memory_manager.save_memory("test_user", "user", "What is the capital of France?", "text")
    memory_manager.save_memory("test_user", "assistant", "The capital of France is Paris.", "text", "gpt-4o")
    
    # Check if follow-up detection works
    followup_question = "How many people live there?"
    is_followup = memory_manager.is_followup_question(followup_question, "test_user")
    print(f"Question: '{followup_question}'")
    print(f"Is follow-up: {is_followup}")
    
    # Get context
    context = memory_manager.get_context_for_followup("test_user", followup_question)
    print(f"\nContext:\n{context}")
    
    # Check memory
    print("\n=== Memory Contents ===")
    memory = memory_manager.get_recent_memory("test_user", 10)
    for i, entry in enumerate(memory):
        print(f"{i+1}. {entry.get('role')}: {entry.get('content')}")

if __name__ == "__main__":
    test_memory_system() 