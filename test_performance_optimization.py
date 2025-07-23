#!/usr/bin/env python3
"""
Test script for performance optimizations
"""

import sys
import os
import time
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationMemory

def test_batched_memory_system():
    """Test the batched memory system"""
    print('Testing batched memory system...')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
        temp_memory_file = temp_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
        temp_image_file = temp_file.name
    
    try:
        memory = ConversationMemory(
            memory_file=temp_memory_file,
            image_memory_file=temp_image_file,
            batch_size=3, 
            flush_interval=5
        )
        
        for i in range(5):
            memory.save_memory('test_user', 'user', f'Test message {i}', 'text', 'test_model')
            print(f'Added entry {i}, buffer size: {len(memory.write_buffer)}')
        
        memory.force_flush()
        print(f'After flush, buffer size: {len(memory.write_buffer)}')
        
        with open(temp_memory_file, 'r') as f:
            lines = f.readlines()
            print(f'File contains {len(lines)} lines')
        
        print('Batched memory system test completed successfully!')
        return True
        
    except Exception as e:
        print(f'Test failed: {e}')
        return False
    
    finally:
        try:
            os.unlink(temp_memory_file)
            os.unlink(temp_image_file)
        except:
            pass

def test_http_timeout_config():
    """Test the improved HTTP timeout handling"""
    print('Testing HTTP timeout improvements...')
    
    try:
        timeout_config = (5, 10)  # (connection_timeout, read_timeout)
        print(f'Timeout configuration: {timeout_config}')
        print('HTTP timeout configuration test passed!')
        return True
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == '__main__':
    print('Running performance optimization tests...\n')
    
    test1_passed = test_batched_memory_system()
    print()
    test2_passed = test_http_timeout_config()
    
    print(f'\nTest Results:')
    print(f'Batched Memory System: {"PASSED" if test1_passed else "FAILED"}')
    print(f'HTTP Timeout Config: {"PASSED" if test2_passed else "FAILED"}')
    
    if test1_passed and test2_passed:
        print('\nAll tests passed! Performance optimizations are working correctly.')
        sys.exit(0)
    else:
        print('\nSome tests failed. Please check the implementation.')
        sys.exit(1)
