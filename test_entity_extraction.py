#!/usr/bin/env python3

import re

def extract_entities_from_response(response: str):
    """
    Extract named entities from an assistant response.
    Returns a dictionary mapping entity types to entity names.
    """
    entities = {}
    
    # Common patterns for entity extraction
    patterns = {
        'capital': r'(\w+)\s+is\s+the\s+capital\s+of\s+(\w+)',
        'capital_simple': r'capital\s+of\s+(\w+)\s+is\s+(\w+)',
        'person': r'(\w+\s+\w+)\s+is\s+(?:a|an)\s+(\w+)',
        'place': r'(?:in|at|from)\s+(\w+)',
        'object': r'(?:the|a|an)\s+(\w+)',
        'number': r'(\d+(?:\.\d+)?)\s+(\w+)',
    }
    
    # Test the patterns and print debug info
    print(f"Extracting entities from response: {response}")
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        print(f"Pattern '{pattern}' found {len(matches)} matches: {matches}")
    
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            if entity_type == 'capital':
                # For "Paris is the capital of France", extract both Paris and France
                entities['capital'] = matches[0][0]  # Paris
                entities['country'] = matches[0][1]  # France
            elif entity_type == 'capital_simple':
                # For "capital of France is Paris", extract both France and Paris
                entities['country'] = matches[0][0]  # France
                entities['capital'] = matches[0][1]  # Paris
            elif entity_type == 'person':
                entities['person'] = matches[0][0]  # Person name
                entities['role'] = matches[0][1]    # Their role
            else:
                entities[entity_type] = matches[0]
    
    return entities

# Test with the actual response
response = "The capital of France is Paris. Paris is not only the political and administrative center of France but also a major cultural and economic hub in Europe."

print("Testing entity extraction...")
entities = extract_entities_from_response(response)
print(f"Extracted entities: {entities}") 