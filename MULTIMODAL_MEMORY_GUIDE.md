# Enhanced Multimodal Memory System

## Overview

The enhanced multimodal memory system provides continuity across text, image, and other formats in your Portal application. It intelligently tracks conversation context, detects cross-modal references, and maintains conversation flow across different modalities.

## Key Features

### 1. **Cross-Modal Reference Detection**
- Automatically detects when users reference previous images in text questions
- Identifies when users reference previous text in image requests
- Uses semantic analysis rather than simple keyword matching

### 2. **Enhanced Context Generation**
- Provides richer context for model calls
- Includes cross-modal reference information
- Maintains conversation flow across modalities

### 3. **Multimodal Context Tracking**
- Tracks current modality (text, image, etc.)
- Records modality switches and cross-references
- Stores metadata for enhanced context

### 4. **Intelligent Task Routing**
- Considers conversation history for task classification
- Detects cross-modal references automatically
- Routes tasks based on context, not just current input

## System Architecture

### Core Components

#### 1. **ConversationMemory Class** (`memory.py`)
```python
class ConversationMemory:
    def __init__(self, memory_file="memory_text.jsonl", image_memory_file="memory_image.jsonl"):
        self.multimodal_context = {}  # Track cross-modal references
        self.task_history = {}        # Track task types for routing
```

#### 2. **Multimodal Context Tracking**
```python
def _update_multimodal_context(self, user_id: str, entry: Dict):
    """Update multimodal context tracking"""
    context = self.multimodal_context[user_id]
    task_type = entry.get('task_type')
    
    # Track modality switches
    if task_type != context['current_modality']:
        context['cross_references'].append({
            'from_modality': context['current_modality'],
            'to_modality': task_type,
            'timestamp': entry['timestamp'],
            'content': entry['content'][:100]
        })
```

#### 3. **Cross-Modal Reference Detection**
```python
def is_cross_modal_reference(self, user_input: str, user_id: str) -> Tuple[bool, str, str]:
    """Detect if user input references content from a different modality"""
    context = self.multimodal_context[user_id]
    input_lower = user_input.lower()
    
    # Check for references to previous image
    if context.get('last_image_prompt'):
        image_ref_indicators = [
            'that image', 'the image', 'this image', 'the picture', 'that picture',
            'the photo', 'that photo', 'the art', 'that art', 'it', 'this', 'that'
        ]
        if any(indicator in input_lower for indicator in image_ref_indicators):
            return True, 'image', context['last_image_prompt']
```

## Usage Examples

### Example 1: Text → Image Cross-Modal Reference
```python
# User asks about cats
memory.save_memory(user_id, "user", "Tell me about cats", "text")
memory.save_memory(user_id, "assistant", "Cats are domesticated mammals...", "text")

# User references the text in an image request
user_input = "Create an image of that"
is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
# Returns: (True, 'text', 'Tell me about cats')
```

### Example 2: Image → Text Cross-Modal Reference
```python
# User creates an image
memory.save_memory(user_id, "user", "Create an image of a cat", "image")
memory.save_memory(user_id, "assistant", "I've created an image of a cat.", "image")

# User asks about the image
user_input = "What breed is that cat?"
is_cross_modal, modality, content = memory.is_cross_modal_reference(user_input, user_id)
# Returns: (True, 'image', 'Create an image of a cat')
```

### Example 3: Enhanced Context Generation
```python
# Get enhanced context for follow-up questions
enhanced_context = memory.get_enhanced_context_for_followup(user_id, current_query)

# Context includes:
# - Previous conversation (text and image)
# - Cross-modal reference information
# - Task type annotations
```

## Integration with Portal

### 1. **Enhanced Task Routing** (`routes.py`)
```python
# Use intelligent task routing with multimodal awareness
task_type, confidence, reasoning = memory_manager.get_intelligent_task_routing(user_id, user_input)

# The system considers:
# - Recent conversation history
# - Cross-modal references
# - Task type patterns
```

### 2. **Improved Context Generation** (`services.py`)
```python
# Get enhanced context for model calls
context = memory_manager.get_enhanced_context_for_followup(user_id, user_input)
if context:
    prompt = f"{context}\n\nUser: {user_input}"
```

### 3. **Memory with Metadata**
```python
# Save memory with enhanced metadata
metadata = {
    'image_style': 'photorealistic',
    'model_used': 'dall-e-3',
    'user_preference': 'detailed'
}
memory.save_memory(user_id, "user", content, task_type, model, metadata)
```

## Real-World Scenarios

### Scenario 1: Educational Workflow
```
User: "Explain neural networks"
Assistant: "Neural networks are computing systems inspired by biological brains..."
User: "Show me a diagram of that"
Assistant: "I've created a neural network diagram."
User: "What are the layers called?"
Assistant: "The layers are input, hidden, and output layers."
```

**Cross-modal references detected:**
- "Show me a diagram of that" → references text about neural networks
- "What are the layers called?" → references the neural network diagram

### Scenario 2: Creative Design Workflow
```
User: "What are good colors for a logo?"
Assistant: "Blue and white convey trust and professionalism..."
User: "Create a logo with those colors"
Assistant: "I've created a blue and white logo."
User: "Make it more modern"
Assistant: "I've updated the logo to be more modern."
```

**Cross-modal references detected:**
- "Create a logo with those colors" → references color discussion
- "Make it more modern" → references the created logo

## Benefits

### 1. **Improved User Experience**
- Users can reference previous content naturally
- System maintains conversation flow across modalities
- Reduces need for explicit repetition

### 2. **Better Context Understanding**
- Models receive richer context information
- Cross-modal references are explicitly identified
- Conversation history includes modality information

### 3. **Enhanced Analytics**
- Tracks modality switches and patterns
- Monitors cross-modal reference frequency
- Provides insights into user behavior

### 4. **Robust Error Handling**
- Handles corrupted memory files gracefully
- Backs up corrupted files automatically
- Continues operation even with data issues

## Testing

### Run the Test Suite
```bash
python test_multimodal_memory.py
```

### Run the Demonstration
```bash
python multimodal_demo.py
```

## Configuration

### Memory Files
- `memory_text.jsonl`: Text conversation memory
- `memory_image.jsonl`: Image generation memory
- `analytics.jsonl`: System analytics and metrics

### Environment Variables
- No additional environment variables required
- Uses existing Portal configuration

## Future Enhancements

### 1. **Audio/Video Support**
- Extend to handle audio and video modalities
- Track audio-visual cross-references
- Support for speech-to-text and text-to-speech

### 2. **Advanced Semantic Analysis**
- Use embeddings for better reference detection
- Semantic similarity for cross-modal matching
- Context-aware reference resolution

### 3. **Multi-User Support**
- Enhanced user isolation
- Shared conversation contexts
- Collaborative multimodal workflows

### 4. **Real-Time Analytics**
- Live modality switching detection
- User behavior pattern analysis
- Performance optimization insights

## Troubleshooting

### Common Issues

1. **Cross-modal references not detected**
   - Check if the referenced modality has recent content
   - Verify the reference indicators are appropriate
   - Ensure memory is being saved correctly

2. **Context not being generated**
   - Verify memory files exist and are readable
   - Check for corrupted JSON lines
   - Ensure user_id is consistent

3. **Performance issues**
   - Monitor memory file sizes
   - Implement memory summarization for long conversations
   - Consider database storage for large-scale deployments

### Debug Commands
```python
# Check multimodal context
context = memory.get_multimodal_context(user_id)
print(context)

# Check recent memory
recent = memory.get_recent_memory(user_id, 10)
for entry in recent:
    print(f"{entry.get('role')}: {entry.get('content')} ({entry.get('task_type')})")

# Test cross-modal detection
is_cross, modality, content = memory.is_cross_modal_reference(user_input, user_id)
print(f"Cross-modal: {is_cross}, Modality: {modality}, Content: {content}")
```

## Conclusion

The enhanced multimodal memory system provides a robust foundation for maintaining conversation continuity across different modalities. It intelligently detects cross-modal references, provides enhanced context for model calls, and improves the overall user experience by reducing the need for explicit repetition.

The system is designed to be extensible and can be easily adapted to support additional modalities and use cases as your Portal application evolves. 