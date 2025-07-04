
"""Utility functions for text processing and language detection."""

from langdetect import detect

def detect_language(text):
    """Detect the language of input text."""
    try:
        return detect(text)
    except:
        return "en"

def classify_task(prompt):
    """Classify the task type based on prompt content."""
    prompt_lower = prompt.lower()
    
    if "translate" in prompt_lower:
        return "translate"
    elif "summary" in prompt_lower or "summarize" in prompt_lower:
        return "summarize"
    elif any(w in prompt_lower for w in [
        "image", "draw", "create", "picture", "generate", "sketch", 
        "headshot", "photo", "realistic", "see what"
    ]):
        return "image"
    else:
        return "text"
