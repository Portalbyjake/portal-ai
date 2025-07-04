
"""Model configuration and selection logic."""

# Core model configuration
MODEL_CONFIG = {
    "text": {"primary": "gpt-4o", "fallback": "gpt-4o"},
    "image": {"primary": "dall-e-3", "fallback": "gpt-4o"},
    "translate": {"primary": "deepl", "fallback": "gpt-4o"},
    "summarize": {"primary": "gpt-4o", "fallback": "gpt-4o"}
}

def classify_image_intent(prompt):
    """Classify the intent of an image generation prompt."""
    prompt = prompt.lower()
    if any(w in prompt for w in ["photo", "realistic", "headshot", "portrait"]):
        return "photorealistic"
    if any(w in prompt for w in ["sketch", "drawing", "illustration", "cartoon"]):
        return "illustration"
    if any(w in prompt for w in ["logo", "symbol", "emblem"]):
        return "logo"
    if any(w in prompt for w in ["fantasy", "dragon", "wizard", "castle"]):
        return "fantasy"
    if any(w in prompt for w in ["cinematic", "movie still", "scene"]):
        return "cinematic"
    return "unclear"

def select_best_model(task_type, prompt, lang="en"):
    """Select the best model for a given task."""
    config = MODEL_CONFIG.get(task_type, {})
    return config.get("primary", "gpt-4o")

def optimize_prompt(task_type, model, prompt):
    """Optimize prompt based on task type and model."""
    if model == "gpt-4o":
        if task_type == "text":
            return f"You are a helpful assistant. Respond clearly to: {prompt}"
        if task_type == "summarize":
            return f"Summarize this in 2-3 sentences: {prompt}"
        if task_type == "translate":
            return f"Translate this clearly: {prompt}"
    if model == "dall-e-3" and task_type == "image":
        style = classify_image_intent(prompt)
        if style == "photorealistic":
            return f"A realistic photo of: {prompt}"
        elif style == "illustration":
            return f"A colorful illustration of: {prompt}"
        elif style == "logo":
            return f"A minimal logo of: {prompt}"
        elif style == "fantasy":
            return f"A fantasy-style scene of: {prompt}"
        elif style == "cinematic":
            return f"A cinematic scene of: {prompt}"
        else:
            return f"A detailed image of: {prompt}"
    elif model == "deepl" and task_type == "translate":
        return prompt
    return prompt
