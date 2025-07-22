from dotenv import load_dotenv
import os

load_dotenv()

import logging
import requests
from openai import OpenAI
from deep_translator import GoogleTranslator
from memory import memory_manager
from utils import detect_language
from classifier.intent_classifier import classify_task, classify_code_complexity
import os
import anthropic
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import re
from openai.types.chat import ChatCompletionUserMessageParam
from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext  # Add this import

logging.info(f"✅ SD KEY FOUND: {os.getenv('STABLE_DIFFUSION_API_KEY')}")

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def check_model_availability(model_name: str) -> bool:
    """
    Check if a model is available based on API key presence and model status.
    Returns True if the model can be used, False otherwise.
    """
    try:
        if model_name.startswith("claude"):
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        elif model_name.startswith("gpt"):
            return bool(os.getenv("OPENAI_API_KEY"))
        elif model_name == "gemini-pro":
            return bool(os.getenv("GOOGLE_API_KEY"))
        elif model_name == "dall-e-3":
            return bool(os.getenv("OPENAI_API_KEY"))
        elif model_name == "stablediffusion":
            return bool(os.getenv("STABLE_DIFFUSION_API_KEY"))
        elif model_name == "anime-diffusion":
            return bool(os.getenv("HUGGINGFACE_API_KEY"))
        elif model_name == "deepl":
            return bool(os.getenv("DEEPL_API_KEY"))
        elif model_name == "whisper":
            return bool(os.getenv("OPENAI_API_KEY"))
        elif model_name == "elevenlabs":
            return bool(os.getenv("ELEVENLABS_API_KEY"))
        elif model_name in ["codellama-70b", "wizardcoder", "phind-codellama"]:
            return bool(os.getenv("HUGGINGFACE_API_KEY"))
        else:
            return True  # Default to available for unknown models
    except Exception as e:
        logging.warning(f"Error checking model availability for {model_name}: {e}")
        return False

def get_available_models_for_task(task_type: str) -> Dict[str, Dict]:
    """
    Get available models for a task type, filtering out unavailable models.
    """
    all_models = get_available_models(task_type)
    available_models = {}
    
    for model_name, model_info in all_models.items():
        if check_model_availability(model_name):
            available_models[model_name] = model_info
        else:
            logging.info(f"Model {model_name} not available (missing API key or service down)")
    
    return available_models

# ============================================================================
# SOPHISTICATED ENTITY RESOLUTION FOR PROMPT REWRITING
# ============================================================================

def detect_pronouns_and_references(question: str) -> Tuple[bool, List[str]]:
    """
    Detect if a question contains pronouns or references that need resolution.
    Returns (needs_resolution, list_of_pronouns_found)
    """
    pronouns = [
        "he", "she", "it", "they", "them", "him", "her",
        "there", "that", "those", "this", "these", "here"
    ]
    
    question_lower = question.lower()
    # Use word boundaries to match whole words only
    import re
    found_pronouns = []
    for pronoun in pronouns:
        if re.search(r'\b' + re.escape(pronoun) + r'\b', question_lower):
            found_pronouns.append(pronoun)
    
    return len(found_pronouns) > 0, found_pronouns

def extract_entities_from_response(response: str) -> Dict[str, str]:
    """
    Extract named entities from an assistant response.
    Returns a dictionary mapping entity types to entity names.
    """
    entities = {}
    
    # Enhanced patterns for better entity extraction - now captures multi-word entities
    patterns = {
        'capital': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+capital\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        'capital_simple': r'capital\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        'capital_mentioned': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?capital',
        'person': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)\s+(\w+)',
        'person_simple': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is|has|had)',
        'place': r'(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        'place_mentioned': r'(?:city|town|country|place)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        'building': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Building|Tower|Center|Plaza|Complex)',
        'landmark': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Monument|Statue|Museum|Park|Bridge)',
        'object': r'(?:the|a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        'number': r'(\d+(?:\.\d+)?)\s+(\w+)',
        'location': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:has|has\s+a|population|people|inhabitants)',
        'topic': r'(?:about|regarding|concerning)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    }
    
    # Test the patterns and print debug info
    logging.info(f"=== ENTITY EXTRACTION DEBUG ===")
    logging.info(f"Extracting entities from response: {response}")
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        logging.info(f"Pattern '{pattern}' found {len(matches)} matches: {matches}")
    logging.info(f"=== END ENTITY EXTRACTION DEBUG ===")
    
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
            elif entity_type == 'capital_mentioned':
                entities['capital'] = matches[0]  # Just the capital name
            elif entity_type == 'person':
                entities['person'] = matches[0][0]  # Person name
                entities['role'] = matches[0][1]    # Their role
            elif entity_type == 'person_simple':
                entities['person'] = matches[0]  # Person name
            elif entity_type == 'location':
                entities['location'] = matches[0]  # Location name
            elif entity_type == 'building':
                entities['building'] = matches[0]  # Building name
            elif entity_type == 'landmark':
                entities['landmark'] = matches[0]  # Landmark name
            elif entity_type == 'topic':
                entities['topic'] = matches[0]  # Topic name
            else:
                entities[entity_type] = matches[0]
    
    return entities

def llm_based_coreference_resolution(question: str, last_response: str) -> str:
    """Use LLM to resolve coreferences in follow-up questions"""
    try:
        # Truncate inputs to prevent token explosion
        question = question[:200]  # Limit question length
        last_response = last_response[:500]  # Limit context length
        
        prompt = f"""Given this context and follow-up question, rewrite the question to be self-contained:

Context: {last_response}

Follow-up question: {question}

Rewrite the question to be self-contained by replacing pronouns and references with specific entities from the context. Return only the rewritten question, nothing else."""

        # Use a cheaper model for this task
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Limit tokens
            temperature=0.1  # Lower temperature for consistency
        )
        rewritten = response.choices[0].message.content.strip() if response.choices[0].message.content else question
        return rewritten
    except Exception as e:
        logging.warning(f"LLM-based resolution failed: {e}")
        return question  # Return original if resolution fails

def resolve_references_in_question(question: str, last_response: str) -> str:
    """
    Resolve pronouns in the question using the last assistant response.
    Returns the rewritten question with pronouns replaced by actual entities.
    """
    needs_resolution, pronouns = detect_pronouns_and_references(question)
    
    if not needs_resolution:
        return question
    
    entities = extract_entities_from_response(last_response)
    rewritten_question = question
    
    # Handle different types of references
    for pronoun in pronouns:
        if pronoun == "there":
            # "there" usually refers to the most recently mentioned place/entity
            if 'capital' in entities:
                rewritten_question = rewritten_question.replace("there", f"in {entities['capital']}")
            elif 'location' in entities:
                rewritten_question = rewritten_question.replace("there", f"in {entities['location']}")
            elif 'place' in entities:
                rewritten_question = rewritten_question.replace("there", f"in {entities['place']}")
            elif 'country' in entities:
                rewritten_question = rewritten_question.replace("there", f"in {entities['country']}")
        
        elif pronoun in ["he", "him"]:
            if 'person' in entities:
                rewritten_question = rewritten_question.replace(pronoun, entities['person'])
        
        elif pronoun in ["she", "her"]:
            if 'person' in entities:
                rewritten_question = rewritten_question.replace(pronoun, entities['person'])
        
        elif pronoun == "it":
            # "it" is more ambiguous - try to find the most relevant entity
            if 'building' in entities:
                rewritten_question = rewritten_question.replace("it", entities['building'])
            elif 'landmark' in entities:
                rewritten_question = rewritten_question.replace("it", entities['landmark'])
            elif 'object' in entities:
                rewritten_question = rewritten_question.replace("it", entities['object'])
            elif 'capital' in entities:
                rewritten_question = rewritten_question.replace("it", entities['capital'])
            elif 'topic' in entities:
                rewritten_question = rewritten_question.replace("it", entities['topic'])
        
        elif pronoun in ["that", "those"]:
            # These are more complex - would need more sophisticated context analysis
            pass
    
    # If regex-based resolution didn't work well, try LLM-based resolution
    if rewritten_question == question and needs_resolution:
        logging.info("Regex-based resolution failed, trying LLM-based resolution")
        return llm_based_coreference_resolution(question, last_response)
    
    return rewritten_question

def get_last_assistant_response(memory: List[Dict]) -> Optional[str]:
    """
    Get the most recent assistant response from memory.
    """
    for entry in reversed(memory):
        if entry.get('role') == 'assistant':
            return entry.get('content', '')
    return None

def rewrite_prompt_with_entity_resolution(prompt: str, memory: List[Dict]) -> str:
    """
    Best-in-class entity resolution: For follow-up questions, send the last assistant response as context with the user's question.
    For non-follow-ups, send the question as-is.
    """
    # Extract the actual question from the optimized prompt
    prefixes = [
        "Provide a clear, comprehensive explanation: ",
        "Create engaging, well-structured content: ",
        "Respond with wit and humor: ",
        "You are a helpful assistant. Provide a clear, engaging response to: ",
        "Provide a thoughtful, nuanced analysis: ",
        "Conduct a thorough analysis: ",
        "Respond with insight and depth: "
    ]
    actual_question = prompt
    for prefix in prefixes:
        if prompt.startswith(prefix):
            actual_question = prompt[len(prefix):]
            break

    # Check if this question needs entity resolution (i.e., is a follow-up)
    needs_resolution, pronouns = detect_pronouns_and_references(actual_question)
    logging.info(f"Entity resolution check: needs_resolution={needs_resolution}, pronouns={pronouns}")
    logging.info(f"Actual question: '{actual_question}'")

    if not needs_resolution:
        return prompt

    # Get the last assistant response for context
    last_response = get_last_assistant_response(memory)
    logging.info(f"Last assistant response: {last_response}")
    if not last_response:
        logging.info("No last response available for entity resolution")
        return prompt  # No context available

    # Compose the new prompt for the LLM
    rewritten_prompt = f"The previous answer was: \"{last_response}\"\nNow answer: {actual_question}"
    logging.info(f"Prompt rewritten for follow-up: '{rewritten_prompt}'")
    return rewritten_prompt

# ============================================================================
# END ENTITY RESOLUTION FUNCTIONS
# ============================================================================


# Claude handler
def call_claude(prompt):
    try:
        anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Updated model name
            max_tokens=1000,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": prompt
            }])
        if response.content and len(response.content) > 0:
            content_block = response.content[0]
            # Check if it's a text block type and safely access 'text'
            if getattr(content_block, 'type', None) == 'text':
                text = getattr(content_block, 'text', None)
                return text or "Claude returned no response."
            # Handle other block types that might not have text
            return "Claude returned no response."
        else:
            return "Claude returned no response."
    except Exception as e:
        logging.error(f"Claude API error: {e}")
        return f"Error calling Claude: {str(e)}"


def llm_select_model(prompt, history=None):
    """
    Use an LLM to select the best image model based on the user's prompt and conversation history.
    Returns (model_name, reason) or (None, None) if LLM fails.
    """
    try:
        system_prompt = (
            "You are an expert AI model selector for image generation. Choose the best model from: "
            "[dall-e-3, stablediffusion, midjourney, anime-diffusion]. "
            "\n\nModel strengths:"
            "\n- dall-e-3: Best for photorealistic images, logos, icons, and structured/commercial images"
            "\n- stablediffusion: Best for artistic, creative, fantasy, surreal, and painting-style images"
            "\n- midjourney: Best for artistic and creative images with unique styles"
            "\n- anime-diffusion: Best for anime, manga, and cartoon-style images"
            "\n\nConsider the user's intent and style preferences. "
            "Respond in JSON: {\"model\": \"model_name\", \"reason\": \"explanation\"}."
        )
        user_message = f"User prompt: {prompt}\n"
        if history:
            user_message += f"Recent history: {history}\n"
        user_message += "Which model is best and why?"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.0
        )
        import json as _json
        content = response.choices[0].message.content or ''
        result = _json.loads(content)
        model = result.get("model")
        reason = result.get("reason")
        if model in ["dall-e-3", "stablediffusion", "midjourney", "anime-diffusion"]:
            return model, reason
        return None, None
    except Exception as e:
        logging.warning(f"LLM model selection failed: {e}")
        return None, None

# Model selector
def select_best_model(task_type, prompt, lang="en", confidence=0.0, history=None):
    """
    Industry-leading dynamic model selector with performance-driven routing.
    Uses MODEL_REGISTRY, performance metrics, and sophisticated task analysis.
    Returns the best model for the given task and prompt.
    """
    prompt_lower = prompt.lower()

    # --- Explicit override logic ---
    if task_type == "image":
        logging.info(f"Image task detected, checking for explicit model selection")
        dalle_phrases = [
            "use dalle", "make with dalle", "generate with dalle", "dall-e", "dalle-3"
        ]
        sd_phrases = [
            "use stable diffusion", "make with stable diffusion", "generate with stable diffusion", "stablediffusion", "stable diffusion"
        ]
        if any(phrase in prompt_lower for phrase in dalle_phrases):
            logging.info("DALL-E phrase detected, returning dall-e-3")
            return "dall-e-3"
        if any(phrase in prompt_lower for phrase in sd_phrases):
            logging.info("Stable Diffusion phrase detected, returning stablediffusion")
            return "stablediffusion"
        # Re-enable LLM model selection for image tasks
        model, reason = llm_select_model(prompt, history)
        if model:
            logging.info(f"LLM model selection: {model} (reason: {reason})")
            return model
        # Fallback to DALL-E if LLM selection fails
        logging.warning("LLM model selection failed or returned None, using DALL-E")
        return "dall-e-3"
    
    # Get available models for this task type (filtered by availability)
    available_models = get_available_models_for_task(task_type)
    
    # If no models available for this task type, fall back to text models
    if not available_models:
        available_models = get_available_models_for_task("text")
    
    # If still no models available, use a basic fallback
    if not available_models:
        logging.warning(f"No available models for task type: {task_type}")
        return "gpt-4o"  # Ultimate fallback
    
    # Score each available model based on multiple factors
    model_scores = {}
    
    for model_name, model_info in available_models.items():
        score = 0.0
        
        # Base score from model strengths
        if task_type in model_info.get("strengths", []):
            score += 0.3
        
        # Performance-based scoring (if orchestrator data available)
        if hasattr(model_orchestrator, 'performance_metrics') and model_name in model_orchestrator.performance_metrics:
            metrics = model_orchestrator.performance_metrics[model_name]
            if metrics.total_requests > 5:  # Only consider models with sufficient data
                # Success rate is most important
                score += metrics.success_rate * 0.4
                # Speed factor (faster is better for most tasks)
                if metrics.avg_response_time < 2.0:
                    score += 0.1
                elif metrics.avg_response_time > 10.0:
                    score -= 0.1
        
        # Task-specific scoring
        if task_type == "code":
            score += _score_code_model(model_name, prompt_lower, confidence)
        elif task_type == "image":
            score += _score_image_model(model_name, prompt_lower)
        elif task_type == "text":
            score += _score_text_model(model_name, prompt_lower, confidence)
        elif task_type == "summarize":
            score += _score_summarize_model(model_name, prompt_lower)
        elif task_type == "translate":
            score += _score_translate_model(model_name, prompt_lower)
        elif task_type == "audio":
            score += _score_audio_model(model_name, prompt_lower)
        elif task_type == "multimodal":
            score += _score_multimodal_model(model_name, prompt_lower)
        
        # Cost consideration (cheaper models get slight boost for similar performance)
        cost = model_info.get("cost", "medium")
        if cost == "low":
            score += 0.05
        elif cost == "very low":
            score += 0.1
        
        # Confidence-based adjustments
        if confidence < 0.5:
            # For low confidence, prefer more reliable models
            if model_name in ["gpt-4o", "claude-sonnet-4"]:
                score += 0.1
        
        model_scores[model_name] = score
    
    # Select the best model
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        logging.info(f"Model selection: {best_model} (score: {model_scores[best_model]:.3f})")
        logging.info(f"All scores: {model_scores}")
        return best_model
    
    # Fallback
    return "gpt-4o"

def _score_code_model(model_name: str, prompt_lower: str, confidence: float) -> float:
    """Score code generation models based on task complexity and model capabilities."""
    score = 0.0
    
    # Use the imported classify_code_complexity function
    complexity, complexity_confidence = classify_code_complexity(prompt_lower)
    
    # Model-specific scoring for code tasks
    if model_name == "claude-3-5-sonnet":
        # Claude 3.5 Sonnet excels at complex code and architecture
        if complexity == "complex":
            score += 0.4
        elif complexity == "medium":
            score += 0.3
        else:
            score += 0.2
        
        # Architecture and design pattern keywords
        if any(word in prompt_lower for word in [
            "architecture", "design pattern", "best practice", "clean code", "refactor",
            "optimize", "performance", "scalable", "maintainable", "test", "unit test",
            "integration", "deployment", "docker", "kubernetes", "microservices"
        ]):
            score += 0.3
    
    elif model_name == "gpt-4o-mini":
        # GPT-4o Mini is good for quick prototyping and simple code
        if complexity == "simple":
            score += 0.4
        elif complexity == "medium":
            score += 0.2
        
        # Quick and simple keywords
        if any(word in prompt_lower for word in [
            "quick", "simple", "basic", "prototype", "example", "demo", "snippet"
        ]):
            score += 0.3
    
    elif model_name == "codellama-70b":
        # CodeLlama is specialized for code generation
        score += 0.3
        if any(word in prompt_lower for word in [
            "python", "javascript", "java", "c++", "go", "rust"
        ]):
            score += 0.2
    
    elif model_name == "gpt-4o":
        # GPT-4o is good for general code tasks
        score += 0.2
    
    return score

def _score_image_model(model_name: str, prompt_lower: str) -> float:
    """Score image generation models based on style and requirements."""
    score = 0.0
    
    if model_name == "dall-e-3":
        # DALL-E 3 excels at photorealistic images and logos
        if any(word in prompt_lower for word in [
            "photorealistic", "photo", "realistic", "natural", "portrait", "logo", "icon"
        ]):
            score += 0.4
    
    elif model_name == "stablediffusion":
        # Stable Diffusion is great for artistic and creative images
        if any(word in prompt_lower for word in [
            "artistic", "creative", "fantasy", "surreal", "painting", "artwork"
        ]):
            score += 0.4
    
    elif model_name == "anime-diffusion":
        # Anime Diffusion for anime/manga style
        if any(word in prompt_lower for word in [
            "anime", "manga", "cartoon", "japanese style"
        ]):
            score += 0.5
    
    return score

def _score_text_model(model_name: str, prompt_lower: str, confidence: float) -> float:
    """Score text generation models based on content type and requirements."""
    score = 0.0
    
    if model_name == "claude-sonnet-4":
        # Claude excels at analysis, reasoning, and academic content
        if any(word in prompt_lower for word in [
            "analyze", "explain", "compare", "contrast", "evaluate", "assess",
            "philosophy", "ethics", "reasoning", "academic", "research"
        ]):
            score += 0.4
    
    elif model_name == "gpt-4o":
        # GPT-4o is great for creative writing and factual Q&A
        if any(word in prompt_lower for word in [
            "creative", "story", "poem", "write", "compose", "humor", "funny"
        ]):
            score += 0.3
        
        # Also good for factual questions
        if any(word in prompt_lower for word in [
            "what is", "who is", "when was", "where is", "how many", "what price", "price of", "forecast", "prediction", "crypto", "cryptocurrency", "stock", "bitcoin", "ethereum", "xrp", "dogecoin"
        ]):
            score += 0.2
    
    # Remove the short-prompt speed bonus for claude-haiku
    # Only add a speed/cost bonus if the prompt is explicitly simple or urgent (not implemented here)
    
    return score

def _score_summarize_model(model_name: str, prompt_lower: str) -> float:
    """Score summarization models."""
    score = 0.0
    
    if model_name == "claude-sonnet-4":
        # Claude is excellent for nuanced understanding and summarization
        score += 0.4
    
    elif model_name == "gpt-4o":
        # GPT-4o is good for structured summaries
        if any(word in prompt_lower for word in [
            "bullets", "bullet points", "list", "points"
        ]):
            score += 0.3
    
    return score

def _score_translate_model(model_name: str, prompt_lower: str) -> float:
    """Score translation models."""
    score = 0.0
    
    if model_name == "deepl":
        # DeepL is specialized for professional translation
        score += 0.5
    
    return score

def _score_audio_model(model_name: str, prompt_lower: str) -> float:
    """Score audio processing models."""
    score = 0.0
    
    if model_name == "whisper":
        # Whisper for speech-to-text
        if any(word in prompt_lower for word in [
            "transcribe", "speech to text", "voice to text"
        ]):
            score += 0.5
    
    elif model_name == "elevenlabs":
        # ElevenLabs for text-to-speech
        if any(word in prompt_lower for word in [
            "text to speech", "speak", "voice", "audio"
        ]):
            score += 0.5
    
    return score

def _score_multimodal_model(model_name: str, prompt_lower: str) -> float:
    """Score multimodal models."""
    score = 0.0
    
    if model_name == "gemini-pro":
        # Gemini Pro is excellent for multimodal tasks
        score += 0.4
    
    elif model_name == "gpt-4o":
        # GPT-4o also handles multimodal well
        score += 0.3
    
    return score

# Model Registry - Comprehensive list of available models
MODEL_REGISTRY = {
    # Text Generation Models
    "gpt-4o": {
        "provider": "openai",
        "strengths": ["creative writing", "humor", "code", "factual Q&A"],
        "cost": "medium",
        "speed": "fast"
    },
    "gpt-4-turbo": {
        "provider": "openai", 
        "strengths": ["general text", "conversation"],
        "cost": "low",
        "speed": "fast"
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "strengths": ["philosophy", "analysis", "reasoning", "academic"],
        "cost": "medium",
        "speed": "medium"
    },
    "claude-haiku": {
        "provider": "anthropic",
        "strengths": ["quick responses", "simple tasks"],
        "cost": "low",
        "speed": "very fast"
    },
    "gemini-pro": {
        "provider": "google",
        "strengths": ["multimodal", "reasoning", "creative"],
        "cost": "low",
        "speed": "fast"
    },
    "llama-3-70b": {
        "provider": "meta",
        "strengths": ["open source", "customizable"],
        "cost": "low",
        "speed": "medium"
    },
    
    # Specialized Code Generation Models
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "strengths": ["code generation", "debugging", "software architecture", "best practices"],
        "cost": "medium",
        "speed": "medium"
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "strengths": ["code", "quick development", "prototyping"],
        "cost": "low",
        "speed": "very fast"
    },
    "codellama-70b": {
        "provider": "meta",
        "strengths": ["code completion", "programming", "multi-language"],
        "cost": "low",
        "speed": "medium"
    },
    "wizardcoder": {
        "provider": "microsoft",
        "strengths": ["code generation", "programming", "software development"],
        "cost": "low",
        "speed": "fast"
    },
    "phind-codellama": {
        "provider": "phind",
        "strengths": ["code generation", "programming", "development"],
        "cost": "low",
        "speed": "fast"
    },
    
    # Image Generation Models
    "dall-e-3": {
        "provider": "openai",
        "strengths": ["photorealistic", "logos", "structured images"],
        "cost": "high",
        "speed": "medium"
    },
    "stablediffusion": {
        "provider": "stability",
        "strengths": ["artistic", "creative", "cost-effective"],
        "cost": "low",
        "speed": "fast"
    },
    "midjourney": {
        "provider": "midjourney",
        "strengths": ["artistic", "3d", "architectural"],
        "cost": "medium",
        "speed": "slow"
    },
    "anime-diffusion": {
        "provider": "huggingface",
        "strengths": ["anime", "manga", "cartoon style"],
        "cost": "very low",
        "speed": "fast"
    },
    "kandinsky": {
        "provider": "sberbank",
        "strengths": ["artistic", "painting style"],
        "cost": "low",
        "speed": "medium"
    },
    
    # Translation Models
    "deepl": {
        "provider": "deepl",
        "strengths": ["professional translation", "100+ languages"],
        "cost": "medium",
        "speed": "fast"
    },
    "google-translate": {
        "provider": "google",
        "strengths": ["general translation", "free tier"],
        "cost": "very low",
        "speed": "very fast"
    },
    "azure-translator": {
        "provider": "microsoft",
        "strengths": ["enterprise translation", "custom models"],
        "cost": "medium",
        "speed": "fast"
    },
    
    # Specialized Models
    "whisper": {
        "provider": "openai",
        "strengths": ["speech-to-text", "transcription"],
        "cost": "low",
        "speed": "medium"
    },
    "elevenlabs": {
        "provider": "elevenlabs",
        "strengths": ["text-to-speech", "voice cloning"],
        "cost": "medium",
        "speed": "fast"
    },
    "replicate": {
        "provider": "replicate",
        "strengths": ["specialized models", "custom workflows"],
        "cost": "variable",
        "speed": "variable"
    }
}

def get_available_models(task_type=None):
    """
    Get available models, optionally filtered by task type.
    """
    if task_type:
        # Filter models by task type
        task_models = {
            "text": ["gpt-4o", "gpt-4-turbo", "claude-sonnet-4", "claude-haiku", "gemini-pro", "llama-3-70b"],
            "code": ["claude-3-5-sonnet", "gpt-4o-mini", "codellama-70b", "wizardcoder", "phind-codellama", "gpt-4o"],
            "image": ["dall-e-3", "stablediffusion", "midjourney", "anime-diffusion", "kandinsky"],
            "translate": ["deepl", "google-translate", "azure-translator"],
            "audio": ["whisper", "elevenlabs"],
            "multimodal": ["gemini-pro", "gpt-4o", "claude-sonnet-4"],
            "summarize": ["gpt-4o", "claude-sonnet-4", "gemini-pro"],
            "debug": ["claude-3-5-sonnet", "gpt-4o", "codellama-70b"],
            "optimize": ["claude-3-5-sonnet", "gpt-4o", "codellama-70b"]
        }
        return {k: MODEL_REGISTRY[k] for k in task_models.get(task_type, [])}
    
    return MODEL_REGISTRY

def get_model_info(model_name):
    """
    Get detailed information about a specific model.
    """
    return MODEL_REGISTRY.get(model_name, None)

def get_last_image_prompt(memory):
    """Get the last image generation prompt from memory"""
    for entry in reversed(memory):
        if entry.get('role') == 'user' and entry.get('task_type') == 'image':
            # Look for the next assistant entry that contains an image URL
            for next_entry in memory[memory.index(entry):]:
                if next_entry.get('role') == 'assistant' and next_entry.get('content', '').startswith('http'):
                    return entry.get('content', '')
    return None

def is_followup_image_prompt(prompt):
    """Check if a prompt is a follow-up to an image generation"""
    prompt_lower = prompt.lower()
    
    # Core modification keywords (high confidence)
    modification_keywords = [
        "make it", "change it", "modify it", "adjust it", "update it",
        "make the", "change the", "modify the", "adjust the", "update the",
        "look more like", "look like", "resemble", "similar to",
        "add", "remove", "include", "exclude", "put", "take away",
        "enhance", "improve", "better", "fix", "correct",
        "same", "similar", "like that", "but", "however", "instead",
        "different", "another", "version", "variation", "style",
        "zoom in", "zoom out", "closer", "farther", "wider", "narrower",
        "brighter", "darker", "lighter", "more colorful", "less colorful",
        "background", "foreground", "setting", "scene", "environment",
        "clothes", "outfit", "dress", "wear", "wearing",
        "hair", "beard", "mustache", "glasses", "hat", "accessories",
        "expression", "smile", "frown", "emotion", "mood",
        "pose", "position", "angle", "view", "perspective",
        "realistic", "cartoon", "anime", "artistic", "photorealistic"
    ]
    
    # Visual editing commands (very high confidence)
    editing_commands = [
        "crop", "resize", "rotate", "flip", "mirror", "invert",
        "blur", "sharpen", "saturate", "desaturate", "brighten", "darken",
        "contrast", "hue", "tint", "shade", "highlight", "shadow",
        "border", "frame", "watermark", "logo", "text overlay",
        "filter", "effect", "style transfer", "color grading"
    ]
    
    # Specific object/feature modifications
    object_modifications = [
        "make him", "make her", "give him", "give her",
        "put a", "add a", "remove the", "change the",
        "replace with", "switch to", "convert to",
        "turn into", "transform into", "make it look like"
    ]
    
    # Style and aesthetic changes
    style_changes = [
        "in the style of", "like a", "as a", "inspired by",
        "vintage", "modern", "classic", "contemporary",
        "minimalist", "detailed", "simple", "complex",
        "professional", "casual", "formal", "informal"
    ]
    
    # Check for any of these patterns
    if any(kw in prompt_lower for kw in modification_keywords):
        return True
    
    if any(cmd in prompt_lower for cmd in editing_commands):
        return True
    
    if any(mod in prompt_lower for mod in object_modifications):
        return True
    
    if any(style in prompt_lower for style in style_changes):
        return True
    
    # Check for pronouns that indicate follow-up (medium confidence)
    pronouns = ["it", "this", "that", "the", "him", "her", "them"]
    if any(pronoun in prompt_lower for pronoun in pronouns):
        # Only consider it a follow-up if it's a short command (likely image edit)
        if len(prompt.split()) <= 8:  # Short commands are more likely to be image edits
            return True
    
    return False

def run_model(model_name, task_type, prompt, user_id=None):

    def get_recent_memory(memory, max_exchanges=3):
        """Get recent memory exchanges, limiting to prevent token explosion"""
        if not memory:
            return []
        
        # Only get the last few exchanges to prevent token explosion
        recent = memory[-max_exchanges*2:]  # *2 because each exchange has user + assistant
        return recent

    def llm_based_coreference_resolution(question: str, last_response: str) -> str:
        """Use LLM to resolve coreferences in follow-up questions"""
        try:
            # Truncate inputs to prevent token explosion
            question = question[:200]  # Limit question length
            last_response = last_response[:500]  # Limit context length
            
            prompt = f"""Given this context and follow-up question, rewrite the question to be self-contained:

Context: {last_response}

Follow-up question: {question}

Rewrite the question to be self-contained by replacing pronouns and references with specific entities from the context. Return only the rewritten question, nothing else."""

            # Use a cheaper model for this task
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Limit tokens
                temperature=0.1
            )
            rewritten = response.choices[0].message.content.strip() if response.choices[0].message.content else question
            return rewritten
        except Exception as e:
            logging.warning(f"LLM-based resolution failed: {e}")
            return question

    def cleanup_memory_if_needed(user_id):
        """Clean up memory if it gets too large"""
        try:
            memory = memory_manager.get_recent_memory(user_id, 10)  # Get limited memory
            if len(memory) > 2:  # If more than 1 exchange (very aggressive)
                # Keep only the last 1 exchange (2 messages)
                recent_memory = memory[-2:]
                # Clear the old memory by saving only recent exchanges
                memory_manager.clear_memory(user_id)
                for exchange in recent_memory:
                    if exchange.get('role') == 'user':
                        memory_manager.save_memory(user_id, 'user', exchange.get('content', ''), 'text', 'gpt-4o')
                    elif exchange.get('role') == 'assistant':
                        memory_manager.save_memory(user_id, 'assistant', exchange.get('content', ''), 'text', 'gpt-4o')
        except Exception as e:
            logging.error(f"Memory cleanup error: {e}")

    # Clean up memory if needed
    if user_id:
        cleanup_memory_if_needed(user_id)

    # Get memory for context
    memory = []
    if user_id:
        memory = memory_manager.get_recent_memory(user_id, 2)  # Only get last 2 exchanges

    # Handle image follow-ups specially
    if task_type == "image" and user_id and is_followup_image_prompt(prompt):
        prev_prompt = get_last_image_prompt(memory)
        if prev_prompt:
            # For image follow-ups, only send the original prompt + modification
            # Don't include the entire conversation history
            combined_prompt = f"{prev_prompt}. {prompt}"
            logging.info(f"[{user_id}] Image follow-up detected. Original: {prev_prompt}, Modification: {prompt}")
            
            # Call image generation with just the combined prompt
            if model_name == "dall-e-3":
                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=combined_prompt,
                        n=1,
                        size="1024x1024"
                    )
                    if response.data and len(response.data) > 0:
                        image_url = response.data[0].url
                        if image_url:  # Check if image_url is not None
                            memory_manager.save_memory(user_id, 'user', combined_prompt, 'image', 'dall-e-3')
                            memory_manager.save_memory(user_id, 'assistant', image_url, 'image', 'dall-e-3')
                            return image_url
                        else:
                            return "Error: No image URL generated"
                    else:
                        return "Error: No image generated"
                except Exception as e:
                    logging.error(f"DALL-E follow-up error: {e}")
                    return f"Error generating image: {e}"
            else:
                # For other image models, use the same approach
                return call_claude(f"Generate an image: {combined_prompt}")

    # For non-image follow-ups, proceed with normal processing
    rewritten_prompt = prompt
    
    # Only do entity resolution for text tasks to save tokens
    if task_type == "text" and memory:
        try:
            # Build conversation context for better entity resolution
            conversation_context = ""
            for entry in memory[-4:]:  # Last 4 entries (2 exchanges)
                role = entry.get('role', '')
                content = entry.get('content', '')
                if role == 'user':
                    conversation_context += f"User: {content}\n"
                elif role == 'assistant':
                    conversation_context += f"Assistant: {content}\n"
            
            if conversation_context and len(conversation_context) < 2000:  # Reasonable size limit
                rewritten_prompt = llm_based_coreference_resolution(prompt, conversation_context)
                logging.info(f"[{user_id}] Entity resolution applied. Original: '{prompt}' -> Resolved: '{rewritten_prompt}'")
        except Exception as e:
            logging.warning(f"Entity resolution failed: {e}")

    # Safety check: if prompt is too long, truncate it
    if len(rewritten_prompt) > 1000:  # Reduced from 2000 to 1000
        logging.warning(f"Prompt too long ({len(rewritten_prompt)} chars), truncating to 1000 chars")
        rewritten_prompt = rewritten_prompt[:1000] + "..."

    # Optimize prompt
    optimizer = IntelligentPromptOptimizer()
    context = OptimizationContext(task_type=task_type, model=model_name, original_prompt=rewritten_prompt)
    optimized_prompt = optimizer.optimize_prompt(context)

    # Select model and call
    if model_name == "gpt-4o":
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[ChatCompletionUserMessageParam(role="user", content=optimized_prompt)],
                temperature=0.7,
                max_tokens=2000  # Reduced from 4000 to 2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            # Try Claude as fallback
            logging.info("Trying Claude as fallback")
            try:
                return call_claude(optimized_prompt)
            except Exception as claude_error:
                logging.error(f"Claude fallback error: {claude_error}")
                return "I'm sorry, I'm experiencing technical difficulties. Please try again in a moment."
    elif model_name == "claude-3-sonnet":
        return call_claude(optimized_prompt)
    elif model_name == "dall-e-3":
        logging.info(f"[{user_id}] Starting DALL-E generation for prompt: {optimized_prompt}")
        
        try:
            response = client.images.generate(model="dall-e-3",
                                              prompt=optimized_prompt,
                                              n=1,
                                              size="1024x1024")
            logging.info(f"[{user_id}] DALL-E response received")
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                if image_url:  # Check if image_url is not None
                    if user_id:
                        memory_manager.save_memory(user_id, 'user', optimized_prompt, 'image', 'dall-e-3')
                        memory_manager.save_memory(user_id, 'assistant', image_url, 'image', 'dall-e-3')
                    return image_url
                else:
                    return "Error: No image URL generated"
            else:
                return "Error: No image generated"
        except Exception as e:
            logging.error(f"DALL-E error: {e}")
            return f"Error generating image: {e}"
    else:
        # Default to GPT-4o
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[ChatCompletionUserMessageParam(role="user", content=optimized_prompt)],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Default model error: {e}")
            return f"Error: {e}"


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    model_name: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    total_requests: int = 0
    cost_per_request: float = 0.0
    last_used: Optional[datetime] = None

class ModelOrchestrator:
    """Intelligent model orchestration for Portal AI"""
    
    def __init__(self):
        self.performance_metrics: Dict[str, ModelPerformance] = {}
        self.model_costs = {
            "gpt-4o": 0.03,  # per 1K tokens
            "gpt-4-turbo": 0.01,
            "claude-sonnet-4": 0.015,
            "claude-haiku": 0.0025,
            "gemini-pro": 0.001,
            "dall-e-3": 0.04,  # per image
            "stablediffusion": 0.02,
        }
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load historical performance data"""
        try:
            with open("model_performance.json", "r") as f:
                data = json.load(f)
                for model_name, metrics in data.items():
                    self.performance_metrics[model_name] = ModelPerformance(**metrics)
        except FileNotFoundError:
            pass
    
    def save_performance_data(self):
        """Save performance data to file"""
        data = {}
        for model_name, metrics in self.performance_metrics.items():
            data[model_name] = {
                "model_name": metrics.model_name,
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.avg_response_time,
                "total_requests": metrics.total_requests,
                "cost_per_request": metrics.cost_per_request,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None
            }
        
        with open("model_performance.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def update_performance(self, model_name: str, success: bool, response_time: float, tokens_used: int = 0):
        """Update model performance metrics"""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = ModelPerformance(model_name)
        
        metrics = self.performance_metrics[model_name]
        metrics.total_requests += 1
        metrics.last_used = datetime.utcnow()
        
        # Update success rate
        if success:
            metrics.success_rate = (metrics.success_rate * (metrics.total_requests - 1) + 1) / metrics.total_requests
        else:
            metrics.success_rate = (metrics.success_rate * (metrics.total_requests - 1)) / metrics.total_requests
        
        # Update average response time
        metrics.avg_response_time = (metrics.avg_response_time * (metrics.total_requests - 1) + response_time) / metrics.total_requests
        
        # Update cost
        if model_name in self.model_costs:
            cost = (tokens_used / 1000) * self.model_costs[model_name]
            metrics.cost_per_request = (metrics.cost_per_request * (metrics.total_requests - 1) + cost) / metrics.total_requests
        
        self.save_performance_data()
    
    def get_best_model_for_task(self, task_type: str, prompt: str, confidence: float, budget_constraint: Optional[float] = None) -> Tuple[str, float]:
        """Enhanced model selection with performance and cost considerations"""
        
        # Get base model selection
        base_model = select_best_model(task_type, prompt, str(confidence))
        
        # Check if we have performance data for this model
        if base_model in self.performance_metrics:
            metrics = self.performance_metrics[base_model]
            
            # If success rate is too low, consider alternatives
            if metrics.success_rate < 0.7 and metrics.total_requests > 10:
                alternatives = self._get_alternative_models(task_type, base_model)
                for alt_model in alternatives:
                    if alt_model in self.performance_metrics:
                        alt_metrics = self.performance_metrics[alt_model]
                        if alt_metrics.success_rate > metrics.success_rate:
                            base_model = alt_model
                            break
        
        # Apply budget constraints
        if budget_constraint and base_model in self.model_costs:
            if self.model_costs[base_model] > budget_constraint:
                # Find cheaper alternative
                for model, cost in sorted(self.model_costs.items(), key=lambda x: x[1]):
                    if cost <= budget_constraint and model != base_model:
                        base_model = model
                        break
        
        return base_model, confidence
    
    def _get_alternative_models(self, task_type: str, current_model: str) -> List[str]:
        """Get alternative models for a task type"""
        alternatives = {
            "text": ["gpt-4o", "claude-sonnet-4", "gemini-pro"],
            "image": ["dall-e-3", "stablediffusion"],
            "summarize": ["claude-sonnet-4", "gpt-4o"],
            "translate": ["deepl", "claude-sonnet-4"]
        }
        
        task_alternatives = alternatives.get(task_type, [])
        return [m for m in task_alternatives if m != current_model]
    
    def get_orchestration_stats(self) -> Dict:
        """Get orchestration statistics"""
        total_requests = sum(m.total_requests for m in self.performance_metrics.values())
        avg_success_rate = sum(m.success_rate for m in self.performance_metrics.values()) / len(self.performance_metrics) if self.performance_metrics else 0
        total_cost = sum(m.cost_per_request * m.total_requests for m in self.performance_metrics.values())
        
        return {
            "total_requests": total_requests,
            "avg_success_rate": avg_success_rate,
            "total_cost": total_cost,
            "models_used": len(self.performance_metrics),
            "performance_metrics": {name: {
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.avg_response_time,
                "total_requests": metrics.total_requests,
                "cost_per_request": metrics.cost_per_request
            } for name, metrics in self.performance_metrics.items()}
        }

# Global orchestrator instance
model_orchestrator = ModelOrchestrator()

@dataclass
class RequestMetrics:
    """Comprehensive metrics for tracking request performance and user experience"""
    timestamp: datetime
    user_id: str
    task_type: str
    model_used: str
    prompt_length: int
    response_length: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    user_feedback: Optional[int] = None  # 1-5 rating
    cost_estimate: Optional[float] = None
    tokens_used: Optional[int] = None
    fallback_used: Optional[str] = None
    confidence_score: float = 0.0

class ComprehensiveMetrics:
    """Industry-leading metrics tracking for Portal AI"""
    
    def __init__(self):
        self.metrics_file = "comprehensive_metrics.jsonl"
        self.performance_cache = {}
        self.user_preferences = {}
    
    def track_request(self, user_id: str, task_type: str, model: str, 
                     prompt: str, response: str, response_time: float, 
                     success: bool, error_message: Optional[str] = None,
                     user_feedback: Optional[int] = None, 
                     confidence_score: float = 0.0,
                     fallback_used: Optional[str] = None):
        """Track comprehensive request metrics"""
        
        metrics = RequestMetrics(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            task_type=task_type,
            model_used=model,
            prompt_length=len(prompt),
            response_length=len(response) if response else 0,
            response_time=response_time,
            success=success,
            error_message=error_message,
            user_feedback=user_feedback,
            confidence_score=confidence_score,
            fallback_used=fallback_used
        )
        
        # Save to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics.__dict__, default=str) + "\n")
        
        # Update performance cache
        self._update_performance_cache(metrics)
        
        # Update user preferences based on feedback
        if user_feedback and user_feedback >= 4:
            self._update_user_preferences(user_id, task_type, model)
    
    def _update_performance_cache(self, metrics: RequestMetrics):
        """Update in-memory performance cache"""
        model_key = metrics.model_used
        if model_key not in self.performance_cache:
            self.performance_cache[model_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "avg_user_feedback": 0.0,
                "total_feedback": 0
            }
        
        cache = self.performance_cache[model_key]
        cache["total_requests"] += 1
        cache["total_response_time"] += metrics.response_time
        
        if metrics.success:
            cache["successful_requests"] += 1
        
        if metrics.user_feedback:
            cache["total_feedback"] += 1
            cache["avg_user_feedback"] = (
                (cache["avg_user_feedback"] * (cache["total_feedback"] - 1) + metrics.user_feedback) 
                / cache["total_feedback"]
            )
        
        cache["success_rate"] = cache["successful_requests"] / cache["total_requests"]
        cache["avg_response_time"] = cache["total_response_time"] / cache["total_requests"]
    
    def _update_user_preferences(self, user_id: str, task_type: str, model: str):
        """Update user preferences based on positive feedback"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_models": {},
                "preferred_task_types": {},
                "last_used": {}
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update preferred models
        if model not in prefs["preferred_models"]:
            prefs["preferred_models"][model] = 0
        prefs["preferred_models"][model] += 1
        
        # Update preferred task types
        if task_type not in prefs["preferred_task_types"]:
            prefs["preferred_task_types"][task_type] = 0
        prefs["preferred_task_types"][task_type] += 1
        
        # Update last used
        prefs["last_used"][task_type] = model
    
    def get_model_performance(self, model: str) -> Dict:
        """Get performance metrics for a specific model"""
        return self.performance_cache.get(model, {})
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences and history"""
        return self.user_preferences.get(user_id, {})
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        total_requests = sum(cache["total_requests"] for cache in self.performance_cache.values())
        avg_success_rate = sum(cache["success_rate"] for cache in self.performance_cache.values()) / len(self.performance_cache) if self.performance_cache else 0
        avg_response_time = sum(cache["avg_response_time"] for cache in self.performance_cache.values()) / len(self.performance_cache) if self.performance_cache else 0
        
        return {
            "total_requests": total_requests,
            "avg_success_rate": avg_success_rate,
            "avg_response_time": avg_response_time,
            "models_in_use": len(self.performance_cache),
            "active_users": len(self.user_preferences)
        }
    
    def get_top_performing_models(self, task_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get top performing models, optionally filtered by task type"""
        model_scores = []
        
        for model, cache in self.performance_cache.items():
            if cache["total_requests"] < 5:  # Only consider models with sufficient data
                continue
            
            # Calculate composite score
            score = (
                cache["success_rate"] * 0.4 +
                (1.0 / (1.0 + cache["avg_response_time"])) * 0.3 +
                cache["avg_user_feedback"] * 0.3
            )
            
            model_scores.append((model, score))
        
        return sorted(model_scores, key=lambda x: x[1], reverse=True)

# Global metrics tracker
comprehensive_metrics = ComprehensiveMetrics()

def run_model_with_fallbacks(model_name, task_type, prompt, user_id=None):
    """
    Run model with comprehensive error handling and fallback logic.
    """
    max_retries = 3
    fallback_models = {
        "claude-3-5-sonnet": ["gpt-4o", "claude-sonnet-4"],
        "gpt-4o": ["claude-sonnet-4", "gpt-4-turbo"],
        "gpt-4o-mini": ["gpt-4o", "claude-haiku"],
        "codellama-70b": ["claude-3-5-sonnet", "gpt-4o"],
        "dall-e-3": ["stablediffusion"],
        "stablediffusion": ["dall-e-3"],
        "claude-sonnet-4": ["gpt-4o", "claude-haiku"],
        "gemini-pro": ["gpt-4o", "claude-sonnet-4"]
    }
    
    # Try the primary model
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}: Trying model {model_name}")
            result = run_model(model_name, task_type, prompt, user_id)
            
            # Check if the result indicates an error
            if isinstance(result, str) and result.startswith("❌"):
                raise Exception(f"Model {model_name} returned error: {result}")
            
            logging.info(f"Successfully used model {model_name}")
            return result
            
        except Exception as e:
            logging.warning(f"Model {model_name} failed (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                # Try fallback models
                fallbacks = fallback_models.get(model_name, ["gpt-4o"])
                for fallback_model in fallbacks:
                    if check_model_availability(fallback_model):
                        try:
                            logging.info(f"Trying fallback model: {fallback_model}")
                            result = run_model(fallback_model, task_type, prompt, user_id)
                            
                            if isinstance(result, str) and not result.startswith("❌"):
                                logging.info(f"Successfully used fallback model {fallback_model}")
                                return result
                        except Exception as fallback_error:
                            logging.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                            continue
    
    # All models failed
    error_msg = f"All models failed for task type {task_type}. Please try again later."
    logging.error(error_msg)
    return error_msg

