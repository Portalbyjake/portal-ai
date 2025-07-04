
"""AI service implementations for different models."""

from openai import OpenAI
import requests
from models import MODEL_CONFIG, optimize_prompt

client = OpenAI()

# Short-term memory store
conversation_memory = []

def run_gpt(prompt):
    """Run GPT model with conversation memory."""
    messages = conversation_memory[-5:]  # Keep last 5 interactions
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    conversation_memory.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def run_dalle(prompt):
    """Run DALL-E image generation."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return response.data[0].url

def run_deepl(prompt, target_lang="EN"):
    """Run DeepL translation service."""
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={"auth_key": "YOUR_DEEPL_API_KEY", "text": prompt, "target_lang": target_lang}
    )
    result = response.json()
    return result["translations"][0]["text"]

def run_model(task_type, model, prompt, user_lang="en"):
    """Run model with fallback support."""
    try:
        if model == "gpt-4o":
            return run_gpt(prompt), None
        elif model == "deepl":
            return run_deepl(prompt, user_lang), None
        elif model == "dall-e-3":
            return run_dalle(prompt), None
        raise Exception("Model not implemented")
    except Exception as e:
        fallback = MODEL_CONFIG.get(task_type, {}).get("fallback")
        if fallback and fallback != model:
            try:
                optimized_fallback_prompt = optimize_prompt(task_type, fallback, prompt)
                if fallback == "gpt-4o":
                    output = run_gpt(optimized_fallback_prompt)
                elif fallback == "deepl":
                    output = run_deepl(optimized_fallback_prompt, user_lang)
                elif fallback == "dall-e-3":
                    output = run_dalle(optimized_fallback_prompt)
                else:
                    raise Exception("Fallback not available")
                return output, f"⚠️ Primary model ({model}) failed. Used fallback model: {fallback}"
            except Exception as e2:
                return f"❌ All models failed. Error: {str(e2)}", None
        return f"❌ Model {model} failed. Error: {str(e)}", None
