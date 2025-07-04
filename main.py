from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from langdetect import detect
from deep_translator import GoogleTranslator
import requests

app = Flask(__name__)
client = OpenAI()

# Short-term memory store
conversation_memory = []

# Core model configuration
MODEL_CONFIG = {
    "text": {"primary": "gpt-4o", "fallback": "gpt-4o"},
    "image": {"primary": "dall-e-3", "fallback": "gpt-4o"},
    "translate": {"primary": "deepl", "fallback": "gpt-4o"},
    "summarize": {"primary": "gpt-4o", "fallback": "gpt-4o"}
}

# Detect input language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Classify image intent
def classify_image_intent(prompt):
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

# Select best model
def select_best_model(task_type, prompt, lang="en"):
    config = MODEL_CONFIG.get(task_type, {})
    return config.get("primary", "gpt-4o")

# Optimize prompt
def optimize_prompt(task_type, model, prompt):
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

# Run model with fallback
def run_model(task_type, model, prompt, user_lang="en"):
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

# GPT function
def run_gpt(prompt):
    messages = conversation_memory[-5:]  # Keep last 5 interactions
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    conversation_memory.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

# Image generation
def run_dalle(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return response.data[0].url

# Translation
def run_deepl(prompt, target_lang="EN"):
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={"auth_key": "YOUR_DEEPL_API_KEY", "text": prompt, "target_lang": target_lang}
    )
    result = response.json()
    return result["translations"][0]["text"]

# Route: homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: intelligent dispatcher
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)

    if "translate" in prompt.lower():
        task = "translate"
    elif "summary" in prompt.lower() or "summarize" in prompt.lower():
        task = "summarize"
    elif any(w in prompt.lower() for w in [
        "image", "draw", "create", "picture", "generate", "sketch", "headshot", "photo", "realistic", "see what"
    ]):
        task = "image"
    else:
        task = "text"

    model = select_best_model(task, prompt, user_lang)
    optimized_prompt = optimize_prompt(task, model, prompt)
    output, notice = run_model(task, model, optimized_prompt, user_lang)

    response = {}
    if task == "image":
        response["image_url"] = output
    elif task == "translate":
        response["translation"] = output
    elif task == "summarize":
        response["summary"] = output
    else:
        response["response"] = output

    if notice:
        response["notice"] = notice

    return jsonify(response)

# Manual fallback routes
@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("text", prompt, user_lang)
    optimized_prompt = optimize_prompt("text", model, prompt)
    output, notice = run_model("text", model, optimized_prompt, user_lang)
    return jsonify({"response": output, "notice": notice})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("image", prompt, user_lang)
    optimized_prompt = optimize_prompt("image", model, prompt)
    output, notice = run_model("image", model, optimized_prompt, user_lang)
    return jsonify({"image_url": output, "notice": notice})

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("translate", prompt, user_lang)
    optimized_prompt = optimize_prompt("translate", model, prompt)
    output, notice = run_model("translate", model, optimized_prompt, user_lang)
    return jsonify({"translation": output, "notice": notice})

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("summarize", prompt, user_lang)
    optimized_prompt = optimize_prompt("summarize", model, prompt)
    output, notice = run_model("summarize", model, optimized_prompt, user_lang)
    return jsonify({"summary": output, "notice": notice})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
