
"""Flask route handlers for the AI assistant."""

from flask import request, jsonify
from utils import detect_language, classify_task
from models import select_best_model, optimize_prompt
from services import run_model

def query():
    """Intelligent dispatcher route."""
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    task = classify_task(prompt)
    
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

def generate_response():
    """Manual text generation route."""
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("text", prompt, user_lang)
    optimized_prompt = optimize_prompt("text", model, prompt)
    output, notice = run_model("text", model, optimized_prompt, user_lang)
    return jsonify({"response": output, "notice": notice})

def generate_image():
    """Manual image generation route."""
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("image", prompt, user_lang)
    optimized_prompt = optimize_prompt("image", model, prompt)
    output, notice = run_model("image", model, optimized_prompt, user_lang)
    return jsonify({"image_url": output, "notice": notice})

def translate_text():
    """Manual translation route."""
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("translate", prompt, user_lang)
    optimized_prompt = optimize_prompt("translate", model, prompt)
    output, notice = run_model("translate", model, optimized_prompt, user_lang)
    return jsonify({"translation": output, "notice": notice})

def summarize_text():
    """Manual summarization route."""
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    model = select_best_model("summarize", prompt, user_lang)
    optimized_prompt = optimize_prompt("summarize", model, prompt)
    output, notice = run_model("summarize", model, optimized_prompt, user_lang)
    return jsonify({"summary": output, "notice": notice})
