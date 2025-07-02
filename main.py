"""
Unified AI MVP - Backend Logic (v1)
This is a simplified Python backend file for routing user input to the optimal AI model
based on task classification and prompt optimization.

Note: This is a mock version. Replace `get_ai_response()` with real API calls later.
"""

from flask import Flask, request, jsonify

app = Flask(__name__)


# Simple task classifier
def classify_task(user_input):
    if any(keyword in user_input.lower()
           for keyword in ["write", "email", "caption", "message"]):
        return "writing"
    elif any(keyword in user_input.lower()
             for keyword in ["summarize", "explain", "research"]):
        return "research"
    elif any(keyword in user_input.lower()
             for keyword in ["code", "python", "build"]):
        return "coding"
    else:
        return "general"


# Model selector
def select_model(task_type):
    if task_type == "writing":
        return "OpenAI GPT-4"
    elif task_type == "research":
        return "Claude"
    elif task_type == "coding":
        return "Gemini"
    else:
        return "OpenAI GPT-4"


# Prompt optimizer (mock logic)
def optimize_prompt(task_type, user_input):
    if task_type == "writing":
        return f"Write a clear and engaging message: {user_input}"
    elif task_type == "research":
        return f"Please summarize or explain this information: {user_input}"
    elif task_type == "coding":
        return f"Write clean and efficient code to solve this: {user_input}"
    else:
        return f"Respond helpfully to: {user_input}"


# Mock AI response generator (replace with real API call)
def get_ai_response(model, optimized_prompt):
    return f"[{model}] → Response for: '{optimized_prompt}'"

@app.route("/")
def home():
    return "Unified AI Backend is Live"
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    user_input = data.get("prompt", "")

    task_type = classify_task(user_input)
    model = select_model(task_type)
    optimized_prompt = optimize_prompt(task_type, user_input)
    ai_response = get_ai_response(model, optimized_prompt)

    return jsonify({
        "task_type": task_type,
        "model": model,
        "optimized_prompt": optimized_prompt,
        "response": ai_response
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
