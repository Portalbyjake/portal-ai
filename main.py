from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from langdetect import detect
from deep_translator import GoogleTranslator
import requests

app = Flask(__name__)
client = OpenAI()

conversation_memory = {}

MODEL_CONFIG = {
    "text": {"primary": "gpt-4o", "fallback": "gpt-4o"},
    "image": {"primary": "dall-e-3", "fallback": "gpt-4o"},
    "translate": {"primary": "deepl", "fallback": "gpt-4o"},
    "summarize": {"primary": "gpt-4o", "fallback": "gpt-4o"}
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def classify_image_intent(prompt):
    prompt = prompt.lower()
    if any(w in prompt for w in ["photo", "realistic", "portrait", "headshot", "snapshot"]):
        return "photorealistic"
    if any(w in prompt for w in ["cartoon", "drawing", "sketch", "illustration"]):
        return "illustration"
    if any(w in prompt for w in ["logo", "icon", "branding"]):
        return "logo"
    if any(w in prompt for w in ["dragon", "castle", "fantasy", "wizard"]):
        return "fantasy"
    return "unclear"

def classify_task(prompt):
    prompt = prompt.lower()
    image_keywords = ["draw", "generate an image", "image of", "create an image", "photo of", "sketch of", "what would it look like", "visual", "show me", "what does", "picture of"]
    translate_keywords = ["translate", "into spanish", "how do you say"]
    summarize_keywords = ["summarize", "tl;dr", "in short"]

    if any(word in prompt for word in translate_keywords):
        return "translate"
    if any(word in prompt for word in summarize_keywords):
        return "summarize"
    if any(word in prompt for word in image_keywords):
        return "image"
    return "text"

def optimize_prompt(task_type, model, prompt):
    if model == "gpt-4o":
        if task_type == "text":
            return f"You are a helpful assistant. Please respond to: {prompt}"
        if task_type == "summarize":
            return f"Summarize this: {prompt}"
        if task_type == "translate":
            return f"Translate: {prompt}"
    elif model == "dall-e-3" and task_type == "image":
        intent = classify_image_intent(prompt)
        if intent == "photorealistic":
            return f"A photorealistic image of: {prompt}"
        if intent == "illustration":
            return f"A stylized cartoon illustration of: {prompt}"
        if intent == "logo":
            return f"A minimal vector-style logo of: {prompt}"
        if intent == "fantasy":
            return f"A fantasy scene of: {prompt}"
        return f"A detailed image of: {prompt}"
    return prompt

def run_model(task_type, model, prompt, user_lang="en"):
    try:
        if model == "gpt-4o":
            return run_gpt(prompt), None
        elif model == "deepl":
            return run_deepl(prompt, user_lang), None
        elif model == "dall-e-3":
            return run_dalle(prompt), None
        else:
            raise Exception("Unsupported model.")
    except Exception as e:
        fallback = MODEL_CONFIG.get(task_type, {}).get("fallback")
        if fallback != model:
            fallback_prompt = optimize_prompt(task_type, fallback, prompt)
            try:
                if fallback == "gpt-4o":
                    return run_gpt(fallback_prompt), f"⚠️ Primary model ({model}) failed. Response by fallback model."
                elif fallback == "deepl":
                    return run_deepl(fallback_prompt, user_lang), f"⚠️ Primary model ({model}) failed. Fallback used."
                elif fallback == "dall-e-3":
                    return run_dalle(fallback_prompt), f"⚠️ Primary model ({model}) failed. Fallback used."
            except Exception as e2:
                return f"❌ All models failed: {str(e2)}", None
        return f"❌ Model failed: {str(e)}", None

def run_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def run_dalle(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return response.data[0].url

def run_deepl(prompt, target_lang="EN"):
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={"auth_key": "YOUR_DEEPL_API_KEY", "text": prompt, "target_lang": target_lang}
    )
    result = response.json()
    return result["translations"][0]["text"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def unified_query():
    data = request.get_json()
    prompt = data.get("prompt")
    user_lang = detect_language(prompt)
    user_id = request.remote_addr
    history = conversation_memory.get(user_id, [])

    if history:
        prompt = f"{' '.join(history[-3:])}
{prompt}"

    task = classify_task(prompt)
    model = MODEL_CONFIG[task]["primary"]
    optimized_prompt = optimize_prompt(task, model, prompt)
    output, notice = run_model(task, model, optimized_prompt, user_lang)

    if output:
        conversation_memory[user_id] = (conversation_memory.get(user_id) or []) + [prompt, output]

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
