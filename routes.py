print('=== DEBUG: routes.py loaded from /Users/jakebaldwin/Desktop/Portal/routes.py ===')
from flask import request, jsonify, render_template, send_file, Response
from openai import OpenAI
from deep_translator import GoogleTranslator
import logging
import requests
import json
from datetime import datetime
import re

from models import select_best_model, call_claude, is_followup_image_prompt, get_last_image_prompt
from services import run_model
from utils import detect_language
from classifier.intent_classifier import classify_task

from memory import memory_manager
client = OpenAI()

def classify_task_with_context(user_input: str, memory: list) -> tuple[str, float]:
    """
    Classify task type considering conversation context and flow.
    Prioritizes conversation continuity over rigid task categorization.
    """
    # Get base classification
    base_task, base_confidence = classify_task(user_input)
    
    # Analyze conversation context
    recent_topics = []
    recent_task_types = []
    
    for entry in memory[-6:]:  # Last 6 entries
        if entry.get('role') == 'user':
            recent_topics.append(entry.get('content', '').lower())
        recent_task_types.append(entry.get('task_type', ''))
    
    # Context-aware adjustments
    input_lower = user_input.lower()
    
    # If recent conversation was about images, favor image tasks
    if any('image' in task_type for task_type in recent_task_types[-3:]):
        if any(word in input_lower for word in ['show', 'see', 'look', 'picture', 'photo', 'draw', 'create']):
            return "image", 0.85
    
    # If recent conversation was about text/analysis, favor text tasks
    if any('text' in task_type for task_type in recent_task_types[-3:]):
        if any(word in input_lower for word in ['explain', 'tell', 'what', 'how', 'why', 'describe']):
            return "text", 0.85
    
    # If user is asking about something mentioned before, it's likely text
    if any(pronoun in input_lower for pronoun in ['it', 'this', 'that', 'they', 'them']):
        if base_task == "text":
            return "text", min(base_confidence + 0.1, 1.0)
    
    # If it's a short question/command, consider context
    if len(user_input.split()) <= 5:
        # Short commands are often follow-ups
        if any(word in input_lower for word in ['more', 'again', 'also', 'too', 'same']):
            # Likely a follow-up to the previous task type
            if recent_task_types:
                last_task = recent_task_types[-1]
                if last_task in ['image', 'text', 'summarize']:
                    return last_task, 0.8
    
    # Default to base classification
    return base_task, base_confidence

def log_analytics(user_id, user_input, task_type, confidence, model, output, processing_time=None):
    """
    Log comprehensive analytics for system performance tracking.
    """
    analytics_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "user_input": user_input,
        "task_type": task_type,
        "confidence": confidence,
        "model_selected": model,
        "output_length": len(str(output)),
        "processing_time": processing_time,
        "success": True
    }
    
    # Save to analytics log
    with open("analytics.jsonl", "a") as f:
        f.write(json.dumps(analytics_entry) + "\n")

def register_routes(app):
    @app.route('/')
    def home():
        return render_template("index_new.html")
    
    @app.route('/dashboard')
    def dashboard():
        return render_template("dashboard.html")
    
    @app.route('/gallery')
    def gallery():
        return render_template("gallery.html")

    @app.route('/query', methods=['POST'])
    def query():
        start_time = datetime.utcnow()
        
        try:
            logging.info("=== QUERY ROUTE CALLED ===")
            data = request.get_json()
            user_input = data.get("input", "").strip()
            user_id = data.get("user_id", "default")
            manual_task = data.get("task", "")

            logging.info(f"[{user_id}] Prompt received: {user_input}")
            if not user_input:
                return jsonify({"error": "Empty input", "details": "Prompt is blank or invalid"}), 400

            # Enhanced task classification with intelligent routing
            if manual_task:
                task_type = manual_task
                confidence = 1.0  # Manual selection is 100% confident
                reasoning = "Manual task selection"
            else:
                # Use intelligent task routing that considers conversation context
                logging.info(f"[{user_id}] DEBUG: About to call get_intelligent_task_routing")
                task_type, confidence, reasoning = memory_manager.get_intelligent_task_routing(user_id, user_input)
                logging.info(f"[{user_id}] DEBUG: get_intelligent_task_routing returned: {task_type}, {confidence}, {reasoning}")
                logging.info(f"[{user_id}] Intelligent routing: {task_type} (confidence: {confidence:.2f}) - {reasoning}")
            
            logging.info(f"[{user_id}] Task type: {task_type} (confidence: {confidence:.2f})")

            # Handle conversational responses differently
            if task_type == "conversation":
                # Get the last response and task type for context
                recent_memory = memory_manager.get_recent_memory(user_id, 2)
                last_response = ""
                last_task_type = "text"
                
                for entry in reversed(recent_memory):
                    if entry.get('role') == 'assistant':
                        last_response = entry.get('content', '')
                        last_task_type = entry.get('task_type', 'text')
                        break
                
                # Generate conversational response
                output = memory_manager.get_conversational_response(user_input, user_id, last_response, last_task_type)
                model = "conversation"
                
                # Store user input and response in memory
                memory_manager.save_memory(user_id, "user", user_input, task_type)
                memory_manager.save_memory(user_id, "assistant", output, task_type, model)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                return jsonify({
                    "model": model,
                    "output": output,
                    "task_type": task_type,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "reasoning": reasoning
                })

            # Language detection (for translation)
            lang = detect_language(user_input)

            # Store user input in memory
            memory_manager.save_memory(user_id, "user", user_input, task_type)

            model = select_best_model(task_type, user_input, lang, confidence)
            print(f"DEBUG: Model selected: {model}")  # Direct print for debugging
            logging.info(f"[{user_id}] Model selected: {model}")
            logging.info(f"[{user_id}] DEBUG: Called select_best_model with task_type={task_type}, user_input={user_input[:50]}...")

            # Use enhanced model-specific prompt optimization
            prompt = user_input  # No model-specific optimization function available
            logging.info(f"[{user_id}] Optimized prompt: {prompt}")

            # Use enhanced model execution with fallbacks
            result = run_model(task_type, model, prompt, user_id)
            output = result[0] if isinstance(result, tuple) else result
            
            # Ensure output is not empty or None
            if not output or output.strip() == "":
                output = "I'm sorry, I couldn't generate a response. Please try asking your question again."
                logging.warning(f"[{user_id}] Empty output from model, using fallback message")
            
            logging.info(f"[{user_id}] Output: {str(output)[:120]}...")

            # Store assistant response in memory
            memory_manager.save_memory(user_id, "assistant", output, task_type, model)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Enhanced analytics and metrics tracking
            log_analytics(user_id, user_input, task_type, confidence, model, output, processing_time)
            
            # Track comprehensive metrics
            from models import comprehensive_metrics
            comprehensive_metrics.track_request(
                user_id=user_id,
                task_type=task_type,
                model=model,
                prompt=user_input,
                response=output,
                response_time=processing_time,
                success=True,
                confidence_score=confidence
            )

            return jsonify({
                "model": model,
                "output": output,
                "task_type": task_type,
                "confidence": confidence,
                "processing_time": processing_time,
                "reasoning": reasoning
            })

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logging.exception("Error in /query route")
            
            # Enhanced error tracking
            log_analytics(user_id, user_input, task_type, 0.0, "error", str(e), processing_time)
            
            # Track error metrics
            from models import comprehensive_metrics
            comprehensive_metrics.track_request(
                user_id=user_id,
                task_type=task_type if 'task_type' in locals() else "unknown",
                model="error",
                prompt=user_input,
                response=str(e),
                response_time=processing_time,
                success=False,
                error_message=str(e),
                confidence_score=0.0
            )
            
            return jsonify({"error": "Something went wrong.", "details": str(e)}), 500

    @app.route('/clear_memory', methods=['POST'])
    def clear_memory():
        """Clear conversation memory for the user"""
        try:
            data = request.get_json()
            user_id = data.get("user_id", "default")
            
            memory_manager.clear_memory(user_id)
            logging.info(f"[{user_id}] Memory cleared")
            
            return jsonify({"success": True, "message": "Memory cleared successfully"})
        except Exception as e:
            logging.exception("Error clearing memory")
            return jsonify({"error": "Failed to clear memory", "details": str(e)}), 500

    @app.route('/memory_stats', methods=['GET'])
    def get_memory_stats():
        """Get memory statistics for monitoring"""
        try:
            user_id = request.args.get("user_id", "default")
            stats = memory_manager.get_memory_stats(user_id)
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": "Failed to get memory stats", "details": str(e)}), 500

    @app.route('/analytics', methods=['GET'])
    def get_analytics():
        """
        Get system analytics for monitoring performance.
        """
        try:
            analytics = []
            with open("analytics.jsonl", "r") as f:
                for line in f:
                    analytics.append(json.loads(line.strip()))
            
            # Calculate summary statistics
            total_queries = len(analytics)
            successful_queries = len([a for a in analytics if a.get("success", False)])
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Model usage statistics
            model_usage = {}
            task_distribution = {}
            avg_confidence = 0
            avg_processing_time = 0
            
            for entry in analytics:
                model = entry.get("model_selected", "unknown")
                task = entry.get("task_type", "unknown")
                confidence = entry.get("confidence", 0)
                processing_time = entry.get("processing_time", 0)
                
                model_usage[model] = model_usage.get(model, 0) + 1
                task_distribution[task] = task_distribution.get(task, 0) + 1
                avg_confidence += confidence
                avg_processing_time += processing_time
            
            if total_queries > 0:
                avg_confidence /= total_queries
                avg_processing_time /= total_queries
            
            return jsonify({
                "total_queries": total_queries,
                "success_rate": success_rate,
                "model_usage": model_usage,
                "task_distribution": task_distribution,
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "recent_queries": analytics[-10:] if analytics else []
            })
            
        except FileNotFoundError:
            return jsonify({"error": "No analytics data available"}), 404
        except Exception as e:
            return jsonify({"error": "Failed to load analytics", "details": str(e)}), 500

    @app.route('/autocomplete')
    def autocomplete():
        query = (request.args.get('query') or '').strip().lower()
        if not query or len(query) < 2:
            return jsonify([])

        # Aggregate recent queries from analytics.jsonl
        suggestions = {}
        try:
            with open('analytics.jsonl') as f:
                for line in f:
                    entry = json.loads(line)
                    user_input = entry.get('user_input', '').lower()
                    if user_input.startswith(query):
                        suggestions[user_input] = suggestions.get(user_input, 0) + 1
        except Exception:
            pass

        # Sort by frequency (commonality) - most common first
        ranked = sorted(suggestions.items(), key=lambda x: -x[1])
        suggestion_list = [s[0] for s in ranked]

        # If not enough suggestions, call GPT-4o for more
        if len(suggestion_list) < 3:
            try:
                from models import client
                prompt = f"Suggest 3 likely ways to complete this user query, based on common user behavior. Only return the completions, one per line.\nQuery: '{query}'\nCompletions:"
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.3
                )
                ai_suggestions = (response.choices[0].message.content or '').strip().split('\n')
                # Clean and filter
                ai_suggestions = [s.strip('- ').strip() for s in ai_suggestions if s.strip()]
                for s in ai_suggestions:
                    if s.lower().startswith(query) and s not in suggestion_list:
                        suggestion_list.append(s)
            except Exception as e:
                print(f"AI autocomplete error: {e}")

        # Return top 5 suggestions in reverse order (most common at bottom)
        return jsonify(suggestion_list[:5][::-1])

    @app.route('/test_dalle')
    def test_dalle():
        from services import run_dalle
        result = run_dalle("a cat")
        return jsonify({"result": result})

    @app.route('/proxy_image')
    def proxy_image():
        url = request.args.get('url')
        # Only allow proxying for OpenAI DALL-E image URLs
        allowed_pattern = re.compile(r'^https://oaidalleapiprodscus\.blob\.core\.windows\.net/private/.*\\.png(\?.*)?$')
        if not url or not allowed_pattern.match(url):
            return Response('Invalid or disallowed image URL', status=400)
        try:
            r = requests.get(url, stream=True, timeout=10)
            r.raise_for_status()
            return Response(r.content, mimetype='image/png')
        except Exception as e:
            return Response(f'Failed to fetch image: {e}', status=502)

   