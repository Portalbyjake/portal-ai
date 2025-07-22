# pyright: ignore[reportOptionalMemberAccess]
from flask import request, jsonify
import logging
from datetime import datetime
import json
from typing import Dict, Optional, cast, Any
from flask import Blueprint

from models import model_orchestrator, run_model
from memory import conversation_memory as memory_manager
from classifier.intent_classifier import classify_task
from prompt_optimizer import IntelligentPromptOptimizer
from utils import detect_language

class APIKeyManager:
    """Manage API keys for Portal AI API-as-a-Service"""
    
    def __init__(self):
        self.api_keys = {}  # In production, use database
        self.usage_limits = {
            "free": {"requests_per_day": 100, "models": ["gpt-4o", "claude-haiku"]},
            "pro": {"requests_per_day": 1000, "models": ["gpt-4o", "claude-sonnet-4", "dall-e-3"]},
            "enterprise": {"requests_per_day": 10000, "models": ["*"]}
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user info"""
        if api_key in self.api_keys:
            return self.api_keys[api_key]
        return None
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if user has exceeded rate limit"""
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False
        
        # Check daily limit
        today = datetime.utcnow().date().isoformat()
        daily_usage = user_info.get("daily_usage", {}).get(today, 0)
        plan = user_info.get("plan", "free")
        limit = self.usage_limits[plan]["requests_per_day"]
        
        return daily_usage < limit
    
    def increment_usage(self, api_key: str):
        """Increment usage counter"""
        if api_key in self.api_keys:
            today = datetime.utcnow().date().isoformat()
            if "daily_usage" not in self.api_keys[api_key]:
                self.api_keys[api_key]["daily_usage"] = {}
            self.api_keys[api_key]["daily_usage"][today] = self.api_keys[api_key]["daily_usage"].get(today, 0) + 1

# Global API key manager
api_key_manager = APIKeyManager()

api = Blueprint('api', __name__)

def register_api_routes(app):
    """Register API routes for Portal AI API-as-a-Service"""
    
    @app.route('/api/v1/query', methods=['POST'])
    def api_query():
        """Main API endpoint for Portal AI orchestration"""
        start_time = datetime.utcnow()
        
        # Validate API key
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        user_info = api_key_manager.validate_api_key(api_key)
        if not user_info:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Check rate limit
        if not api_key_manager.check_rate_limit(api_key):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        try:
            data = request.get_json()
            user_input = data.get("input", "").strip()
            user_id = data.get("user_id", user_info.get("user_id", "api_user"))
            manual_task = data.get("task", "")
            budget_constraint = data.get("budget_constraint")
            
            if not user_input:
                return jsonify({"error": "Empty input"}), 400
            
            # Task classification
            if manual_task:
                task_type = manual_task
                confidence = 1.0
            else:
                task_type, confidence = classify_task(user_input)
            
            # Language detection
            lang = detect_language(user_input)
            
            # Store in memory
            memory_manager.save_memory(user_id, "user", user_input, task_type)
            
            # Enhanced model selection with orchestration
            model, confidence = model_orchestrator.get_best_model_for_task(
                task_type, user_input, confidence, budget_constraint
            )
            
            # Prompt optimization
            optimizer = IntelligentPromptOptimizer()
            prompt = optimizer.optimize_prompt(user_input)
            
            # Run model with performance tracking
            model_start_time = datetime.utcnow()
            output = run_model(model, task_type, prompt, user_id)
            model_response_time = (datetime.utcnow() - model_start_time).total_seconds()
            
            # Update performance metrics
            success = not str(output).startswith("❌")
            model_orchestrator.update_performance(model, success, model_response_time)
            
            # Store response
            memory_manager.save_memory(user_id, "assistant", output, task_type, model)
            
            # Increment usage
            api_key_manager.increment_usage(api_key)
            
            # Calculate total processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return jsonify({
                "model": model,
                "output": output,
                "task_type": task_type,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_response_time": model_response_time,
                "success": success
            })
            
        except Exception as e:
            logging.exception("API query error")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    @app.route('/api/v1/models', methods=['GET'])
    def api_models():
        """Get available models and their capabilities"""
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        user_info = api_key_manager.validate_api_key(api_key)
        if not user_info:
            return jsonify({"error": "Invalid API key"}), 401
        
        plan = user_info.get("plan", "free")
        available_models = api_key_manager.usage_limits[plan]["models"]
        
        # Get model information
        models_info = {
            "gpt-4o": {
                "provider": "OpenAI",
                "capabilities": ["text", "code", "analysis"],
                "cost_per_1k_tokens": 0.03,
                "max_tokens": 128000
            },
            "claude-sonnet-4": {
                "provider": "Anthropic",
                "capabilities": ["text", "analysis", "reasoning"],
                "cost_per_1k_tokens": 0.015,
                "max_tokens": 200000
            },
            "dall-e-3": {
                "provider": "OpenAI",
                "capabilities": ["image"],
                "cost_per_image": 0.04,
                "resolution": "1024x1024"
            },
            "stablediffusion": {
                "provider": "Stability AI",
                "capabilities": ["image"],
                "cost_per_image": 0.02,
                "resolution": "1024x1024"
            }
        }
        
        # Filter based on plan
        if available_models != ["*"]:
            models_info = {k: v for k, v in models_info.items() if k in available_models}
        
        return jsonify({
            "available_models": models_info,
            "plan": plan,
            "rate_limit": api_key_manager.usage_limits[plan]["requests_per_day"]
        })
    
    @app.route('/api/v1/usage', methods=['GET'])
    def api_usage():
        """Get API usage statistics"""
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        user_info = api_key_manager.validate_api_key(api_key)
        if not user_info:
            return jsonify({"error": "Invalid API key"}), 401
        
        plan = user_info.get("plan", "free")
        limit = api_key_manager.usage_limits[plan]["requests_per_day"]
        today = datetime.utcnow().date().isoformat()
        used = user_info.get("daily_usage", {}).get(today, 0)
        
        return jsonify({
            "plan": plan,
            "daily_limit": limit,
            "used_today": used,
            "remaining": max(0, limit - used),
            "reset_time": "00:00 UTC"
        })
    
    @app.route('/api/v1/analytics', methods=['GET'])
    def api_analytics():
        """Get orchestration analytics"""
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        user_info = api_key_manager.validate_api_key(api_key)
        if not user_info:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Get orchestration stats
        stats = model_orchestrator.get_orchestration_stats()
        
        return jsonify({
            "orchestration_stats": stats,
            "user_plan": user_info.get("plan", "free")
        })
    
    @app.route('/api/dashboard', methods=['GET'])
    def api_dashboard():
        """Get dashboard analytics data"""
        try:
            # Load analytics data
            analytics = []
            try:
                with open("analytics.jsonl", "r") as f:
                    for line in f:
                        analytics.append(json.loads(line.strip()))
            except FileNotFoundError:
                analytics = []
            
            # Calculate statistics
            total_requests = len(analytics)
            successful_requests = len([a for a in analytics if a.get("success", False)])
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate average response time
            response_times = [a.get("processing_time", 0) for a in analytics if a.get("processing_time")]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Model usage statistics
            model_usage = {}
            for entry in analytics:
                model = entry.get("model_selected", "unknown")
                if model not in model_usage:
                    model_usage[model] = {"requests": 0, "success_rate": 0, "avg_time": 0}
                
                model_usage[model]["requests"] += 1
                if entry.get("success", False):
                    model_usage[model]["success_rate"] += 1
            
            # Calculate success rates
            for model in model_usage:
                total = model_usage[model]["requests"]
                successful = model_usage[model]["success_rate"]
                model_usage[model]["success_rate"] = (successful / total * 100) if total > 0 else 0
            
            # Recent activity (last 10 entries)
            recent_activity = []
            for entry in analytics[-10:]:
                recent_activity.append({
                    "task_type": entry.get("task_type", "unknown"),
                    "model": entry.get("model_selected", "unknown"),
                    "timestamp": entry.get("timestamp", ""),
                    "duration": round(entry.get("processing_time", 0) * 1000)  # Convert to ms
                })
            
            # Active users (users with activity in last 24 hours)
            from datetime import datetime, timedelta
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            active_users = len(set([
                entry.get("user_id", "unknown") 
                for entry in analytics 
                if entry.get("timestamp", "") > cutoff
            ]))
            
            return jsonify({
                "total_requests": total_requests,
                "success_rate": round(success_rate, 1),
                "avg_response_time": round(avg_response_time, 2),
                "active_users": active_users,
                "model_usage": model_usage,
                "recent_activity": recent_activity
            })
            
        except Exception as e:
            logging.exception("Dashboard API error")
            return jsonify({"error": "Failed to load dashboard data", "details": str(e)}), 500
    
    @app.route('/api/gallery', methods=['GET'])
    def api_gallery():
        """Get gallery of generated images"""
        try:
            # Load memory to find image entries
            images = []
            try:
                with open("memory_text.jsonl", "r") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get("task_type") == "image" and entry.get("role") == "assistant":
                            # Extract image URL from content
                            content = entry.get("content", "")
                            if content and (content.startswith("http") or content.startswith("data:")):
                                images.append({
                                    "url": content,
                                    "prompt": entry.get("user_input", "Generated Image"),
                                    "model": entry.get("model", "unknown"),
                                    "timestamp": entry.get("timestamp", "")
                                })
            except FileNotFoundError:
                pass
            
            # Sort by timestamp (newest first)
            images.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return jsonify({
                "images": images,
                "total_images": len(images)
            })
            
        except Exception as e:
            logging.exception("Gallery API error")
            return jsonify({"error": "Failed to load gallery data", "details": str(e)}), 500 

@api.route('/image/undo', methods=['POST'])
def undo_last_image():
    user_id = request.json.get('user_id', 'default')
    undone = memory_manager.undo_last_image(user_id)
    if undone:
        return jsonify({"success": True, "undone": undone}), 200
    else:
        return jsonify({"success": False, "error": "No image to undo."}), 400

@api.route('/image/history', methods=['GET'])
def get_image_history():
    user_id = request.args.get('user_id', 'default')
    history = memory_manager.get_image_history(user_id)
    return jsonify({"history": history}), 200

@api.route('/image/last_prompt', methods=['GET'])
def get_last_image_prompt():
    user_id = request.args.get('user_id', 'default')
    last_entry = memory_manager.get_last_image_entry(user_id)
    if last_entry is not None and isinstance(last_entry, dict):
        entry: Dict[str, Any] = cast(Dict[str, Any], last_entry)
        return jsonify({
            "prompt": entry.get('prompt'),
            "summary": entry.get('summary'),
            "url": entry.get('url')
        }), 200
    return jsonify({"error": "No image found."}), 404 