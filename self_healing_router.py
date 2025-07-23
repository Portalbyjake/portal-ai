import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from models import run_model_with_fallbacks, check_model_availability
import hashlib

class SelfHealingRouter:
    def __init__(self):
        self.response_cache = {}
        self.model_health = {}
        self.circuit_breakers = {}
        self.cache_ttl = 300  # 5 minutes
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_timeout = 60  # 1 minute
        
    def route_with_healing(self, model_name: str, task_type: str, prompt: str, user_id: str = None) -> str:
        """Route request with self-healing capabilities including caching and circuit breakers."""
        try:
            if self._is_circuit_open(model_name):
                logging.warning(f"Circuit breaker open for {model_name}, using fallback")
                return self._use_cached_or_fallback(model_name, task_type, prompt, user_id)
            
            cache_key = self._generate_cache_key(prompt, model_name, task_type)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if self._is_cache_valid(cached_response):
                    logging.info(f"Cache hit for {model_name}")
                    return cached_response['response']
                else:
                    del self.response_cache[cache_key]
            
            start_time = time.time()
            try:
                result = run_model_with_fallbacks(model_name, task_type, prompt, user_id)
                response_time = time.time() - start_time
                
                self._update_model_health(model_name, True, response_time)
                
                self._cache_response(cache_key, result, response_time)
                
                logging.info(f"Successfully routed to {model_name} in {response_time:.2f}s")
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                logging.error(f"Model {model_name} failed: {e}")
                
                self._update_model_health(model_name, False, response_time)
                self._increment_circuit_breaker(model_name)
                
                return self._use_cached_or_fallback(model_name, task_type, prompt, user_id)
                
        except Exception as e:
            logging.error(f"Self-healing router error: {e}")
            return run_model_with_fallbacks(model_name, task_type, prompt, user_id)
    
    def _generate_cache_key(self, prompt: str, model: str, task_type: str) -> str:
        """Generate a cache key for the request."""
        content = f"{prompt}:{model}:{task_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_response: Dict) -> bool:
        """Check if cached response is still valid."""
        cache_time = cached_response.get('timestamp', 0)
        return time.time() - cache_time < self.cache_ttl
    
    def _cache_response(self, cache_key: str, response: str, response_time: float):
        """Cache a successful response."""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time(),
            'response_time': response_time
        }
        
        if len(self.response_cache) > 1000:
            sorted_cache = sorted(self.response_cache.items(), 
                                key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_cache[:100]:
                del self.response_cache[key]
    
    def _is_circuit_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for a model."""
        if model_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[model_name]
        if breaker['failures'] >= self.circuit_breaker_threshold:
            if time.time() - breaker['last_failure'] > self.circuit_breaker_timeout:
                self.circuit_breakers[model_name] = {'failures': 0, 'last_failure': 0}
                return False
            return True
        return False
    
    def _increment_circuit_breaker(self, model_name: str):
        """Increment circuit breaker failure count."""
        if model_name not in self.circuit_breakers:
            self.circuit_breakers[model_name] = {'failures': 0, 'last_failure': 0}
        
        self.circuit_breakers[model_name]['failures'] += 1
        self.circuit_breakers[model_name]['last_failure'] = time.time()
    
    def _update_model_health(self, model_name: str, success: bool, response_time: float):
        """Update model health metrics."""
        if model_name not in self.model_health:
            self.model_health[model_name] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0.0,
                'avg_response_time': 0.0,
                'success_rate': 0.0
            }
        
        health = self.model_health[model_name]
        health['total_requests'] += 1
        health['total_response_time'] += response_time
        
        if success:
            health['successful_requests'] += 1
            if model_name in self.circuit_breakers:
                self.circuit_breakers[model_name]['failures'] = 0
        
        health['success_rate'] = health['successful_requests'] / health['total_requests']
        health['avg_response_time'] = health['total_response_time'] / health['total_requests']
    
    def _use_cached_or_fallback(self, model_name: str, task_type: str, prompt: str, user_id: str = None) -> str:
        """Use cached response or fallback to another model."""
        cache_key_base = hashlib.md5(f"{prompt}:{task_type}".encode()).hexdigest()
        
        for key, cached in self.response_cache.items():
            if cache_key_base in key and self._is_cache_valid(cached):
                logging.info(f"Using cached response as fallback for {model_name}")
                return cached['response']
        
        logging.info(f"No cache available, using original fallback for {model_name}")
        return run_model_with_fallbacks(model_name, task_type, prompt, user_id)
    
    def get_health_stats(self) -> Dict:
        """Get health statistics for all models."""
        return {
            'model_health': self.model_health,
            'circuit_breakers': self.circuit_breakers,
            'cache_size': len(self.response_cache)
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        logging.info("Response cache cleared")
