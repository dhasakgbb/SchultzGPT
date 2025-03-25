"""
State and cache models for SchultzGPT.
Manages application state, caching, and persistence.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import json
import os
import hashlib
import time
import pickle
from datetime import datetime
from functools import wraps


class SchultzState:
    """Class to manage the application state for SchultzGPT."""
    
    def __init__(self, state_file: str = "state.json"):
        self.state_file = state_file
        self.mood = "neutral"
        self.persona_level = 1  # Base persona level (1-3)
        self.last_interaction = datetime.now()
        self.caching_enabled = True
        self.debug_mode = False
        self.retrieval_store_enabled = True
        self.context_window_size = 10  # Number of messages to include in context
        self.temperature_modifier = 0.0
        self.performance_metrics: Dict[str, Any] = {
            "api_calls": 0,
            "avg_response_time": 0.0,
            "total_tokens_used": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        self.custom_settings: Dict[str, Any] = {}
        
        # Load state from file if it exists
        self.load_state()
    
    def save_state(self) -> None:
        """Save the current state to a file."""
        state_data = {
            "mood": self.mood,
            "persona_level": self.persona_level,
            "last_interaction": self.last_interaction.isoformat(),
            "caching_enabled": self.caching_enabled,
            "debug_mode": self.debug_mode,
            "retrieval_store_enabled": self.retrieval_store_enabled,
            "context_window_size": self.context_window_size,
            "temperature_modifier": self.temperature_modifier,
            "performance_metrics": self.performance_metrics,
            "custom_settings": self.custom_settings
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def load_state(self) -> None:
        """Load state from a file if it exists."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
                
            self.mood = state_data.get("mood", "neutral")
            self.persona_level = state_data.get("persona_level", 1)
            self.last_interaction = datetime.fromisoformat(
                state_data.get("last_interaction", datetime.now().isoformat())
            )
            self.caching_enabled = state_data.get("caching_enabled", True)
            self.debug_mode = state_data.get("debug_mode", False)
            
            # Handle both old and new keys for backward compatibility
            self.retrieval_store_enabled = state_data.get(
                "retrieval_store_enabled", 
                state_data.get("vector_store_enabled", True)
            )
            
            self.context_window_size = state_data.get("context_window_size", 10)
            self.temperature_modifier = state_data.get("temperature_modifier", 0.0)
            self.performance_metrics = state_data.get("performance_metrics", {})
            self.custom_settings = state_data.get("custom_settings", {})
            
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def update_mood(self, new_mood: str) -> None:
        """Update the current mood."""
        self.mood = new_mood
        self.last_interaction = datetime.now()
        self.save_state()
    
    def toggle_caching(self) -> bool:
        """Toggle caching on/off and return new state."""
        self.caching_enabled = not self.caching_enabled
        self.save_state()
        return self.caching_enabled
    
    def toggle_debug(self) -> bool:
        """Toggle debug mode on/off and return new state."""
        self.debug_mode = not self.debug_mode
        self.save_state()
        return self.debug_mode
    
    def toggle_retrieval_store(self) -> bool:
        """Toggle retrieval store on/off and return new state."""
        self.retrieval_store_enabled = not self.retrieval_store_enabled
        self.save_state()
        return self.retrieval_store_enabled
    
    def set_context_window(self, size: int) -> None:
        """Set the context window size."""
        if size > 0:
            self.context_window_size = size
            self.save_state()
    
    def set_temperature_modifier(self, modifier: float) -> None:
        """Set the temperature modifier within reasonable bounds."""
        # Keep the modifier in a reasonable range
        if -0.5 <= modifier <= 0.5:
            self.temperature_modifier = modifier
            self.save_state()
    
    def update_performance_metric(self, metric: str, value: Any) -> None:
        """Update a specific performance metric."""
        if metric in self.performance_metrics:
            if metric in ["api_calls", "successful_requests", "failed_requests"]:
                self.performance_metrics[metric] += 1
            elif metric == "total_tokens_used" and isinstance(value, int):
                self.performance_metrics[metric] += value
            elif metric == "avg_response_time" and isinstance(value, (int, float)):
                # Calculate running average
                curr_avg = self.performance_metrics[metric]
                curr_count = self.performance_metrics["api_calls"]
                if curr_count > 0:
                    new_avg = ((curr_avg * (curr_count - 1)) + value) / curr_count
                    self.performance_metrics[metric] = new_avg
        
        # Save after significant changes (e.g., every 5 API calls)
        if (self.performance_metrics.get("api_calls", 0) % 5) == 0:
            self.save_state()
    
    def get_performance_report(self) -> str:
        """Get a formatted performance report."""
        return (
            f"Performance Metrics:\n"
            f"- API Calls: {self.performance_metrics.get('api_calls', 0)}\n"
            f"- Success Rate: {self._calculate_success_rate():.1f}%\n"
            f"- Avg Response Time: {self.performance_metrics.get('avg_response_time', 0):.2f}s\n"
            f"- Total Tokens Used: {self.performance_metrics.get('total_tokens_used', 0)}\n"
        )
    
    def _calculate_success_rate(self) -> float:
        """Calculate the API success rate as a percentage."""
        successful = self.performance_metrics.get("successful_requests", 0)
        failed = self.performance_metrics.get("failed_requests", 0)
        total = successful + failed
        if total == 0:
            return 100.0
        return (successful / total) * 100.0
    
    def set_custom_setting(self, key: str, value: Any) -> None:
        """Set a custom setting value."""
        self.custom_settings[key] = value
        self.save_state()
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get a custom setting value with optional default."""
        return self.custom_settings.get(key, default)
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.performance_metrics = {
            "api_calls": 0,
            "avg_response_time": 0.0,
            "total_tokens_used": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        self.save_state()


class ResponseCache:
    """Cache for API responses to reduce API calls and latency."""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 86400):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.enabled = True
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_key(self, data: Any) -> str:
        """Generate a unique key from input data."""
        # Convert data to a consistent string representation
        if isinstance(data, dict):
            # Sort dictionary keys for consistent hashing
            serialized = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            serialized = json.dumps(data)
        else:
            serialized = str(data)
        
        # Create hash from the serialized data
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and isn't expired."""
        if not self.enabled:
            return None
            
        cache_path = os.path.join(self.cache_dir, f"{key}.cache")
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            timestamp = cache_data.get('timestamp', 0)
            value = cache_data.get('value')
            
            # Check if cache has expired
            if time.time() - timestamp > self.ttl:
                # Remove expired cache
                os.remove(cache_path)
                return None
                
            return value
        except Exception as e:
            # If any error occurs, return None
            print(f"Cache error: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        if not self.enabled:
            return
            
        cache_path = os.path.join(self.cache_dir, f"{key}.cache")
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'value': value
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Cache write error: {str(e)}")
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache."""
        cache_path = os.path.join(self.cache_dir, f"{key}.cache")
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                return True
            except Exception:
                return False
        return False
    
    def clear_all(self) -> int:
        """Clear all cached responses. Returns count of deleted files."""
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    count += 1
            return count
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return count
    
    def toggle(self) -> bool:
        """Toggle cache on/off. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
        total_size = 0
        
        for file in files:
            file_path = os.path.join(self.cache_dir, file)
            total_size += os.path.getsize(file_path)
        
        return {
            "enabled": self.enabled,
            "entries": len(files),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "ttl_seconds": self.ttl
        }


def cached(cache_instance: ResponseCache):
    """
    Decorator for caching function results.
    
    Usage:
        cache = ResponseCache()
        
        @cached(cache)
        def expensive_function(param1, param2):
            # ...
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache_instance.enabled:
                return func(*args, **kwargs)
            
            # Create a dictionary of the function call
            call_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            
            # Generate a cache key
            cache_key = cache_instance._generate_key(call_data)
            
            # Check if result is in cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute the function and cache the result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result
            
        return wrapper
    return decorator 