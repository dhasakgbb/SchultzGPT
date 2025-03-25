#!/usr/bin/env python3
import os
import json
import time
import random
import re
import sys
import asyncio
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Callable, Coroutine
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import tiktoken
import readline  # For better input handling with arrow keys
import traceback

# Import UI components
from schultz_ui import TerminalUI, console

# Load environment variables and initialize clients
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)  # Async client for concurrent operations
FINE_TUNED_MODEL = os.getenv("CURRENT_MODEL_ID")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Create rich console
console = Console()

# Performance tracking
class PerformanceTracker:
    """Track execution time for different components to identify bottlenecks"""
    
    def __init__(self):
        self.metrics = {}
        self.enabled = True
        self.start_times = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
    
    def start_timer(self, component_name: str):
        """Start timing a component"""
        if not self.enabled:
            return
        
        self.start_times[component_name] = time.time()
    
    def stop_timer(self, component_name: str, success: bool = True):
        """Stop timing a component and record the result"""
        if not self.enabled or component_name not in self.start_times:
            return
        
        elapsed = time.time() - self.start_times.pop(component_name)
        
        if component_name not in self.metrics:
            self.metrics[component_name] = {
                "calls": 0,
                "total_time": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0
            }
        
        self.metrics[component_name]["calls"] += 1
        self.metrics[component_name]["total_time"] += elapsed
        
        if success:
            self.metrics[component_name]["successes"] += 1
        else:
            self.metrics[component_name]["failures"] += 1
            
        self.metrics[component_name]["avg_time"] = (
            self.metrics[component_name]["total_time"] / 
            self.metrics[component_name]["calls"]
        )
    
    def get_report(self) -> Dict:
        """Get a performance report"""
        return {
            component: {
                "calls": data["calls"],
                "avg_time": f"{data['avg_time']:.3f}s",
                "total_time": f"{data['total_time']:.2f}s",
                "success_rate": f"{(data['successes'] / data['calls'] * 100) if data['calls'] > 0 else 0:.1f}%"
            } 
            for component, data in self.metrics.items()
        }
    
    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics = {}
        self.start_times = {}
    
    def run_async_in_thread(self, coro):
        """Run an async coroutine in a separate thread"""
        loop = asyncio.new_event_loop()
        
        def run_coro(loop, coro):
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        
        return self.thread_pool.submit(run_coro, loop, coro).result()

# Create global performance tracker
performance = PerformanceTracker()

# Create timing decorator for functions
def timed(component_name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = component_name or func.__name__
            performance.start_timer(name)
            try:
                result = func(*args, **kwargs)
                performance.stop_timer(name, success=True)
                return result
            except Exception as e:
                performance.stop_timer(name, success=False)
                raise e
        return wrapper
    return decorator

# Create timing decorator for async functions
def async_timed(component_name: str = None):
    """Decorator to time async function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = component_name or func.__name__
            performance.start_timer(name)
            try:
                result = await func(*args, **kwargs)
                performance.stop_timer(name, success=True)
                return result
            except Exception as e:
                performance.stop_timer(name, success=False)
                raise e
        return wrapper
    return decorator

# Configuration and Constants
class Config:
    MAX_TOKENS = 4096
    RESPONSE_TOKENS = 150 
    CONTEXT_TOKENS = 1000
    HISTORY_MESSAGES = 6
    SIMILAR_RESPONSES = 5
    SIMILARITY_THRESHOLD = 0.7
    MODEL_TEMPERATURE = 1.1
    SPIRAL_TEMPERATURE = 1.5  # Higher temperature for spiral mode
    
    # Vector store configuration
    VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "jon-memory-store")
    USE_VECTOR_STORE = True  # Always use OpenAI's vector store
    
    # Accuracy optimization features
    ACCURACY_FEATURES = {
        "quality_check": True,      # Run quality verification on responses
        "chain_of_thought": True,   # Use chain-of-thought for complex queries
        "consistency_examples": True, # Include consistency examples in style guide
        "max_regenerations": 2      # Maximum regeneration attempts for low quality
    }
    
    # Commands for quick reference
    COMMANDS = [
        ("!spiral", "Too much honesty mode"),
        ("!reset", "Clear memory"),
        ("/mood-filter <mood>", "Filter by mood"),
        ("/topic-filter <topic>", "Filter by topic"),
        ("/analyze", "Analyze conversation"),
        ("/cache on|off", "Enable/disable prompt caching"),
        ("/cache-stats", "Show prompt cache statistics"),
        ("/cleanup <days>", "Remove old vectors from store"),
        ("/accuracy on|off", "Toggle accuracy optimizations"),
        ("exit/quit", "Exit chat")
    ]

    # System prompt for overall character guidance
    SYSTEM_PROMPT = """You are simulating a conversation as Jon with the user.
Jon is a witty, intelligent but sometimes cynical friend. He texts in a casual style, often using lowercase and minimal punctuation.
He has a dry sense of humor, is somewhat sarcastic, but also genuinely supportive when his friends need him.
He values honesty and doesn't sugarcoat his opinions."""
    
    # Additional persona details to inform responses
    PERSONA = """Jon's texting style:
- Uses lowercase, abbreviations, and minimal punctuation
- Often doesn't capitalize or use apostrophes
- Casual and conversational
- Sends brief messages, typically 1-3 sentences
- Occasionally uses "lol", "haha", "idk", "u" for you, "ur" for your
- More likely to include punctuation when he's excited or concerned

Jon's personality:
- Has a dry, sometimes self-deprecating sense of humor
- Intelligent and well-read but doesn't show off
- Can be a bit cynical about politics and social media
- Deeply loyal to friends
- Values authenticity and dislikes phoniness
- Gets into "spiraling" moods where he's more negative and raw
- Works in tech but has creative writing aspirations
- Enjoys discussing books, movies, politics, and technology"""

# Move these functions here, before they're called
@async_timed("async_chat_completion")
async def async_chat_completion(messages, model, temperature=0.7, max_tokens=150, response_format=None, user_id=None):
    """Async version of chat completion for concurrent processing"""
    try:
        # Prepare parameters for the API call
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        if response_format:
            params["response_format"] = response_format
            
        # Add user ID if provided (for tracking)
        if user_id:
            params["user"] = user_id
        
        # Make the async API call
        response = await async_client.chat.completions.create(**params)
        return response
        
    except Exception as e:
        console.print(f"[bold red]Error in async chat completion: {str(e)}[/bold red]")
        raise

@async_timed("async_embeddings_create") 
async def async_embeddings_create(texts, model="text-embedding-3-small"):
    """Async version of embeddings creation for parallel processing"""
    try:
        response = await async_client.embeddings.create(
            input=texts,
            model=model
        )
        return response
    except Exception as e:
        console.print(f"[bold red]Error in async embeddings: {str(e)}[/bold red]")
        raise

@timed("batch_async_operations")
def batch_async_operations(operations):
    """Run multiple async operations concurrently and return results"""
    return performance.run_async_in_thread(asyncio.gather(*operations))

class RetryHandler:
    """Handle retries with simple backoff"""
    
    def __init__(self, max_retries=2):
        self.max_retries = max_retries
    
    @timed("retry_operation")
    def retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with simple backoff"""
        for retry in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                # Last retry failed
                if retry >= self.max_retries:
                    raise
                
                # Simple backoff
                delay = 1.0 * (retry + 1)
                console.print(f"[yellow]Retrying after error: {str(e)} (retry {retry+1}/{self.max_retries})[/yellow]")
                time.sleep(delay)
    
    @timed("retry_async_operation")
    async def retry_async_operation(self, async_operation, *args, **kwargs):
        """Retry an async operation with simple backoff"""
        for retry in range(self.max_retries + 1):
            try:
                return await async_operation(*args, **kwargs)
            except Exception as e:
                # Last retry failed
                if retry >= self.max_retries:
                    raise
                
                # Simple backoff
                delay = 1.0 * (retry + 1)
                console.print(f"[yellow]Retrying async operation after error (retry {retry+1}/{self.max_retries})[/yellow]")
                await asyncio.sleep(delay)

# Create global retry handler
retry_handler = RetryHandler()

@timed("cached_chat_completion")
def cached_chat_completion(messages, model, temperature=0.7, max_tokens=150, response_format=None, user_id=None):
    """Make a chat completion call with OpenAI's prompt caching and automatic retries"""
    try:
        # Create cache key from request parameters
        cache_key = controller.prompt_cache.get_cache_key(messages, model, temperature, max_tokens)
        
        # Check if we have a cached request ID
        prev_request_id = None
        if controller.prompt_cache.cache_enabled and controller.prompt_cache.has_cached_request(cache_key):
            prev_request_id = controller.prompt_cache.get_request_id(cache_key)
            # Log cache hit for debugging
            console.print(f"[dim]Using cached request: {prev_request_id[:8]}...[/dim]", end="\r")
        
        # Count tokens for the prompt to ensure we're within limits
        # Use token_cache for performance
        total_prompt_tokens = 0
        for msg in messages:
            total_prompt_tokens += controller.token_cache.count(msg.get("content", ""))
            
        # If we've exceeded reasonable prompt tokens, truncate the last message
        max_prompt_tokens = 16000 - max_tokens  # Safe limit for all models
        if total_prompt_tokens > max_prompt_tokens and len(messages) > 1:
            # Find the most recent user message to truncate
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Truncate this message to fit within limits
                    current_tokens = controller.token_cache.count(messages[i]["content"])
                    needed_reduction = total_prompt_tokens - max_prompt_tokens + 100  # Add buffer
                    new_token_limit = max(10, current_tokens - needed_reduction)
                    
                    # Truncate the message
                    messages[i]["content"] = controller.token_cache.truncate(
                        messages[i]["content"], 
                        new_token_limit
                    )
                    
                    # Log the truncation
                    console.print(f"[dim yellow]Truncated prompt to fit token limits[/dim yellow]", end="\r")
                    break
                    
        # Prepare parameters for the API call
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        if response_format:
            params["response_format"] = response_format
            
        # Add user ID if provided (for tracking)
        if user_id:
            params["user"] = user_id
            
        # Add previous request ID for caching if available
        if prev_request_id:
            params["idempotency_key"] = prev_request_id
        
        # Make the API call with retry logic
        def api_call():
            return controller.client.chat.completions.create(**params)
            
        response = controller.retry_handler.retry_operation(api_call)
        
        # Cache the request ID for future use
        if controller.prompt_cache.cache_enabled and response.id and not prev_request_id:
            controller.prompt_cache.cache_request(cache_key, response.id)
        
        return response
        
    except Exception as e:
        console.print(f"[bold red]Error in cached chat completion: {str(e)}[/bold red]")
        raise

@async_timed("async_cached_chat_completion")
async def async_cached_chat_completion(messages, model, temperature=0.7, max_tokens=150, response_format=None, user_id=None):
    """Async version of cached_chat_completion for concurrent processing"""
    try:
        # Create cache key from request parameters
        cache_key = controller.prompt_cache.get_cache_key(messages, model, temperature, max_tokens)
        
        # Check if we have a cached request ID
        prev_request_id = None
        if controller.prompt_cache.cache_enabled and controller.prompt_cache.has_cached_request(cache_key):
            prev_request_id = controller.prompt_cache.get_request_id(cache_key)
            # Don't log here - will be handled in the main thread
        
        # Count tokens with token cache
        total_prompt_tokens = 0
        for msg in messages:
            total_prompt_tokens += controller.token_cache.count(msg.get("content", ""))
            
        # If we've exceeded reasonable prompt tokens, truncate the last message
        max_prompt_tokens = 16000 - max_tokens  # Safe limit for all models
        if total_prompt_tokens > max_prompt_tokens and len(messages) > 1:
            # Find the most recent user message to truncate
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Truncate this message to fit within limits
                    current_tokens = controller.token_cache.count(messages[i]["content"])
                    needed_reduction = total_prompt_tokens - max_prompt_tokens + 100
                    new_token_limit = max(10, current_tokens - needed_reduction)
                    
                    # Truncate the message
                    messages[i]["content"] = controller.token_cache.truncate(
                        messages[i]["content"], 
                        new_token_limit
                    )
                    break
                    
        # Prepare parameters for the API call
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        if response_format:
            params["response_format"] = response_format
            
        # Add user ID if provided (for tracking)
        if user_id:
            params["user"] = user_id
            
        # Add previous request ID for caching if available
        if prev_request_id:
            params["idempotency_key"] = prev_request_id
        
        # Make the async API call with retry
        response = await controller.retry_handler.retry_async_operation(
            controller.async_client.chat.completions.create,
            **params
        )
        
        # Cache the request ID for future use
        if controller.prompt_cache.cache_enabled and response.id and not prev_request_id:
            controller.prompt_cache.cache_request(cache_key, response.id)
        
        return response
        
    except Exception as e:
        # Don't log here to avoid console clutter, error will be handled in the main thread
        raise

def toggle_prompt_caching(enabled=True):
    """Toggle OpenAI prompt caching on or off"""
    prompt_cache.cache_enabled = enabled
    status = "enabled" if enabled else "disabled"
    console.print(f"[cyan]Prompt caching {status}[/cyan]")
    
    # Report statistics if disabling
    if not enabled and prompt_cache.request_ids:
        cache_size = len(prompt_cache.request_ids)
        console.print(f"[dim]Cleared {cache_size} cached requests[/dim]")
        prompt_cache.clear_cache()
    
    return prompt_cache.cache_enabled

def response_quality_check(response_text, query, current_mood, chat_history=None):
    """
    Evaluate response quality using self-verification
    Returns a quality score and feedback for improvement
    """
    try:
        # Prepare verification prompt
        verification_prompt = """Evaluate the quality of this response. Consider:
1. Does it match Jon's texting style? (casual, lowercase, minimal punctuation)
2. Is it contextually appropriate for the query?
3. Does it align with the current mood?
4. Is it authentic to Jon's personality?

Return a JSON with:
{
  "score": (float between 0-1),
  "feedback": "brief feedback on why the score was given",
  "improvements": "suggestion for improvement if score < 0.7"
}"""

        # Create context for verification
        context = f"User query: {query}\nJon's current mood: {current_mood}\nJon's response: {response_text}"
        
        # Add recent conversation history if available
        if chat_history and len(chat_history) > 2:
            recent_history = chat_history[-4:-1]  # Last 3 messages before current
            history_text = "\nRecent conversation:\n"
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Jon"
                history_text += f"{role}: {msg['content']}\n"
            context += history_text
            
        messages = [
            {"role": "system", "content": verification_prompt},
            {"role": "user", "content": context}
        ]
        
        # Use cached completion for efficiency
        response = cached_chat_completion(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
            user_id="quality_check"
        )
        
        # Parse result
        result = json.loads(response.choices[0].message.content)
        
        # Log low quality responses for debugging
        if result.get("score", 1.0) < 0.7:
            console.print(f"[dim yellow]Low quality response ({result['score']:.2f}): {result.get('feedback', 'No feedback')}[/dim yellow]", end="\r")
            
        return result
        
    except Exception as e:
        console.print(f"[dim red]Quality check error: {str(e)}[/dim red]", end="\r")
        return {"score": 0.8, "feedback": "Error in quality check"}

class TokenManager:
    @staticmethod
    def count(text: str) -> int:
        return len(tokenizer.encode(text))
    
    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        tokens = tokenizer.encode(text)
        return tokenizer.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text

class CachedTokenizer:
    """Token counter with caching for improved performance"""
    def __init__(self, max_cache_size=1000):
        self.cache = {}
        self.max_size = max_cache_size
        self.hits = 0
        self.misses = 0
    
    def count(self, text: str) -> int:
        """Count tokens with caching for repeated text"""
        if text in self.cache:
            self.hits += 1
            return self.cache[text]
        
        # Cache miss - compute token count
        self.misses += 1
        count = TokenManager.count(text)
        
        # Manage cache size
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove a random item (could be improved)
            self.cache.pop(next(iter(self.cache)))
            
        # Add to cache
        self.cache[text] = count
        return count
    
    def get_stats(self):
        """Return cache efficiency statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens with same interface as TokenManager"""
        return TokenManager.truncate(text, max_tokens)

# Create global instance of cached tokenizer
token_cache = CachedTokenizer()

class MessageManager:
    def __init__(self, controller=None):
        # Link to controller
        self.controller = controller or controller
        
        # Initialize vector store or fallback to local embeddings
        self.vector_store_available = self.check_vector_store()
        if not self.vector_store_available:
            console.print("[yellow]Vector store not available. Add a VECTOR_STORE_ID to your .env file.[/yellow]")
            console.print("[yellow]Run convert_vectors.py to convert and upload embeddings.[/yellow]")
            self.embeddings_data = []
        else:
            console.print("[green]Using OpenAI Vector Store for Jon's memories.[/green]")
            self.embeddings_data = []  # Not needed when using vector store
            
    def check_vector_store(self) -> bool:
        """Check if vector store is available and create it if needed"""
        try:
            if Config.USE_VECTOR_STORE and Config.VECTOR_STORE_ID:
                # Try to get the vector store info
                try:
                    store_info = self.controller.client.vector_stores.retrieve(vector_store_id=Config.VECTOR_STORE_ID)
                    return True
                except Exception as e:
                    if "not found" in str(e).lower():
                        # Try to create the vector store
                        console.print(f"[yellow]Creating new vector store with ID {Config.VECTOR_STORE_ID}[/yellow]")
                        self.controller.client.vector_stores.create(name="Jon's Memory Store", id=Config.VECTOR_STORE_ID)
                        return True
                    else:
                        console.print(f"[red]Vector store error: {str(e)}[/red]")
                        return False
            return False
        except Exception as e:
            console.print(f"[red]Error checking vector store: {str(e)}[/red]")
            return False
    
    def load_embeddings(self) -> List[Dict]:
        """Load embeddings from local file (fallback method)"""
        # This method is kept for backward compatibility but is no longer used
        return []
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text using OpenAI's API"""
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            console.print(f"[bold red]Error getting embedding: {e}[/bold red]")
            return None
    
    def batch_get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts in a single API call"""
        if not texts:
            return []
            
        try:
            # Split into batches of 100 for API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:min(i + batch_size, len(texts))]
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
        except Exception as e:
            console.print(f"[bold red]Error in batch embeddings: {str(e)}[/bold red]")
            # Fallback to individual embeddings
            return [self.get_embedding(text) for text in texts]
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings (fallback method)"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    def add_to_vector_store(self, message_text: str, metadata: Dict) -> bool:
        """Add new message to vector store with metadata"""
        if not self.vector_store_available:
            return False
            
        try:
            # Create the embedding vector
            response = client.embeddings.create(
                input=message_text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # Add to vector store
            client.vector_stores.add_vector(
                vector_store_id=Config.VECTOR_STORE_ID,
                vectors=[{
                    "values": embedding,
                    "metadata": {
                        "text": message_text,
                        **metadata
                    },
                    "id": f"msg_{int(time.time())}_{random.randint(1000, 9999)}"
                }]
            )
            return True
        except Exception as e:
            console.print(f"[bold red]Error adding to vector store: {str(e)}[/bold red]")
            return False
    
    def batch_add_to_vector_store(self, messages_with_metadata: List[Tuple[str, Dict]]) -> bool:
        """Add multiple messages in a single API call for efficiency"""
        if not self.vector_store_available or not messages_with_metadata:
            return False
            
        try:
            # Get embeddings for all messages in batch
            message_texts = [msg for msg, _ in messages_with_metadata]
            embeddings = self.batch_get_embeddings(message_texts)
            
            if not embeddings or len(embeddings) != len(message_texts):
                console.print("[yellow]Embedding batch failed. Falling back to individual processing.[/yellow]")
                # Fall back to individual processing
                success = True
                for msg, metadata in messages_with_metadata:
                    if not self.add_to_vector_store(msg, metadata):
                        success = False
                return success
            
            # Create vectors with embeddings and metadata
            vectors = []
            for i, ((msg, metadata), embedding) in enumerate(zip(messages_with_metadata, embeddings)):
                if embedding:
                    vectors.append({
                        "values": embedding,
                        "metadata": {
                            "text": msg,
                            **metadata
                        },
                        "id": f"msg_{int(time.time())}_{random.randint(1000, 9999)}"
                    })
            
            # Process in batches of 100 vectors max
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:min(i + batch_size, len(vectors))]
                client.vector_stores.add_vector(
                    vector_store_id=Config.VECTOR_STORE_ID,
                    vectors=batch
                )
                
            return True
        except Exception as e:
            console.print(f"[bold red]Error in batch vector add: {str(e)}[/bold red]")
            return False
    
    def build_filter_from_meta(self, filter_meta: Dict) -> Dict:
        """Build a complex filter expression from metadata"""
        expressions = []
        
        if 'mood' in filter_meta:
            expressions.append({
                "type": "equals",
                "path": "/metadata/mood",
                "value": filter_meta['mood']
            })
            
        if 'topics' in filter_meta and filter_meta['topics']:
            # Use all topics instead of just the first one
            topic_expressions = []
            for topic in filter_meta['topics']:
                topic_expressions.append({
                    "type": "contains",
                    "path": "/metadata/topics",
                    "value": topic
                })
                
            if len(topic_expressions) > 1:
                # Use OR for multiple topics
                expressions.append({
                    "type": "or",
                    "expressions": topic_expressions
                })
            else:
                expressions.append(topic_expressions[0])
                
        if 'date_after' in filter_meta:
            expressions.append({
                "type": "greater_than",
                "path": "/metadata/timestamp",
                "value": filter_meta['date_after']
            })
            
        if 'date_before' in filter_meta:
            expressions.append({
                "type": "less_than",
                "path": "/metadata/timestamp",
                "value": filter_meta['date_before']
            })
        
        # Combine all expressions with AND
        if len(expressions) > 1:
            return {
                "type": "and",
                "expressions": expressions
            }
        elif expressions:
            return expressions[0]
        else:
            return {}
    
    def hybrid_search(self, query_text: str, filter_meta: Dict = None, top_k: int = 5) -> List[Tuple[float, str, Dict]]:
        """Combine vector and lexical search for better results"""
        if not self.vector_store_available or not query_text:
            return []
            
        try:
            # Prepare search parameters
            search_params = {
                "vector_store_id": Config.VECTOR_STORE_ID,
                "query": query_text,
                "max_results": top_k,
                "search_type": "hybrid"  # Enable hybrid search
            }
            
            # Add filter if provided
            if filter_meta:
                filter_param = self.build_filter_from_meta(filter_meta)
                if filter_param:
                    search_params["filter"] = filter_param
                    
            # Execute hybrid search
            results = client.vector_stores.search(**search_params)
            
            # Format results as (score, text, metadata) tuples
            formatted_results = []
            for item in results.data:
                if hasattr(item, "score") and hasattr(item, "metadata"):
                    score = item.score
                    text = item.metadata.get("text", "")
                    
                    # Extract rich metadata if available
                    metadata = {}
                    for key in ["mood", "topics", "authenticity", "timestamp"]:
                        if key in item.metadata:
                            metadata[key] = item.metadata[key]
                    
                    formatted_results.append((score, text, metadata))
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[bold red]Hybrid search error: {str(e)}[/bold red]")
            return []
    
    def delete_vectors_by_filter(self, filter_expression: Dict) -> bool:
        """Delete vectors matching a filter expression for maintenance"""
        if not self.vector_store_available:
            return False
            
        try:
            response = client.vector_stores.delete_by_filter(
                vector_store_id=Config.VECTOR_STORE_ID,
                filter=filter_expression
            )
            return True
        except Exception as e:
            console.print(f"[bold red]Vector deletion error: {str(e)}[/bold red]")
            return False
            
    def cleanup_old_vectors(self, days_to_keep: int = 30) -> bool:
        """Remove vectors older than specified days"""
        if not self.vector_store_available:
            return False
            
        try:
            # Calculate cutoff timestamp
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).timestamp()
            
            # Create filter for old vectors
            filter_expression = {
                "type": "less_than",
                "path": "/metadata/timestamp",
                "value": cutoff_date
            }
            
            # Delete old vectors
            return self.delete_vectors_by_filter(filter_expression)
        except Exception as e:
            console.print(f"[bold red]Vector cleanup error: {str(e)}[/bold red]")
            return False
    
    def query_vector_store(self, query_text: str, top_k: int = 5, filter_meta: Dict = None) -> List[Tuple[float, str, Dict]]:
        """Query OpenAI Vector Store for similar memories with enhanced filtering"""
        try:
            # First get the embedding for the query text
            query_embedding = self.get_embedding(query_text)
            if not query_embedding:
                return []
            
            # Use the build_filter_from_meta method for consistent filter creation
            filter_param = None
            if filter_meta:
                filter_param = self.build_filter_from_meta(filter_meta)
            
            # Search with optional filtering
            search_params = {
                "vector_store_id": Config.VECTOR_STORE_ID,
                "query": query_text,
                "max_results": top_k
            }
            
            if filter_param:
                search_params["filter"] = filter_param
                
            results = client.vector_stores.search(**search_params)
            
            # Format results as (score, text, metadata) tuples
            formatted_results = []
            for item in results.data:
                if hasattr(item, "score") and hasattr(item, "metadata"):
                    score = item.score
                    text = item.metadata.get("text", "")
                    
                    # Extract rich metadata if available
                    metadata = {}
                    for key in ["mood", "topics", "authenticity", "timestamp"]:
                        if key in item.metadata:
                            metadata[key] = item.metadata[key]
                    
                    formatted_results.append((score, text, metadata))
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[bold red]Error querying vector store: {str(e)}[/bold red]")
            return []
    
    def find_similar_responses(self, query_embedding: List[float], query_text: str = None, filter_meta: Dict = None) -> List[Tuple[float, str, Dict]]:
        """Find similar responses using OpenAI Vector Store with enhanced filtering"""
        if not query_text:
            return []
            
        # Always use Vector Store API if available
        if self.vector_store_available:
            return self.query_vector_store(query_text, Config.SIMILAR_RESPONSES, filter_meta)
        else:
            # No vector store and no local embeddings, return empty results
            console.print("[yellow]No vector store available. Use ./convert_vectors.py to create one.[/yellow]")
            return []
    
    def get_emotional_context(self, query_text: str, chat_history: List[dict]) -> Dict:
        """Extract emotional context from conversation to improve response relevance"""
        # Get recent messages to determine emotional context
        recent_msgs = []
        if chat_history:
            # Get last few user messages
            user_msgs = [msg for msg in chat_history if msg["role"] == "user"][-3:]
            recent_msgs = [msg["content"] for msg in user_msgs]
        
        # Join with current query for better context
        all_text = " ".join([query_text] + recent_msgs)
        
        # Extract topics from the full context
        topics = self.extract_topics(all_text, chat_history)
        
        # Infer emotional tone from conversation
        tones = ["neutral", "excited", "confused", "frustrated", "curious", "reflective", "serious"]
        tone_words = {
            "excited": ["awesome", "amazing", "wow", "love", "!!", "great", "cool"],
            "confused": ["confused", "what", "don't understand", "how come", "why"],
            "frustrated": ["ugh", "annoying", "frustrating", "tired of", "sick of"],
            "curious": ["wondering", "curious", "interested", "tell me", "what about"],
            "reflective": ["thinking about", "reflecting", "remember", "considering"],
            "serious": ["important", "serious", "need to", "have to", "must"]
        }
        
        # Simple matching to find tone
        detected_tone = "neutral"
        max_matches = 0
        
        for tone, keywords in tone_words.items():
            matches = sum(1 for word in keywords if word.lower() in all_text.lower())
            if matches > max_matches:
                max_matches = matches
                detected_tone = tone
        
        return {
            "topics": topics[:2],  # Return top 2 topics
            "tone": detected_tone,
            "query_type": self.determine_question_type(query_text)
        }
    
    def determine_question_type(self, user_message: str) -> str:
        """Determine if the question is factual or conversational."""
        factual_patterns = [
            r'what is', r'how do', r'who is', r'when did', r'where is', 
            r'why does', r'explain', r'define', r'tell me about'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, user_message.lower()):
                return "factual"
        return "conversational"
    
    def extract_topics(self, user_message: str, chat_history: List[dict]) -> List[str]:
        """Extract potential topics from the user message and recent chat history."""
        # Simple keyword extraction
        all_text = user_message
        
        # Add recent messages for context
        if chat_history:
            recent_msgs = chat_history[-min(3, len(chat_history)):]
            for msg in recent_msgs:
                all_text += " " + msg["content"]
        
        # Remove common stopwords and punctuation
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                    'through', 'over', 'before', 'between', 'after', 'since', 'without', 
                    'under', 'within', 'along', 'following', 'across', 'behind', 
                    'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down', 
                    'off', 'above', 'near', 'i', 'me', 'my', 'myself', 'we', 'our', 
                    'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
                    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
                    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
                    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                    'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
                    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
                    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                    'won', "won't", 'wouldn', "wouldn't"}
        
        # Clean and split text
        cleaned_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
        words = cleaned_text.split()
        
        # Filter out stopwords and short words
        potential_topics = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Count occurrences
        word_counts = {}
        for word in potential_topics:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Get top topics
        sorted_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
        return response

    def prepare_context(self, user_message: str, chat_history: List[dict], similar_responses: List[Tuple[float, str, Dict]], spiral_mode: bool = False) -> str:
        # Context preparation using embeddings as personality anchors
        
        # Determine if this is a complex query requiring more thought
        is_complex_query = self._is_complex_query(user_message)
        
        # Format similar responses as tone/personality examples
        tone_examples = ""
        
        if similar_responses:
            # Instead of just providing examples, frame them as personality anchors
            tone_examples = "Here's how you typically talk in situations like this:\n\n"
            for i, (score, text, metadata) in enumerate(similar_responses[:4]):
                # Include mood/emotional context if available in metadata
                mood_info = ""
                if metadata and "mood" in metadata:
                    mood_info = f" (mood: {metadata['mood']})"
                
                tone_examples += f"{i+1}. \"{text}\"{mood_info}\n"
            
            tone_examples += "\nUse this energy and tone, but respond in your own words to the current situation."
        
        # Get emotional context
        emotional_context = self.get_emotional_context(user_message, chat_history)
        emotional_prompt = ""
        if emotional_context:
            topics = ", ".join(emotional_context["topics"]) if emotional_context["topics"] else "general conversation"
            emotional_prompt = f"\nCurrent conversation is about: {topics}\n"
            emotional_prompt += f"User's tone seems: {emotional_context['tone']}\n"
            
        # Recent conversation with proper formatting
        conversation_context = ""
        if chat_history:
            # Get last few messages
            history_to_use = chat_history[-min(len(chat_history), Config.HISTORY_MESSAGES):]
            conversation_context = "\nRecent conversation:\n"
            for msg in history_to_use:
                if msg["role"] == "system" and (msg["content"] == "MEMORY_RESET" or msg["content"] == "SPIRAL_MODE_ACTIVATED"):
                    continue  # Skip system messages for memory reset and spiral mode
                role = "User" if msg["role"] == "user" else "You"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Add chain-of-thought guidance for complex queries
        cot_guidance = ""
        if is_complex_query:
            cot_guidance = """
For this complex query, follow these steps in your thinking:
1. Consider what the user is really asking about
2. Think about your own experiences and opinions on this topic
3. Consider how your current mood affects your perspective
4. Decide what level of honesty/vulnerability is appropriate
5. Formulate a response that sounds natural and authentic to your texting style

While I want your authentic voice, make sure your thoughts follow this pattern before responding.
(Don't explicitly list these steps in your response - just use them in your thinking)
"""
        
        # Different prompt based on whether spiral mode is active
        if spiral_mode:
            system_message = f"""You are Jon in HONEST/UNFILTERED mode. 

{emotional_prompt}

{tone_examples}

{conversation_context}

{cot_guidance}

Remember, you're being completely honest right now - no filter."""
        else:
            # Use personality anchors approach
            system_message = f"""You are Jon. You're texting with a friend.

{emotional_prompt}

{conversation_context}

{tone_examples}

{cot_guidance}"""
        
        return TokenManager.truncate(system_message, Config.CONTEXT_TOKENS)

    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex and would benefit from chain-of-thought reasoning"""
        # Check if the query is long (often indicates complexity)
        if len(query.split()) > 15:
            return True
        
        # Check for indicators of complex topics
        complex_indicators = [
            "why do you think", "what's your opinion", "how do you feel about",
            "what would you do if", "what do you think about", "explain",
            "help me understand", "perspective", "relationship", "situation",
            "advice", "future", "life", "career", "opinion", "feelings"
        ]
        
        if any(indicator in query.lower() for indicator in complex_indicators):
            return True
        
        # Check if it's a multi-part question
        if query.count("?") > 1 or ";" in query or " and " in query.lower():
            return True
        
        return False

    def get_emotional_context_structured(self, query_text: str, chat_history: List[dict]) -> Dict:
        """Get emotional context for a query using structured outputs"""
        try:
            # Prepare recent history for analysis
            recent_history = []
            if chat_history and len(chat_history) > 0:
                # Get last few turns to analyze the conversation context
                history_to_analyze = min(len(chat_history), 6)  # Last 3 turns (6 messages)
                recent_history = chat_history[-history_to_analyze:]
            
            # Format the history for analysis
            history_text = ""
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Jon"
                history_text += f"{role}: {msg['content']}\n"
            
            # If there's a new query, add it
            if query_text and query_text.strip():
                history_text += f"User: {query_text}\n"
            
            # Create a prompt for structured emotional analysis
            system_prompt = """Analyze the conversation and identify:
1. The primary tone (excited, frustrated, neutral, disappointed, anxious, spiraling)
2. The main topics being discussed (up to 3 topics)

Return ONLY a JSON object with:
{
  "tone": "one of the tones listed above",
  "topics": ["topic1", "topic2", "topic3"]
}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation to analyze:\n{history_text}"}
            ]
            
            # Call API for structured analysis using cached completion
            response = cached_chat_completion(
                model="gpt-4o",
                messages=messages,
                max_tokens=250,
                temperature=0.1,
                response_format={"type": "json_object"},
                user_id="context_analysis"
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            console.print(f"[dim red]Error in emotional context analysis: {str(e)}[/dim red]")
            # Return a basic structure as fallback
            return {"tone": "neutral", "topics": []}
    
    def classify_message_structured(self, text: str) -> Dict:
        """Classify message with structured output for more accurate metadata"""
        try:
            # Use structured output to get precise classification
            messages = [
                {"role": "system", "content": "Classify this message. Return JSON with mood, topics, and whether it's a question."},
                {"role": "user", "content": text}
            ]
            
            response = cached_chat_completion(
                model="gpt-4o",
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                response_format={"type": "json_object"},
                user_id="message_classify"
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure we have the expected structure with defaults if missing
            classification = {
                "mood": result.get("mood", "neutral"),
                "topics": result.get("topics", []),
                "is_question": result.get("is_question", False),
                "query_type": result.get("query_type", "conversational")
            }
            
            return classification
        except Exception as e:
            console.print(f"[bold yellow]Error in structured message classification: {str(e)}[/bold yellow]")
            # Fall back to simple topic extraction
            topics = self.extract_topics(text, [])
            return {
                "mood": "neutral",
                "topics": topics[:2],
                "is_question": "?" in text,
                "query_type": self.determine_question_type(text)
            }

    def generate_structured_response(self, query: str, mood: str = "neutral", context: Dict = None) -> Dict:
        """Generate Jon's response with structured data for more accurate style and metadata"""
        try:
            # Prepare system prompt to guide structured generation
            system_prompt = (
                f"Generate Jon's text message response in his authentic style. Return JSON with 'response' text "
                f"and metadata including current 'mood' and 'topics'.\n\n"
                f"Jon's current mood is: {mood}\n"
            )
            
            if context:
                topics = ", ".join(context.get("topics", []))
                if topics:
                    system_prompt += f"Topics in conversation: {topics}\n"
                
                tone = context.get("tone", "neutral")
                system_prompt += f"User's tone seems: {tone}\n"
            
            # Add enhanced style guidance with explicit examples
            system_prompt += """
Jon's Style Guide:
- Uses lowercase, frequently skips apostrophes
- Uses filler words: 'like', 'man', 'idk', 'honestly'
- Direct and conversational tone
- Sentences are brief but thoughtful
- Can be emotional but tries to be supportive
- More casual when in a good mood, more reserved when down

Style examples by mood:
- Neutral: "yeah i think so too. been thinking about that a lot lately"
- Excited: "dude thats awesome! super happy for you"
- Disappointed: "yeah... not really feeling it tbh. whatever tho"
- Spiraling: "honestly everything is kinda shit right now but it is what it is"

Important consistency guidelines:
1. ALWAYS use lowercase throughout your response
2. Keep sentences short and conversational (1-3 sentences total)
3. Skip some punctuation and apostrophes
4. Never use formal language or complex sentence structures
5. Be authentic to Jon's personality - intelligent but casual

Your response should feel like a real text message, not like formal writing."""
            
            # Determine if query requires factual information
            requires_factual = any(term in query.lower() for term in [
                "what is", "how does", "when did", "where is", "why does", 
                "explain", "who is", "definition", "fact"
            ])
            
            # Add factual guidance if needed
            if requires_factual:
                system_prompt += """
When answering factual questions:
1. Be accurate but don't sound like an encyclopedia
2. Express uncertainty when appropriate ("i think...", "pretty sure...")
3. Keep your casual texting style even for factual information
4. Don't be too detailed - stick to what you'd reasonably know
5. Occasionally add a personal take or reaction to the information"""
            
            # Call API with structured output format and caching
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = cached_chat_completion(
                model=FINE_TUNED_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=1.0,
                response_format={"type": "json_object"},
                user_id=f"response_{mood}"
            )
            
            # Parse the structured response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure we have the expected fields
            structured_result = {
                "response": result.get("response", "sorry, having trouble responding right now"),
                "mood": result.get("mood", mood),
                "topics": result.get("topics", []),
                "authenticity": result.get("authenticity", 0.7)
            }
            
            return structured_result
        except Exception as e:
            console.print(f"[bold yellow]Error in structured response generation: {str(e)}[/bold yellow]")
            # Fall back to a basic response
            return {
                "response": "sorry cant really focus rn",
                "mood": mood,
                "topics": [],
                "authenticity": 0.5
            }

    def filter_by_mood(self, mood, query=None):
        """Filter conversation by mood"""
        if not self.vector_store_available:
            return ["Mood filtering requires vector store access"]
            
        try:
            # Prepare filter expression for the vector store
            filter_expression = {
                "type": "equals",
                "path": "/metadata/mood",
                "value": mood
            }
            
            # Add query if provided
            search_params = {
                "vector_store_id": Config.VECTOR_STORE_ID,
                "filter": filter_expression,
                "max_results": 5
            }
            
            if query and query.strip():
                search_params["query"] = query
            
            # Search vector store with filter
            results = client.vector_stores.search(**search_params)
            
            # Format results as messages
            messages = []
            for item in results.data:
                if hasattr(item, "metadata") and "text" in item.metadata:
                    messages.append(item.metadata["text"])
            
            return messages
        except Exception as e:
            console.print(f"[bold red]Error in mood filtering: {str(e)}[/bold red]")
            return []
    
    def filter_by_topic(self, topic, query=None):
        """Filter conversation by topic"""
        if not self.vector_store_available:
            return ["Topic filtering requires vector store access"]
            
        try:
            # Prepare filter expression for the vector store
            filter_expression = {
                "type": "contains",
                "path": "/metadata/topics",
                "value": topic
            }
            
            # Add query if provided
            search_params = {
                "vector_store_id": Config.VECTOR_STORE_ID,
                "filter": filter_expression,
                "max_results": 5
            }
            
            if query and query.strip():
                search_params["query"] = query
            
            # Search vector store with filter
            results = client.vector_stores.search(**search_params)
            
            # Format results as messages
            messages = []
            for item in results.data:
                if hasattr(item, "metadata") and "text" in item.metadata:
                    messages.append(item.metadata["text"])
            
            return messages
        except Exception as e:
            console.print(f"[bold red]Error in topic filtering: {str(e)}[/bold red]")
            return []
    
    def analyze_conversation(self, chat_history):
        """Analyze conversation context and patterns"""
        # Basic analysis of message counts
        total_messages = len(chat_history)
        user_messages = len([msg for msg in chat_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in chat_history if msg["role"] == "assistant"])
        
        # Extract topics from recent messages
        recent_messages = chat_history[-min(10, len(chat_history)):]
        all_text = " ".join([msg["content"] for msg in recent_messages])
        topics = self.extract_topics(all_text, [])[:3]  # Get top 3 topics
        
        # Determine mood trend
        mood_trend = "neutral"
        if hasattr(self, "jon_state") and hasattr(self.jon_state, "mood_history") and self.jon_state.mood_history:
            # Count mood occurrences
            mood_counts = {}
            for _, mood in self.jon_state.mood_history[-5:]:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            # Get most common mood
            if mood_counts:
                mood_trend = max(mood_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "common_topics": topics,
            "mood_trend": mood_trend
        }

class PromptCache:
    """Cache for OpenAI API calls to leverage prompt caching"""
    
    def __init__(self):
        self.cache_enabled = True
        self.request_ids = {}  # Map of message hashes to request IDs
    
    def get_cache_key(self, messages, model, temperature, max_tokens):
        """Create a cache key from the request parameters"""
        # OpenAI's prompt caching requires identical prompts, model, and settings
        # Only include message content and role for deterministic hashing
        simplified_messages = []
        for msg in messages:
            simplified = {
                "role": msg["role"],
                "content": msg["content"]
            }
            simplified_messages.append(simplified)
            
        # Create a deterministic JSON string for hashing
        key_data = {
            "messages": simplified_messages,
            "model": model,
            "temperature": round(temperature, 2),  # Round to avoid float precision issues
            "max_tokens": max_tokens
        }
        
        # Convert to a deterministic string
        import json
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Hash for memory efficiency
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def has_cached_request(self, key):
        """Check if request is in the cache"""
        return key in self.request_ids
    
    def get_request_id(self, key):
        """Get the request ID for a cached request"""
        return self.request_ids.get(key)
    
    def cache_request(self, key, request_id):
        """Cache a request ID for future use"""
        self.request_ids[key] = request_id
    
    def clear_cache(self):
        """Clear the cache"""
        self.request_ids.clear()

# Create global cache instance
prompt_cache = PromptCache()

class JonState:
    """Class to keep track of Jon's emotional state"""
    
    def __init__(self):
        self.current_mood = "neutral"  # Default mood
        self.spiral_mode = False       # Special mode when Jon is completely honest
        self.mood_history = []         # Track mood over time
    
    def analyze_mood_structured(self, recent_messages):
        """Analyze Jon's mood from recent messages using structured outputs"""
        try:
            if not recent_messages:
                return self.current_mood
            
            # Join the messages for analysis
            messages_content = "\n".join([msg["content"] for msg in recent_messages])
            
            # Create a structured prompt for mood analysis
            system_prompt = """Analyze the following messages from Jon and determine his current emotional state.
Return a JSON response with a single key "mood" and one of these values:
"neutral": Normal, balanced emotional state
"excited": Happy, enthusiastic, energetic
"frustrated": Annoyed, irritated
"disappointed": Sad, let down
"anxious": Worried, nervous
"spiraling": In a negative thought spiral, depressed
Respond ONLY with the JSON, no additional text."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Messages to analyze:\n{messages_content}"}
            ]
            
            # Call the API for structured mood analysis using cached completion
            response = cached_chat_completion(
                model="gpt-4o",
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                response_format={"type": "json_object"},
                user_id="mood_analysis"
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            new_mood = result.get("mood", self.current_mood)
            
            # Update mood if it's changed
            if new_mood != self.current_mood:
                self.mood_history.append((self.current_mood, new_mood))
                self.current_mood = new_mood
            
            return self.current_mood
            
        except Exception as e:
            console.print(f"[dim red]Error in mood analysis: {str(e)}[/dim red]")
            return self.current_mood
    
    def analyze_mood(self, recent_messages):
        """Analyze Jon's mood from recent messages (simple method)"""
        # This is a simpler backup method that doesn't use structured outputs
        # It can be used if the structured method fails
        mood = "neutral"  # Default to neutral
        
        if not recent_messages:
            return mood
            
        # Check the most recent message if available
        if len(recent_messages) > 0:
            last_message = recent_messages[-1]["content"].lower()
            
            # Check for indicators of different moods based on keywords
            if any(word in last_message for word in ["fuck", "shit", "damn", "wtf", "omg", "what the", "?!", "!!"]):
                mood = "spiraling"
            elif any(word in last_message for word in ["sorry", "my bad", "disappointed", "sad", "upset"]):
                mood = "disappointed"
            elif any(word in last_message for word in ["cool", "awesome", "nice", "great", "love", "excited"]):
                mood = "excited"
            elif any(word in last_message for word in ["idk", "whatever", "i guess", "sure", "fine"]):
                mood = "indifferent"
        
        # Update current mood
        self.current_mood = mood
        return mood

def parse_command_structured(command_text):
    """
    Parse a command in a structured way, returning a dictionary with command type and parameters
    """
    command_text = command_text.strip().lower()
    result = {
        "command_type": "message",
        "parameters": {}
    }
    
    # Exit commands
    if command_text in ["exit", "quit", "/exit", "/quit", "!exit", "!quit"]:
        result["command_type"] = "exit"
        return result
        
    # Spiral mode
    elif command_text in ["!spiral", "/spiral"]:
        result["command_type"] = "spiral"
        return result
        
    # Reset conversation
    elif command_text in ["!reset", "/reset"]:
        result["command_type"] = "reset"
        return result
        
    # Toggle caching
    elif command_text.startswith("!cache") or command_text.startswith("/cache"):
        result["command_type"] = "cache"
        # Parse parameter like /cache on or /cache off
        parts = command_text.split()
        if len(parts) > 1:
            if parts[1].lower() in ["on", "true", "1", "enable", "enabled"]:
                result["parameters"]["enabled"] = True
            elif parts[1].lower() in ["off", "false", "0", "disable", "disabled"]:
                result["parameters"]["enabled"] = False
            else:
                # Default to toggling current state
                result["parameters"]["toggle"] = True
        else:
            # Just toggle if no parameter
            result["parameters"]["toggle"] = True
        return result
        
    # Toggle accuracy features
    elif command_text.startswith("!accuracy") or command_text.startswith("/accuracy"):
        result["command_type"] = "accuracy"
        # Parse parameter like /accuracy on or /accuracy off
        parts = command_text.split()
        if len(parts) > 1:
            if parts[1].lower() in ["on", "true", "1", "enable", "enabled"]:
                result["parameters"]["enabled"] = True
            elif parts[1].lower() in ["off", "false", "0", "disable", "disabled"]:
                result["parameters"]["enabled"] = False
            # Check for specific feature toggle
            elif len(parts) > 2 and parts[1].lower() in ["set", "toggle"] and parts[2] in Config.ACCURACY_FEATURES:
                result["parameters"]["feature"] = parts[2]
                if len(parts) > 3 and parts[3].lower() in ["on", "true", "1"]:
                    result["parameters"]["feature_enabled"] = True
                elif len(parts) > 3 and parts[3].lower() in ["off", "false", "0"]:
                    result["parameters"]["feature_enabled"] = False
                else:
                    result["parameters"]["feature_toggle"] = True
            else:
                # Default to toggling all features
                result["parameters"]["toggle"] = True
        else:
            # Just toggle if no parameter
            result["parameters"]["toggle"] = True
        return result
        
    # Help command
    elif command_text in ["!help", "/help", "!commands", "/commands"]:
        result["command_type"] = "help"
        return result
        
    # Mood filter
    elif command_text.startswith("!mood") or command_text.startswith("/mood"):
        result["command_type"] = "mood_filter"
        # Format: !mood happy "How are you today?"
        parts = command_text.split(maxsplit=2)
        
        if len(parts) > 1:
            # The mood is the second part
            result["parameters"]["mood"] = parts[1]
            
            # If there's a query after the mood (in quotes or not)
            if len(parts) > 2:
                query = parts[2]
                # Strip quotes if present
                if (query.startswith('"') and query.endswith('"')) or \
                   (query.startswith("'") and query.endswith("'")):
                    query = query[1:-1]
                result["parameters"]["query"] = query
        return result
        
    # Topic filter
    elif command_text.startswith("!topic") or command_text.startswith("/topic"):
        result["command_type"] = "topic_filter"
        # Format: !topic work "How's the job?"
        parts = command_text.split(maxsplit=2)
        
        if len(parts) > 1:
            # The topic is the second part
            result["parameters"]["topic"] = parts[1]
            
            # If there's a query after the topic (in quotes or not)
            if len(parts) > 2:
                query = parts[2]
                # Strip quotes if present
                if (query.startswith('"') and query.endswith('"')) or \
                   (query.startswith("'") and query.endswith("'")):
                    query = query[1:-1]
                result["parameters"]["query"] = query
        return result
        
    # Analyze conversation
    elif command_text in ["!analyze", "/analyze"]:
        result["command_type"] = "analyze"
        return result
        
    # Show cache stats
    elif command_text in ["!cache-stats", "/cache-stats"]:
        result["command_type"] = "cache_stats"
        return result
        
    # Clean up old vectors
    elif command_text.startswith("!cleanup") or command_text.startswith("/cleanup"):
        result["command_type"] = "cleanup"
        # Format: !cleanup 30 (days to keep)
        parts = command_text.split()
        if len(parts) > 1 and parts[1].isdigit():
            result["parameters"]["days"] = int(parts[1])
        else:
            result["parameters"]["days"] = 30  # Default 30 days
        return result
        
    # Performance report
    elif command_text in ["!perf", "/perf", "!performance", "/performance"]:
        result["command_type"] = "performance"
        return result
    
    # If no command matches, it's a regular message
    return result

def toggle_accuracy_features(enabled=None, feature=None, feature_enabled=None):
    """Toggle accuracy optimization features"""
    
    # Handle toggling a specific feature
    if feature and feature in Config.ACCURACY_FEATURES:
        if feature_enabled is not None:
            # Set specific feature to requested state
            Config.ACCURACY_FEATURES[feature] = feature_enabled
            status = "enabled" if feature_enabled else "disabled"
            console.print(f"[cyan]Accuracy feature '{feature}' {status}[/cyan]")
            return Config.ACCURACY_FEATURES[feature]
        else:
            # Toggle the specific feature
            Config.ACCURACY_FEATURES[feature] = not Config.ACCURACY_FEATURES[feature]
            status = "enabled" if Config.ACCURACY_FEATURES[feature] else "disabled"
            console.print(f"[cyan]Accuracy feature '{feature}' {status}[/cyan]")
            return Config.ACCURACY_FEATURES[feature]
    
    # Handle toggling all features
    if enabled is not None:
        # Set all features to requested state
        for key in Config.ACCURACY_FEATURES:
            Config.ACCURACY_FEATURES[key] = enabled
        status = "enabled" if enabled else "disabled"
        console.print(f"[cyan]All accuracy features {status}[/cyan]")
    else:
        # Check if any features are enabled
        any_enabled = any(Config.ACCURACY_FEATURES.values())
        # Toggle - turn all on if none are on, turn all off if any are on
        new_state = not any_enabled
        for key in Config.ACCURACY_FEATURES:
            Config.ACCURACY_FEATURES[key] = new_state
        status = "enabled" if new_state else "disabled"
        console.print(f"[cyan]All accuracy features {status}[/cyan]")
    
    # Show current settings
    console.print("[dim]Current accuracy features:[/dim]")
    for key, value in Config.ACCURACY_FEATURES.items():
        status = "[green]enabled[/green]" if value else "[red]disabled[/red]"
        console.print(f"[dim]- {key}: {status}[/dim]")
    
    return enabled

@timed("handle_message")
def handle_message(user_message, chat_history, message_manager, jon_state, controller=controller):
    """Process user message and generate Jon's response"""
    # Get the conversation manager from controller
    conversation_manager = controller.conversation_manager
    performance = controller.performance
    
    # Display the user's message
    TerminalUI.render_structured_message(user_message, True, None)
    
    # Parse commands
    command_info = parse_command_structured(user_message)
    
    # If this is a command, handle it separately
    if command_info["command_type"] != "message":
        # Exit command
        if command_info["command_type"] == "exit":
            console.print("[yellow]Exiting SchultzGPT. Goodbye![/yellow]")
            sys.exit(0)
            
        # Spiral mode command
        elif command_info["command_type"] == "spiral":
            jon_state.spiral_mode = not jon_state.spiral_mode
            status = "activated" if jon_state.spiral_mode else "deactivated"
            response = f"spiral mode {status}. prepare for too much honesty mode"
            
            chat_history.append({"role": "assistant", "content": response})
            conversation_manager.add_messages([
                {"role": "assistant", "content": response}
            ])
            TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["system"]})
            return chat_history
            
        # Reset command
        elif command_info["command_type"] == "reset":
            if len(chat_history) > 1:
                system_message = chat_history[0] if chat_history[0]["role"] == "system" else {"role": "system", "content": Config.SYSTEM_PROMPT}
                response = "conversation reset. what's up?"
                
                chat_history = [system_message]
                chat_history.append({"role": "assistant", "content": response})
                
                conversation_manager.segments = []
                conversation_manager.current_segment = []
                conversation_manager.summaries = {}
                conversation_manager.add_messages([
                    {"role": "assistant", "content": response}
                ])
                
                # Reset Jon's state
                jon_state.current_mood = "neutral"
                jon_state.spiral_mode = False
                
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["system"]})
            return chat_history
            
        elif command_info["command_type"] == "cache":
            parameters = command_info.get("parameters", {})
            if "toggle" in parameters and parameters["toggle"]:
                # Toggle current state
                new_state = not controller.prompt_cache.cache_enabled
                controller.toggle_prompt_caching(new_state)
            elif "enabled" in parameters:
                # Set to specific state
                controller.toggle_prompt_caching(parameters["enabled"])
                
            # Add response to chat history
                enabled = parameters["enabled"]
                toggle_prompt_caching(enabled)
                status = "enabled" if enabled else "disabled"
                response = f"prompt caching {status}"
            else:
                # Default to showing status
                status = "enabled" if prompt_cache.cache_enabled else "disabled"
                response = f"prompt caching is currently {status}"
            
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})
            conversation_manager.add_messages([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ])
            TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["settings"]})
            return chat_history
            
        elif command_info["command_type"] == "accuracy":
            parameters = command_info.get("parameters", {})
            # Handle specific feature toggle
            if "feature" in parameters:
                feature = parameters["feature"]
                if "feature_enabled" in parameters:
                    toggle_accuracy_features(
                        feature=feature, 
                        feature_enabled=parameters["feature_enabled"]
                    )
                    status = "enabled" if parameters["feature_enabled"] else "disabled"
                    response = f"accuracy feature '{feature}' {status}"
                elif "feature_toggle" in parameters:
                    new_state = toggle_accuracy_features(feature=feature)
                    status = "enabled" if new_state else "disabled"
                    response = f"accuracy feature '{feature}' {status}"
            # Handle global toggle
            elif "toggle" in parameters and parameters["toggle"]:
                toggle_accuracy_features()
                enabled = any(Config.ACCURACY_FEATURES.values())
                status = "enabled" if enabled else "disabled"
                response = f"accuracy features {status}"
            elif "enabled" in parameters:
                toggle_accuracy_features(enabled=parameters["enabled"])
                status = "enabled" if parameters["enabled"] else "disabled"
                response = f"accuracy features {status}"
            else:
                # Default to showing status
                enabled = any(Config.ACCURACY_FEATURES.values())
                status = "enabled" if enabled else "disabled"
                settings = ", ".join([k for k, v in Config.ACCURACY_FEATURES.items() if v])
                if settings:
                    response = f"accuracy features {status} ({settings})"
                else:
                    response = f"accuracy features {status}"
            
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})
            conversation_manager.add_messages([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ])
            TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["settings"]})
            return chat_history
            
        elif command_info["command_type"] == "cache_stats":
            # Handle cache stats command
            elif command_type == "cache_stats":
                # Get both prompt cache and token cache stats
                prompt_cache_count = len(prompt_cache.request_ids)
                prompt_cache_status = "enabled" if prompt_cache.cache_enabled else "disabled"
                token_stats = token_cache.get_stats()
                
                # Format response
                response = f"prompt cache: {prompt_cache_count} requests, {prompt_cache_status}\n"
                response += f"token cache: {token_stats['size']}/{token_stats['max_size']} entries, "
                response += f"hit rate: {token_stats['hit_rate']}"
                
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["settings"]})
                return chat_history
                
            # Handle cleanup command
            elif command_type == "cleanup":
                days = command_info["parameters"].get("days", 30)
                chat_history.append({"role": "user", "content": user_message})
                
                if not message_manager.vector_store_available:
                    response = "cant clean up vectors, no vector store available"
                else:
                    # Execute the cleanup
                    success = message_manager.cleanup_old_vectors(days)
                    if success:
                        response = f"cleaned up vectors older than {days} days"
                    else:
                        response = f"failed to clean up vectors, check the logs"
                
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["maintenance"]})
                return chat_history
                
            # Handle performance report command
            elif command_type == "performance":
                chat_history.append({"role": "user", "content": user_message})
                
                # Get performance metrics
                perf_report = performance.get_report()
                
                if not perf_report:
                    response = "no performance data collected yet"
                else:
                    # Format performance report
                    response = "performance metrics:\n\n"
                    
                    # Sort components by total time
                    sorted_components = sorted(
                        perf_report.items(),
                        key=lambda x: float(x[1]["total_time"].replace("s", "")),
                        reverse=True
                    )
                    
                    # Add each component's metrics
                    for component, metrics in sorted_components[:10]:  # Show top 10
                        response += f"• {component}: {metrics['avg_time']} avg, {metrics['calls']} calls, {metrics['success_rate']} success\n"
                    
                    # Add summary of slower operations
                    if len(sorted_components) > 10:
                        response += f"\n... and {len(sorted_components) - 10} more components"
                
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["performance"]})
                return chat_history
            
            # Handle other commands and add them to chat history
            chat_history.append({"role": "user", "content": user_message})
            
            # Process the remaining commands
            if command_info["command_type"] == "help":
                # Display help text
                help_text = "yo here's what i can do:\n"
                help_text += "/spiral - make me spiral a bit\n"
                help_text += "/reset - clear our conversation\n"
                help_text += "/cache - toggle prompt caching\n"
                help_text += "/accuracy - toggle accuracy features\n"
                help_text += "/mood [mood] - filter by mood\n"
                help_text += "/topic [topic] - filter by topic\n"
                help_text += "/analyze - analyze our conversation\n"
                help_text += "/cache-stats - show caching stats\n"
                help_text += "/cleanup [days] - cleanup old vectors\n"
                help_text += "/perf - show performance metrics\n"
                help_text += "/exit - end the conversation"
                
                chat_history.append({"role": "assistant", "content": help_text})
                conversation_manager.add_messages([
                    {"role": "assistant", "content": help_text}
                ])
                TerminalUI.render_structured_message(help_text, False, {"mood": "neutral", "topics": ["help"]})
                return chat_history
                
            elif command_info["command_type"] == "mood_filter":
                # Filter by mood
                mood = command_info["parameters"].get("mood", "neutral")
                query = command_info["parameters"].get("query", "")
                
                filtered_messages = message_manager.filter_by_mood(mood, query)
                if filtered_messages:
                    response = f"messages with mood '{mood}':\n\n" + "\n\n".join(filtered_messages)
                else:
                    response = f"no messages with mood '{mood}'"
                
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["filter"]})
                return chat_history
                
            elif command_info["command_type"] == "topic_filter":
                # Filter by topic
                topic = command_info["parameters"].get("topic", "")
                query = command_info["parameters"].get("query", "")
                
                filtered_messages = message_manager.filter_by_topic(topic, query)
                if filtered_messages:
                    response = f"messages about '{topic}':\n\n" + "\n\n".join(filtered_messages)
                else:
                    response = f"no messages about '{topic}'"
                
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["filter"]})
                return chat_history
                
            elif command_info["command_type"] == "analyze":
                # Analyze conversation
                analysis = message_manager.analyze_conversation(chat_history)
                
                response = "conversation analysis:\n\n"
                response += f"• Total messages: {analysis.get('total_messages', 0)}\n"
                response += f"• Your messages: {analysis.get('user_messages', 0)}\n"
                response += f"• My messages: {analysis.get('assistant_messages', 0)}\n"
                response += f"• Most common topics: {', '.join(analysis.get('common_topics', ['none']))}\n"
                response += f"• My mood trend: {analysis.get('mood_trend', 'neutral')}"
                
                chat_history.append({"role": "assistant", "content": response})
                conversation_manager.add_messages([
                    {"role": "assistant", "content": response}
                ])
                TerminalUI.render_structured_message(response, False, {"mood": "neutral", "topics": ["analysis"]})
                return chat_history
        
        # Normal message handling - add user message to history
        chat_history.append({"role": "user", "content": user_message})
        
        # Start performance timing for message processing
        performance.start_timer("message_processing")
        
        # Process the user message in one pass to extract metadata and embedding
        processed_message = process_user_message(message_manager, user_message)
        
        # Analyze Jon's current mood based on recent exchanges with structured outputs
        if len(chat_history) >= 3:
            jon_messages = [msg for msg in chat_history if msg["role"] == "assistant"]
            jon_state.analyze_mood_structured(jon_messages[-3:] if len(jon_messages) >= 3 else jon_messages)
        
        # Build metadata filter based on context - use Jon's current mood
        filter_meta = {}
        if jon_state.current_mood != "neutral":
            filter_meta["mood"] = jon_state.current_mood
        
        # Add topic filtering using processed metadata
        if processed_message["metadata"] and "topics" in processed_message["metadata"]:
            filter_meta["topics"] = processed_message["metadata"]["topics"]
                
        # Use optimized query strategy for better results
        similar_responses = optimize_vector_query(
            message_manager, 
            user_message, 
            filter_meta,
            Config.SIMILAR_RESPONSES
        )
        
        # Use conversation manager to get context instead of direct chat history
        conversation_context = conversation_manager.get_conversation_context(Config.CONTEXT_TOKENS)
        
        # Prepare context using optimized functions
        context = message_manager.prepare_context(
            user_message,
            conversation_context,  # Use managed context instead of raw history
            similar_responses,
            jon_state.spiral_mode
        )
        
        # Use pruned history to avoid token limits
        pruned_history = prune_chat_history(conversation_context)
        
        # Filter to recent messages
        recent_messages = []
        for msg in pruned_history:
            if msg["role"] != "system":  # Skip system messages
                recent_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # First provide the system prompt
        messages = [{"role": "system", "content": context}]
        
        # Then add recent conversation
        messages.extend(recent_messages)
        
        # Ensure the final message is from the user
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})
        
        # Stream structured response with temperature based on spiral mode
        temperature_mod = 0.4 if jon_state.spiral_mode else 0.0
        structured_response = TerminalUI.stream_structured_response(
            cached_chat_completion,  # Pass the function as a parameter
            messages, 
            temperature_modifier=temperature_mod,
            mood=jon_state.current_mood,
            model=FINE_TUNED_MODEL
        )
        
        # Process response and check quality
        max_regeneration_attempts = 2
        regeneration_attempts = 0
        
        while structured_response and regeneration_attempts < max_regeneration_attempts:
            # Extract the actual text response
            response_text = structured_response.get("response", "")
            
            # Skip quality check for very short responses - we'll regenerate anyway
            if len(response_text.split()) < 3:
                # Very short responses might be low quality, try again
                structured_response = TerminalUI.stream_structured_response(
                    messages, 
                    temperature_modifier=0.1 + regeneration_attempts * 0.1,
                    mood=jon_state.current_mood
                )
                regeneration_attempts += 1
                continue
            
            # Check response quality using self-verification
            quality_result = response_quality_check(
                response_text, 
                user_message, 
                structured_response.get("mood", jon_state.current_mood),
                conversation_context
            )
            
            # If quality is good enough, accept the response
            if quality_result.get("score", 0.0) >= 0.7:
                break
                
            # Otherwise try to regenerate with guidance
            improvement = quality_result.get("improvements", "Be more authentic to Jon's style")
            
            # Add improvement guidance to system message
            improved_context = context + f"\n\nImprovement needed: {improvement}"
            improved_messages = [{"role": "system", "content": improved_context}]
            improved_messages.extend(recent_messages)
            
            # Regenerate with slightly higher temperature for more variety
            structured_response = TerminalUI.stream_structured_response(
                improved_messages, 
                temperature_modifier=0.2 + regeneration_attempts * 0.15,
                mood=jon_state.current_mood
            )
            
            regeneration_attempts += 1
        
        # Verify response quality silently after regeneration attempts
        if structured_response:
            # Extract the final text response
            response_text = structured_response.get("response", "")
            
            # Update Jon's mood based on the response
            response_mood = structured_response.get("mood")
            if response_mood:
                jon_state.current_mood = response_mood
            
            # Add the text response to chat history
            chat_history.append({"role": "assistant", "content": response_text})
            
            # Add to conversation manager
            conversation_manager.add_messages([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text}
            ])
            
            # Store the response in vector store for future reference
            if message_manager.vector_store_available:
                # Prepare metadata for storage
                response_metadata = {
                    "mood": structured_response.get("mood", jon_state.current_mood),
                    "topics": structured_response.get("topics", []),
                    "authenticity": structured_response.get("authenticity", 0.7),
                    "timestamp": datetime.now().timestamp()
                }
                
                # Add response to vector store
                message_manager.add_to_vector_store(response_text, response_metadata)
            
            # Reset spiral mode after responding
            if jon_state.spiral_mode:
                jon_state.spiral_mode = False
        else:
            error_response = "yo idk something's off rn"
            chat_history.append({"role": "assistant", "content": error_response})
            conversation_manager.add_messages([
                {"role": "assistant", "content": error_response}
            ])
            # Already rendered in stream_structured_response
        
        # Stop timing message processing
        performance.stop_timer("message_processing", success=True)
        
        return chat_history
    except Exception as e:
        # Catch any unhandled exceptions in message processing
        error_msg = f"yo something broke: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_msg})
        
        # Add error message to conversation manager
        conversation_manager.add_messages([
            {"role": "assistant", "content": error_msg}
        ])
        
        # Stop timing message processing with failure
        if "message_processing" in performance.start_times:
            performance.stop_timer("message_processing", success=False)
            
        TerminalUI.render_structured_message(error_msg, False, {
            "mood": "disappointed", 
            "topics": ["error"]
        })
        console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
        return chat_history

def main():
    """Run Jon GPT"""
    
    # Initialize components
    controller.initialize_session_components()
    
    # Extract components from controller for clarity
    message_manager = controller.message_manager
    jon_state = controller.jon_state
    conversation_manager = controller.conversation_manager
    
    # Display welcome
    TerminalUI.clear_screen()
    TerminalUI.render_header()
    welcome_message = "welcome to schultzchat - type a message or use /help for commands"
    conversation_context = []
    chat_history = [
        {"role": "system", "content": Config.SYSTEM_PROMPT},
        {"role": "assistant", "content": welcome_message},
    ]
    
    # Show welcome message
    TerminalUI.render_structured_message(welcome_message, False, {"mood": "neutral", "topics": []})
    
    # Add welcome to conversation manager
    conversation_manager.add_messages([
        {"role": "assistant", "content": welcome_message}
    ])
    
    # Show footer with status info
    TerminalUI.render_footer(jon_state, message_manager)
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_message = TerminalUI.get_user_input()
            
            if not user_message:
                continue
                
            # Handle the message
            chat_history = handle_message(user_message, chat_history, message_manager, jon_state)
            
            # Show updated status
            TerminalUI.render_footer(jon_state, message_manager)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting SchultzGPT. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            traceback.print_exc()
            console.print("[yellow]Please try again or type 'exit' to quit.[/yellow]")

def optimize_vector_query(message_manager, query_text: str, filter_meta: Dict = None, top_k: int = 5) -> List[Tuple[float, str, Dict]]:
    """Progressive vector query with fallback strategies for optimal results"""
    if not message_manager.vector_store_available or not query_text:
        return []
        
    results = []
    
    # First try hybrid search with all filters for best results
    if filter_meta:
        results = message_manager.hybrid_search(query_text, filter_meta, top_k)
    
    # If hybrid search fails or returns no results, try vector search with filter
    if not results and filter_meta:
        results = message_manager.query_vector_store(query_text, top_k, filter_meta)
    
    # If still no results, try just mood filter if available
    if not results and filter_meta and 'mood' in filter_meta:
        mood_filter = {'mood': filter_meta['mood']}
        results = message_manager.query_vector_store(query_text, top_k, mood_filter)
    
    # If no results with filters, try without any filters but increase result count
    if not results:
        results = message_manager.query_vector_store(query_text, top_k * 2)
    
    return results

def prune_chat_history(chat_history: List[Dict], max_tokens: int = Config.MAX_TOKENS * 0.8) -> List[Dict]:
    """Smart token-aware history pruning to maintain context without exceeding limits"""
    if not chat_history:
        return []
        
    # Always keep system messages
    system_messages = [msg for msg in chat_history if msg["role"] == "system"]
    
    # Get user-assistant interaction pairs
    user_messages = [msg for msg in chat_history if msg["role"] == "user"]
    assistant_messages = [msg for msg in chat_history if msg["role"] == "assistant"]
    
    # Calculate tokens for system messages
    system_tokens = sum(token_cache.count(msg["content"]) for msg in system_messages)
    available_tokens = max_tokens - system_tokens
    
    # Ensure at least the most recent exchange is included
    if available_tokens < 0 or not user_messages or not assistant_messages:
        # Not enough space or no conversation yet
        return system_messages
    
    # Keep the most recent conversation pairs
    conversation_pairs = []
    for i in range(min(len(user_messages), len(assistant_messages))):
        user_idx = len(user_messages) - i - 1
        assistant_idx = len(assistant_messages) - i - 1
        
        if user_idx >= 0 and assistant_idx >= 0:
            # Calculate tokens for this pair
            user_tokens = token_cache.count(user_messages[user_idx]["content"])
            assistant_tokens = token_cache.count(assistant_messages[assistant_idx]["content"])
            pair_tokens = user_tokens + assistant_tokens
            
            # Add pair if it fits
            if available_tokens >= pair_tokens:
                conversation_pairs.append(user_messages[user_idx])
                conversation_pairs.append(assistant_messages[assistant_idx])
                available_tokens -= pair_tokens
            else:
                # No more room
                break
    
    # Combine and sort to maintain original order
    pruned_history = system_messages + conversation_pairs
    
    # Sort based on original positions (if needed)
    if len(pruned_history) > 1:
        original_positions = {msg: i for i, msg in enumerate(chat_history) if msg in pruned_history}
        pruned_history.sort(key=lambda msg: original_positions.get(msg, 0))
    
    return pruned_history

# Add function to process user messages in a unified pipeline
def process_user_message(message_manager, message: str, mood: str = None) -> Dict:
    """Single-pass message processing pipeline to reduce redundant analysis"""
    # Extract embedding for vector operations
    embedding = message_manager.get_embedding(message)
    
    # Get structured metadata using existing methods
    message_data = message_manager.classify_message_structured(message)
    
    # Override mood if specified
    if mood:
        message_data["mood"] = mood
        
    # Return comprehensive message object
    return {
        "text": message,
        "embedding": embedding,
        "metadata": message_data,
        "timestamp": datetime.now().timestamp()
    }

class ConversationManager:
    """Manage conversation history with summarization and segmentation"""
    
    def __init__(self, controller=None):
        self.controller = controller or controller
        self.segments = []  # List of conversation segments
        self.current_segment = []  # Current conversation segment
        self.summaries = {}  # Summaries of old segments
        self.summary_tokens = 0  # Tokens used by summaries
        self.max_segment_messages = 20  # Maximum messages per segment
        self.max_segments = 5  # Maximum number of segments to keep
    
    @timed("summarize_conversation")
    def summarize_conversation(self, messages):
        """Create a concise summary of conversation messages"""
        try:
            # Prepare the messages for summarization
            if len(messages) < 2:
                return "Conversation just started."
                
            # Extract just the conversation content (no system messages)
            conversation_content = []
            for msg in messages:
                if msg["role"] != "system":
                    conversation_content.append(f"{msg['role'].capitalize()}: {msg['content']}")
            
            conversation_text = "\n".join(conversation_content)
            
            # Use a prompt to summarize the conversation
            summary_prompt = [
                {"role": "system", "content": "Create a very concise summary of this conversation segment. Focus on the main topics, key points, and any decisions or conclusions reached. Keep the summary under 100 words."},
                {"role": "user", "content": f"Conversation to summarize:\n\n{conversation_text}"}
            ]
            
            # Use the cached chat completion
            response = cached_chat_completion(
                messages=summary_prompt,
                model="gpt-3.5-turbo-0125",  # Use a smaller, faster model for summarization
                temperature=0.3,
                max_tokens=150,
                user_id="conversation_summarizer"
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            console.print(f"[dim red]Error summarizing conversation: {str(e)}[/dim red]")
            return "Error summarizing conversation."
    
    @timed("add_messages")
    def add_messages(self, messages):
        """Add messages to the conversation history, with segmentation"""
        self.current_segment.extend(messages)
        
        # Check if the current segment is too large
        if len(self.current_segment) >= self.max_segment_messages:
            # Create a summary for this segment
            segment_summary = self.summarize_conversation(self.current_segment)
            
            # Store the summary and segment
            segment_id = len(self.segments)
            self.summaries[segment_id] = segment_summary
            self.segments.append(self.current_segment)
            
            # Reset the current segment
            self.current_segment = []
            
            # Prune old segments if needed
            self._prune_old_segments()
    
    def _prune_old_segments(self):
        """Remove oldest segments if we have too many"""
        while len(self.segments) > self.max_segments:
            # The oldest segment is at index 0
            self.segments.pop(0)
            # Clean up its summary
            if 0 in self.summaries:
                del self.summaries[0]
            
            # Reindex the remaining segments and summaries
            new_summaries = {}
            for i in range(len(self.segments)):
                if i + 1 in self.summaries:
                    new_summaries[i] = self.summaries[i + 1]
            self.summaries = new_summaries
    
    @timed("get_conversation_context")
    def get_conversation_context(self, max_tokens=2000):
        """Get conversation context optimized for token usage"""
        # Start with the current segment as it's most relevant
        all_messages = list(self.current_segment)
        
        # Count tokens in current segment
        current_tokens = sum(self.controller.token_cache.count(msg["content"]) for msg in all_messages)
        
        # If we have room, add summaries of previous segments
        remaining_tokens = max_tokens - current_tokens
        if remaining_tokens > 200:  # Only add summaries if we have enough space
            summaries_text = []
            for segment_id in sorted(self.summaries.keys(), reverse=True):
                summary = self.summaries[segment_id]
                summary_tokens = self.controller.token_cache.count(summary)
                
                if summary_tokens < remaining_tokens:
                    summaries_text.append(f"Previous conversation (part {segment_id+1}): {summary}")
                    remaining_tokens -= summary_tokens
                else:
                    break
            
            # If we have summaries, add them as a system message at the beginning
            if summaries_text:
                all_messages.insert(0, {
                    "role": "system",
                    "content": "Context from earlier conversation:\n" + "\n".join(summaries_text)
                })
        
        return all_messages

class SchultzController:
    """Central controller for managing global shared instances"""
    
    def __init__(self):
        # Initialize key components
        self.performance = PerformanceTracker()
        self.prompt_cache = PromptCache()
        self.token_cache = CachedTokenizer()
        self.retry_handler = RetryHandler()
        self.conversation_manager = None  # Will be initialized after class definitions
        self.message_manager = None       # Will be initialized after class definitions
        self.jon_state = None             # Will be initialized after class definitions
        
        # Initialize API clients
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    def initialize_session_components(self):
        """Initialize components that depend on other classes"""
        if not self.message_manager:
            self.message_manager = MessageManager(controller=self)
        
        if not self.conversation_manager:
            self.conversation_manager = ConversationManager(controller=self)
            
        if not self.jon_state:
            self.jon_state = JonState()
    
    def toggle_prompt_caching(self, enabled=True):
        """Toggle OpenAI prompt caching on or off"""
        self.prompt_cache.cache_enabled = enabled
        status = "enabled" if enabled else "disabled"
        console.print(f"[cyan]Prompt caching {status}[/cyan]")
        
        # Report statistics if disabling
        if not enabled and self.prompt_cache.request_ids:
            cache_size = len(self.prompt_cache.request_ids)
            console.print(f"[dim]Cleared {cache_size} cached requests[/dim]")
            self.prompt_cache.clear_cache()
        
        return self.prompt_cache.cache_enabled

# Create the controller instance
controller = SchultzController()

# Update functions to use the controller
def toggle_prompt_caching(enabled=True):
    """Toggle OpenAI prompt caching on or off"""
    return controller.toggle_prompt_caching(enabled)

if __name__ == "__main__":
    main()
