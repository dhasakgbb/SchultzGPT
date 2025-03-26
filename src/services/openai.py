"""
OpenAI service module for SchultzGPT.
Provides both synchronous and asynchronous client interfaces.
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage

from src.models.state import ResponseCache, cached


# Initialize clients
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Setup cache
response_cache = ResponseCache(".cache/openai")
response_cache._ensure_cache_dir()


@cached(response_cache)
def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None
) -> ChatCompletion:
    """
    Get a chat completion from OpenAI.
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        response_format: Optional format specification
        
    Returns:
        ChatCompletion object
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if response_format is not None:
            kwargs["response_format"] = response_format
            
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}")


async def async_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None
) -> ChatCompletion:
    """
    Get a chat completion from OpenAI asynchronously.
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        response_format: Optional format specification
        
    Returns:
        ChatCompletion object
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if response_format is not None:
            kwargs["response_format"] = response_format
            
        # First check cache
        if response_cache.enabled:
            cache_key = response_cache._generate_key({
                'func': 'async_chat_completion',
                'args': (messages,),
                'kwargs': {'model': model, 'temperature': temperature, 
                          'max_tokens': max_tokens, 'response_format': response_format}
            })
            
            cached_result = response_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Get response asynchronously
        response = await async_client.chat.completions.create(**kwargs)
        
        # Cache the result if caching is enabled
        if response_cache.enabled:
            response_cache.set(cache_key, response)
            
        return response
    except Exception as e:
        raise RuntimeError(f"Async OpenAI API error: {str(e)}")


def chat_completion_create(
    messages: List[Dict[str, str]],
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False
) -> ChatCompletion:
    """
    Create a chat completion using the OpenAI API.
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use
        temperature: Response variability
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        
    Returns:
        OpenAI chat completion response
    """
    try:
        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Make API call
        response = client.chat.completions.create(**params)
        return response
        
    except Exception as e:
        raise RuntimeError(f"OpenAI Chat API error: {str(e)}")


async def async_chat_completion_create(
    messages: List[Dict[str, str]],
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False
) -> ChatCompletion:
    """
    Create a chat completion asynchronously using the OpenAI API.
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use
        temperature: Response variability
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        
    Returns:
        OpenAI chat completion response
    """
    try:
        # Log the request for debugging
        request_log = {
            'func': 'async_chat_completion_create',
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream,
            'message_count': len(messages)
        }
        
        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Make async API call
        response = await async_client.chat.completions.create(**params)
        return response
        
    except Exception as e:
        raise RuntimeError(f"Async OpenAI Chat API error: {str(e)}")


async def batch_async_operations(
    operations: List[Callable[..., Awaitable[Any]]],
    max_concurrency: int = 5
) -> List[Any]:
    """
    Run multiple async operations with controlled concurrency.
    
    Args:
        operations: List of async callables to execute
        max_concurrency: Maximum number of concurrent operations
        
    Returns:
        List of operation results
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore(operation):
        async with semaphore:
            return await operation
    
    # Create tasks with semaphore control
    tasks = [run_with_semaphore(op) for op in operations]
    
    # Execute all tasks and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results to re-raise exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        processed_results.append(result)
        
    return processed_results


def run_async(async_func):
    """
    Decorator to run an async function in a synchronous context.
    
    Usage:
        @run_async
        async def my_async_function():
            ...
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper 