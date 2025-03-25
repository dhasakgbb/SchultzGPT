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

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.embedding import Embedding

from models.state import ResponseCache, cached


# Initialize clients
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Setup cache
response_cache = ResponseCache(cache_dir=".cache/openai")


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


def embeddings_create(
    input: Union[str, List[str]],
    model: str = "text-embedding-ada-002"
) -> List[List[float]]:
    """
    Create embeddings for the given input.
    
    Args:
        input: String or list of strings to embed
        model: The embedding model to use
        
    Returns:
        List of embedding vectors
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=input
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"OpenAI Embeddings API error: {str(e)}")


async def async_embeddings_create(
    input: Union[str, List[str]],
    model: str = "text-embedding-ada-002"
) -> List[List[float]]:
    """
    Create embeddings for the given input asynchronously.
    
    Args:
        input: String or list of strings to embed
        model: The embedding model to use
        
    Returns:
        List of embedding vectors
    """
    try:
        # First check cache
        if response_cache.enabled:
            cache_key = response_cache._generate_key({
                'func': 'async_embeddings_create',
                'args': (input,),
                'kwargs': {'model': model}
            })
            
            cached_result = response_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Get response asynchronously
        response = await async_client.embeddings.create(
            model=model,
            input=input
        )
        
        result = [item.embedding for item in response.data]
        
        # Cache the result if caching is enabled
        if response_cache.enabled:
            response_cache.set(cache_key, result)
            
        return result
    except Exception as e:
        raise RuntimeError(f"Async OpenAI Embeddings API error: {str(e)}")


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