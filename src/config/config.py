"""
Configuration module for SchultzGPT.
Contains all settings, constants, and environment variable loading.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for SchultzGPT"""
    
    # API Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Context Settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    RESPONSE_TOKENS = int(os.getenv("RESPONSE_TOKENS", "150"))
    CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS", "1000"))
    HISTORY_MESSAGES = 6
    SIMILAR_RESPONSES = 5
    SIMILARITY_THRESHOLD = 0.7
    MODEL_TEMPERATURE = 1.1
    SPIRAL_TEMPERATURE = 1.5  # Higher temperature for spiral mode
    
    # Vector Store Configuration
    VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "jon-memory-store")
    USE_VECTOR_STORE = True
    
    # Debug Settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    PERFORMANCE_TRACKING = os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true"
    
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