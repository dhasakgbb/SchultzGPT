"""
Configuration module for SchultzGPT.
Contains all settings, constants, and environment variable loading.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")

# Output directory for data generation
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data_generation/output")

# Jon's persona configuration
PERSONA = """Jon is a millennial who works at a retirement community. He's married to Chelsea (a therapist) and has two cats. He's dealing with relationship issues and currently lives in his mom's basement. He has dyslexia but loves reading fantasy books like Game of Thrones. He enjoys being a dungeon master for D&D games and playing video games in the evening. He's afraid of both driving and flying. He has big dreams but often struggles with follow-through. He shows codependent tendencies and has an anxious attachment style in relationships. He can be mopey and get stuck in negative thought patterns, but he's also self-aware and working on personal growth through therapy. He loves meat and barbecue, and occasionally enjoys cigars and scotch."""

class Config:
    """Configuration class for SchultzGPT"""
    
    # API Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "gpt-4-turbo-preview")
    
    # Data Generation Settings
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "data_generation/output")
    MEMORY_LOGS_DIR = os.getenv("MEMORY_LOGS_DIR", "memory_logs")
    DEFAULT_CHECKPOINT_SIZE = int(os.getenv("DEFAULT_CHECKPOINT_SIZE", "100"))
    DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "20"))
    CHECKPOINT_FREQUENCY: int = int(os.getenv("CHECKPOINT_FREQUENCY", "10"))
    CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    
    # Model Settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    RESPONSE_TOKENS = int(os.getenv("RESPONSE_TOKENS", "150"))
    CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS", "1000"))
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    SPIRAL_TEMPERATURE = float(os.getenv("SPIRAL_TEMPERATURE", "1.5"))
    
    # Retrieval API Settings
    RETRIEVAL_MODEL: str = os.getenv("RETRIEVAL_MODEL", "gpt-4-turbo-preview")
    RETRIEVAL_TEMPERATURE: float = float(os.getenv("RETRIEVAL_TEMPERATURE", "0.7"))
    RETRIEVAL_ASSISTANT_ID: str = os.getenv("RETRIEVAL_ASSISTANT_ID", "")
    USE_RETRIEVAL_API = True
    
    # Debug Settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PERFORMANCE_TRACKING = os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
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
        ("/cleanup <days>", "Remove old entries from retrieval store"),
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