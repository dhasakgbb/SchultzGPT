#!/usr/bin/env python
"""
Jon Data Generator

This module generates synthetic data in three formats:
- Q&A pairs for the Retrieval API
- Conversation data for fine-tuning
- Standalone statements for the Retrieval API

The generator uses real Jon data as examples to ensure high fidelity
and maintains Jon's authentic voice and style.
"""

import os
import sys
import json
import random
import argparse
import time
import threading
import gc
import glob
import requests
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import traceback
from requests.exceptions import RequestException
from time import sleep
from collections import defaultdict
from threading import Lock

# Add Rich library for colorized terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import print as rprint

# Create console object for rich output
console = Console()

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Optional memory tracking
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

# Global configuration variables
CHECKPOINT_FREQUENCY = int(os.getenv("CHECKPOINT_FREQUENCY", "100"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data_generation/output")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(OUTPUT_DIR, "checkpoints")) # Define checkpoint dir
MEMORY_TRACKING = False

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
# Update the import path for Config to work from the new directory location
try:
    from src.config.config import Config
except ImportError:
    # Add parent project directories to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config.config import Config
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Initialize OpenAI client
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Debug logging for API configuration
print("\nOpenAI API Configuration:")
print(f"API Key: {os.environ.get('OPENAI_API_KEY')[:10]}...")
print(f"Organization ID: {os.environ.get('OPENAI_ORG_ID')}")

# Initialize client with project-scoped key and organization
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID"),
    default_headers={
        "OpenAI-Organization": os.environ.get("OPENAI_ORG_ID")
    }
)

# Initialize Responses client
responses_client = client.beta

# Track API usage and errors
api_calls = {
    "qa_pairs": 0,
    "conversations": 0,
    "statements": 0,
    "variations": 0,
    "total_tokens": 0,
    "total_cost": 0,
    "batched_calls": 0,
    "individual_calls": 0,
    "errors": [],
    "retries": 0,
    "total_calls": 0
}

# Thread-local storage for tracking API calls
thread_local = threading.local()

# Global tracking
api_calls_lock = threading.Lock()

# Path to real Jon data
REAL_JON_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'jon_training_data_from_convo.jsonl'))

# Load real Jon conversations
def load_real_jon_data():
    """Load real Jon conversations from the JSONL file"""
    real_jon_data = []
    try:
        if os.path.exists(REAL_JON_DATA_PATH):
            with open(REAL_JON_DATA_PATH, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        real_jon_data.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Count total Jon messages in the data
            jon_msg_count = 0
            for item in real_jon_data:
                if isinstance(item, dict) and "messages" in item:
                    for msg in item["messages"]:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            jon_msg_count += 1
            
            console.print(f"[green]Loaded [bold]{len(real_jon_data)}[/bold] conversation entries with [bold]{jon_msg_count}[/bold] Jon messages from {REAL_JON_DATA_PATH}[/green]")
        else:
            console.print(f"[yellow]Warning: [bold]Real Jon data file not found[/bold] at {REAL_JON_DATA_PATH}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading real Jon data: {e}[/red]")
    
    return real_jon_data

# Load the real Jon data
REAL_JON_DATA = load_real_jon_data()

# Extract Jon's messages from real data
def extract_jon_messages():
    """Extract representative Jon messages from the real data"""
    global JON_REAL_MESSAGES
    
    if not REAL_JON_DATA:
        return []
    
    messages = []
    
    # Extract Jon's messages from the JSONL conversation format
    for item in REAL_JON_DATA:
        if isinstance(item, dict) and "messages" in item:
            for msg in item["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and len(msg.get("content", "")) > 20:
                    messages.append(msg["content"])
    
    # Deduplicate and filter messages
    unique_messages = list(set(messages))
    
    # Filter for longer, more representative messages
    filtered_messages = [m for m in unique_messages if len(m) > 50 and len(m) < 500]
    
    # Limit to a reasonable number
    JON_REAL_MESSAGES = filtered_messages[:50] if len(filtered_messages) > 50 else filtered_messages
    
    console.print(f"[cyan]Extracted [bold]{len(JON_REAL_MESSAGES)}[/bold] representative Jon messages from real data[/cyan]")
    return JON_REAL_MESSAGES

# Jon's real messages for reference
JON_REAL_MESSAGES = extract_jon_messages()

# Constants for data generation
TOPICS = [
    # Personal topics
    "work stress", "favorite books", "weekend plans", 
    "family relationships", "personal goals", "living at parents", 
    "exercise routine", "sleep habits", "hobbies", "travel experiences",
    "marriage struggles", "couples therapy", "relationship issues",
    
    # Geek/Nerd Interests
    "science fiction", "fantasy novels", "comic books", "manga",
    "anime", "indie games", "tabletop games", "board games", "card games",
    "video games", "dungeon mastering", "D&D", "Game of Thrones", 
    "fantasy books", "dwarves", "superhero movies", "Marvel", "DC",
    "pop culture references", "graphic novels",
    
    # Creative Interests
    "graphic design", "drawing", "art supplies", "character design",
    "digital art", "sketching", "illustration", "art frustration",
    
    # Lifestyle
    "cigars", "scotch", "barbecue", "meat dishes", "energy drinks",
    "junk food", "fast food", "snacks", "pizza", "sushi", "buffets", 
    "restaurant visits", "beard care", "beard oil", "beard grooming",
    "spa treatments", "massages", "man care", "self-care day",
    
    # Entertainment
    "comedy shows", "stand-up comedy", "comedy movies", "sitcoms",
    "funny YouTube videos", "memes", "silly jokes",
    
    # Situational
    "giving advice", "responding to a joke", "providing support", 
    "discussing a problem", "sharing an opinion", "making plans", 
    "catching up", "reminiscing", "struggling with dyslexia", 
    "dealing with codependency", "struggling with follow-through", 
    "big dreams", "spiraling", "procrastination", "awkward situations",
    "embarrassing moments", "being misunderstood",
    
    # Emotional
    "feeling anxious", "feeling excited", "feeling cynical", "feeling thoughtful",
    "feeling disappointed", "feeling amused", "feeling motivated", "feeling tired", 
    "feeling mopey", "feeling codependent", "feeling self-doubt", "feeling self-esteem", 
    "feeling self-confidence", "feeling self-improvement", "feeling personal growth"
]

JON_STYLE_ELEMENTS = [
    # General style characteristics 
    "Uses lowercase almost exclusively",
    "Skips apostrophes in contractions (dont, cant, wont, Im)",
    "Makes brief, choppy sentences",
    "Adds 'haha', 'lol', or multiple exclamation points frequently",
    "Makes many spelling errors and typos due to dyslexia",
    "Often substitutes 'u' for 'you'",
    "Sometimes double-texts or sends multiple sequential messages",
    "Uses minimal punctuation, often missing periods and commas",
    "Occasionally adds emojis, but not excessively",
    "Sometimes repeats words or phrases when emphasizing a point",
    "Uses 'like' and 'just' frequently as filler words",
    "Often starts sentences with 'so' or 'and'",
    "Makes typing errors like 'tbe' for 'the' or 'inwanted' for 'I wanted'",
    "Occasionally uses all caps for emphasis",
    "Makes phonetic spelling errors (writhing instead of writing)",
    "Sometimes accidentally types the same word twice",
    "Mixes up letters in words (certi\ufb01cate instead of certificate)",
    "Occasionally accidentally joins words together",
    "Misplaces or omits spaces between words",
    "Often lacks logical transitions between topics"
]

MOODS = [
    "neutral", "sarcastic", "thoughtful", "cynical", "supportive", 
    "amused", "irritated", "relaxed", "tired", "energetic", "mopey", "anxious", "motivated", "depressed", "confident", "self-doubt", "self-esteem", "self-confidence", "self-improvement", "personal growth"
]

# Group topics into semantic clusters for better organization
TOPIC_CLUSTERS = {
    "personal_life": ["work stress", "family relationships", "personal goals", 
                     "living at parents", "exercise routine", "sleep habits", "struggling with dyslexia",
                     "dealing with codependency", "struggling with follow-through", 
                     "marriage struggles", "couples therapy", "relationship issues"],
    "entertainment": ["favorite books", "science fiction", "fantasy novels", "indie games", 
                     "films", "video games", "dungeon mastering", "D&D", 
                     "Game of Thrones", "fantasy books", "anime", "manga", 
                     "superhero movies", "comic books", "graphic novels",
                     "comedy shows", "stand-up comedy", "comedy movies", "sitcoms"],
    "creative_pursuits": ["graphic design", "drawing", "sketching", "illustration", 
                         "character design", "digital art", "art frustration"],
    "social": ["weekend plans", "hobbies", "travel experiences", "social media", 
              "giving advice", "making plans", "catching up", "cigars", "scotch", 
              "barbecue", "meat dishes", "board games", "tabletop games",
              "awkward situations", "embarrassing moments"],
    "emotional_states": ["feeling anxious", "feeling excited", "feeling cynical", 
                        "feeling thoughtful", "feeling disappointed", "feeling amused", 
                        "feeling motivated", "feeling tired", "feeling mopey",
                        "dealing with procrastination", "big dreams"],
    "self_care": ["beard care", "beard oil", "beard grooming", "spa treatments", 
                 "massages", "man care", "self-care day", "sushi", "buffets", 
                 "restaurant visits"]
}

# Entities that Jon might discuss - for metadata enrichment
ENTITIES = {
    "people": ["Chelsea", "mom", "Karen", "Prit", "Gaga", "dad", "Tom", "Chris", "therapist", "Ray", "Meg", "Gator"],
    "activities": ["writing", "reading", "working out", "push-ups", "gaming", "playing video games", 
                  "D&D", "being a dungeon master", "barbecuing", "eating meat", "drinking scotch", 
                  "smoking cigars", "drawing", "sketching", "collecting action figures",
                  "going to spas", "getting massages", "beard grooming", "eating sushi", "going to buffets",
                  "couples therapy", "marriage counseling"],
    "retirement_community": ["activities calendar", "event coordination", "residents", "staff", "scheduling"],
    "thrift_store": ["furniture", "mattresses", "donations", "promotions", "carrying heavy items"],
    "relationship_concepts": ["couples counseling", "therapy", "avoidant attachment", "anxious attachment", 
                             "emotional support", "codependency", "marriage struggles", "separation"],
    "personal_challenges": ["weight loss", "relationship issues", "therapy", "personal growth", 
                           "job transition", "fear of driving", "fear of flying", "dyslexia", 
                           "following through on plans", "mopey moods", "art frustration", 
                           "creative blocks", "social awkwardness", "living with parents"],
    "places": ["mom's basement", "parents' house", "retirement community", "thrift store", 
              "mom's house", "game store", "comic book shop", "sushi restaurant", 
              "buffet", "spa", "massage place", "couples therapist's office"],
    "pets": ["cats"],
    "media": ["Game of Thrones", "fantasy books", "fantasy series", "video games", "Marvel", "DC", 
             "comic books", "anime", "manga", "graphic novels", "action figures", "board games", 
             "comedy movies", "stand-up specials", "sitcoms"],
    "personality_traits": ["millennial", "anxious", "codependent", "big dreamer", "procrastinator", 
                          "geek", "nerd", "failed artist", "awkward", "silly", "self-conscious"],
    "self_care": ["beard oil", "beard balm", "beard comb", "massage", "spa treatment", 
                 "man care products", "self-care routines"],
    "food_preferences": ["sushi", "buffets", "barbecue", "meat dishes", "energy drinks", 
                        "junk food", "fast food", "pizza"]
}

JON_FACTS = [
    "is in his mid-30s and has struggled to adjust to adult life responsibilities",
    "is an event coordinator at a retirement community who recently started the job",
    "previously worked at a thrift store carrying furniture and unloading mattress trucks which helped him lose weight",
    "recently lost weight (from 340 to 298 pounds)",
    "enjoys writing and uses AI for spell checking and improving his wording",
    "is currently living in his mom's basement due to marriage issues with Chelsea",
    "has been working on personal improvement and goes to therapy regularly",
    "is married to Chelsea (a therapist) but they're going through a difficult period",
    "they haven't been intimate in about 2 years, which bothers Jon significantly",
    "values authenticity and emotional support in relationships",
    "can get into 'spiraling' moods where he's more negative and raw",
    "has a good sense of humor and often uses 'haha' or 'lol' in messages",
    "speaks in a casual, sometimes choppy style with minimal punctuation",
    "has recently become more interested in reading",
    "is trying to establish better exercise habits and has asked about workout routines",
    "expresses determination but sometimes struggles with follow-through",
    "sends messages with many spelling errors and typos due to his dyslexia",
    "communicates with an anxious attachment style and feels Chelsea has avoidant tendencies",
    "can be self-deprecating but is working on recognizing his own worth",
    "has two cats that are currently staying with Chelsea at their previous home",
    "is afraid of driving and doesn't drive",
    "is afraid of flying",
    "has a brother named Chris",
    "has a mom named Karen (also called Prit or Gaga)",
    "has a father named Tom who works as a security guard but thinks he's a cop",
    "likes to play video games, especially in the evenings",
    "enjoys cigars occasionally",
    "drinks scotch when socializing",
    "has big dreams but rarely follows through with them",
    "can be mopey and get stuck in negative thought patterns",
    "is a millennial",
    "likes to be a dungeon master for D&D games with friends",
    "loves to eat meat and barbecue",
    "shows codependent tendencies in relationships",
    "is dyslexic which affects his reading and writing significantly",
    "has been working on a book and enjoying the writing process despite his dyslexia",
    "loves Game of Thrones and other fantasy series despite his dyslexia",
    "has severe ADD and struggles significantly with executive dysfunction",
    "demonstrates failure to launch tendencies and difficulty achieving independence",
    "is emotionally very needy and constantly seeks validation and reassurance",
    "feels like Chelsea keeps him at a distance emotionally",
    "complains that Chelsea works then plays games all night with her friends",
    "feels like a 'toy on a shelf waiting to be played with' in his relationship",
    "often feels Chelsea avoids deep conversations about their relationship",
    "has a 6-month timeline in mind for seeing improvements in his marriage",
    "regularly experiences intense anxiety about rejection and abandonment",
    "often gets stuck in cycles of procrastination and perfectionism",
    "finds it difficult to maintain consistent daily routines without external structure or accountability",
    "struggles with basic adult responsibilities like making his bed every morning",
    "genuinely trying to grow and improve himself, even if progress is slow and difficult",
    "uses humor and self-deprecation as primary coping mechanisms to mask insecurity",
    "is highly sensitive to criticism and tends to internalize negative feedback deeply",
    "feels ongoing frustration and disappointment about unmet personal goals",
    "relies heavily on escapist behaviors like video games, fantasy series, and indulgent foods to soothe anxiety",
    "frequently struggles with emotional regulation and can be easily overwhelmed by stress",
    "is introspective and self-aware, but often feels powerless to change his patterns of behavior",
    "deeply fears disappointing others, leading to avoidance of responsibilities or difficult conversations",
    "has an underlying fear of inadequacy and often doubts his own abilities and worth",
    "tends to withdraw emotionally during conflict rather than confronting issues directly",
    "is overly dependent on family support structures, particularly his mother, during stressful times",
    "often has bursts of motivation followed by periods of emotional burnout and low productivity",
    "is currently in a significant period of transition, trying actively to reshape his identity and habits",
    "struggles with boundaries, often agreeing to things he doesn't actually want to do, then becoming resentful or overwhelmed",
    "is highly empathetic, easily absorbing the emotional states of those around him, which sometimes drains his own energy",
    "avoids conflict at almost any cost, preferring harmony even if it means sacrificing his own needs or feelings",
    "tends to catastrophize or exaggerate negative outcomes, intensifying his anxiety and stress",
    "may subconsciously fear success, worrying that achieving goals would bring increased responsibility or scrutiny he's not prepared for",
    "experiences imposter syndrome regularly, particularly in social or professional situations where he feels evaluated",
    "relies heavily on routine and familiar comforts, feeling disproportionately unsettled or anxious when routines are disrupted",
    "studied graphic design in college but dropped out before completing his degree",
    "considers himself a failed artist and occasionally tries to draw but gets frustrated quickly",
    "has strong opinions about superhero movies, comic book adaptations, and fantasy franchises",
    "collects action figures and graphic novels",
    "avoids news, politics, and current events because they stress him out",
    "prefers fantasy worlds and fictional scenarios to real-world topics",
    "can speak at length about niche nerdy topics but feels out of his depth in academic discussions",
    "is knowledgeable about pop culture but not about history, philosophy, or politics",
    "has tried coding and programming but quickly gave up, finding it too difficult",
    "has a beard that he takes pride in and regularly maintains with beard oil and grooming products",
    "loves sushi and considers himself a sushi connoisseur despite his otherwise 'basic' food preferences",
    "gets excited about buffets and all-you-can-eat restaurants",
    "enjoys occasional spa days and massages as 'man care' self-care activities",
    "is socially awkward in new situations and often says the wrong thing",
    "has a silly, sometimes childish sense of humor that emerges when he's comfortable",
    "enjoys comedy movies and stand-up specials, frequently quotes his favorite comedians",
    "gets overly excited about small things and can ramble when talking about his interests",
    "sometimes laughs inappropriately in serious situations due to nervousness",
    "creates elaborate, often impractical plans for self-improvement that rarely materialize",
    "feels both frustrated and relieved to be temporarily living with his parents again",
    "worries about the future of his marriage but is committed to trying to make it work for at least 6 months",
    "misses his cats who are still with Chelsea at their previous shared home",
    "feels like he has changed aspects of himself to 'keep the peace' in his relationship",
    "appreciates that Chelsea has supported them financially but feels emotional burden has been on him"
]

def robust_api_call(api_func, max_retries=3, initial_delay=1):
    """
    Execute an API call with robust error handling and retry logic.
    
    Args:
        api_func: Function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        API response or None if all retries fail
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return api_func()
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Handle rate limits
            if "rate limit" in error_msg or "429" in error_msg:
                console.print(f"\n[yellow]Rate limit hit[/yellow] (attempt {attempt + 1}/{max_retries})")
                console.print(f"[dim]Waiting {delay} seconds before retry...[/dim]")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            
            # Handle API errors
            if "api" in error_msg or "openai" in error_msg:
                console.print(f"\n[orange1]API error[/orange1] (attempt {attempt + 1}/{max_retries}): {str(e)}")
                console.print(f"[dim]Waiting {delay} seconds before retry...[/dim]")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            
            # Handle network errors
            if "network" in error_msg or "connection" in error_msg:
                console.print(f"\n[red]Network error[/red] (attempt {attempt + 1}/{max_retries}): {str(e)}")
                console.print(f"[dim]Waiting {delay} seconds before retry...[/dim]")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            
            # For other errors, print and retry
            console.print(f"\n[red]Error[/red] (attempt {attempt + 1}/{max_retries}): {str(e)}")
            console.print(f"[dim]Waiting {delay} seconds before retry...[/dim]")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    console.print(f"\n[bold red]All {max_retries} attempts failed.[/bold red] Last error: {str(last_error)}")
    return None

def get_token_estimate(text: str) -> int:
    """Roughly estimate token count for text sizing"""
    # Approximation: average English word is ~1.3 tokens
    return int(len(text.split()) * 1.3)

def track_api_call(call_type, tokens_used=0, batch_size=1):
    """Track API usage statistics"""
    with api_calls_lock:
        api_calls[call_type] = api_calls.get(call_type, 0) + 1
        api_calls["total_tokens"] += tokens_used
        
        # Simple cost estimate based on token count for GPT-4
        estimated_cost = tokens_used * 0.00001  # Very rough approximation
        api_calls["total_cost"] += estimated_cost
        
        # Track batched vs individual calls
        if batch_size > 1:
            api_calls["batched_calls"] += 1
        else:
            api_calls["individual_calls"] += 1
            
        # Update total calls counter
        api_calls["total_calls"] += 1

def calculate_optimal_batch_size(content_type, complexity_factor=1.0, target_token_limit=3500):
    """
    Dynamically calculate optimal batch size based on content type and complexity
    
    Args:
        content_type: Type of content ('qa', 'conversation', 'statement')
        complexity_factor: Multiplier for content complexity (higher = more complex)
        target_token_limit: Target token limit per API call
        
    Returns:
        Optimal batch size
    """
    # Base token estimates per item type
    token_estimates = {
        "qa": 500,  # Average Q&A pair tokens
        "conversation": 1200,  # Average conversation tokens
        "statement": 200,  # Average statement tokens
    }
    
    # Calculate batch size
    base_tokens = token_estimates.get(content_type, 500)
    adjusted_tokens = base_tokens * complexity_factor
    
    # Add overhead for prompt and formatting
    with_overhead = adjusted_tokens * 1.2
    
    # Calculate and clamp batch size
    optimal_size = max(1, int(target_token_limit / with_overhead))
    
    # Clamp to reasonable limits
    if content_type == "qa":
        return min(20, optimal_size)
    elif content_type == "conversation":
        return min(5, optimal_size)
    else:
        return min(30, optimal_size)

def calculate_real_data_benchmarks():
    """Calculate style metrics from real Jon data"""
    real_metrics = {
        "typo_frequency": 0,
        "abbreviation_rate": 0,
        "apostrophe_omission": 0,
        "ellipsis_usage": 0,
        "laughter_markers": 0,
        "avg_sentence_length": 0,
        "unique_word_ratio": 0
    }
    
    if not JON_REAL_MESSAGES:
        return real_metrics
        
    total_chars = 0
    total_words = 0
    unique_words = set()
    
    for msg in JON_REAL_MESSAGES:
        # Typos and style markers
        real_metrics["typo_frequency"] += len(re.findall(r'\b(u|ur|dont|wont|cant)\b', msg.lower()))
        real_metrics["abbreviation_rate"] += len(re.findall(r'\b(lol|haha|smh|btw)\b', msg.lower()))
        real_metrics["apostrophe_omission"] += len(re.findall(r"\w+nt\b", msg.lower()))  # matches "dont", "wont"
        real_metrics["ellipsis_usage"] += msg.count('...')
        real_metrics["laughter_markers"] += len(re.findall(r'\b(haha|hahaha|lol|lmao)\b', msg.lower()))
        
        # Linguistic features
        words = msg.split()
        total_words += len(words)
        unique_words.update(words)
        total_chars += len(msg)
        
    # Calculate averages
    num_messages = len(JON_REAL_MESSAGES)
    real_metrics = {
        key: value/num_messages for key, value in real_metrics.items()
    }
    
    # Additional metrics
    real_metrics["avg_sentence_length"] = total_words/num_messages if num_messages else 0
    real_metrics["unique_word_ratio"] = len(unique_words)/total_words if total_words else 0
    real_metrics["avg_word_length"] = total_chars/total_words if total_words else 0
    
    return real_metrics

def analyze_data_quality(data_items, item_type="qa"):
    """
    Analyze the quality of generated data
    
    Args:
        data_items: List of generated data items
        item_type: Type of data (qa, conversation, statement)
    
    Returns:
        Quality metrics dictionary
    """
    # Return early with default metrics if no data
    if not data_items:
        console.print("[yellow]Warning: No data items to analyze[/yellow]")
        return {
            "vocabulary_richness": 0,
            "metadata_completeness": 0,
            "typo_frequency": 0,
            "laughter_markers": 0,
            "avg_token_count": 0,
        }
    
    metrics = {
        "token_counts": [],
        "topic_distribution": {},
        "sentiment_distribution": {},
        "unique_entities": set(),
        "vocabulary_richness": 0,
        "redundancy_score": 0,
        "metadata_completeness": 0,
        "style_consistency": 0,
        "error_count": 0,
        "validation_errors": [],
        "typo_frequency": 0,
        "abbreviation_rate": 0,
        "apostrophe_omission": 0,
        "ellipsis_usage": 0,
        "laughter_markers": 0,
    }
    
    all_text = ""
    valid_items = 0
    
    # Add real data comparison
    real_benchmarks = calculate_real_data_benchmarks()
    
    for item in data_items:
        try:
            # Basic validation
            if not isinstance(item, dict):
                metrics["error_count"] += 1
                metrics["validation_errors"].append(f"Invalid item type: {type(item)}")
                continue
            
            # Extract text based on item type
            text = extract_text(item, item_type)
            if not text:
                metrics["error_count"] += 1
                metrics["validation_errors"].append(f"Missing text in {item_type} item")
                continue
            
            # Token count
            token_count = get_token_estimate(text)
            metrics["token_counts"].append(token_count)
            
            # Topic distribution
            if "metadata" in item and "topics" in item["metadata"]:
                topics = item["metadata"]["topics"]
                if isinstance(topics, list):
                    for topic in topics:
                        metrics["topic_distribution"][topic] = metrics["topic_distribution"].get(topic, 0) + 1
            
            # Also check the singular "topic" field for backward compatibility
            if "metadata" in item and "topic" in item["metadata"]:
                topic = item["metadata"]["topic"]
                if isinstance(topic, str):
                    metrics["topic_distribution"][topic] = metrics["topic_distribution"].get(topic, 0) + 1
            
            # Sentiment distribution - only process if present
            if "metadata" in item and "sentiment" in item["metadata"]:
                sentiment = item["metadata"]["sentiment"]
                metrics["sentiment_distribution"][sentiment] = metrics["sentiment_distribution"].get(sentiment, 0) + 1
            
            # Entity tracking
            if "metadata" in item and "entities" in item["metadata"]:
                entities = item["metadata"]["entities"]
                if isinstance(entities, list):
                    metrics["unique_entities"].update(entities)
            
            # Metadata completeness check
            required_fields = {
                "qa": ["question", "answer", "metadata"],
                "conversation": ["messages", "metadata"],
                "statement": ["statement", "metadata"]
            }
            
            item_fields = required_fields.get(item_type, [])
            if item_fields:
                missing_fields = [field for field in item_fields if field not in item]
                if missing_fields:
                    metrics["validation_errors"].append(f"Missing required fields: {missing_fields}")
            
            # Style consistency check
            if item_type == "qa":
                # Check for Jon's style in answers
                style_markers = ["lowercase", "minimal punctuation", "casual tone"]
                style_score = sum(1 for marker in style_markers if any(marker in text.lower() for char in marker))
                metrics["style_consistency"] += style_score / len(style_markers)
            
            all_text += text + " "
            valid_items += 1
            
            # Style metrics
            metrics["typo_frequency"] += len(re.findall(r'\b(u|ur|dont|wont|cant)\b', text))
            metrics["abbreviation_rate"] += len(re.findall(r'\b(lol|haha|smh|btw)\b', text))
            metrics["apostrophe_omission"] += len(re.findall(r"\w+nt\b", text))
            metrics["ellipsis_usage"] += text.count('...')
            metrics["laughter_markers"] += len(re.findall(r'\b(haha|hahaha|lol|lmao)\b', text))
            
        except Exception as e:
            metrics["error_count"] += 1
            metrics["validation_errors"].append(f"Error processing item: {str(e)}")
            console.print(f"[red]Error in data quality analysis: {e}[/red]")
    
    # Calculate final metrics
    if valid_items > 0:
        # Average token count
        metrics["avg_token_count"] = sum(metrics["token_counts"]) / len(metrics["token_counts"]) if metrics["token_counts"] else 0
        
        # Topic distribution percentage
        total_topics = sum(metrics["topic_distribution"].values())
        if total_topics > 0:
            metrics["topic_distribution"] = {
                topic: count / total_topics * 100 
                for topic, count in metrics["topic_distribution"].items()
            }
        
        # Sentiment distribution percentage
        total_sentiments = sum(metrics["sentiment_distribution"].values())
        if total_sentiments > 0:
            metrics["sentiment_distribution"] = {
                sentiment: count / total_sentiments * 100 
                for sentiment, count in metrics["sentiment_distribution"].items()
            }
        
        # Vocabulary richness (unique words / total words)
        words = all_text.lower().split()
        unique_words = set(words)
        metrics["vocabulary_richness"] = len(unique_words) / len(words) if words else 0
        
        # Metadata completeness
        metrics["metadata_completeness"] = (valid_items - metrics["error_count"]) / valid_items * 100 if valid_items else 0
        
        # Style consistency average
        metrics["style_consistency"] = metrics["style_consistency"] / valid_items * 100 if valid_items else 0
        
        # Redundancy score (based on text similarity)
        if len(data_items) > 1:
            similarity_scores = []
            for i in range(len(data_items)):
                for j in range(i + 1, len(data_items)):
                    text1 = extract_text(data_items[i], item_type)
                    text2 = extract_text(data_items[j], item_type)
                    if text1 and text2:
                        similarity = calculate_text_similarity(text1, text2)
                        similarity_scores.append(similarity)
            metrics["redundancy_score"] = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        # Normalize metrics
        num_items = len(data_items)
        if num_items > 0:
            for key in ["typo_frequency", "abbreviation_rate", "apostrophe_omission",
                       "ellipsis_usage", "laughter_markers"]:
                metrics[key] = metrics[key]/num_items
        
        # Only add comparison if we have real style metrics
        if real_benchmarks and all(key in real_benchmarks for key in ["typo_frequency", "laughter_markers"]):
            try:
                comparison_data = {
                    "metric": ["Typos/Message", "Abbreviations", "Missing Apostrophes",
                              "Ellipsis Usage", "Laughter Markers"],
                    "real_data": [
                        real_benchmarks.get("typo_frequency", 0),
                        real_benchmarks.get("abbreviation_rate", 0),
                        real_benchmarks.get("apostrophe_omission", 0), 
                        real_benchmarks.get("ellipsis_usage", 0),
                        real_benchmarks.get("laughter_markers", 0)
                    ],
                    "generated_data": [
                        metrics["typo_frequency"],
                        metrics["abbreviation_rate"],
                        metrics["apostrophe_omission"], 
                        metrics["ellipsis_usage"],
                        metrics["laughter_markers"]
                    ]
                }
                metrics["real_data_comparison"] = comparison_data
            except Exception as e:
                console.print(f"[yellow]Warning: Could not create real data comparison: {e}[/yellow]")
    
    # Convert sets to lists for JSON serialization
    metrics["unique_entities"] = list(metrics["unique_entities"])
    
    return metrics

def extract_text(item, item_type):
    """Extract text content from a data item based on its type"""
    if item_type == "qa":
        return item.get("question", "") + " " + item.get("answer", "")
    elif item_type == "conversation":
        return " ".join([m.get("content", "") for m in item.get("messages", [])])
    elif item_type == "statement":
        return item.get("statement", "")
    return ""

def calculate_text_similarity(text1, text2):
    """Enhanced similarity calculation using character-level Jaccard"""
    # Existing word-level Jaccard
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    word_sim = len(words1 & words2)/len(words1 | words2) if words1 | words2 else 0
    
    # Character 3-gram similarity
    def get_ngrams(text, n=3):
        return {text[i:i+n] for i in range(len(text)-n+1)}
    
    chars1 = get_ngrams(text1.lower())
    chars2 = get_ngrams(text2.lower())
    char_sim = len(chars1 & chars2)/len(chars1 | chars2) if chars1 | chars2 else 0
    
    # Weighted combination favoring character similarity
    return 0.3*word_sim + 0.7*char_sim

def enrich_metadata(item, item_type, metrics=None):
    """
    Enrich item metadata with additional useful information
    
    Args:
        item: The data item to enrich
        item_type: Type of data (qa, conversation, statement)
        metrics: Optional quality metrics for context
    
    Returns:
        Enriched item with enhanced metadata
    """
    # Deep copy to avoid modifying original
    enriched = dict(item)
    
    # Initialize metadata if not present
    if "metadata" not in enriched:
        enriched["metadata"] = {}
    
    # Add generation timestamp
    enriched["metadata"]["generated_at"] = datetime.now().isoformat()
    
    # Add token estimates if not present
    if "token_estimate" not in enriched["metadata"]:
        if item_type == "qa":
            enriched["metadata"]["token_estimate"] = get_token_estimate(item.get("answer", ""))
        elif item_type == "conversation":
            enriched["metadata"]["token_estimate"] = sum(get_token_estimate(m.get("content", "")) 
                                                     for m in item.get("messages", []))
        elif item_type == "statement":
            enriched["metadata"]["token_estimate"] = get_token_estimate(item.get("statement", ""))
    
    # Add complexity score (ratio of rare words)
    if item_type == "qa":
        text = item.get("answer", "")
    elif item_type == "conversation":
        text = " ".join([m.get("content", "") for m in item.get("messages", [])])
    elif item_type == "statement":
        text = item.get("statement", "")
        
    words = text.lower().split()
    common_words = {"the", "and", "a", "to", "of", "in", "is", "it", "you", "that", "was", "for", "on", "are"}
    if words:
        complexity = sum(1 for word in words if word not in common_words) / len(words)
        enriched["metadata"]["complexity_score"] = round(complexity, 2)
    
    # Add uniqueness score if metrics provided
    if metrics and "redundancy_score" in metrics:
        enriched["metadata"]["uniqueness_score"] = round(1.0 - metrics["redundancy_score"], 2)
    
    # Add data version
    enriched["metadata"]["version"] = "2.0"
    enriched["metadata"]["generator"] = "jon_data_generator"
    
    return enriched

def extract_entities(text: str) -> List[str]:
    """Extract relevant entities from text that match Jon's known entities.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        List of found entities
    """
    found_entities = []
    text_lower = text.lower()
    
    # Check each entity category
    for category, entities in ENTITIES.items():
        for entity in entities:
            if entity.lower() in text_lower:
                found_entities.append(entity)
    
    return list(set(found_entities))  # Remove duplicates

def extract_topics(text: str) -> List[str]:
    """Extract relevant topics from text that match Jon's known topics.
    
    Args:
        text: Text to extract topics from
        
    Returns:
        List of found topics
    """
    found_topics = []
    text_lower = text.lower()
    
    # Check each topic
    for topic in TOPICS:
        if topic.lower() in text_lower:
            found_topics.append(topic)
    
    # Check topic clusters
    for cluster, topics in TOPIC_CLUSTERS.items():
        for topic in topics:
            if topic.lower() in text_lower:
                found_topics.append(topic)
    
    return list(set(found_topics))  # Remove duplicates

def build_prompt(topic: Optional[str] = None, style: str = "casual", use_real_data: bool = True) -> str:
    """Build a prompt for generating Q&A pairs.
    
    Args:
        topic: Optional topic to focus on
        style: Response style (default: casual)
        use_real_data: Whether to use real data examples
        
    Returns:
        Formatted prompt string
    """
    # Get real Jon examples if enabled
    real_examples = ""
    if use_real_data and JON_REAL_MESSAGES:
        # Format examples with line breaks and quotation marks for better clarity
        sample_messages = random.sample(JON_REAL_MESSAGES, min(3, len(JON_REAL_MESSAGES)))
        formatted_examples = "\n\n".join([f"Example: \"{msg}\"" for msg in sample_messages])
        real_examples = f"\nReal Jon examples:\n{formatted_examples}"
    
    # Build topic-specific prompt
    topic_prompt = ""
    if topic:
        topic_prompt = f"Focus on the topic: {topic}\n"
    
    # Get random style elements
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    
    # Get random facts about Jon
    jon_facts = random.sample(JON_FACTS, min(3, len(JON_FACTS)))
    
    prompt = f"""You are Jon. Generate a Q&A pair that reflects Jon's authentic voice and personality.

{Config.PERSONA}

{topic_prompt}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Relevant facts about Jon:
{' '.join([f"- {fact}" for fact in jon_facts])}

IMPORTANT GUIDELINES:
- Jon is a nerd/geek who loves fantasy, sci-fi, comic books, and video games
- Jon studied graphic design but dropped out and considers himself a failed artist
- Jon is intelligent but NOT well-read - he doesn't discuss literature, philosophy, or academic topics
- Jon NEVER talks about programming, coding, or technical subjects - he tried and gave up
- Jon AVOIDS politics and current events completely - they stress him out
- Jon sticks to topics he's comfortable with: games, fantasy, relationships, personal struggles, pop culture
- Jon is dyslexic - he makes many spelling errors, typos and writing mistakes

{real_examples}

Generate a Q&A pair in this format:
Q: [user's question]
A: [Jon's response]

The response should be in Jon's authentic voice and style."""

    return prompt

def apply_dyslexic_typos(text, severity=0.4):
    """
    Apply realistic dyslexic typos to text to match Jon's writing style.
    
    Args:
        text: The text to modify
        severity: How severe the dyslexia symptoms should be (0.0-1.0)
        
    Returns:
        Text with authentic dyslexic writing patterns
    """
    if not text or severity <= 0:
        return text
        
    # Common dyslexic typo patterns
    typo_patterns = [
        # Letter reversals (more common in dyslexia)
        (r'\b(\w*?)([aeiou][bcdfghjklmnpqrstvwxyz])(\w*?)\b', lambda m: m.group(1) + m.group(2)[::-1] + m.group(3), 0.15),
        
        # Common misspellings
        (r'\bdefinitely\b', 'defintely', 0.7),
        (r'\bprobably\b', 'probly', 0.6), 
        (r'\bexperience\b', 'experence', 0.6),
        (r'\binteresting\b', 'intresting', 0.6),
        (r'\btomorrow\b', 'tommorow', 0.6),
        (r'\baccommodate\b', 'acommodate', 0.7),
        (r'\bfriend\b', 'freind', 0.5),
        (r'\btheir\b', 'thier', 0.5),
        (r'\bgoing to\b', 'gona', 0.4),
        (r'\bgonna\b', 'gona', 0.4),
        
        # Joining words
        (r'\ba lot\b', 'alot', 0.6),
        (r'\beach other\b', 'eachother', 0.6),
        (r'\bin front\b', 'infront', 0.5),
        
        # Missing apostrophes (already common in Jon's writing)
        (r'\bdon\'t\b', 'dont', 0.9), 
        (r'\bcan\'t\b', 'cant', 0.9),
        (r'\bwon\'t\b', 'wont', 0.9),
        (r'\bdidn\'t\b', 'didnt', 0.9),
        (r'\bisn\'t\b', 'isnt', 0.9),
        (r'\bwouldn\'t\b', 'wouldnt', 0.9),
        (r'\bcouldn\'t\b', 'couldnt', 0.9),
        (r'\bshouldn\'t\b', 'shouldnt', 0.9),
        (r'\bI\'m\b', 'im', 0.9),
        (r'\byou\'re\b', 'youre', 0.8),
        (r'\bthey\'re\b', 'theyre', 0.8),
        (r'\bwe\'re\b', 'were', 0.8),
        
        # Common 'b' and 'd' confusions
        (r'\bdefinitely\b', 'befinitely', 0.1),
        (r'\bbad\b', 'dad', 0.1),
        
        # Omitting small words
        (r'\b(the|a|to)\s+', '', 0.05),
        
        # Doubling letters incorrectly
        (r'\b(\w+?)([aeioulmnrst])([^aeioulmnrst\s])', lambda m: m.group(1) + m.group(2) + m.group(2) + m.group(3), 0.1)
    ]
    
    # Apply transformations randomly based on severity
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for pattern, replacement, probability in typo_patterns:
            # Adjust probability by severity
            if random.random() < probability * severity:
                if callable(replacement):
                    # For regex substitution with a function
                    line = re.sub(pattern, replacement, line)
                else:
                    # For simple string replacement
                    line = re.sub(pattern, replacement, line)
                    
        # Randomly omit capitalization (50% chance * severity)
        if random.random() < 0.5 * severity and line and line[0].isupper():
            line = line[0].lower() + line[1:]
            
        lines[i] = line
        
    return '\n'.join(lines)

def save_data(
    qa_data: List[Dict[str, Any]],
    conversation_data: List[Dict[str, Any]],
    statement_data: List[Dict[str, Any]],
    output_dir: str,
    verify: bool = False
) -> Dict[str, str]:
    """
    Save generated data in multiple formats.
    
    Args:
        qa_data: List of Q&A pairs
        conversation_data: List of conversations
        statement_data: List of statements
        output_dir: Directory to save files
        verify: Whether to verify the output data
        
    Returns:
        Dictionary with paths to saved files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        raw_data = {
            "qa_data": qa_data,
            "conversation_data": conversation_data,
            "statement_data": statement_data,
            "metadata": {
                "timestamp": timestamp,
                "version": "2.0",
                "generator": "jon_data_generator"
            }
        }
        
        raw_file = os.path.join(output_dir, f"jon_data_raw_{timestamp}.json")
        try:
            with open(raw_file, 'w') as f:
                json.dump(raw_data, f, indent=2)
        except IOError as e:
            print(f"Error saving raw data file: {e}")
            raise
        
        # Save retrieval format
        retrieval_data = []
        
        # Convert QA pairs
        for qa in qa_data:
            retrieval_data.append({
                "text": f"Q: {qa['question']}\nA: {qa['answer']}",
                "metadata": {
                    "type": "qa_pair",
                    "topics": qa.get("metadata", {}).get("topics", []),
                    "version": "2.0"
                }
            })
        
        # Convert statements
        for stmt in statement_data:
            retrieval_data.append({
                "text": stmt["statement"],
                "metadata": {
                    "type": "statement",
                    "topics": stmt.get("metadata", {}).get("topics", []),
                    "version": "2.0"
                }
            })
        
        retrieval_file = os.path.join(output_dir, f"jon_retrieval_data_{timestamp}.jsonl")
        try:
            with open(retrieval_file, 'w') as f:
                for item in retrieval_data:
                    f.write(json.dumps(item) + '\n')
        except IOError as e:
            print(f"Error saving retrieval data file: {e}")
            raise
        
        # Save fine-tuning format
        fine_tuning_data = []
        
        # Convert QA pairs
        for qa in qa_data:
            fine_tuning_data.append({
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            })
        
        # Add conversations
        fine_tuning_data.extend(conversation_data)
        
        fine_tuning_file = os.path.join(output_dir, f"jon_fine_tuning_data_{timestamp}.jsonl")
        try:
            with open(fine_tuning_file, 'w') as f:
                for item in fine_tuning_data:
                    f.write(json.dumps(item) + '\n')
        except IOError as e:
            print(f"Error saving fine-tuning data file: {e}")
            raise
        
        # Verify data if requested
        if verify:
            try:
                # Add the project root to the Python path
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Try to import test functions, but don't fail if they don't exist
                test_functions = {}
                try:
                    from data_generation.utils.test_jon_data import (
                        test_qa_data,
                        test_conversation_data,
                        test_statement_data,
                        test_retrieval_data
                    )
                    test_functions = {
                        "qa": test_qa_data,
                        "conversation": test_conversation_data,
                        "statement": test_statement_data,
                        "retrieval": test_retrieval_data
                    }
                except ImportError as e:
                    print(f"Warning: Could not import test functions: {e}")
                    print("Skipping data verification.")
                    return {
                        "raw_file": raw_file,
                        "retrieval_file": retrieval_file,
                        "fine_tuning_file": fine_tuning_file
                    }
                
                print("\nVerifying generated data...")
                verification_errors = []
                
                # Run tests if available
                if "qa" in test_functions:
                    try:
                        test_functions["qa"](qa_data)
                    except Exception as e:
                        verification_errors.append(f"QA data verification failed: {e}")
                
                if "conversation" in test_functions:
                    try:
                        test_functions["conversation"](conversation_data)
                    except Exception as e:
                        verification_errors.append(f"Conversation data verification failed: {e}")
                
                if "statement" in test_functions:
                    try:
                        test_functions["statement"](statement_data)
                    except Exception as e:
                        verification_errors.append(f"Statement data verification failed: {e}")
                
                if "retrieval" in test_functions:
                    try:
                        test_functions["retrieval"](retrieval_file)
                    except Exception as e:
                        verification_errors.append(f"Retrieval data verification failed: {e}")
                
                if verification_errors:
                    print("\nVerification Warnings:")
                    for error in verification_errors:
                        print(f"- {error}")
                    print("\nSome data verification checks failed. Please review the warnings above.")
                else:
                    print("\nAll data verification checks passed successfully.")
                
            except Exception as e:
                print(f"Warning: Error during data verification: {e}")
                print("Skipping data verification.")
        
        return {
            "raw_file": raw_file,
            "retrieval_file": retrieval_file,
            "fine_tuning_file": fine_tuning_file
        }
    except Exception as e:
        print(f"Error in save_data: {e}")
        raise

# --- Checkpointing Functions ---

def save_checkpoint(data: Dict[str, List[Any]], checkpoint_dir: str):
    """Saves the current generation state to checkpoint files."""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for key, items in data.items():
            if items: # Only save if there's data
                filename = os.path.join(checkpoint_dir, f"checkpoint_{key}_{timestamp}.jsonl")
                # Save only the latest checkpoint, remove older ones for this type
                for old_file in glob.glob(os.path.join(checkpoint_dir, f"checkpoint_{key}_*.jsonl")):
                    if old_file != filename:
                        try:
                            os.remove(old_file)
                        except OSError:
                            pass # Ignore if file is already gone
                with open(filename, 'w') as f:
                    for item in items:
                        f.write(json.dumps(item) + '\n')
        console.print(f"\n[green]Checkpoint saved at {timestamp}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error saving checkpoint: {e}[/red]")

def load_checkpoint(checkpoint_dir: str) -> Dict[str, List[Any]]:
    """Loads the most recent generation state from checkpoint files."""
    loaded_data = {"qa_data": [], "conversation_data": [], "statement_data": []}
    try:
        if not os.path.isdir(checkpoint_dir):
            return loaded_data # No checkpoint directory exists

        for key in loaded_data.keys():
            checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, f"checkpoint_{key}_*.jsonl")), reverse=True)
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[0]
                console.print(f"[cyan]Loading checkpoint for {key} from {os.path.basename(latest_checkpoint)}[/cyan]")
                with open(latest_checkpoint, 'r') as f:
                    for line in f:
                        try:
                            loaded_data[key].append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            console.print(f"[yellow]Warning: Skipping invalid line in checkpoint file {latest_checkpoint}[/yellow]")
                console.print(f"[green]Loaded {len(loaded_data[key])} items for {key}[/green]")

    except Exception as e:
        console.print(f"\n[red]Error loading checkpoint: {e}. Starting fresh.[/red]")
        # Reset data if loading fails catastrophically
        return {"qa_data": [], "conversation_data": [], "statement_data": []}
    return loaded_data

# Add this function after robust_api_call function
def process_batch(batch_tasks, client, temperature=0.7, max_tokens=1000, typo_severity=0.4, max_retries=3):
    """
    Process multiple tasks using OpenAI's Responses API.
    
    Args:
        batch_tasks: List of (task_type, prompt) tuples
        client: OpenAI client
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        typo_severity: Severity of dyslexic typos
        max_retries: Maximum retry attempts
        
    Returns:
        List of (task_type, result) tuples
    """
    if not batch_tasks:
        return []
    
    # Extract prompts and task types
    task_types = [task[0] for task in batch_tasks]
    prompts = [task[1] for task in batch_tasks]
    
    # Print batch information with colors
    task_counts = {}
    for task_type in task_types:
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    # Use different colors for different task types
    task_summary_parts = []
    if task_counts.get("qa", 0) > 0:
        task_summary_parts.append(f"[cyan]{task_counts['qa']} QA pairs[/cyan]")
    if task_counts.get("conv", 0) > 0:
        task_summary_parts.append(f"[green]{task_counts['conv']} conversations[/green]")
    if task_counts.get("stmt", 0) > 0:
        task_summary_parts.append(f"[yellow]{task_counts['stmt']} statements[/yellow]")
    
    task_summary = ", ".join(task_summary_parts)
    
    console.print(f"\n[bold]Processing batch:[/bold] {task_summary}")
    
    # Track API usage
    track_api_call("batch_gen_start")
    api_calls["batched_calls"] += 1
    
    def batch_api_call():
        try:
            start_time = time.time()
            # Make a single API call with multiple prompts using the Responses API
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt} for prompt in prompts],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "text"}
            )
            
            elapsed_time = time.time() - start_time
            
            # Track token usage - safely extract token usage if available
            total_tokens = 0
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                total_tokens = response.usage.total_tokens
            
            track_api_call("batch_gen", total_tokens)
            console.print(f"  [green]✓[/green] Batch completed in [bold]{elapsed_time:.2f}s[/bold]. Used [bold]{total_tokens}[/bold] tokens.")
            
            # Process results
            results = []
            for i, choice in enumerate(response.choices):
                task_type = task_types[i]
                # Safely extract content from the message if it exists
                content = ""
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                
                if task_type == "qa":
                    # Extract question and answer
                    question_match = re.search(r"Q:\s*(.*?)(?=\nA:|$)", content, re.DOTALL | re.IGNORECASE)
                    answer_match = re.search(r"A:\s*(.*?)(?=\n\n|$)", content, re.DOTALL | re.IGNORECASE)
                    
                    if not question_match or not answer_match:
                        results.append((task_type, None))
                        continue
                    
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    
                    # Apply dyslexic typos only to Jon's answer
                    answer = apply_dyslexic_typos(answer, typo_severity)
                    
                    # Validate response quality
                    if len(question) < 10 or len(answer) < 20:
                        results.append((task_type, None))
                        continue
                    
                    # Extract entities and topics
                    entities = extract_entities(answer)
                    topics = extract_topics(answer)
                    
                    # Generate metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "model": "gpt-4-turbo-preview",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "style": "casual",
                        "entities": entities,
                        "topics": topics,
                        "version": "2.0",
                        "generator": "jon_data_generator",
                        "attempt": 1
                    }
                    
                    results.append((task_type, {
                        "question": question,
                        "answer": answer,
                        "metadata": metadata
                    }))
                    
                elif task_type == "conv":
                    # Extract conversation messages
                    messages = []
                    user_pattern = r"User:\s*(.*?)(?=\nJon:|$)"
                    jon_pattern = r"Jon:\s*(.*?)(?=\nUser:|$)"
                    
                    user_matches = re.finditer(user_pattern, content, re.DOTALL | re.IGNORECASE)
                    jon_matches = re.finditer(jon_pattern, content, re.DOTALL | re.IGNORECASE)
                    
                    for user_match, jon_match in zip(user_matches, jon_matches):
                        user_msg = user_match.group(1).strip()
                        jon_msg = jon_match.group(1).strip()
                        
                        # Apply dyslexic typos only to Jon's messages
                        jon_msg = apply_dyslexic_typos(jon_msg, typo_severity)
                        
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": jon_msg})
                    
                    if not messages:
                        results.append((task_type, None))
                        continue
                    
                    # Extract entities and topics
                    all_text = " ".join(m["content"] for m in messages)
                    entities = extract_entities(all_text)
                    topics = extract_topics(all_text)
                    
                    # Generate metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "model": "gpt-4-turbo-preview",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "style": "casual",
                        "entities": entities,
                        "topics": topics,
                        "version": "2.0",
                        "generator": "jon_data_generator",
                        "attempt": 1
                    }
                    
                    results.append((task_type, {
                        "messages": messages,
                        "metadata": metadata
                    }))
                    
                elif task_type == "stmt":
                    # Just use the content as the statement
                    statement = content.strip()
                    
                    # Apply dyslexic typos to statement
                    statement = apply_dyslexic_typos(statement, typo_severity)
                    
                    # Validate response quality
                    if len(statement) < 20:
                        results.append((task_type, None))
                        continue
                    
                    # Extract entities and topics
                    entities = extract_entities(statement)
                    topics = extract_topics(statement)
                    
                    # Generate metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "model": "gpt-4-turbo-preview",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "style": "casual",
                        "entities": entities,
                        "topics": topics,
                        "sentiment": "neutral",  # Default sentiment
                        "version": "2.0",
                        "generator": "jon_data_generator",
                        "attempt": 1
                    }
                    
                    results.append((task_type, {
                        "statement": statement,
                        "metadata": metadata
                    }))
            
            return results
            
        except Exception as e:
            # Log the error
            api_calls["errors"].append(str(e))
            console.print(f"[red]Error in batch processing: {str(e)}[/red]")
            return None
    
    # Use robust API call with retry logic
    results = robust_api_call(batch_api_call, max_retries=max_retries)
    if results is None:
        console.print(f"[bold red]Processing failed after {max_retries} attempts[/bold red]")
        return [(task_type, None) for task_type in task_types]
    
    return results

# --- Now modify the main function to use batch processing ---
def main(args=None):
    """Main entry point for the data generator using parallel execution with the Responses API."""
    global CHECKPOINT_FREQUENCY, OUTPUT_DIR, CHECKPOINT_DIR

    # Parse command line arguments if not provided
    if args is None:
        args = parse_args()

    # Update global settings from args
    CHECKPOINT_FREQUENCY = args.checkpoint_frequency
    OUTPUT_DIR = args.output_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    max_workers = args.max_workers
    current_workers = max_workers  # Track current worker count for dynamic adjustment
    typo_severity = args.typo_severity
    batch_size = args.batch_size  # Number of items to process in each API call

    # Print colorful configuration header using Rich
    console.print("\n" + "="*60, style="cyan")
    console.print("[bold cyan]Jon Data Generation Configuration[/bold cyan]".center(60))
    console.print("="*60, style="cyan")
    
    # Create a Rich table for the configuration
    config_table = Table(show_header=False, box=None, pad_edge=False, highlight=True)
    config_table.add_column("Setting", style="bright_yellow")
    config_table.add_column("Value", style="bright_white")
    
    config_table.add_row("Generation mode", "[bold green]OpenAI Responses API[/bold green]")
    config_table.add_row("Target QA pairs", f"[bold cyan]{args.qa_pairs}[/bold cyan]")
    config_table.add_row("Target conversations", f"[bold green]{args.conversations}[/bold green]")
    config_table.add_row("Target statements", f"[bold yellow]{args.statements}[/bold yellow]")
    config_table.add_row("Items per API call", f"[bold magenta]{batch_size}[/bold magenta]")
    config_table.add_row("Max concurrent workers", f"[bold blue]{max_workers}[/bold blue]")
    config_table.add_row("Using real Jon data", f"[bold]{'Yes' if args.use_real_data else 'No'}[/bold]")
    config_table.add_row("Dyslexic typo severity", f"[bold]{typo_severity:.1f}/1.0[/bold]")
    config_table.add_row("Checkpointing frequency", f"[bold]{CHECKPOINT_FREQUENCY}[/bold] items")
    config_table.add_row("Checkpoint directory", f"[dim]{CHECKPOINT_DIR}[/dim]")
    config_table.add_row("Output directory", f"[dim]{OUTPUT_DIR}[/dim]")
    config_table.add_row("Verify output", f"[bold]{'Yes' if args.verify else 'No'}[/bold]")
    config_table.add_row("Dry run (no saving)", f"[bold]{'Yes' if args.dry_run else 'No'}[/bold]")
    
    console.print(config_table)
    console.print("="*60 + "\n", style="cyan")

    # --- Load from checkpoint with colorful output ---
    console.print("[bold blue]Attempting to load data from checkpoint...[/bold blue]")
    all_data = load_checkpoint(CHECKPOINT_DIR)
    qa_data = all_data["qa_data"]
    conversation_data = all_data["conversation_data"]
    statement_data = all_data["statement_data"]
    console.print("-" * 30, style="dim")

    # Calculate remaining items needed
    qa_needed = max(0, args.qa_pairs - len(qa_data))
    conv_needed = max(0, args.conversations - len(conversation_data))
    stmt_needed = max(0, args.statements - len(statement_data))
    total_needed = qa_needed + conv_needed + stmt_needed

    if total_needed == 0:
        console.print("[bold green]All requested data already generated (found in checkpoints).[/bold green]")
    else:
        console.print(f"Need to generate: [cyan]{qa_needed}[/cyan] QA pairs, [green]{conv_needed}[/green] conversations, [yellow]{stmt_needed}[/yellow] statements.")
        console.print("[bold]Starting generation...[/bold]")

        # Prepare task prompts
        batch_prompts = []
        
        # Prepare prompts for QA pairs, conversations, and statements
        # (Existing code remains unchanged)
        # ... existing code ...
        
        # Shuffle all prompts to mix types
        random.shuffle(batch_prompts)
        
        # Process in batches with rich progress display
        items_generated_since_checkpoint = 0
        
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "[cyan]{task.completed}/{task.total}[/cyan]",
            "[bright_yellow]{task.fields[stats]}[/bright_yellow]",
        ) as progress:
            gen_task = progress.add_task("[green]Generating data[/green]", total=total_needed, stats="")
            
            # Process in batches of batch_size
            for i in range(0, len(batch_prompts), batch_size):
                batch = batch_prompts[i:i+batch_size]
                
                # Update stats in progress bar
                progress.update(gen_task, stats=f"Batch {i//batch_size + 1}/{(len(batch_prompts)+batch_size-1)//batch_size}")
                
                # Process batch
                results = process_batch(
                    batch, 
                    client=client,  # Use the main OpenAI client which has the Responses API
                    temperature=0.7,
                    max_tokens=1000,
                    typo_severity=typo_severity
                )
                
                # Process results
                for (task_type, result) in results:
                    if result:
                        if task_type == "qa":
                            qa_data.append(result)
                        elif task_type == "conv":
                            conversation_data.append(result)
                        elif task_type == "stmt":
                            statement_data.append(result)
                        
                        items_generated_since_checkpoint += 1
                        progress.update(gen_task, advance=1)
                
                # Checkpoint after batch if needed
                if items_generated_since_checkpoint >= CHECKPOINT_FREQUENCY:
                    if not args.dry_run:
                        current_data = {
                            "qa_data": qa_data,
                            "conversation_data": conversation_data,
                            "statement_data": statement_data
                        }
                        save_checkpoint(current_data, CHECKPOINT_DIR)
                        progress.update(gen_task, stats=f"Checkpointed ({items_generated_since_checkpoint} items)")
                    items_generated_since_checkpoint = 0

        console.print("\n[bold green]Generation finished.[/bold green]")

    # --- Final Save with colorful output---
    if not args.dry_run:
        console.print("\n[bold blue]Saving final data...[/bold blue]")
        # Ensure final data is saved, even if checkpoint frequency wasn't hit
        final_data = {
            "qa_data": qa_data,
            "conversation_data": conversation_data,
            "statement_data": statement_data
        }
        save_checkpoint(final_data, CHECKPOINT_DIR) # Save final state as checkpoint too

        output_files = save_data(
            qa_data=qa_data[:args.qa_pairs], # Trim excess from checkpoints if needed
            conversation_data=conversation_data[:args.conversations],
            statement_data=statement_data[:args.statements],
            output_dir=OUTPUT_DIR,
            verify=args.verify
        )
        console.print("\n[bold green]Final data saved![/bold green]")
        if output_files:
            console.print(f"[dim]Raw data saved to:[/dim] [cyan]{output_files.get('raw_file', 'N/A')}[/cyan]")
            console.print(f"[dim]Retrieval data saved to:[/dim] [cyan]{output_files.get('retrieval_file', 'N/A')}[/cyan]")
            console.print(f"[dim]Fine-tuning data saved to:[/dim] [cyan]{output_files.get('fine_tuning_file', 'N/A')}[/cyan]")
    else:
        console.print("\n[yellow]Dry run: Skipping final data saving.[/yellow]")

    # --- Data Quality Analysis with colorful tables---
    console.print("\n[bold blue]Analyzing generated data quality...[/bold blue]")
    try:
        if qa_data:
            qa_metrics = analyze_data_quality(qa_data[:args.qa_pairs], item_type="qa")
            console.print("\n[bold cyan]QA Data Quality Metrics:[/bold cyan]")
            
            # Create a Rich table for metrics
            metrics_table = Table(show_header=True, header_style="bold cyan")
            metrics_table.add_column("Metric")
            metrics_table.add_column("Value")
            
            # Add only important metrics to the table
            key_metrics = [
                ("avg_token_count", "Average token count"),
                ("vocabulary_richness", "Vocabulary richness"),
                ("metadata_completeness", "Metadata completeness (%)"),
                ("typo_frequency", "Typo frequency"),
                ("laughter_markers", "Laughter markers")
            ]
            
            for key, display_name in key_metrics:
                if key in qa_metrics:
                    value = qa_metrics[key]
                    # Format value based on type
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    metrics_table.add_row(display_name, formatted_value)
                    
            console.print(metrics_table)
            
        # Similar tables for conversation and statement data
        # ... existing code ...
            
        # Print real data comparison if available
        if qa_data:
            console.print("\n[bold magenta]Real Data Comparison:[/bold magenta]")
            if "real_data_comparison" in qa_metrics:
                comparison_data = qa_metrics["real_data_comparison"]
                
                # Create a Rich table for comparison
                comparison_table = Table(show_header=True, header_style="bold magenta")
                comparison_table.add_column("Metric")
                comparison_table.add_column("Real Data")
                comparison_table.add_column("Generated")
                comparison_table.add_column("Variance")
                
                for i, metric in enumerate(comparison_data["metric"]):
                    real = comparison_data["real_data"][i]
                    generated = comparison_data["generated_data"][i]
                    variance = ((generated - real) / real * 100) if real > 0 else 0
                    variance_str = f"{variance:+.1f}%" if real > 0 else "N/A"
                    
                    # Choose color based on variance
                    if -10 <= variance <= 10:  # Within 10% is good
                        variance_style = "[green]"
                    elif -20 <= variance <= 20:  # Within 20% is acceptable 
                        variance_style = "[yellow]"
                    else:  # Beyond 20% variance
                        variance_style = "[red]"
                        
                    comparison_table.add_row(
                        metric,
                        f"{real:.2f}",
                        f"{generated:.2f}",
                        f"{variance_style}{variance_str}[/]"
                    )
                    
                console.print(comparison_table)
    except Exception as e:
        console.print(f"[red]Could not perform data quality analysis: {e}[/red]")


    # --- Print API usage statistics with colorful Rich table ---
    console.print("\n[bold blue]API Usage Statistics:[/bold blue]")
    # Calculate total calls based on successful generations + errors
    api_calls["total_calls"] = len(qa_data) + len(conversation_data) + len(statement_data) + len(api_calls["errors"])
    # Format cost
    cost_str = f"${api_calls['total_cost']:.4f}" if api_calls['total_cost'] > 0 else "$0.00"

    # Create a Rich table for API stats
    stats_table = Table(show_header=True, header_style="bold blue")
    stats_table.add_column("Statistic", style="bright_yellow")
    stats_table.add_column("Value", style="bright_white")
    
    stats_table.add_row("Total Successful Generations", f"[green]{len(qa_data) + len(conversation_data) + len(statement_data)}[/green]")
    stats_table.add_row("Total API Calls Attempted", f"{api_calls['total_calls']}")
    stats_table.add_row("Total Tokens Used", f"[cyan]{api_calls['total_tokens']:,}[/cyan]")
    stats_table.add_row("Estimated Cost", f"[green]{cost_str}[/green]")
    stats_table.add_row("Retries", f"{api_calls['retries']}")
    stats_table.add_row("Errors Logged", f"[red]{len(api_calls['errors'])}[/red]")
    
    console.print(stats_table)

    if api_calls['errors']:
        console.print(f"\n[bold red]First 5 Errors Encountered:[/bold red]")
        for i, error in enumerate(api_calls['errors'][:5]):
            console.print(f"[red]{i+1}. {error}[/red]")

    console.print("\n[bold green]Generation process complete![/bold green]")

    # Colorful sample data display
    if qa_data:
        console.print("\n[bold cyan]Sample QA Pair:[/bold cyan]")
        sample_qa = qa_data[0]
        qa_panel = Panel(
            f"[bright_white]Q: {sample_qa.get('question', 'N/A')}[/bright_white]\n\n" +
            f"[green]A: {sample_qa.get('answer', 'N/A')}[/green]",
            title="[cyan]Jon QA Sample[/cyan]",
            border_style="cyan"
        )
        console.print(qa_panel)
    else:
        console.print("\n[yellow]No QA pairs were generated.[/yellow]")

    # Add similar panels for conversation and statement samples
    # ... existing code ...

def parse_args():
    """Parse command line arguments for the Responses API data generator."""
    parser = argparse.ArgumentParser(description="Generate Jon's data using the OpenAI Responses API")
    parser.add_argument("--qa-pairs", type=int, default=400, help="Number of QA pairs to generate")
    parser.add_argument("--conversations", type=int, default=300, help="Number of conversations to generate")
    parser.add_argument("--statements", type=int, default=300, help="Number of statements to generate")
    parser.add_argument("--max-workers", type=int, default=12, help="Maximum number of concurrent workers")
    parser.add_argument("--typo-severity", type=float, default=0.4, help="Severity of dyslexic typos (0.0-1.0)")
    parser.add_argument("--use-real-data", action="store_true", help="Use real Jon data for examples")
    parser.add_argument("--verify", action="store_true", help="Verify output data")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving data")
    parser.add_argument("--topic", type=str, help="Specific topic to focus on")
    parser.add_argument("--checkpoint-frequency", type=int, default=100, help="Save checkpoint every N items")
    parser.add_argument("--output-dir", type=str, default="data_generation/output", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="data_generation/output/checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of prompts to process in each API call")
    return parser.parse_args()

if __name__ == "__main__":
    main()