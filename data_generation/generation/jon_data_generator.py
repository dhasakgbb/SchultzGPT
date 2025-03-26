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
import traceback
from requests.exceptions import RequestException
from time import sleep
from collections import defaultdict

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

# Add a function for robust API calls with retry logic
def robust_api_call(call_func, max_retries=3, backoff_factor=2):
    """
    Make a robust API call with retry logic for transient errors
    
    Args:
        call_func: Function that makes the actual API call
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        
    Returns:
        API response or None if all retries failed
    """
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            return call_func()
        except Exception as e:
            last_error = e
            retry_count += 1
            api_calls["errors"].append(str(e))
            
            # Don't retry if it's an authentication or permission error
            if hasattr(e, 'code') and e.code in (401, 403):
                print(f"Authentication error: {e}")
                break
                
            if retry_count <= max_retries:
                wait_time = backoff_factor ** retry_count
                api_calls["retries"] += 1
                print(f"API call failed: {e}. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"API call failed after {max_retries} retries: {e}")
    
    return None

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
            print(f"Loaded {len(real_jon_data)} real Jon conversations")
        else:
            print(f"Warning: Real Jon data file not found at {REAL_JON_DATA_PATH}")
    except Exception as e:
        print(f"Error loading real Jon data: {e}")
    
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
    
    # Extract Jon's messages from different formats
    for item in REAL_JON_DATA:
        if isinstance(item, dict):
            # QA format
            if "answer" in item and len(item["answer"]) > 20:
                messages.append(item["answer"])
            
            # Conversation format
            if "messages" in item:
                for msg in item["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and len(msg.get("content", "")) > 20:
                        messages.append(msg["content"])
            
            # Statement format
            if "statement" in item and len(item["statement"]) > 20:
                messages.append(item["statement"])
    
    # Deduplicate and filter messages
    unique_messages = list(set(messages))
    
    # Filter for longer, more representative messages
    filtered_messages = [m for m in unique_messages if len(m) > 50 and len(m) < 500]
    
    # Limit to a reasonable number
    JON_REAL_MESSAGES = filtered_messages[:50] if len(filtered_messages) > 50 else filtered_messages
    
    print(f"Extracted {len(JON_REAL_MESSAGES)} representative Jon messages from real data")
    return JON_REAL_MESSAGES

# Jon's real messages for reference
JON_REAL_MESSAGES = extract_jon_messages()

# Constants for data generation
TOPICS = [
    # Personal topics
    "work stress", "favorite books", "weekend plans", "dating life",
    "family relationships", "personal goals", "apartment living",
    "exercise routine", "sleep habits", "hobbies", "travel experiences",
    
    # Interests
    "science fiction", "literary fiction", "technology news", "programming",
    "creative writing", "indie games", "films", "music", "philosophy",
    "psychology", "history", "politics", "social media", "video games", 
    "dungeon mastering", "D&D", "Game of Thrones", "fantasy books",
    
    # Lifestyle
    "cigars", "scotch", "barbecue", "meat dishes",
    
    # Situational
    "giving advice", "reacting to news", "responding to a joke",
    "providing support", "discussing a problem", "debating ideas",
    "sharing an opinion", "making plans", "catching up", "reminiscing",
    "struggling with dyslexia", "dealing with codependency", "struggling with follow-through", "big dreams", "spiraling", "procrastination",

    
    # Emotional
    "feeling anxious", "feeling excited", "feeling cynical", "feeling thoughtful",
    "feeling disappointed", "feeling amused", "feeling motivated", "feeling tired", "feeling mopey", "feeling codependent", 
    "feeling self-doubt", "feeling self-esteem", "feeling self-confidence", "feeling self-improvement", "feeling personal growth"
]

MOODS = [
    "neutral", "sarcastic", "thoughtful", "cynical", "supportive", 
    "amused", "irritated", "relaxed", "tired", "energetic", "mopey", "anxious", "motivated", "depressed", "confident", "self-doubt", "self-esteem", "self-confidence", "self-improvement", "personal growth"
]

# Group topics into semantic clusters for better organization
TOPIC_CLUSTERS = {
    "personal_life": ["work stress", "dating life", "family relationships", "personal goals", 
                     "apartment living", "exercise routine", "sleep habits", "struggling with dyslexia",
                     "dealing with codependency", "struggling with follow-through"],
    "entertainment": ["favorite books", "science fiction", "literary fiction", "indie games", 
                     "films", "music", "video games", "dungeon mastering", "D&D", 
                     "Game of Thrones", "fantasy books"],
    "intellectual": ["philosophy", "psychology", "history", "politics", "technology news", 
                    "programming", "creative writing"],
    "social": ["weekend plans", "hobbies", "travel experiences", "social media", 
              "giving advice", "making plans", "catching up", "cigars", "scotch", 
              "barbecue", "meat dishes"],
    "emotional_states": ["feeling anxious", "feeling excited", "feeling cynical", 
                        "feeling thoughtful", "feeling disappointed", "feeling amused", 
                        "feeling motivated", "feeling tired", "feeling mopey",
                        "dealing with procrastination", "big dreams"]
}

# Entities that Jon might discuss - for metadata enrichment
ENTITIES = {
    "people": ["Chelsea", "mom", "Karen", "Prit", "Gaga", "dad", "Tom", "Chris", "therapist", "Ray", "Meg", "Gator"],
    "activities": ["writing", "reading", "working out", "push-ups", "gaming", "playing video games", "D&D", "being a dungeon master", "barbecuing", "eating meat", "drinking scotch", "smoking cigars"],
    "retirement_community": ["activities calendar", "event coordination", "residents", "staff", "scheduling"],
    "thrift_store": ["furniture", "mattresses", "donations", "promotions", "carrying heavy items"],
    "relationship_concepts": ["couples counseling", "therapy", "avoidant attachment", "anxious attachment", "emotional support", "codependency"],
    "personal_challenges": ["weight loss", "relationship issues", "therapy", "personal growth", "job transition", "fear of driving", "fear of flying", "dyslexia", "following through on plans", "mopey moods"],
    "places": ["basement", "apartment", "retirement community", "thrift store", "mom's house"],
    "pets": ["cats"],
    "media": ["Game of Thrones", "fantasy books", "fantasy series", "video games"],
    "personality_traits": ["millennial", "anxious", "codependent", "big dreamer", "procrastinator"]
}

JON_STYLE_ELEMENTS = [
    "uses lowercase almost exclusively, rarely capitalizes even names",
    "skips apostrophes in contractions (writes 'dont' instead of 'don't', 'im' instead of 'I'm')",
    "uses minimal punctuation and often runs sentences together",
    "adds 'lol' or 'haha' or 'hahahah' for lighter moments",
    "uses 'dude', 'man', or 'buddy' when addressing friends...uses buddy more than dude",
    "makes brief, choppy sentences with incomplete thoughts",
    "adds occasional typos or misspellings ('wrigjting' instead of 'writing')",
    "makes typing errors like repeated words or phrases",
    "uses text abbreviations like 'u' for 'you' and 'ur' for 'your'",
    "starts messages with 'yeah' or 'yup' when agreeing",
    "uses ellipses... to trail off thoughts",
    "types 'like' as a filler word similar to spoken speech",
    "sometimes uses ALL CAPS for emphasis",
    "shows anxious thought patterns with questions and self-doubt",
    "balances self-deprecation with moments of determination",
    "uses 'inknow' instead of 'I know' occasionally",
    "refers to serious topics bluntly and directly",
    "abruptly changes topics within messages",
    "tends to use periods sparingly, creating run-on sentences",
    "uses 'like' as a filler word similar to spoken speech",
    "sometimes uses the wrong version of a word such as 'knew' or 'new'",
    
    
]

JON_FACTS = [
    "is an event coordinator at a retirement community who recently started the job",
    "previously worked at a thrift store carrying furniture and unloading mattress trucks",
    "recently lost weight (from 340 to 298 pounds)",
    "enjoys writing and uses AI for spell checking and improving his wording",
    "is currently staying in his mom's basement due to relationship issues",
    "has been working on personal improvement and is going to therapy",
    "is dealing with relationship challenges with Chelsea",
    "values authenticity and emotional support in relationships",
    "can get into 'spiraling' moods where he's more negative and raw",
    "has a good sense of humor and often uses 'haha' or 'lol' in messages",
    "speaks in a casual, sometimes choppy style with minimal punctuation",
    "has recently become more interested in reading",
    "is trying to establish better exercise habits and has asked about workout routines",
    "expresses determination but sometimes struggles with follow-through",
    "sends messages with occasional spelling errors and typos",
    "communicates with an anxious attachment style in relationships",
    "can be self-deprecating but is working on recognizing his own worth",
    "is married to Chelsea who is a therapist",
    "has two cats",
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
    "is dyslexic which affects his reading and writing",
    "loves Game of Thrones and other fantasy series despite his dyslexia",
    "has severe ADD and struggles significantly with executive dysfunction",
    "demonstrates failure to launch tendencies and difficulty achieving independence",
    "is emotionally very needy and constantly seeks validation and reassurance",
    "tends to idealize partners and friends, often placing unrealistic expectations on relationships",
    "regularly experiences intense anxiety about rejection and abandonment",
    "often gets stuck in cycles of procrastination and perfectionism",
    "finds it difficult to maintain consistent daily routines without external structure or accountability",
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
    "might idealize past events or relationships, nostalgically overlooking negative aspects as a coping mechanism",
    "frequently engages in passive-aggressive communication, stemming from discomfort with direct confrontation or expression of dissatisfaction",
    "displays perfectionistic tendencies that paradoxically contribute to procrastination and avoidance of starting or completing projects"
]

# Thread-local storage for tracking API calls
thread_local = threading.local()

# Global tracking
api_calls_lock = threading.Lock()

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

def analyze_data_quality(data_items, item_type="qa"):
    """
    Analyze the quality of generated data
    
    Args:
        data_items: List of generated data items
        item_type: Type of data (qa, conversation, statement)
    
    Returns:
        Quality metrics dictionary
    """
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
        "validation_errors": []
    }
    
    all_text = ""
    valid_items = 0
    
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
            if "metadata" in item and "topic" in item["metadata"]:
                topic = item["metadata"]["topic"]
                metrics["topic_distribution"][topic] = metrics["topic_distribution"].get(topic, 0) + 1
            
            # Sentiment distribution
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
            
            missing_fields = [field for field in required_fields[item_type] if field not in item]
            if missing_fields:
                metrics["validation_errors"].append(f"Missing required fields: {missing_fields}")
            
            # Style consistency check
            if item_type == "qa":
                # Check for Jon's style in answers
                style_markers = ["lowercase", "minimal punctuation", "casual tone"]
                style_score = sum(1 for marker in style_markers if any(marker in text.lower()))
                metrics["style_consistency"] += style_score / len(style_markers)
            
            all_text += text + " "
            valid_items += 1
            
        except Exception as e:
            metrics["error_count"] += 1
            metrics["validation_errors"].append(f"Error processing item: {str(e)}")
    
    # Calculate final metrics
    if valid_items > 0:
        # Average token count
        metrics["avg_token_count"] = sum(metrics["token_counts"]) / len(metrics["token_counts"])
        
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
        metrics["metadata_completeness"] = (valid_items - metrics["error_count"]) / valid_items * 100
        
        # Style consistency average
        metrics["style_consistency"] = metrics["style_consistency"] / valid_items * 100
        
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
    """
    Calculate simplified text similarity using Jaccard similarity of word sets
    Returns a value between 0 (completely different) and 1 (identical)
    """
    if not text1 or not text2:
        return 0
        
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

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

def generate_qa_pair(
    topic: Optional[str] = None,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Generate a single Q&A pair with metadata.
    
    Args:
        topic: Optional topic to focus on
        style: Response style (default: casual)
        use_real_data: Whether to use real data examples
        max_retries: Maximum number of retries
        temperature: Response variability
        max_tokens: Maximum tokens in response
        client: Optional OpenAI client
        
    Returns:
        Dictionary containing the Q&A pair and metadata
    """
    # Use global client if none provided
    if client is None:
        client = globals()["client"]
    
    # Build prompt with real data examples if enabled
    prompt = build_prompt(topic, style, use_real_data)
    
    # Track API usage
    track_api_call("qa_pair_gen_start")
    
    for attempt in range(max_retries + 1):
        try:
            # Generate response with exponential backoff
            if attempt > 0:
                backoff_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                print(f"Retry attempt {attempt}/{max_retries}, waiting {backoff_time} seconds...")
                time.sleep(backoff_time)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track successful API call
            track_api_call("qa_pair_gen", response.usage.total_tokens)
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract question and answer with improved regex
            question_match = re.search(r"Q:\s*(.*?)(?=\nA:|$)", content, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"A:\s*(.*?)(?=\n\n|$)", content, re.DOTALL | re.IGNORECASE)
            
            if not question_match or not answer_match:
                raise ValueError("Could not extract Q&A from response")
                
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()
            
            # Validate response quality
            if len(question) < 10 or len(answer) < 20:
                raise ValueError("Response too short")
            
            if len(answer) > max_tokens * 4:  # Rough estimate
                raise ValueError("Response too long")
            
            # Extract entities and topics
            entities = extract_entities(answer)
            topics = extract_topics(answer)
            
            # Generate metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4-turbo-preview",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "style": style,
                "entities": entities,
                "topics": topics,
                "version": "2.0",
                "generator": "jon_data_generator",
                "attempt": attempt + 1
            }
            
            return {
                "question": question,
                "answer": answer,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error generating Q&A pair (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt == max_retries:
                raise
            continue
    
    raise Exception(f"Failed to generate Q&A pair after {max_retries + 1} attempts")

def generate_contextual_variations(
    base_qa: Dict[str, Any],
    num_variations: int = 3,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> List[Dict[str, Any]]:
    """Generate contextual variations of a base Q&A pair
    
    Args:
        base_qa: Base Q&A pair to generate variations from
        num_variations: Number of variations to generate
        style: Writing style (casual, formal, etc.)
        use_real_data: Whether to use real data examples
        max_retries: Maximum number of retries
        temperature: Response variability
        max_tokens: Maximum tokens in response
        client: Optional OpenAI client
        
    Returns:
        List of Q&A pairs with variations
    """
    if client is None:
        client = OpenAI()
    
    variations = []
    for i in range(num_variations):
        try:
            # Generate variation
            variation = generate_qa_pair(
                topic=base_qa["metadata"]["topics"][0],  # Use first topic
                style=style,
                use_real_data=use_real_data,
                max_retries=max_retries,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client
            )
            
            # Add variation metadata
            variation["metadata"]["variation_of"] = base_qa["question"]
            variation["metadata"]["variation_index"] = i
            variation["metadata"]["version"] = "2.0"
            variation["metadata"]["generator"] = "jon_data_generator"
            
            variations.append(variation)
            
        except Exception as e:
            print(f"Error generating variation {i}: {e}")
            if max_retries > 0:
                return generate_contextual_variations(
                    base_qa=base_qa,
                    num_variations=num_variations,
                    style=style,
                    use_real_data=use_real_data,
                    max_retries=max_retries - 1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    client=client
                )
            raise
    
    return variations

def generate_bulk_conversations(count: int, turns_per_conversation: int = 6) -> List[Dict[str, Any]]:
    """
    Generate multiple conversations in a single API call
    
    Args:
        count: Number of conversations to generate
        turns_per_conversation: Number of message exchanges per conversation
    """
    # Select random topics from different clusters for variety
    topics = []
    clusters = list(TOPIC_CLUSTERS.keys())
    for _ in range(count):
        cluster = random.choice(clusters)
        topic = random.choice(TOPIC_CLUSTERS[cluster])
        topics.append(topic)
    
    # Style elements and moods
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    mood_options = random.sample(MOODS, min(5, len(MOODS)))
    
    prompt = f"""
    Generate {count} complete conversations between User and Jon.
    
    Jon's persona:
    {Config.PERSONA}
    
    Each conversation should:
    1. Have approximately {turns_per_conversation} total messages (alternating between User and Jon)
    2. Begin with a user message and end with a Jon message
    3. Have a natural flow and tone
    
    Topics to use (one per conversation):
    {', '.join(topics)}
    
    Jon's style elements to incorporate:
    {' '.join([f"- {element}" for element in style_elements])}
    
    Potential moods for Jon:
    {', '.join(mood_options)}
    
    Format your response as a JSON array of conversation objects.
    Each conversation object should contain:
    - "topic": The conversation topic
    - "messages": Array of message objects, each with:
      - "role": Either "user" or "assistant" (for Jon)
      - "content": The message content
      - "mood": Jon's mood (only for assistant messages)
    
    Your response should be ONLY a valid JSON array without explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=4000  # Increased token limit for multiple conversations
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Check if result is already an array or if it's wrapped in an object
        if isinstance(result_json, dict) and "conversations" in result_json:
            conversations = result_json["conversations"]
        elif isinstance(result_json, list):
            conversations = result_json
        else:
            # Try to find an array in the response
            for key in result_json:
                if isinstance(result_json[key], list):
                    conversations = result_json[key]
                    break
            else:
                raise ValueError("Could not find conversations array in response")
        
        return conversations
    except Exception as e:
        print(f"Error generating bulk conversations: {e}")
        # Return empty conversations to avoid breaking the loop
        return [{"topic": "", "messages": [], "error": str(e)} for _ in range(count)]

def generate_bulk_statements(count: int, topic: str = None) -> List[Dict[str, Any]]:
    """
    Generate multiple Jon statements in a single API call
    
    Args:
        count: Number of statements to generate
        topic: Optional topic to focus on
    """
    # Select topic if not provided
    if not topic:
        cluster = random.choice(list(TOPIC_CLUSTERS.keys()))
        topic = random.choice(TOPIC_CLUSTERS[cluster])
    
    # For topic clustering
    topic_cluster = None
    for cluster, topics in TOPIC_CLUSTERS.items():
        if topic in topics:
            topic_cluster = cluster
            break
    
    # Style elements to include
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    
    prompt = f"""
    Generate {count} standalone statements from Jon on the topic of {topic}.
    
    Jon's persona:
    {Config.PERSONA}
    
    Jon's style elements to incorporate:
    {' '.join([f"- {element}" for element in style_elements])}
    
    Each statement should:
    1. Be in Jon's authentic voice
    2. Express an opinion, insight, or perspective
    3. Be between 1-3 sentences long
    4. Vary in tone and mood
    
    Format your response as a JSON array of statement objects, each with:
    - "statement": The text of Jon's statement
    - "topic": The specific aspect of {topic} addressed
    - "mood": Jon's mood (e.g., {', '.join(random.sample(MOODS, 3))})
    - "sentiment": Emotional tone (positive, negative, neutral, mixed)
    
    Your response should be ONLY a valid JSON array without explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Check if result is already an array or if it's wrapped in an object
        if isinstance(result_json, dict) and "statements" in result_json:
            statements = result_json["statements"]
        elif isinstance(result_json, list):
            statements = result_json
        else:
            # Try to find an array in the response
            for key in result_json:
                if isinstance(result_json[key], list):
                    statements = result_json[key]
                    break
            else:
                raise ValueError("Could not find statements array in response")
        
        # Add metadata to each statement
        for statement in statements:
            statement["metadata"] = {
                "topic": statement.get("topic", topic),
                "topic_cluster": topic_cluster,
                "mood": statement.get("mood", "neutral"),
                "sentiment": statement.get("sentiment", "neutral"),
                "token_estimate": get_token_estimate(statement["statement"])
            }
        
        return statements
    except Exception as e:
        print(f"Error generating bulk statements: {e}")
        return [{"statement": "", "error": str(e), "metadata": {}} for _ in range(count)]

def generate_parallel(func, count, batch_size, **kwargs):
    """
    Generate data in parallel using thread pool
    
    Args:
        func: Generation function to call
        count: Total items to generate
        batch_size: Items per batch
        **kwargs: Additional arguments for generation function
    
    Returns:
        List of generated items
    """
    results = []
    
    # Calculate number of full batches and remainder
    num_batches = count // batch_size
    remainder = count % batch_size
    
    # Create batches with appropriate sizes
    batches = [batch_size] * num_batches
    if remainder > 0:
        batches.append(remainder)
    
    # Define worker function
    def worker(batch_size, **kwargs):
        try:
            return func(batch_size, **kwargs)
        except Exception as e:
            print(f"Error in worker thread: {e}")
            return []
    
    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(len(batches), 3)) as executor:
        futures = [executor.submit(worker, size, **kwargs) for size in batches]
        
        # Process results as they complete
        for future in tqdm(futures, total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            results.extend(batch_results)
            
            # Add small delay between batches to avoid rate limits
            time.sleep(0.5)
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate Jon data for training and retrieval")
    parser.add_argument("--qa-pairs", type=int, default=0, help="Number of QA pairs to generate")
    parser.add_argument("--conversations", type=int, default=0, help="Number of conversations to generate")
    parser.add_argument("--statements", type=int, default=0, help="Number of statements to generate")
    parser.add_argument("--topic", type=str, help="Topic to focus on")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated data")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for API calls")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent API calls")
    parser.add_argument("--use-real-data", action="store_true", help="Use real Jon data as examples")
    parser.add_argument("--one-shot", action="store_true", help="Use one-shot generation for all data types")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory for checkpoint files")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N items")
    parser.add_argument("--verify", action="store_true", help="Verify output files")
    parser.add_argument("--no-verify", action="store_true", help="Skip output verification")
    return parser.parse_args()

def add_batch_metadata(items, item_type, batch_num=0, generation_method="batch_api"):
    """Add consistent metadata to each item generated in a batch
    
    This is a centralized function to ensure metadata is consistent across all generation methods.
    
    Args:
        items: List of generated items (QA pairs, conversations, or statements)
        item_type: Type of item ("qa", "conversation", or "statement")
        batch_num: Batch number for tracking
        generation_method: Method used to generate data ("batch_api", "one_shot", "standard", etc.)
        
    Returns:
        The original items list with metadata added to each item
    """
    timestamp = datetime.now().isoformat()
    
    for item in items:
        if not isinstance(item, dict):
            print(f"Warning: Received non-dictionary {item_type}: {item}")
            continue
            
        if item_type == "qa":
            # Find topic cluster
            topic_cluster = "other"
            if "topic" in item:
                topic = item.get("topic", "")
                for cluster, topics in TOPIC_CLUSTERS.items():
                    if topic.lower() in [t.lower() for t in topics]:
                        topic_cluster = cluster
                        break
            
            # Get entities, defaulting to empty list
            entities = item.get("entities", [])
            if isinstance(entities, str):
                # Sometimes entities might come as comma-separated string
                entities = [e.strip() for e in entities.split(",") if e.strip()]
            
            item["metadata"] = {
                "topic": item.get("topic", ""),
                "topic_cluster": topic_cluster,
                "entities": entities,
                "sentiment": item.get("sentiment", "neutral"),
                "token_estimate": get_token_estimate(item.get("answer", "")),
                "generated_at": timestamp,
                "generation_method": generation_method,
                "batch": batch_num
            }
        
        elif item_type == "statement":
            # Find topic cluster
            topic_cluster = "other"
            if "topic" in item:
                topic = item.get("topic", "")
                for cluster, topics in TOPIC_CLUSTERS.items():
                    if topic.lower() in [t.lower() for t in topics]:
                        topic_cluster = cluster
                        break
            
            item["metadata"] = {
                "topic": item.get("topic", ""),
                "topic_cluster": topic_cluster,
                "sentiment": item.get("sentiment", "neutral"),
                "token_estimate": get_token_estimate(item.get("statement", "")),
                "generated_at": timestamp,
                "generation_method": generation_method,
                "batch": batch_num
            }
        
        elif item_type == "conversation":
            item["metadata"] = {
                "topic": item.get("topic", ""),
                "turns": len(item.get("messages", [])),
                "entities": [],
                "generated_at": timestamp,
                "generation_method": generation_method,
                "batch": batch_num
            }
            
            # Add metadata to each message
            for msg in item.get("messages", []):
                if isinstance(msg, dict):
                    msg["metadata"] = {
                        "topic": item.get("topic", ""),
                        "token_estimate": get_token_estimate(msg.get("content", "")),
                        "generated_at": timestamp
                    }
    
    return items

def create_batch_prompt(qa_count, conv_count, stmt_count, use_real_data=True):
    """Create a prompt for batch generation that works with the Batch API"""
    # Get real Jon examples if requested
    real_examples = ""
    if use_real_data:
        real_examples = extract_jon_messages()
        if real_examples:
            real_examples = "\nReal Jon examples:\n" + "\n".join(real_examples[:3])  # Limit to 3 examples
    
    # Create a clear, structured prompt for batch processing
    prompt = f"""You are Jon. Generate the following data in JSON format:
- {qa_count} Q&A pairs
- {conv_count} conversations
- {stmt_count} statements

{real_examples}

Generate the data in this exact JSON format:
{{
    "qa_pairs": [
        {{
            "question": "user question here",
            "answer": "jon's answer here",
            "metadata": {{
                "topic": "topic here",
                "entities": ["entity1", "entity2"],
                "sentiment": "positive/negative/neutral"
            }}
        }}
    ],
    "conversations": [
        {{
            "messages": [
                {{"role": "user", "content": "user message"}},
                {{"role": "assistant", "content": "jon's response"}}
            ],
            "metadata": {{
                "topic": "topic here",
                "mood": "mood here",
                "entities": ["entity1", "entity2"]
            }}
        }}
    ],
    "statements": [
        {{
            "statement": "jon's statement here",
            "metadata": {{
                "topic": "topic here",
                "entities": ["entity1", "entity2"],
                "sentiment": "positive/negative/neutral"
            }}
        }}
    ]
}}

Important:
1. Each response must be valid JSON
2. Include metadata for each item
3. Maintain Jon's casual, lowercase style with minimal punctuation
4. Use topics from: {', '.join(random.sample(TOPICS, min(3, len(TOPICS))))}
5. Include entities from: {', '.join(random.sample([item for sublist in ENTITIES.values() for item in sublist], 3))}
6. Keep responses concise and authentic to Jon's voice
7. Ensure all required fields are present in the JSON structure"""
    
    return prompt

def generate_one_shot_bulk(qa_count, conv_count, stmt_count, batch_num=0, use_real_data=True):
    """Generate data using one-shot bulk generation method
    
    This function is used as a fallback when the Batch API is not available.
    It sends a single prompt requesting multiple items to be generated at once.
    
    Args:
        qa_count: Number of QA pairs to generate
        conv_count: Number of conversations to generate  
        stmt_count: Number of statements to generate
        batch_num: Batch number for tracking
        use_real_data: Whether to include real Jon examples in the prompt
        
    Returns:
        Dictionary containing the generated data
    """
    track_api_call("one_shot_bulk_gen_start")
    
    # Create the prompt
    prompt = create_batch_prompt(
        qa_count=qa_count,
        conv_count=conv_count,
        stmt_count=stmt_count,
        use_real_data=use_real_data
    )
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=4000
        )
        
        # Process the response
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Extract and validate data
        qa_data = result_json.get("qa_pairs", [])
        statement_data = result_json.get("statements", [])
        conversation_data = result_json.get("conversations", [])
        
        # Add metadata
        qa_data = add_batch_metadata(qa_data, "qa", batch_num, "one_shot_bulk")
        statement_data = add_batch_metadata(statement_data, "statement", batch_num, "one_shot_bulk")
        conversation_data = add_batch_metadata(conversation_data, "conversation", batch_num, "one_shot_bulk")
        
        # Track API usage
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
        total_tokens = prompt_tokens + completion_tokens
        
        track_api_call("one_shot_bulk_gen", total_tokens, 
                       batch_size=qa_count + conv_count + stmt_count)
        
        print(f"Generated {len(qa_data)} QA pairs, {len(conversation_data)} conversations, and {len(statement_data)} statements")
        
        return {
            "qa_data": qa_data,
            "statement_data": statement_data,
            "conversation_data": conversation_data
        }
        
    except Exception as e:
        print(f"Error in one-shot bulk generation: {e}")
        # Return empty data as a fallback
        return {
            "qa_data": [],
            "statement_data": [],
            "conversation_data": []
        }

def generate_with_batch_api(qa_pairs, conversations, statements, batch_size=15, max_concurrent=8, use_real_data=True):
    """Generate Jon data using OpenAI's API with batching for efficiency"""
    # Calculate batch sizes for each data type
    qa_per_batch = min(15, batch_size)
    conv_per_batch = min(5, max(1, batch_size // 3))
    stmt_per_batch = min(8, max(1, batch_size // 2))
    
    # Calculate number of batches
    qa_batches = (qa_pairs + qa_per_batch - 1) // qa_per_batch
    conv_batches = (conversations + conv_per_batch - 1) // conv_per_batch
    stmt_batches = (statements + stmt_per_batch - 1) // stmt_per_batch
    
    total_batches = max(qa_batches, conv_batches, stmt_batches)
    
    print(f"\nStarting Batch API Generation:")
    print(f"Total batches: {total_batches}")
    print(f"Batch sizes: {qa_per_batch} QA pairs, {conv_per_batch} conversations, {stmt_per_batch} statements")
    
    # Initialize data containers
    qa_data = []
    conversation_data = []
    statement_data = []
    
    # Track failed attempts
    failed_attempts = {
        "qa": 0,
        "conversations": 0,
        "statements": 0
    }
    max_failed_attempts = 3
    
    # Generate data sequentially
    try:
        # Generate QA pairs
        if qa_pairs > 0 and failed_attempts["qa"] < max_failed_attempts:
            print(f"\nGenerating {qa_pairs} Q&A pairs...")
            qa_data = generate_bulk_qa_pairs(qa_pairs, batch_size=qa_per_batch)
            if not qa_data:
                print("Warning: No QA pairs were generated")
                failed_attempts["qa"] += 1
            else:
                failed_attempts["qa"] = 0
        
        # Generate conversations
        if conversations > 0 and failed_attempts["conversations"] < max_failed_attempts:
            print(f"\nGenerating {conversations} conversations...")
            try:
                conversation_data = generate_bulk_conversations(conversations)
                if not conversation_data:
                    print("Warning: No conversations were generated")
                    failed_attempts["conversations"] += 1
                else:
                    failed_attempts["conversations"] = 0
            except Exception as e:
                print(f"Error generating conversations: {e}")
                failed_attempts["conversations"] += 1
                conversation_data = []
        
        # Generate statements
        if statements > 0 and failed_attempts["statements"] < max_failed_attempts:
            print(f"\nGenerating {statements} statements...")
            try:
                statement_data = generate_bulk_statements(statements)
                if not statement_data:
                    print("Warning: No statements were generated")
                    failed_attempts["statements"] += 1
                else:
                    failed_attempts["statements"] = 0
            except Exception as e:
                print(f"Error generating statements: {e}")
                failed_attempts["statements"] += 1
                statement_data = []
        
    except Exception as e:
        print(f"Error during generation: {e}")
        # Return whatever data we managed to generate
        return qa_data, conversation_data, statement_data
    
    # Print final status
    print("\nGeneration Status:")
    print(f"QA Pairs: {len(qa_data)}/{qa_pairs}")
    print(f"Conversations: {len(conversation_data)}/{conversations}")
    print(f"Statements: {len(statement_data)}/{statements}")
    
    return qa_data, conversation_data, statement_data

def verify_data_output(qa_data, conversation_data, statement_data, output_files):
    """Verify the output files contain the expected data"""
    print("\nVerifying output files...")
    
    # Check raw data
    raw_file = output_files.get("raw_file")
    if raw_file and os.path.exists(raw_file):
        try:
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
                qa_count = len(raw_data.get("qa_data", []))
                conv_count = len(raw_data.get("conversation_data", []))
                stmt_count = len(raw_data.get("statement_data", []))
                
                if qa_count == len(qa_data) and conv_count == len(conversation_data) and stmt_count == len(statement_data):
                    print(f"✅ Raw data verified: {qa_count} QA pairs, {conv_count} conversations, {stmt_count} statements")
                else:
                    print(f"❌ Raw data verification failed: counts don't match")
        except Exception as e:
            print(f"❌ Raw data verification failed: {e}")
    else:
        print(f"❌ Raw data file not found or empty")
    
    # Check retrieval data
    retrieval_file = output_files.get("retrieval_file")
    if retrieval_file and os.path.exists(retrieval_file):
        try:
            with open(retrieval_file, 'r') as f:
                lines = f.readlines()
                retrieval_count = len(lines)
                expected_count = len(qa_data) + len(statement_data)
                
                if retrieval_count >= expected_count:
                    print(f"✅ Retrieval data verified: {retrieval_count} items")
                else:
                    print(f"❌ Retrieval data verification failed: {retrieval_count} items found, expected at least {expected_count}")
        except Exception as e:
            print(f"❌ Retrieval data verification failed: {e}")
    else:
        print(f"❌ Retrieval file not found or empty")
    
    # Check fine-tuning data
    fine_tuning_file = output_files.get("fine_tuning_file")
    if fine_tuning_file and os.path.exists(fine_tuning_file):
        try:
            with open(fine_tuning_file, 'r') as f:
                lines = f.readlines()
                ft_count = len(lines)
                
                if ft_count == len(conversation_data):
                    print(f"✅ Fine-tuning data verified: {ft_count} conversations")
                else:
                    print(f"❌ Fine-tuning data verification failed: {ft_count} conversations found, expected {len(conversation_data)}")
        except Exception as e:
            print(f"❌ Fine-tuning data verification failed: {e}")
    else:
        print(f"❌ Fine-tuning file not found or empty")

def save_data(qa_data, conversation_data, statement_data, output_dir=None, timestamp=None, verify=True):
    """Save generated data in appropriate formats with validation and quality checks"""
    if output_dir is None:
        output_dir = "data_generation/output"
    os.makedirs(output_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare file paths
    raw_file = os.path.join(output_dir, f"jon_raw_data_{timestamp}.json")
    retrieval_file = os.path.join(output_dir, f"jon_retrieval_data_{timestamp}.jsonl")
    fine_tuning_file = os.path.join(output_dir, f"jon_fine_tuning_data_{timestamp}.jsonl")
    
    # Validate data before saving
    validation_results = {
        "qa_data": {"valid": 0, "invalid": 0, "errors": []},
        "conversation_data": {"valid": 0, "invalid": 0, "errors": []},
        "statement_data": {"valid": 0, "invalid": 0, "errors": []}
    }
    
    # Validate QA data
    for qa in qa_data:
        try:
            if not isinstance(qa, dict):
                raise ValueError("Not a dictionary")
            if "question" not in qa or "answer" not in qa:
                raise ValueError("Missing required fields")
            if len(qa["answer"]) < 20:
                raise ValueError("Answer too short")
            if "metadata" not in qa:
                raise ValueError("Missing metadata")
            validation_results["qa_data"]["valid"] += 1
        except Exception as e:
            validation_results["qa_data"]["invalid"] += 1
            validation_results["qa_data"]["errors"].append(str(e))
    
    # Validate conversation data
    for conv in conversation_data:
        try:
            if not isinstance(conv, dict):
                raise ValueError("Not a dictionary")
            if "messages" not in conv:
                raise ValueError("Missing messages")
            if len(conv["messages"]) < 2:
                raise ValueError("Too few messages")
            validation_results["conversation_data"]["valid"] += 1
        except Exception as e:
            validation_results["conversation_data"]["invalid"] += 1
            validation_results["conversation_data"]["errors"].append(str(e))
    
    # Validate statement data
    for stmt in statement_data:
        try:
            if not isinstance(stmt, dict):
                raise ValueError("Not a dictionary")
            if "statement" not in stmt:
                raise ValueError("Missing statement")
            if len(stmt["statement"]) < 10:
                raise ValueError("Statement too short")
            validation_results["statement_data"]["valid"] += 1
        except Exception as e:
            validation_results["statement_data"]["invalid"] += 1
            validation_results["statement_data"]["errors"].append(str(e))
    
    # Print validation results
    print("\nData Validation Results:")
    for data_type, results in validation_results.items():
        total = results["valid"] + results["invalid"]
        if total > 0:
            success_rate = (results["valid"] / total) * 100
            print(f"\n{data_type}:")
            print(f"Valid: {results['valid']}/{total} ({success_rate:.1f}%)")
            if results["invalid"] > 0:
                print(f"Invalid: {results['invalid']}")
                print("Common errors:")
                error_counts = defaultdict(int)
                for error in results["errors"]:
                    error_counts[error] += 1
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"- {error}: {count} times")
    
    # Save raw data with validation results
    raw_data = {
        "version": "2.0",
        "timestamp": timestamp,
        "model": "gpt-4-turbo-preview",
        "qa_count": len(qa_data),
        "conversation_count": len(conversation_data),
        "statement_count": len(statement_data),
        "validation_results": validation_results,
        "qa_data": qa_data,
        "conversation_data": conversation_data,
        "statement_data": statement_data
    }
    
    try:
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
    except Exception as e:
        print(f"Error saving raw data: {e}")
        return None
    
    # Format and save retrieval data
    try:
        retrieval_data = format_for_retrieval_store(qa_data, "qa_pair")
        retrieval_data += "\n" + format_for_retrieval_store(statement_data, "statement")
        
        with open(retrieval_file, 'w') as f:
            f.write(retrieval_data)
    except Exception as e:
        print(f"Error saving retrieval data: {e}")
        return None
    
    # Format and save fine-tuning data
    try:
        fine_tuning_data = format_for_fine_tuning(conversation_data)
        
        with open(fine_tuning_file, 'w') as f:
            f.write(fine_tuning_data)
    except Exception as e:
        print(f"Error saving fine-tuning data: {e}")
        return None
    
    # Verify output files if requested
    if verify:
        verify_data_output(qa_data, conversation_data, statement_data, {
            "raw_file": raw_file,
            "retrieval_file": retrieval_file,
            "fine_tuning_file": fine_tuning_file
        })
    
    return {
        "raw_file": raw_file,
        "retrieval_file": retrieval_file,
        "fine_tuning_file": fine_tuning_file,
        "validation_results": validation_results
    }

def save_checkpoint(qa_data, conversation_data, statement_data, checkpoint_dir, batch_num=None):
    """Save a checkpoint of the generated data
    
    Args:
        qa_data: List of QA data items
        conversation_data: List of conversation data items
        statement_data: List of statement data items
        checkpoint_dir: Directory to save checkpoint in
        batch_num: Optional batch number to include in filename
        
    Returns:
        Path to the checkpoint file
    """
    # Create checkpoint directory if needed
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate timestamp for checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint filename
    batch_suffix = f"_batch{batch_num}" if batch_num is not None else ""
    checkpoint_filename = f"jon_checkpoint_{timestamp}{batch_suffix}_{len(qa_data)}qa_{len(conversation_data)}conv_{len(statement_data)}stmt.json"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    try:
        checkpoint_data = {
            'qa_data': qa_data,
            'conversation_data': conversation_data,
            'statement_data': statement_data,
            'timestamp': datetime.now().isoformat(),
            'api_calls': api_calls,
            'metadata': {
                'qa_count': len(qa_data),
                'conversation_count': len(conversation_data),
                'statement_count': len(statement_data)
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        print(f"\nCheckpoint saved to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return None
        
def load_checkpoint(checkpoint_path):
    """Load data from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing the checkpoint data or None if loading failed
    """
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            
        # Validate the checkpoint data
        if not isinstance(checkpoint_data, dict):
            print(f"Error: Checkpoint file does not contain a valid dictionary")
            return None
            
        required_keys = ['qa_data', 'conversation_data', 'statement_data']
        if not all(key in checkpoint_data for key in required_keys):
            print(f"Error: Checkpoint file is missing required keys: {', '.join(required_keys)}")
            return None
            
        # Update global api_calls counter if available
        if 'api_calls' in checkpoint_data:
            global api_calls
            api_calls.update(checkpoint_data['api_calls'])
            
        return checkpoint_data
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def get_memory_usage():
    """Get current memory usage in MB"""
    if HAVE_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    return 0

def log_memory(stage_name):
    """Log memory usage at a given stage of execution
    
    Args:
        stage_name: Name of the current processing stage
    """
    if not MEMORY_TRACKING or not HAVE_PSUTIL:
        return
        
    try:
        memory_usage = get_memory_usage()
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "stage": stage_name,
            "memory_mb": memory_usage,
            "memory_gb": memory_usage / 1024
        }
        
        # Print memory usage
        print(f"Memory usage at {stage_name}: {memory_usage:.2f} MB ({memory_usage/1024:.3f} GB)")
        
        # Log to file if memory_logs directory exists
        log_dir = os.path.join(OUTPUT_DIR, "memory_logs")
        if os.path.exists(log_dir):
            log_file = os.path.join(log_dir, f"memory_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging memory: {e}")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the given directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoint files
        
    Returns:
        Path to the latest checkpoint file or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "jon_checkpoint_*.json"))
    if not checkpoint_files:
        return None
        
    # Sort by modification time, newest first
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def generate_bulk_qa_pairs(count: int, batch_size: int = 15) -> List[Dict[str, Any]]:
    """Generate multiple Q&A pairs in batches
    
    Args:
        count: Number of Q&A pairs to generate
        batch_size: Number of pairs per batch
        
    Returns:
        List of Q&A pairs with metadata
    """
    qa_pairs = []
    failed_attempts = 0
    max_failed_attempts = 3
    consecutive_failures = 0
    max_consecutive_failures = 3
    quality_metrics = {
        "total_attempts": 0,
        "successful_pairs": 0,
        "failed_pairs": 0,
        "avg_length": 0,
        "topic_distribution": defaultdict(int),
        "entity_distribution": defaultdict(int)
    }
    
    # Calculate number of batches
    num_batches = (count + batch_size - 1) // batch_size
    
    # Create progress bar
    pbar = tqdm(total=count, desc="Generating Q&A pairs")
    
    for batch_num in range(num_batches):
        # Calculate batch size
        remaining = count - len(qa_pairs)
        current_batch_size = min(batch_size, remaining)
        
        # Skip if we've hit too many failures
        if failed_attempts >= max_failed_attempts or consecutive_failures >= max_consecutive_failures:
            print(f"\nStopping after {failed_attempts} failed attempts and {consecutive_failures} consecutive failures")
            break
            
        try:
            # Generate batch
            batch_pairs = []
            batch_metrics = {
                "successful": 0,
                "failed": 0,
                "total_length": 0
            }
            
            for i in range(current_batch_size):
                quality_metrics["total_attempts"] += 1
                try:
                    # Add exponential backoff for rate limiting
                    if consecutive_failures > 0:
                        backoff_time = min(2 ** consecutive_failures, 30)  # Cap at 30 seconds
                        pbar.set_description(f"Rate limit backoff: waiting {backoff_time}s...")
                        time.sleep(backoff_time)
                    
                    qa_pair = generate_qa_pair(
                        max_retries=2,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    if qa_pair and "question" in qa_pair and "answer" in qa_pair:
                        # Validate quality
                        if len(qa_pair["answer"]) < 20:
                            raise ValueError("Answer too short")
                        
                        # Update metrics
                        batch_metrics["successful"] += 1
                        batch_metrics["total_length"] += len(qa_pair["answer"])
                        
                        # Track topics and entities
                        for topic in qa_pair["metadata"]["topics"]:
                            quality_metrics["topic_distribution"][topic] += 1
                        for entity in qa_pair["metadata"]["entities"]:
                            quality_metrics["entity_distribution"][entity] += 1
                        
                        batch_pairs.append(qa_pair)
                        consecutive_failures = 0  # Reset on success
                    else:
                        raise ValueError("Invalid Q&A pair format")
                        
                except Exception as e:
                    print(f"\nError generating Q&A pair in batch {batch_num}, item {i}: {e}")
                    batch_metrics["failed"] += 1
                    consecutive_failures += 1
                    continue
            
            # Add successful pairs
            if batch_pairs:
                qa_pairs.extend(batch_pairs)
                failed_attempts = 0  # Reset counter on success
                consecutive_failures = 0  # Reset consecutive failures
                
                # Update progress
                pbar.update(len(batch_pairs))
                pbar.set_description(f"Generated {len(qa_pairs)}/{count} pairs")
            else:
                failed_attempts += 1
                consecutive_failures += 1
                print(f"\nBatch {batch_num} failed to generate any pairs")
            
            # Add small delay between batches to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"\nError in batch {batch_num}: {e}")
            failed_attempts += 1
            consecutive_failures += 1
            continue
    
    # Close progress bar
    pbar.close()
    
    # Calculate final metrics
    quality_metrics["successful_pairs"] = len(qa_pairs)
    quality_metrics["failed_pairs"] = quality_metrics["total_attempts"] - len(qa_pairs)
    
    if qa_pairs:
        total_length = sum(len(pair["answer"]) for pair in qa_pairs)
        quality_metrics["avg_length"] = total_length / len(qa_pairs)
    
    # Print final status with quality metrics
    print(f"\nQ&A Generation Complete:")
    print(f"Successfully generated: {len(qa_pairs)}/{count} pairs")
    print(f"Failed attempts: {failed_attempts}")
    print(f"Consecutive failures: {consecutive_failures}")
    print(f"\nQuality Metrics:")
    print(f"Total attempts: {quality_metrics['total_attempts']}")
    print(f"Success rate: {(quality_metrics['successful_pairs'] / quality_metrics['total_attempts'] * 100):.1f}%")
    print(f"Average answer length: {quality_metrics['avg_length']:.1f} characters")
    
    # Print top topics and entities
    print("\nTop Topics:")
    for topic, count in sorted(quality_metrics["topic_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {topic}: {count}")
    
    print("\nTop Entities:")
    for entity, count in sorted(quality_metrics["entity_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {entity}: {count}")
    
    return qa_pairs

def build_prompt(topic: Optional[str] = None, style: str = "casual", use_real_data: bool = True) -> str:
    """Build a prompt for generating a Q&A pair
    
    Args:
        topic: Optional topic to focus on
        style: Response style (casual, formal, etc.)
        use_real_data: Whether to use real Jon examples
        
    Returns:
        Prompt string for the API
    """
    # Get real Jon examples if requested
    real_examples = ""
    if use_real_data and JON_REAL_MESSAGES:
        # Use cached messages if available
        real_examples = "\nReal Jon examples:\n" + "\n".join(JON_REAL_MESSAGES[:2])  # Limit to 2 examples
    
    # Select topic and related topics for context
    if not topic:
        # Select a random topic cluster
        cluster = random.choice(list(TOPIC_CLUSTERS.keys()))
        topic = random.choice(TOPIC_CLUSTERS[cluster])
        related_topics = [t for t in TOPIC_CLUSTERS[cluster] if t != topic][:2]
    else:
        # Find related topics from the same cluster
        related_topics = []
        for cluster, topics in TOPIC_CLUSTERS.items():
            if topic in topics:
                related_topics = [t for t in topics if t != topic][:2]
                break
    
    # Select style elements based on topic and mood
    style_elements = []
    if style == "casual":
        style_elements = random.sample([e for e in JON_STYLE_ELEMENTS if "lowercase" in e or "minimal punctuation" in e], 2)
    else:
        style_elements = random.sample(JON_STYLE_ELEMENTS, 2)
    
    # Add Jon's persona for context
    persona = Config.PERSONA
    
    # Build the prompt with enhanced context
    prompt = f"""You are Jon. Generate a question and answer pair about {topic}.

{persona}

{real_examples}

Your response should:
1. Be in Jon's authentic voice
2. Include these style elements:
{' '.join([f"- {element}" for element in style_elements])}
3. Be formatted exactly as:
Q: [user's question]
A: [jon's answer]

Keep the response concise and natural. Make sure Jon's answer reflects his personality and writing style."""
    
    return prompt

def format_for_retrieval_store(data: List[Dict[str, Any]], data_type: str) -> str:
    """Format data for the retrieval store
    
    Args:
        data: List of data items (QA pairs or statements)
        data_type: Type of data ("qa_pair" or "statement")
        
    Returns:
        JSONL formatted string
    """
    formatted_data = []
    
    for item in data:
        if data_type == "qa_pair":
            # Format Q&A pair
            text = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
            metadata = item.get("metadata", {})
            metadata["type"] = "qa_pair"
            
            formatted_item = {
                "text": text,
                "metadata": metadata
            }
            
        elif data_type == "statement":
            # Format statement
            text = item.get("statement", "")
            metadata = item.get("metadata", {})
            metadata["type"] = "statement"
            
            formatted_item = {
                "text": text,
                "metadata": metadata
            }
        
        formatted_data.append(json.dumps(formatted_item))
    
    return "\n".join(formatted_data)

def format_for_fine_tuning(conversations: List[Dict[str, Any]]) -> str:
    """Format conversations for fine-tuning
    
    Args:
        conversations: List of conversation data
        
    Returns:
        JSONL formatted string for fine-tuning
    """
    formatted_data = []
    
    for conv in conversations:
        messages = conv.get("messages", [])
        if not messages:
            continue
            
        # Format each message pair as a training example
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i]
            assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
            
            if not assistant_msg:
                continue
                
            training_example = {
                "messages": [
                    {"role": "system", "content": Config.PERSONA},
                    {"role": "user", "content": user_msg.get("content", "")},
                    {"role": "assistant", "content": assistant_msg.get("content", "")}
                ]
            }
            
            formatted_data.append(json.dumps(training_example))
    
    return "\n".join(formatted_data)

def extract_entities(text: str) -> List[str]:
    """Extract entities from text using predefined entity lists"""
    entities = []
    text_lower = text.lower()
    
    # Check for each entity type
    for entity_type, entity_list in ENTITIES.items():
        for entity in entity_list:
            if entity.lower() in text_lower:
                entities.append(entity)
    
    return list(set(entities))

def extract_topics(text: str) -> List[str]:
    """Extract topics from text using predefined topic lists"""
    topics = []
    text_lower = text.lower()
    
    # Check each topic
    for topic in TOPICS:
        if topic.lower() in text_lower:
            topics.append(topic)
    
    return list(set(topics))

def generate_conversation(
    topic: Optional[str] = None,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Generate a single conversation with metadata.
    
    Args:
        topic: Optional topic to focus on
        style: Response style (default: casual)
        use_real_data: Whether to use real data examples
        max_retries: Maximum number of retries
        temperature: Response variability
        max_tokens: Maximum tokens in response
        client: Optional OpenAI client
        
    Returns:
        Dictionary containing the conversation and metadata
    """
    # Use global client if none provided
    if client is None:
        client = globals()["client"]
    
    # Build prompt with real data examples if enabled
    prompt = build_prompt(topic, style, use_real_data)
    
    # Track API usage
    track_api_call("conversation_gen_start")
    
    for attempt in range(max_retries + 1):
        try:
            # Generate response with exponential backoff
            if attempt > 0:
                backoff_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                print(f"Retry attempt {attempt}/{max_retries}, waiting {backoff_time} seconds...")
                time.sleep(backoff_time)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track successful API call
            track_api_call("conversation_gen", response.usage.total_tokens)
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract messages with improved regex
            message_pattern = r"(?:User|Assistant):\s*(.*?)(?=\n(?:User|Assistant):|$)"
            messages = re.findall(message_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not messages or len(messages) < 2:
                raise ValueError("Could not extract enough messages from response")
            
            # Format messages
            formatted_messages = []
            for i, msg in enumerate(messages):
                role = "user" if i % 2 == 0 else "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.strip()
                })
            
            # Extract entities and topics from all messages
            all_content = " ".join(msg["content"] for msg in formatted_messages)
            entities = extract_entities(all_content)
            topics = extract_topics(all_content)
            
            # Generate metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4-turbo-preview",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "style": style,
                "entities": entities,
                "topics": topics,
                "version": "2.0",
                "generator": "jon_data_generator",
                "attempt": attempt + 1
            }
            
            return {
                "messages": formatted_messages,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error generating conversation (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt == max_retries:
                raise
            continue
    
    raise Exception(f"Failed to generate conversation after {max_retries + 1} attempts")

def main(args=None):
    """Main entry point for the data generator"""
    global CHECKPOINT_FREQUENCY, OUTPUT_DIR
    
    # Parse command line arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Generate synthetic Jon data")
        parser.add_argument("--qa-pairs", type=int, default=0, help="Number of Q&A pairs to generate")
        parser.add_argument("--conversations", type=int, default=0, help="Number of conversations to generate")
        parser.add_argument("--statements", type=int, default=0, help="Number of statements to generate")
        parser.add_argument("--batch-size", type=int, default=15, help="Batch size for API calls")
        parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent API calls")
        parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
        parser.add_argument("--checkpoint-frequency", type=int, default=CHECKPOINT_FREQUENCY, help="Save checkpoint every N items")
        parser.add_argument("--use-real-data", action="store_true", help="Use real Jon data for examples")
        parser.add_argument("--one-shot", action="store_true", help="Use one-shot generation instead of batch")
        parser.add_argument("--verify", action="store_true", help="Verify output data")
        parser.add_argument("--dry-run", action="store_true", help="Skip retrieval store functionality")
        args = parser.parse_args()
    
    # Update global settings
    CHECKPOINT_FREQUENCY = args.checkpoint_frequency
    OUTPUT_DIR = args.output_dir
    
    # Print configuration
    print("\n" + "="*60)
    print("             Jon Data Generation Configuration              ".center(60))
    print("="*60)
    print(f"Generation mode: {'Batch API' if not args.one_shot else 'Sequential'}")
    print(f"Target quantities: {args.qa_pairs} QA pairs, {args.conversations} conversations, {args.statements} statements")
    print(f"Using real Jon data: {'Yes' if args.use_real_data else 'No'}")
    print(f"Checkpointing every {CHECKPOINT_FREQUENCY} items")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Generate data
    if args.one_shot:
        # Use one-shot bulk generation
        data = generate_one_shot_bulk(
            qa_count=args.qa_pairs,
            conv_count=args.conversations,
            stmt_count=args.statements,
            use_real_data=args.use_real_data
        )
        qa_data = data["qa_data"]
        conversation_data = data["conversation_data"]
        statement_data = data["statement_data"]
    else:
        # Use batch API
        qa_data, conversation_data, statement_data = generate_with_batch_api(
            qa_pairs=args.qa_pairs,
            conversations=args.conversations,
            statements=args.statements,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            use_real_data=args.use_real_data
        )
    
    # Save data
    output_files = save_data(
        qa_data=qa_data,
        conversation_data=conversation_data,
        statement_data=statement_data,
        output_dir=OUTPUT_DIR,
        verify=args.verify
    )
    
    print("\nGeneration complete!")
    print(f"Raw data saved to: {output_files['raw_file']}")
    print(f"Retrieval data saved to: {output_files['retrieval_file']}")
    print(f"Fine-tuning data saved to: {output_files['fine_tuning_file']}")
    
    # Print API usage statistics
    print("\nAPI Usage Statistics:")
    print(f"Total API calls: {api_calls['total_calls']}")
    print(f"Total tokens used: {api_calls['total_tokens']}")
    print(f"Estimated cost: ${api_calls['total_cost']:.2f}")
    print(f"Batched calls: {api_calls['batched_calls']}")
    print(f"Individual calls: {api_calls['individual_calls']}")
    if api_calls['errors']:
        print(f"\nErrors encountered: {len(api_calls['errors'])}")
        for error in api_calls['errors'][:5]:  # Show first 5 errors
            print(f"- {error}")

if __name__ == "__main__":
    main() 