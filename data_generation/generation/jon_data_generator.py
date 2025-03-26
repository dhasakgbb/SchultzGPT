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
    "sometimes uses the wrong version of a word such as 'knew' or 'new'"
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
        real_examples = "\nReal Jon examples:\n" + "\n".join(random.sample(JON_REAL_MESSAGES, min(3, len(JON_REAL_MESSAGES))))
    
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

{real_examples}

Generate a Q&A pair in this format:
Q: [user's question]
A: [Jon's response]

The response should be in Jon's authentic voice and style."""

    return prompt

def generate_conversation(
    topic: Optional[str] = None,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Generate a single conversation with metadata."""
    # Use global client if none provided
    if client is None:
        client = globals()["client"]
    
    # Build prompt with real data examples if enabled
    real_examples = ""
    if use_real_data and JON_REAL_MESSAGES:
        real_examples = "\nReal Jon examples:\n" + "\n".join(random.sample(JON_REAL_MESSAGES, min(3, len(JON_REAL_MESSAGES))))
    
    # Build topic-specific prompt
    topic_prompt = ""
    if topic:
        topic_prompt = f"Focus on the topic: {topic}\n"
    
    # Get random style elements
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    
    # Get random facts about Jon
    jon_facts = random.sample(JON_FACTS, min(3, len(JON_FACTS)))
    
    prompt = f"""You are Jon. Generate a conversation that reflects Jon's authentic voice and personality.

{Config.PERSONA}

{topic_prompt}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Relevant facts about Jon:
{' '.join([f"- {fact}" for fact in jon_facts])}

{real_examples}

Generate a conversation with 3-4 exchanges in this format:
User: [user's message]
Jon: [Jon's response]
User: [user's follow-up]
Jon: [Jon's response]
...

The responses should be in Jon's authentic voice and style."""

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
            
            # Extract messages
            messages = []
            user_pattern = r"User:\s*(.*?)(?=\nJon:|$)"
            jon_pattern = r"Jon:\s*(.*?)(?=\nUser:|$)"
            
            user_matches = re.finditer(user_pattern, content, re.DOTALL | re.IGNORECASE)
            jon_matches = re.finditer(jon_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for user_match, jon_match in zip(user_matches, jon_matches):
                messages.append({"role": "user", "content": user_match.group(1).strip()})
                messages.append({"role": "assistant", "content": jon_match.group(1).strip()})
            
            if not messages:
                raise ValueError("Could not extract conversation from response")
            
            # Validate response quality
            if len(messages) < 2:
                raise ValueError("Conversation too short")
            
            if len(messages) > 8:  # Cap at 4 exchanges
                messages = messages[:8]
            
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
                "style": style,
                "entities": entities,
                "topics": topics,
                "version": "2.0",
                "generator": "jon_data_generator",
                "attempt": attempt + 1
            }
            
            return {
                "messages": messages,
                "metadata": metadata
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for authentication errors
            if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
                print("Authentication error detected. Please check your API key.")
                raise
            
            # Check for rate limit errors
            if "429" in error_msg or "rate limit" in error_msg:
                if attempt < max_retries:
                    backoff_time = min(2 ** (attempt + 1), 60)  # Cap at 60 seconds for rate limits
                    print(f"Rate limit hit. Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded after all retries")
            
            print(f"Error generating conversation (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt == max_retries:
                raise
            continue
    
    raise Exception(f"Failed to generate conversation after {max_retries + 1} attempts")

def generate_statement(
    topic: Optional[str] = None,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Generate a single statement with metadata."""
    # Use global client if none provided
    if client is None:
        client = globals()["client"]
    
    # Build prompt with real data examples if enabled
    real_examples = ""
    if use_real_data and JON_REAL_MESSAGES:
        real_examples = "\nReal Jon examples:\n" + "\n".join(random.sample(JON_REAL_MESSAGES, min(3, len(JON_REAL_MESSAGES))))
    
    # Build topic-specific prompt
    topic_prompt = ""
    if topic:
        topic_prompt = f"Focus on the topic: {topic}\n"
    
    # Get random style elements
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    
    # Get random facts about Jon
    jon_facts = random.sample(JON_FACTS, min(3, len(JON_FACTS)))
    
    prompt = f"""You are Jon. Generate a statement that reflects Jon's authentic voice and personality.

{Config.PERSONA}

{topic_prompt}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Relevant facts about Jon:
{' '.join([f"- {fact}" for fact in jon_facts])}

{real_examples}

Generate a statement in Jon's authentic voice and style."""

    # Track API usage
    track_api_call("statement_gen_start")
    
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
            track_api_call("statement_gen", response.usage.total_tokens)
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Validate response quality
            if len(content) < 20:
                raise ValueError("Statement too short")
            
            if len(content) > max_tokens * 4:  # Rough estimate
                raise ValueError("Statement too long")
            
            # Extract entities and topics
            entities = extract_entities(content)
            topics = extract_topics(content)
            
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
                "statement": content,
                "metadata": metadata
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for authentication errors
            if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
                print("Authentication error detected. Please check your API key.")
                raise
            
            # Check for rate limit errors
            if "429" in error_msg or "rate limit" in error_msg:
                if attempt < max_retries:
                    backoff_time = min(2 ** (attempt + 1), 60)  # Cap at 60 seconds for rate limits
                    print(f"Rate limit hit. Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded after all retries")
            
            print(f"Error generating statement (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt == max_retries:
                raise
            continue
    
    raise Exception(f"Failed to generate statement after {max_retries + 1} attempts")

def generate_qa_pair(
    topic: Optional[str] = None,
    style: str = "casual",
    use_real_data: bool = True,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Generate a single Q&A pair with metadata."""
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
            error_msg = str(e).lower()
            
            # Check for authentication errors
            if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
                print("Authentication error detected. Please check your API key.")
                raise
            
            # Check for rate limit errors
            if "429" in error_msg or "rate limit" in error_msg:
                if attempt < max_retries:
                    backoff_time = min(2 ** (attempt + 1), 60)  # Cap at 60 seconds for rate limits
                    print(f"Rate limit hit. Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded after all retries")
            
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
        client = globals()["client"]
    
    variations = []
    failed_attempts = 0
    
    while len(variations) < num_variations and failed_attempts < max_retries:
        try:
            # Generate variation
            variation = generate_qa_pair(
                topic=base_qa["metadata"]["topics"][0],  # Use first topic
                style=style,
                use_real_data=use_real_data,
                max_retries=2,  # Use fewer retries for variations
                temperature=temperature,
                max_tokens=max_tokens,
                client=client
            )
            
            # Add variation metadata
            variation["metadata"]["variation_of"] = base_qa["question"]
            variation["metadata"]["variation_index"] = len(variations)
            variation["metadata"]["version"] = "2.0"
            variation["metadata"]["generator"] = "jon_data_generator"
            
            variations.append(variation)
            failed_attempts = 0  # Reset on success
            
        except Exception as e:
            print(f"Error generating variation {len(variations)}: {e}")
            failed_attempts += 1
            if failed_attempts >= max_retries:
                print(f"Failed to generate variations after {max_retries} attempts")
                break
            time.sleep(2 ** failed_attempts)  # Exponential backoff
    
    return variations

def generate_bulk_conversations(count: int, turns_per_conversation: int = 6) -> List[Dict[str, Any]]:
    """
    Generate multiple conversations in a single API call
    
    Args:
        count: Number of conversations to generate
        turns_per_conversation: Number of message exchanges per conversation
    """
    conversations = []
    failed_attempts = 0
    max_failed_attempts = 3
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    # Create progress bar
    pbar = tqdm(total=count, desc="Generating conversations")
    
    while len(conversations) < count:
        try:
            # Add exponential backoff for rate limiting
            if consecutive_failures > 0:
                backoff_time = min(2 ** consecutive_failures, 30)  # Cap at 30 seconds
                pbar.set_description(f"Rate limit backoff: waiting {backoff_time}s...")
                time.sleep(backoff_time)
            
            # Generate a single conversation
            conversation = generate_conversation(
                max_retries=2,
                temperature=0.7,
                max_tokens=1000
            )
            
            if conversation and "messages" in conversation and len(conversation["messages"]) >= 2:
                conversations.append(conversation)
                consecutive_failures = 0  # Reset on success
                pbar.update(1)
                pbar.set_description(f"Generated {len(conversations)}/{count} conversations")
            else:
                raise ValueError("Invalid conversation format")
                
        except Exception as e:
            print(f"\nError generating conversation: {e}")
            failed_attempts += 1
            consecutive_failures += 1
            
            if failed_attempts >= max_failed_attempts or consecutive_failures >= max_consecutive_failures:
                print(f"\nStopping after {failed_attempts} failed attempts and {consecutive_failures} consecutive failures")
                break
            
            continue
    
    # Close progress bar
    pbar.close()
    
    print(f"\nConversation Generation Complete:")
    print(f"Successfully generated: {len(conversations)}/{count} conversations")
    print(f"Failed attempts: {failed_attempts}")
    print(f"Consecutive failures: {consecutive_failures}")
    
    return conversations

def generate_bulk_statements(count: int, topic: str = None) -> List[Dict[str, Any]]:
    """
    Generate multiple Jon statements in a single API call
    
    Args:
        count: Number of statements to generate
        topic: Optional topic to focus on
    """
    statements = []
    failed_attempts = 0
    max_failed_attempts = 3
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    # Create progress bar
    pbar = tqdm(total=count, desc="Generating statements")
    
    while len(statements) < count:
        try:
            # Add exponential backoff for rate limiting
            if consecutive_failures > 0:
                backoff_time = min(2 ** consecutive_failures, 30)  # Cap at 30 seconds
                pbar.set_description(f"Rate limit backoff: waiting {backoff_time}s...")
                time.sleep(backoff_time)
            
            # Generate a single statement
            statement = generate_statement(
                topic=topic,
                max_retries=2,
                temperature=0.7,
                max_tokens=1000
            )
            
            if statement and "statement" in statement and len(statement["statement"]) >= 20:
                statements.append(statement)
                consecutive_failures = 0  # Reset on success
                pbar.update(1)
                pbar.set_description(f"Generated {len(statements)}/{count} statements")
            else:
                raise ValueError("Invalid statement format")
                
        except Exception as e:
            print(f"\nError generating statement: {e}")
            failed_attempts += 1
            consecutive_failures += 1
            
            if failed_attempts >= max_failed_attempts or consecutive_failures >= max_consecutive_failures:
                print(f"\nStopping after {failed_attempts} failed attempts and {consecutive_failures} consecutive failures")
                break
            
            continue
    
    # Close progress bar
    pbar.close()
    
    print(f"\nStatement Generation Complete:")
    print(f"Successfully generated: {len(statements)}/{count} statements")
    print(f"Failed attempts: {failed_attempts}")
    print(f"Consecutive failures: {consecutive_failures}")
    
    return statements

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
    
    # Track consecutive errors for backoff
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    def handle_api_error(e, batch_type, batch_num):
        nonlocal consecutive_errors
        error_msg = str(e).lower()
        
        # Check for authentication errors
        if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
            print(f"Authentication error in {batch_type} batch {batch_num}. Please check your API key.")
            raise
        
        # Check for rate limit errors
        if "429" in error_msg or "rate limit" in error_msg:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive rate limit errors in {batch_type} generation. Stopping.")
                return True
            
            backoff_time = min(2 ** consecutive_errors, 60)  # Cap at 60 seconds
            print(f"Rate limit hit in {batch_type} batch {batch_num}. Waiting {backoff_time} seconds...")
            time.sleep(backoff_time)
            return False
        
        # Check for quota errors
        if "quota" in error_msg or "insufficient" in error_msg:
            print(f"Quota exceeded in {batch_type} batch {batch_num}. Please check your API quota.")
            raise
        
        # Handle other errors
        consecutive_errors += 1
        if consecutive_errors >= max_consecutive_errors:
            print(f"Too many consecutive errors in {batch_type} generation. Stopping.")
            return True
        
        print(f"Error in {batch_type} batch {batch_num}: {e}")
        time.sleep(2 ** consecutive_errors)  # Exponential backoff
        return False
    
    # Generate QA pairs in batches
    if qa_pairs > 0:
        print(f"\nGenerating {qa_pairs} Q&A pairs...")
        for batch_start in range(0, qa_pairs, qa_per_batch):
            batch_size = min(qa_per_batch, qa_pairs - batch_start)
            batch_num = batch_start // qa_per_batch + 1
            
            # Create batch prompt
            topics = [random.choice(TOPICS) for _ in range(batch_size)]
            style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
            
            prompt = f"""Generate {batch_size} Q&A pairs about the following topics: {', '.join(topics)}

Jon's persona:
{Config.PERSONA}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Format your response as a JSON array of objects, each with:
- "question": The user's question
- "answer": Jon's response
- "topic": The topic addressed

Each answer should be in Jon's authentic voice and style.
Your response should be ONLY a valid JSON array."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=2000
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    batch_qa = result if isinstance(result, list) else result.get("qa_pairs", [])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response in QA batch {batch_num}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many JSON parsing errors. Stopping.")
                        break
                    time.sleep(2 ** consecutive_errors)
                    continue
                
                # Process and validate each QA pair
                for qa in batch_qa:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        # Add metadata
                        qa["metadata"] = {
                            "topic": qa.get("topic", ""),
                            "entities": extract_entities(qa["answer"]),
                            "topics": extract_topics(qa["answer"]),
                            "timestamp": datetime.now().isoformat(),
                            "model": "gpt-4-turbo-preview",
                            "version": "2.0",
                            "generator": "jon_data_generator"
                        }
                        qa_data.append(qa)
                
                track_api_call("qa_batch_gen", response.usage.total_tokens, batch_size)
                consecutive_errors = 0  # Reset on success
                
            except Exception as e:
                if handle_api_error(e, "QA", batch_num):
                    break
                continue
    
    # Generate conversations in batches
    if conversations > 0:
        print(f"\nGenerating {conversations} conversations...")
        for batch_start in range(0, conversations, conv_per_batch):
            batch_size = min(conv_per_batch, conversations - batch_start)
            batch_num = batch_start // conv_per_batch + 1
            
            # Create batch prompt
            topics = [random.choice(TOPICS) for _ in range(batch_size)]
            style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
            
            prompt = f"""Generate {batch_size} conversations between User and Jon about these topics: {', '.join(topics)}

Jon's persona:
{Config.PERSONA}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Format your response as a JSON array of conversation objects, each with:
- "messages": Array of message objects with "role" (user/assistant) and "content"
- "topic": The conversation topic

Each conversation should have 3-4 exchanges. Keep Jon's responses authentic to his voice.
Your response should be ONLY a valid JSON array."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=2000
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    batch_convs = result if isinstance(result, list) else result.get("conversations", [])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response in conversation batch {batch_num}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many JSON parsing errors. Stopping.")
                        break
                    time.sleep(2 ** consecutive_errors)
                    continue
                
                # Process and validate each conversation
                for conv in batch_convs:
                    if isinstance(conv, dict) and "messages" in conv and len(conv["messages"]) >= 2:
                        # Add metadata
                        conv["metadata"] = {
                            "topic": conv.get("topic", ""),
                            "entities": extract_entities(" ".join(m["content"] for m in conv["messages"])),
                            "topics": extract_topics(" ".join(m["content"] for m in conv["messages"])),
                            "timestamp": datetime.now().isoformat(),
                            "model": "gpt-4-turbo-preview",
                            "version": "2.0",
                            "generator": "jon_data_generator"
                        }
                        conversation_data.append(conv)
                
                track_api_call("conversation_batch_gen", response.usage.total_tokens, batch_size)
                consecutive_errors = 0  # Reset on success
                
            except Exception as e:
                if handle_api_error(e, "conversation", batch_num):
                    break
                continue
    
    # Generate statements in batches
    if statements > 0:
        print(f"\nGenerating {statements} statements...")
        for batch_start in range(0, statements, stmt_per_batch):
            batch_size = min(stmt_per_batch, statements - batch_start)
            batch_num = batch_start // stmt_per_batch + 1
            
            # Create batch prompt
            topics = [random.choice(TOPICS) for _ in range(batch_size)]
            style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
            
            prompt = f"""Generate {batch_size} standalone statements from Jon about these topics: {', '.join(topics)}

Jon's persona:
{Config.PERSONA}

Style elements to incorporate:
{' '.join([f"- {element}" for element in style_elements])}

Format your response as a JSON array of statement objects, each with:
- "statement": Jon's statement
- "topic": The topic addressed

Each statement should be 2-3 sentences in Jon's authentic voice.
Your response should be ONLY a valid JSON array."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=2000
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    batch_stmts = result if isinstance(result, list) else result.get("statements", [])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response in statement batch {batch_num}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many JSON parsing errors. Stopping.")
                        break
                    time.sleep(2 ** consecutive_errors)
                    continue
                
                # Process and validate each statement
                for stmt in batch_stmts:
                    if isinstance(stmt, dict) and "statement" in stmt and len(stmt["statement"]) >= 20:
                        # Add metadata
                        stmt["metadata"] = {
                            "topic": stmt.get("topic", ""),
                            "entities": extract_entities(stmt["statement"]),
                            "topics": extract_topics(stmt["statement"]),
                            "sentiment": "neutral",  # Default sentiment
                            "timestamp": datetime.now().isoformat(),
                            "model": "gpt-4-turbo-preview",
                            "version": "2.0",
                            "generator": "jon_data_generator"
                        }
                        statement_data.append(stmt)
                
                track_api_call("statement_batch_gen", response.usage.total_tokens, batch_size)
                consecutive_errors = 0  # Reset on success
                
            except Exception as e:
                if handle_api_error(e, "statement", batch_num):
                    break
                continue
    
    # Print generation status
    print("\nGeneration Status:")
    print(f"QA Pairs: {len(qa_data)}/{qa_pairs}")
    print(f"Conversations: {len(conversation_data)}/{conversations}")
    print(f"Statements: {len(statement_data)}/{statements}")
    
    return qa_data, conversation_data, statement_data

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
    
    try:
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
                
    except Exception as e:
        print(f"\nError in main: {e}")
        print("Stack trace:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 