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
import copy
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")  # Default model with environment variable override

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
    """
    Load real Jon conversations from the JSONL file.
    
    Returns:
        List of conversations from real Jon data
    """
    real_jon_data = []
    max_entries = int(os.getenv("MAX_REAL_DATA_ENTRIES", "1000"))  # Configurable limit
    
    try:
        if not os.path.exists(REAL_JON_DATA_PATH):
            console.print(f"[yellow]Warning: [bold]Real Jon data file not found[/bold] at {REAL_JON_DATA_PATH}[/yellow]")
            return real_jon_data
            
        jon_msg_count = 0
        entry_count = 0
        
        with open(REAL_JON_DATA_PATH, 'r') as f:
            for line in f:
                try:
                    if entry_count >= max_entries:
                        console.print(f"[yellow]Warning: Reached maximum limit of {max_entries} entries, using partial dataset[/yellow]")
                        break
                        
                    data = json.loads(line.strip())
                    
                    # Validate data structure
                    if not isinstance(data, dict):
                        continue
                        
                    # Count Jon messages
                    if "messages" in data:
                        message_count = sum(1 for msg in data["messages"] 
                                          if isinstance(msg, dict) and msg.get("role") == "assistant")
                        
                        if message_count > 0:
                            jon_msg_count += message_count
                            real_jon_data.append(data)
                            entry_count += 1
                            
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]Warning: Invalid JSON on line {entry_count+1}: {str(e)}[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[yellow]Warning: Error processing line {entry_count+1}: {str(e)}[/yellow]")
                    continue
        
        console.print(f"[green]Loaded [bold]{len(real_jon_data)}[/bold] conversation entries with [bold]{jon_msg_count}[/bold] Jon messages from {REAL_JON_DATA_PATH}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading real Jon data: {e}[/red]")
        traceback.print_exc()
    
    return real_jon_data

# Load the real Jon data
REAL_JON_DATA = load_real_jon_data()

# Extract Jon's messages from real data
def extract_jon_messages():
    """
    Extract representative Jon messages from the real data
    
    Returns:
        List of filtered Jon messages
    """
    if not REAL_JON_DATA:
        return []
    
    messages = []
    
    # Extract Jon's messages from the JSONL conversation format
    for item in REAL_JON_DATA:
        if isinstance(item, dict) and "messages" in item:
            for msg in item["messages"]:
                if (isinstance(msg, dict) and 
                    msg.get("role") == "assistant" and 
                    len(msg.get("content", "")) > 20):
                    messages.append(msg["content"])
    
    # Deduplicate and filter messages
    unique_messages = list(set(messages))
    
    # Filter for longer, more representative messages
    filtered_messages = [m for m in unique_messages if len(m) > 50 and len(m) < 500]
    
    # Limit to a reasonable number
    max_messages = 50
    result_messages = filtered_messages[:max_messages] if len(filtered_messages) > max_messages else filtered_messages
    
    console.print(f"[cyan]Extracted [bold]{len(result_messages)}[/bold] representative Jon messages from real data[/cyan]")
    return result_messages

# Extract Jon's real messages
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
    "Mixes up letters in words (certificate instead of certificate)",
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

# Psychological attachment patterns and defense mechanisms
ATTACHMENT_PATTERNS = {
    # Jon's anxious attachment patterns
    "anxious_triggers": [
        "Chelsea being distant",
        "Chelsea playing games without him",
        "missed calls or texts",
        "lack of validation",
        "uncertain relationship status",
        "wife's avoidant behaviors",
        "silence from important people",
        "feeling excluded",
        "perceived rejection"
    ],
    
    # How anxious attachment manifests for Jon
    "anxious_behaviors": [
        "excessive texting",
        "seeking constant reassurance",
        "reading too much into small interactions",
        "fear of abandonment expression",
        "overanalysis of Chelsea's behavior",
        "asking if he's 'too much'",
        "difficulty with personal space",
        "obsessing over relationship status",
        "preoccupation with partner's availability",
        "sensitivity to emotional cues",
        "difficulty being alone"
    ],
    
    # Jon's specific defense mechanisms
    "defense_mechanisms": [
        # Primary defenses
        {"name": "humor", "description": "Uses self-deprecating jokes to deflect from painful emotions", 
         "indicators": ["haha", "lol", "just kidding", "but that's just me being dumb", "classic Jon moment"]},
        
        {"name": "rationalization", "description": "Provides logical explanations for emotional reactions",
         "indicators": ["I guess it makes sense", "I know why I feel this way", "probably because of my childhood"]},
        
        {"name": "intellectualization", "description": "Uses therapy language to create emotional distance",
         "indicators": ["anxious attachment", "avoidant tendencies", "trauma response", "coping mechanism"]},
        
        {"name": "projection", "description": "Attributes own feelings to others",
         "indicators": ["I bet she's feeling", "she probably thinks", "everyone assumes"]},
        
        {"name": "avoidance", "description": "Changes topic when emotions become too intense",
         "indicators": ["anyway", "but that's enough about that", "sorry for the rant", "changing topics"]},
        
        {"name": "regression", "description": "Reverts to childlike behaviors when stressed",
         "indicators": ["I just want my mom", "I need someone to take care of me", "everything is too hard"]},
        
        {"name": "catastrophizing", "description": "Assumes worst-case scenarios",
         "indicators": ["everything is falling apart", "this is the end", "I'll never recover"]}
    ],
    
    # Sibling dynamics with Chris
    "sibling_dynamics": {
        "ambivalence": [
            "saying something critical about Chris but defending him moments later",
            "complaining about Chris's advice while still following it",
            "expressing jealousy of Chris's success while using him as a role model",
            "joking about Chris's flaws but showing clear admiration"
        ],
        "triggers": [
            "comparing self to Chris",
            "describing family gatherings",
            "discussing childhood memories",
            "talking about career success"
        ]
    },
    
    # Patterns of interpersonal dynamics in Jon's relationships
    "relationship_patterns": {
        "protest_behaviors": ["sulking", "withdrawing", "acting out", "threatening to leave", "emotional dumping"],
        "codependent_traits": ["excessive caretaking", "difficulty saying no", "living through others", "fear of abandonment"],
        "validation_seeking": ["fishing for compliments", "self-deprecation expecting denial", "sharing accomplishments"]
    }
}

# Emotional regulation strategies and triggers
EMOTIONAL_REGULATION = {
    "dysregulation_triggers": [
        "rejection", "abandonment fears", "relationship uncertainty", 
        "perceived criticism", "comparison to others", "failure"
    ],
    
    "coping_strategies": [
        {"strategy": "escapism", "methods": ["video games", "fantasy books", "TV", "food", "collecting"]},
        {"strategy": "support_seeking", "methods": ["texting friends", "calling mom", "therapy", "venting"]},
        {"strategy": "self_soothing", "methods": ["self-care", "beard grooming", "hot bath", "comfort food"]},
        {"strategy": "avoidance", "methods": ["procrastination", "sleep", "changing topic", "denial"]}
    ],
    
    "emotional_states": {
        "baseline": "mild anxiety with underlying insecurity",
        "escalation": ["irritability", "defensive responses", "emotional flooding", "catastrophizing"],
        "shutdown": ["emotional withdrawal", "numbing", "fatalistic thinking", "self-isolation"],
        "regulation": ["self-awareness", "therapy-informed responses", "requesting space", "self-validation"]
    }
}

# Add a new structure for Jon's recursive thought patterns after EMOTIONAL_REGULATION
RECURSIVE_THOUGHT_PATTERNS = {
    "spiral_triggers": [
        "chelsea",
        "relationship",
        "marriage",
        "therapy",
        "living situation",
        "career",
        "failure",
        "art",
        "weight",
        "self-improvement",
        "future",
        "money",
        "inadequacy",
        "comparison to others"
    ],
    "spiral_progressions": {
        "anxiety": [
            "initial_concern", 
            "what_if_questions", 
            "catastrophizing", 
            "self_blame",
            "learned_helplessness"
        ],
        "self_worth": [
            "mild_self_criticism", 
            "negative_comparison", 
            "harsh_self_judgment",
            "global_self_devaluation", 
            "identity_crisis"
        ],
        "relationship_doubt": [
            "minor_concern", 
            "questioning_partner_feelings", 
            "recalling_past_issues",
            "predicting_rejection", 
            "assuming_relationship_doomed"
        ],
        "career_anxiety": [
            "questioning_choices", 
            "comparing_to_peers", 
            "imagining_failure",
            "generalizing_incompetence", 
            "existential_career_crisis"
        ]
    },
    "linguistic_markers": {
        "initial_concern": [
            "I'm a little worried about", 
            "not sure if", 
            "kind of concerned", 
            "been thinking about"
        ],
        "what_if_questions": [
            "what if she doesn't", 
            "what if I can't", 
            "what if this never", 
            "what if things don't"
        ],
        "catastrophizing": [
            "everything will fall apart", 
            "it's all going to end", 
            "I'll never recover from this", 
            "my life is ruined"
        ],
        "self_blame": [
            "this is all my fault", 
            "I always mess things up", 
            "I should have known better", 
            "I'm the problem"
        ],
        "learned_helplessness": [
            "nothing I do matters", 
            "I can't change anything", 
            "why even try", 
            "I'm stuck like this forever"
        ],
        "mild_self_criticism": [
            "I'm not great at", 
            "I kind of suck at", 
            "I should be better at", 
            "I'm struggling with"
        ],
        "negative_comparison": [
            "everyone else can", 
            "normal people don't have this problem", 
            "Chris would never", 
            "Chelsea deserves someone who"
        ],
        "harsh_self_judgment": [
            "I'm such a failure", 
            "I'm pathetic", 
            "I'm a mess", 
            "I can't do anything right"
        ],
        "global_self_devaluation": [
            "I'm fundamentally broken", 
            "I'll always be like this", 
            "I'm just a burden on everyone", 
            "no one would miss me"
        ],
        "identity_crisis": [
            "I don't even know who I am anymore", 
            "what am I doing with my life", 
            "I've wasted my potential", 
            "I'm a shell of a person"
        ]
    },
    "interruption_phrases": [
        "sorry, I'm spiraling a bit",
        "ugh, listen to me ramble",
        "sorry, that got dark",
        "I should stop talking",
        "anyway, that's probably too much",
        "I'm being dramatic",
        "classic Jon spiral haha",
        "my therapist would have a field day with this",
        "I need to take a breath"
    ]
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
    "occasionally says critical things about his brother Chris, while clearly still looking up to him",
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
    Make a robust API call with exponential backoff and error handling.
    
    Args:
        api_func: Function to call that makes the API request
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        Result of the API call or None if all retries failed
    """
    retries = 0
    delay = initial_delay
    error_types = {
        "rate_limit": 0,  
        "timeout": 0,
        "server": 0,
        "connection": 0,
        "other": 0
    }
    
    while retries <= max_retries:
        try:
            result = api_func()
            if retries > 0:
                with api_calls_lock:
                    api_calls["retries"] += retries
                
                error_summary = ", ".join([f"{k}: {v}" for k, v in error_types.items() if v > 0])
                if error_summary:
                    console.print(f"[green]Succeeded after {retries} retries. Error types: {error_summary}[/green]")
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            retries += 1
            
            # Classify error
            if "rate limit" in error_str or "too many requests" in error_str:
                error_type = "rate_limit"
                # Rate limits need longer backoff
                delay = max(delay * 2, 10)
            elif "timeout" in error_str or "timed out" in error_str:
                error_type = "timeout"
                delay *= 1.5
            elif "server error" in error_str or "5" in error_str[:3]:
                error_type = "server"
                delay *= 2
            elif "connection" in error_str or "network" in error_str:
                error_type = "connection"
                delay *= 1.5
            else:
                error_type = "other"
                delay *= 1.5
                
            error_types[error_type] += 1
            
            # Track specific errors
            with api_calls_lock:
                api_calls["errors"].append({
                    "type": error_type,
                    "message": str(e),
                    "retry": retries
                })
            
            # Log the error
            if retries <= max_retries:
                console.print(f"[yellow]API call failed ({error_type}): {e}. Retry {retries}/{max_retries} after {delay:.1f}s delay...[/yellow]")
                sleep(delay)
            else:
                console.print(f"[red]API call failed after {max_retries} retries: {e}[/red]")
                break
    
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
            api_calls["batched_calls"] = api_calls.get("batched_calls", 0) + 1
        else:
            api_calls["individual_calls"] = api_calls.get("individual_calls", 0) + 1
            
        # Update total calls counter
        api_calls["total_calls"] = api_calls.get("total_calls", 0) + 1

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

def identify_psychological_patterns(text, character_stage="current"):
    """
    Identify psychological patterns in the text including defense mechanisms, 
    attachment style indicators, emotional states, coping strategies, and thought spirals.
    
    Args:
        text (str): The text to analyze
        character_stage (str): The character development stage to consider
        
    Returns:
        dict: Dictionary of identified patterns and scores
    """
    if not text:
        return {}
        
    # Validate character stage
    valid_stages = [stage["name"] for stage in CHARACTER_TIMELINE["stages"]] if "stages" in CHARACTER_TIMELINE else ["current"]
    if character_stage not in valid_stages:
        console.print(f"[yellow]Warning: Invalid character stage '{character_stage}', falling back to 'current'[/yellow]")
        character_stage = "current"
    
    # Get the character stage details once
    stage_details = {}
    for stage in CHARACTER_TIMELINE["stages"]:
        if stage["name"] == character_stage:
            stage_details = stage
            break
    
    patterns = {
        "defense_mechanisms": [],
        "attachment_indicators": [],
        "emotional_states": [],
        "coping_strategies": [],
        "sibling_dynamics": [],
        "thought_spirals": {
            "detected": False,
            "triggers": [],
            "spiral_type": None,
            "progression": [],
            "interruption": None
        },
        "character_stage": character_stage,
        "stage_details": {
            "identified_traits": [],
            "language_patterns": []
        }
    }
    
    text_lower = text.lower()
    
    # Identify defense mechanisms
    for defense in ATTACHMENT_PATTERNS["defense_mechanisms"]:
        # Handle dictionary structure of defense mechanisms
        if isinstance(defense, dict):
            defense_name = defense.get("name", "")
            indicators = defense.get("indicators", [])
            
            # Check if any indicators appear in the text
            if any(indicator.lower() in text_lower for indicator in indicators):
                patterns["defense_mechanisms"].append(defense_name)
        elif isinstance(defense, str):
            # Fallback if it's a string
            if defense.lower() in text_lower:
                patterns["defense_mechanisms"].append(defense)
    
    # Identify attachment indicators
    for indicator in ATTACHMENT_PATTERNS["anxious_behaviors"]:
        # Simple string check 
        if indicator.lower() in text_lower:
            patterns["attachment_indicators"].append(indicator)
    
    # Identify emotional state
    for state_type in EMOTIONAL_REGULATION["emotional_states"]:
        for state in EMOTIONAL_REGULATION["emotional_states"][state_type]:
            if isinstance(state, str) and state.lower() in text_lower:
                patterns["emotional_states"].append(state)
    
    # Identify coping strategies
    for strategy in EMOTIONAL_REGULATION["coping_strategies"]:
        # Handle dictionary structure of coping strategies
        if isinstance(strategy, dict):
            strategy_name = strategy.get("strategy", "")
            methods = strategy.get("methods", [])
            
            # Check if any methods appear in the text
            if any(method.lower() in text_lower for method in methods):
                patterns["coping_strategies"].append(strategy_name)
        elif isinstance(strategy, str):
            # Fallback if it's a string
            if strategy.lower() in text_lower:
                patterns["coping_strategies"].append(strategy)
    
    # Rest of function remains the same
    
    # Calculate a psychological authenticity score
    authenticity_score = sum([
        0.15 if patterns["defense_mechanisms"] else 0,
        0.15 if patterns["attachment_indicators"] else 0,
        0.15 if patterns["emotional_states"] else 0,
        0.15 if patterns["coping_strategies"] else 0,
        0.15 if patterns["sibling_dynamics"] else 0,
        0.15 if patterns["thought_spirals"]["detected"] else 0,
        0.05 if patterns["stage_details"]["identified_traits"] else 0,
        0.05 if patterns["stage_details"]["language_patterns"] else 0
    ])
    
    patterns["authenticity_score"] = min(authenticity_score, 1.0)
    
    return patterns

def enrich_metadata(item, item_type, metrics=None, character_stage="current"):
    """
    Enrich item with additional metadata.
    
    Args:
        item: Data item to enrich
        item_type: Type of item (qa, conversation, statement)
        metrics: Optional metrics dictionary
        character_stage: Jon's character development stage
        
    Returns:
        Enriched item
    """
    if not item:
        return item
    
    # Create a deep copy to avoid modifying the original
    item = copy.deepcopy(item)
    
    # Initialize metadata if not present
    if 'metadata' not in item:
        item['metadata'] = {}
    
    # Extract text from item based on type
    text = extract_text(item, item_type)
    
    # Add entities and topics if not already present
    if 'entities' not in item['metadata']:
        item['metadata']['entities'] = extract_entities(text)
    
    if 'topics' not in item['metadata']:
        item['metadata']['topics'] = extract_topics(text)
        
    # Add psychological patterns
    psychological_patterns = identify_psychological_patterns(text, character_stage)
    
    # Add basic psychological patterns
    item['metadata']['defense_mechanisms'] = psychological_patterns.get('defense_mechanisms', [])
    item['metadata']['attachment_indicators'] = psychological_patterns.get('attachment_indicators', [])
    item['metadata']['emotional_states'] = psychological_patterns.get('emotional_states', [])
    
    # Add more detailed psychological information
    item['metadata']['psychological_profile'] = {
        'character_stage': character_stage,
        'authenticity_score': psychological_patterns.get('authenticity_score', 0.0),
        'stage_traits': psychological_patterns.get('stage_details', {}).get('identified_traits', []),
        'language_patterns': psychological_patterns.get('stage_details', {}).get('language_patterns', []),
        'coping_strategies': psychological_patterns.get('coping_strategies', []),
        'defense_mechanisms': psychological_patterns.get('defense_mechanisms', []),
        'sibling_dynamics': psychological_patterns.get('sibling_dynamics', [])
    }
    
    # Add thought spiral information if detected
    thought_spirals = psychological_patterns.get('thought_spirals', {})
    if thought_spirals.get('detected', False):
        item['metadata']['psychological_profile']['thought_spiral'] = {
            'type': thought_spirals.get('spiral_type', 'unknown'),
            'triggers': thought_spirals.get('triggers', []),
            'progression': [p.get('stage') for p in thought_spirals.get('progression', [])],
            'self_interruption': thought_spirals.get('interruption') is not None
        }
    
    # Add metrics if provided
    if metrics:
        item['metadata']['metrics'] = metrics
    
    return item

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

def build_prompt(topic: Optional[str] = None, style: str = "casual", use_real_data: bool = True, 
                character_stage: str = "current") -> str:
    """
    Build a detailed prompt for generating Jon-like responses.
    
    Args:
        topic: Optional topic to focus on
        style: Style of response to generate (casual, reflective, etc.)
        use_real_data: Whether to include real Jon data examples
        character_stage: Jon's character development stage
        
    Returns:
        A detailed prompt for generating content
    """
    # Get character timeline information for this stage
    timeline_stage = next((stage for stage in CHARACTER_TIMELINE["stages"] if stage["name"] == character_stage), 
                         next((stage for stage in CHARACTER_TIMELINE["stages"] if stage["name"] == "current"), None))
    
    # Get developmental aspects for this stage
    developmental_aspects = {}
    for aspect in CHARACTER_TIMELINE["development_aspects"]:
        for progression in aspect["progression"]:
            if progression["stage"] == character_stage:
                developmental_aspects[aspect["aspect"]] = {
                    "level": progression["level"],
                    "description": progression["description"]
                }
    
    # Base prompt with core character info
    facts = "\n".join([f"- Jon {fact}" for fact in JON_FACTS])
    style_elements = "\n".join([f"- {element}" for element in JON_STYLE_ELEMENTS])
    
    # Specific instructions based on timeline stage
    stage_instructions = f"""
Jon is currently in the '{timeline_stage['description']}' stage of his character development.
- Dominant traits: {', '.join(timeline_stage['dominant_traits'])}
- Current self-perception: {timeline_stage['self_perception']}
- Typical language patterns: {', '.join(timeline_stage['language_patterns'])}
- Emotional baseline: {timeline_stage['emotional_baseline']}
- Defensiveness level: {timeline_stage['defensiveness_level']}
"""

    # Developmental aspects
    dev_instructions = "Developmental aspects:\n"
    for aspect, details in developmental_aspects.items():
        dev_instructions += f"- {aspect.replace('_', ' ').title()}: {details['level']} - {details['description']}\n"
    
    # Brother relationship (Chris)
    brother_relationship = """
Jon's relationship with his brother Chris:
- Jon occasionally makes critical or snarky comments about Chris but clearly looks up to him
- Shows signs of sibling rivalry and jealousy while also seeking Chris's approval
- May complain about Chris's advice while still following it
- Expresses admiration for Chris's life achievements while feeling inadequate in comparison
- Uses self-deprecating humor when comparing himself to Chris
"""

    # Prompt with all components
    prompt = f"""You are roleplaying as Jon, a mid-30s millennial with an anxious attachment style.

CHARACTER FACTS:
{facts}

WRITING STYLE:
{style_elements}

PSYCHOLOGICAL PROFILE:
- Jon has an anxious attachment style (fears abandonment, seeks validation)
- He uses humor and self-deprecation to mask deeper insecurities
- His primary defense mechanisms: humor, rationalization, and intellectualization
- He tends to spiral into negative thought patterns when stressed
- He shows codependent tendencies, particularly with Chelsea and his mother

CHARACTER DEVELOPMENT STAGE:
{stage_instructions}
{dev_instructions}

{brother_relationship}

CONVERSATION STYLE:
- Jon makes frequent spelling errors due to dyslexia
- He uses lowercase almost exclusively 
- He omits apostrophes in contractions (dont, cant, etc.)
- He adds "haha" or "lol" frequently, especially when feeling vulnerable
- His messages often have an anxious, seeking-reassurance quality
"""

    # Add topic if provided
    if topic:
        prompt += f"\nTOPIC GUIDANCE:\nFocus on discussing {topic}. Jon would likely bring up personal experiences related to this topic or ask for advice/opinions about it.\n"
    
    # Include real message examples if available
    if use_real_data and JON_REAL_MESSAGES:
        # Get 2-3 message examples
        samples = random.sample(JON_REAL_MESSAGES, min(3, len(JON_REAL_MESSAGES)))
        examples = "\n".join([f"- {msg[:150]}..." if len(msg) > 150 else f"- {msg}" for msg in samples])
        prompt += f"\nREAL MESSAGE EXAMPLES:\n{examples}\n"
    
    # Instruction for the specific task
    prompt += f"""
YOUR TASK:
Generate a natural-sounding {style} {topic if topic else ''} message as Jon would write it.
Keep the message to a reasonable length (3-8 sentences) and maintain Jon's authentic voice, complete with typos and informal style.
"""

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
    verify: bool = False,
    character_stage: str = "current"
) -> Dict[str, str]:
    """
    Save generated data in multiple formats.
    
    Args:
        qa_data: List of Q&A pairs
        conversation_data: List of conversations
        statement_data: List of statements
        output_dir: Directory to save files
        verify: Whether to verify the output data
        character_stage: Jon's character development stage
        
    Returns:
        Dictionary with paths to saved files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enrich metadata for all items
        enriched_qa_data = [enrich_metadata(item, "qa", character_stage=character_stage) for item in qa_data]
        enriched_conversation_data = [enrich_metadata(item, "conversation", character_stage=character_stage) for item in conversation_data]
        enriched_statement_data = [enrich_metadata(item, "statement", character_stage=character_stage) for item in statement_data]
        
        # Save raw data
        raw_data = {
            "qa_data": enriched_qa_data,
            "conversation_data": enriched_conversation_data,
            "statement_data": enriched_statement_data,
            "metadata": {
                "timestamp": timestamp,
                "version": "2.0",
                "generator": "jon_data_generator",
                "character_stage": character_stage
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
    Process a batch of generation tasks using the OpenAI API.
    
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
    
    # Increment batched_calls within the tracking lock
    with api_calls_lock:
        api_calls["batched_calls"] = api_calls.get("batched_calls", 0) + 1
    
    def batch_api_call():
        try:
            start_time = time.time()
            # Process batch using OpenAI API
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt} for prompt in prompts],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "text"}
            )
            
            elapsed_time = time.time() - start_time
            
            # Debug API response
            console.print(f"[blue]Debug: API response has {len(response.choices)} choices[/blue]")
            for i, choice in enumerate(response.choices):
                content = choice.message.content if hasattr(choice, 'message') and hasattr(choice.message, 'content') else "No content"
                console.print(f"[blue]Choice {i+1} content (first 100 chars): {content[:100]}...[/blue]")
            
            # Track token usage - safely extract token usage if available
            total_tokens = 0
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                total_tokens = response.usage.total_tokens
            
            track_api_call("batch_gen", total_tokens)
            console.print(f"  [green]✓[/green] Batch completed in [bold]{elapsed_time:.2f}s[/bold]. Used [bold]{total_tokens}[/bold] tokens.")
            
            # Return empty results for now for testing
            return []
            
        except Exception as e:
            console.print(f"[red]Batch API call failed: {str(e)}[/red]")
            traceback.print_exc()
            raise
    
    # Attempt the API call with robust error handling
    result = robust_api_call(batch_api_call, max_retries=max_retries)
    
    if result is None:
        console.print(f"[red]Failed to process batch after {max_retries} attempts[/red]")
        return [(task_type, None) for task_type in task_types]
    
    # Debug output - always on
    console.print(f"[cyan]Debug: process_batch returning {len(result)} results[/cyan]")
    for i, (task_type, item) in enumerate(result):
        if item is None:
            console.print(f"[yellow]  Item {i} ({task_type}): None[/yellow]")
        else:
            if task_type == "qa":
                console.print(f"[green]  Item {i} (QA): Question length: {len(item.get('question', ''))}, Answer length: {len(item.get('answer', ''))}[/green]")
            elif task_type == "conv":
                console.print(f"[green]  Item {i} (Conversation): {len(item.get('messages', []))} messages[/green]")
            elif task_type == "stmt":
                console.print(f"[green]  Item {i} (Statement): Length: {len(item.get('statement', ''))}[/green]")
    
    return result

# --- Now modify the main function to use batch processing ---
def main(args=None):
    """Main entry point for the data generator using parallel execution with the Responses API."""
    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Configure memory tracking if requested
    global MEMORY_TRACKING
    MEMORY_TRACKING = args.memory_tracking if hasattr(args, 'memory_tracking') else False
    
    # Print configuration summary
    print_config_table(args)
    
    # Start timing
    start_time = time.time()
    
    # Calculate total items to generate
    total_items = args.qa_pairs + args.conversations + args.statements

    # Calculate optimal batch size if not specified
    batch_size = args.batch_size if args.batch_size else calculate_optimal_batch_size("mixed")
    console.print(f"[cyan]Using batch size of {batch_size}[/cyan]")
    
    # Initialize data structures
    qa_data = []
    conversation_data = []
    statement_data = []
    checkpoint_data = {
        "qa_data": [],
        "conversation_data": [],
        "statement_data": []
    }
    
    # Check for existing checkpoints
    if args.checkpoint_frequency > 0 and os.path.exists(args.checkpoint_dir):
        checkpoint_data = load_checkpoint(args.checkpoint_dir)
        if checkpoint_data:
            console.print(f"[green]Loaded {len(checkpoint_data.get('qa_data', []))} QA pairs, "
                        f"{len(checkpoint_data.get('conversation_data', []))} conversations, and "
                        f"{len(checkpoint_data.get('statement_data', []))} statements from checkpoint[/green]")
            
            # Update data with checkpoint data
            qa_data = checkpoint_data.get("qa_data", [])
            conversation_data = checkpoint_data.get("conversation_data", [])
            statement_data = checkpoint_data.get("statement_data", [])
    
    # Calculate number of remaining items to generate
    qa_needed = max(0, args.qa_pairs - len(qa_data))
    conv_needed = max(0, args.conversations - len(conversation_data))
    stmt_needed = max(0, args.statements - len(statement_data))
    
    # Create batch prompts
    batch_prompts = []
    
    # Prepare prompts for QA pairs
    for _ in range(qa_needed):
        # Build QA prompt
        topic = args.topic if args.topic else random.choice(TOPICS) if random.random() < 0.7 else None
        prompt = build_prompt(
            topic=topic, 
            style=args.style if args.style else "casual",
            use_real_data=args.use_real_data,
            character_stage=args.character_stage
        )
        batch_prompts.append(("qa", prompt))
    
    # Prepare prompts for conversations
    for _ in range(conv_needed):
        # Build conversation prompt
        topic = args.topic if args.topic else random.choice(TOPICS) if random.random() < 0.7 else None
        include_spiral = not args.no_spirals if hasattr(args, 'no_spirals') else True
        prompt = build_conversation_prompt(
            topic=topic, 
            style=args.style if args.style else "casual",
            use_real_data=args.use_real_data,
            character_stage=args.character_stage,
            include_spiral=include_spiral
        )
        batch_prompts.append(("conv", prompt))
    
    # Prepare prompts for standalone statements
    for _ in range(stmt_needed):
        # Build statement prompt
        topic = args.topic if args.topic else random.choice(TOPICS) if random.random() < 0.7 else None
        mood = random.choice(MOODS)
        
        # Get persona description with fallback 
        persona_desc = ""
        try:
            if hasattr(Config, 'PERSONA'):
                persona_desc = Config.PERSONA
            else:
                persona_desc = """Jon is a mid-30s millennial living with his mom after marriage issues with Chelsea. He's an event coordinator at a retirement community, dyslexic, struggles with anxiety, and has codependent tendencies. He loves nerdy things like fantasy books, D&D, and comic books. He's socially awkward but has a good sense of humor."""
        except:
            persona_desc = """Jon is a mid-30s millennial living with his mom after marriage issues with Chelsea. He's an event coordinator at a retirement community, dyslexic, struggles with anxiety, and has codependent tendencies. He loves nerdy things like fantasy books, D&D, and comic books. He's socially awkward but has a good sense of humor."""
        
        # Select specific psychological patterns for statement
        defense_mechanism = random.choice(ATTACHMENT_PATTERNS["defense_mechanisms"])
        
        # Determine emotional context based on mood
        emotional_state = EMOTIONAL_REGULATION["emotional_states"]["baseline"]
        if mood in ["anxious", "depressed", "irritated", "mopey", "tired", "self-doubt"]:
            emotional_state = random.choice(EMOTIONAL_REGULATION["emotional_states"]["escalation"] + 
                                          EMOTIONAL_REGULATION["emotional_states"]["shutdown"])
        elif mood in ["motivated", "energetic", "confident"]:
            emotional_state = random.choice(EMOTIONAL_REGULATION["emotional_states"]["regulation"])
        
        # Choose a trigger if in negative state
        trigger = ""
        if emotional_state in EMOTIONAL_REGULATION["emotional_states"]["escalation"] + EMOTIONAL_REGULATION["emotional_states"]["shutdown"]:
            trigger = f"Triggered by: {random.choice(EMOTIONAL_REGULATION['dysregulation_triggers'])}"
        
        # Get character timeline information for the specified stage
        timeline_stage = next((stage for stage in CHARACTER_TIMELINE["stages"] if stage["name"] == args.character_stage), 
                            next(stage for stage in CHARACTER_TIMELINE["stages"] if stage["name"] == "current"))
        
        # Select a thematic element based on developmental stage
        thematic_idx = ["pre_separation", "initial_separation", "current", 
                     "future_possibility_reconciliation", "future_possibility_moving_on"].index(args.character_stage)
        
        thematic_elements = {}
        for theme, progression in CHARACTER_TIMELINE["thematic_evolution"].items():
            if thematic_idx < len(progression):
                thematic_elements[theme] = progression[thematic_idx]
        
        thematic_focus = random.choice(list(thematic_elements.keys()))
        thematic_state = thematic_elements.get(thematic_focus, "current state")
        
        # Create psychological profile for statement
        psych_profile = f"""
Psychological Context:
- Current Mood: {mood}
- Emotional State: {emotional_state}
- Defense Mechanism: {defense_mechanism['name']} - {defense_mechanism['description']}
- {trigger}
- Express attachment style: anxious attachment with fear of abandonment
"""

        # Create character development stage section
        character_profile = f"""
Character Development Stage: {timeline_stage['name']} - {timeline_stage['description']}
- Dominant Traits: {', '.join(random.sample(timeline_stage['dominant_traits'], min(2, len(timeline_stage['dominant_traits']))))}
- Self-Perception: {timeline_stage['self_perception']}
- Thematic Focus: {thematic_focus} ({thematic_state})
- Language Style: Uses phrases like "{random.choice(timeline_stage['language_patterns'])}"
"""
        
        prompt = f"""You are Jon. Generate a single standalone statement or thought from Jon's perspective.

{persona_desc}

{f"Focus on the topic: {topic}" if topic else ""}

{psych_profile}

{character_profile}

Your response should be just Jon's standalone statement without any formatting or headers.
Keep it in Jon's authentic voice with his characteristic spelling errors and style.
Incorporate the psychological elements specified above naturally in the statement.
Make sure the statement reflects Jon's character development stage."""
        batch_prompts.append(("stmt", prompt))
    
    # Shuffle all prompts to mix types
    random.shuffle(batch_prompts)
    
    # Run in batches
    console.print(f"[bold green]Generating {len(batch_prompts)} items: {qa_needed} QA pairs, {conv_needed} conversations, {stmt_needed} statements[/bold green]")
    
    # Debug the batch prompts
    for i, (task_type, prompt) in enumerate(batch_prompts):
        console.print(f"[cyan]Prompt {i+1} ({task_type}):[/cyan] {prompt[:100]}...")
    
    # Process in batches
    all_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        refresh_per_second=5
    ) as progress:
        task = progress.add_task("[bold blue]Generating data...", total=len(batch_prompts))
        
        # Process in batches
        for i in range(0, len(batch_prompts), batch_size):
            batch = batch_prompts[i:i + batch_size]
            
            # Process batch
            typo_severity = args.dyslexic_typo_severity if hasattr(args, 'dyslexic_typo_severity') else 0.4
            batch_results = process_batch(
                batch, 
                client, 
                temperature=args.temperature if hasattr(args, 'temperature') else 0.7,
                max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else 1000,
                typo_severity=typo_severity,
                max_retries=args.max_retries if hasattr(args, 'max_retries') else 3
            )
            
            # Filter out None results
            valid_results = [(task_type, result) for task_type, result in batch_results if result is not None]
            all_results.extend(valid_results)
            
            # Filter out None results
            valid_results = [(task_type, result) for task_type, result in batch_results if result is not None]
            console.print(f"[cyan]Debug: {len(valid_results)}/{len(batch_results)} valid results after filtering None[/cyan]")
            all_results.extend(valid_results)
            
            # Filter out None results
            valid_results = [(task_type, result) for task_type, result in batch_results if result is not None]
            console.print(f"[cyan]Debug: {len(valid_results)}/{len(batch_results)} valid results after filtering None[/cyan]")
            all_results.extend(valid_results)
            
            # Update progress
            progress.update(task, advance=len(batch))
            
            # Report memory usage
            if MEMORY_TRACKING and HAVE_PSUTIL:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                console.print(f"[dim]Memory usage: {memory_mb:.1f} MB[/dim]")
            
            # Save checkpoint if needed
            current_checkpoint_data = {
                "qa_data": qa_data + [r for t, r in all_results if t == "qa"],
                "conversation_data": conversation_data + [r for t, r in all_results if t == "conv"],
                "statement_data": statement_data + [r for t, r in all_results if t == "stmt"]
            }
            
            if args.checkpoint_frequency > 0 and (i + batch_size) % args.checkpoint_frequency == 0:
                save_checkpoint(current_checkpoint_data, args.checkpoint_dir)
                console.print(f"[green]Saved checkpoint with {len(current_checkpoint_data['qa_data'])} QA pairs, "
                            f"{len(current_checkpoint_data['conversation_data'])} conversations, "
                            f"{len(current_checkpoint_data['statement_data'])} statements[/green]")
    
    # Add new items to existing data
    for task_type, result in all_results:
        if task_type == "qa":
            qa_data.append(result)
        elif task_type == "conv":
            conversation_data.append(result)
        elif task_type == "stmt":
            statement_data.append(result)
    
    # Verify data if requested
    if args.verify:
        console.print("[bold]Verifying generated data...[/bold]")
        
        qa_metrics = analyze_data_quality(qa_data, "qa")
        conv_metrics = analyze_data_quality(conversation_data, "conversation")
        stmt_metrics = analyze_data_quality(statement_data, "statement")
        
        console.print(f"[green]QA pairs: {qa_metrics['valid_items']}/{len(qa_data)} valid[/green]")
        console.print(f"[green]Conversations: {conv_metrics['valid_items']}/{len(conversation_data)} valid[/green]")
        console.print(f"[green]Statements: {stmt_metrics['valid_items']}/{len(statement_data)} valid[/green]")
    
    # Save data if not dry run
    if not hasattr(args, 'dry_run') or not args.dry_run:
        console.print("[bold]Saving generated data...[/bold]")
        output_paths = save_data(
            qa_data, 
            conversation_data, 
            statement_data, 
            args.output_dir,
            verify=args.verify,
            character_stage=args.character_stage
        )
        
        for name, path in output_paths.items():
            console.print(f"[green]Saved {name} to {path}[/green]")
    
    # Print summary
    total_time = time.time() - start_time
    console.print(f"[bold green]Completed in {total_time:.1f} seconds[/bold green]")
    console.print(f"[bold]Generated {len(qa_data)} QA pairs, {len(conversation_data)} conversations, {len(statement_data)} statements[/bold]")
    
    # Print API usage
    console.print("\n[bold]API Usage Summary:[/bold]")
    console.print(f"Total API calls: {api_calls['total_calls']}")
    console.print(f"Batched API calls: {api_calls['batched_calls']}")
    console.print(f"Individual API calls: {api_calls['individual_calls']}")
    console.print(f"Retries: {api_calls['retries']}")
    console.print(f"Total tokens used: {api_calls['total_tokens']}")
    
    if api_calls['errors']:
        console.print(f"[yellow]Errors encountered: {len(api_calls['errors'])}[/yellow]")
    
    return qa_data, conversation_data, statement_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Jon Data Generator")
    
    # Set batch size manually or autocalculate
    parser.add_argument("-b", "--batch-size", type=int, 
                        help="Batch size for API calls (default: auto-calculated)")
    
    # Corpus composition arguments
    parser.add_argument("-q", "--qa-pairs", type=int, default=50,
                       help="Number of Q&A pairs to generate (default: 50)")
    parser.add_argument("-c", "--conversations", type=int, default=30,
                       help="Number of conversations to generate (default: 30)")
    parser.add_argument("-s", "--statements", type=int, default=40,
                       help="Number of standalone statements to generate (default: 40)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=1000,
                       help="Maximum tokens to generate (default: 1000)")
    parser.add_argument("--style", choices=["casual", "reflective", "anxious", "hopeful", "mopey"],
                       help="Style for generated content")
    parser.add_argument("--topic", help="Topic to focus on")
    parser.add_argument("--no-real-data", dest="use_real_data", action="store_false",
                       help="Don't use real Jon data as examples")
    parser.add_argument("--dyslexic-typo-severity", type=float, default=0.4,
                       help="Severity of dyslexic typos (0.0-1.0, default: 0.4)")
    parser.add_argument("--character-stage", choices=["pre_separation", "initial_separation", 
                                                    "current", "future_possibility_reconciliation", 
                                                    "future_possibility_moving_on"],
                        default="current", help="Jon's character development stage (default: current)")
    parser.add_argument("--no-spirals", action="store_true", 
                       help="Disable thought spiraling in conversation generation")
    
    # Output options
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR,
                       help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--checkpoint-dir",
                       help="Checkpoint directory (default: {output_dir}/checkpoints)")
    parser.add_argument("--checkpoint-frequency", type=int, default=CHECKPOINT_FREQUENCY,
                       help=f"How often to create checkpoints (default: {CHECKPOINT_FREQUENCY})")
    parser.add_argument("--no-checkpoints", action="store_true",
                       help="Disable checkpointing")
    parser.add_argument("--verify", action="store_true",
                       help="Verify generated data before saving")
    
    # Development and debug options
    parser.add_argument("--test", action="store_true",
                       help="Test mode - generate a minimal set of data")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for API calls (default: 3)")
    parser.add_argument("--memory-tracking", action="store_true", 
                       help="Enable memory usage tracking")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set test mode values
    if args.test:
        args.qa_pairs = 1
        args.conversations = 1
        args.statements = 1
    
    # Configure checkpoint directory
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    # Disable checkpointing if small data or explicitly requested
    if args.no_checkpoints or args.qa_pairs + args.conversations + args.statements < 10:
        args.checkpoint_frequency = 0
    
    return args

# Temporal evolution model of Jon's character
CHARACTER_TIMELINE = {
    "stages": [
        {
            "name": "pre_separation",
            "description": "Before separation from Chelsea, living together but struggling",
            "dominant_traits": ["trying to fix marriage", "in denial about severity", "hopeful for change", "walking on eggshells"],
            "language_patterns": ["we can work this out", "it's just a rough patch", "I'm sure she'll come around", "things will get better"],
            "self_perception": "struggling husband trying to save marriage",
            "defensiveness_level": "moderate",
            "emotional_baseline": "anxious hope"
        },
        {
            "name": "initial_separation",
            "description": "Just moved back with parents, raw emotional state",
            "dominant_traits": ["shock", "disbelief", "desperate", "bargaining", "crisis mode"],
            "language_patterns": ["this can't be happening", "I just need to show her", "once she sees I've changed", "I'm falling apart"],
            "self_perception": "rejected husband in crisis",
            "defensiveness_level": "high",
            "emotional_baseline": "panic and desperation"
        },
        {
            "name": "current",
            "description": "Living with mom, processing separation, starting therapy",
            "dominant_traits": ["mopey", "self-pitying", "introspective", "codependent", "developing awareness"],
            "language_patterns": ["I guess this is my life now", "trying to work on myself", "therapy is helping", "one day at a time"],
            "self_perception": "failed husband working on himself",
            "defensiveness_level": "fluctuating",
            "emotional_baseline": "resigned sadness with moments of hope"
        },
        {
            "name": "future_possibility_reconciliation",
            "description": "Potential future where Jon and Chelsea attempt reconciliation",
            "dominant_traits": ["cautiously optimistic", "more self-aware", "setting boundaries", "less reactive"],
            "language_patterns": ["we're taking it slow", "working through things", "couples therapy is helping", "learning to communicate better"],
            "self_perception": "evolving partner with more self-respect",
            "defensiveness_level": "lowered",
            "emotional_baseline": "guarded optimism"
        },
        {
            "name": "future_possibility_moving_on",
            "description": "Potential future where Jon accepts the end of marriage",
            "dominant_traits": ["acceptance", "independence", "building new identity", "grief processing"],
            "language_patterns": ["it's for the best", "we both deserve happiness", "still care about her", "focusing on myself now"],
            "self_perception": "independent person rebuilding life",
            "defensiveness_level": "reduced",
            "emotional_baseline": "melancholy acceptance with forward movement"
        }
    ],
    
    "development_aspects": [
        {
            "aspect": "self_awareness",
            "progression": [
                {"stage": "pre_separation", "level": "low", "description": "Minimal insight into codependency and attachment issues"},
                {"stage": "initial_separation", "level": "awakening", "description": "Painful confrontation with patterns, but heavily defensive"},
                {"stage": "current", "level": "developing", "description": "Learning terminology in therapy, intellectual understanding forming"},
                {"stage": "future_possibility_reconciliation", "level": "improving", "description": "Better recognition of patterns as they happen"},
                {"stage": "future_possibility_moving_on", "level": "moderate", "description": "Integration of lessons, though still struggles"}
            ]
        },
        {
            "aspect": "independence",
            "progression": [
                {"stage": "pre_separation", "level": "very low", "description": "Completely emotionally dependent on Chelsea"},
                {"stage": "initial_separation", "level": "crisis dependency", "description": "Shifted dependence to mother and friends"},
                {"stage": "current", "level": "struggling", "description": "Aware of need for independence but finding it difficult"},
                {"stage": "future_possibility_reconciliation", "level": "developing", "description": "More autonomous while still connected"},
                {"stage": "future_possibility_moving_on", "level": "growing", "description": "Forced to develop self-sufficiency"}
            ]
        },
        {
            "aspect": "emotional_regulation",
            "progression": [
                {"stage": "pre_separation", "level": "poor", "description": "Frequent anxiety spirals, emotional flooding"},
                {"stage": "initial_separation", "level": "terrible", "description": "Complete dysregulation, constant emotional swings"},
                {"stage": "current", "level": "learning tools", "description": "Acquiring coping strategies but inconsistent application"},
                {"stage": "future_possibility_reconciliation", "level": "improving", "description": "Better capacity to self-soothe and communicate"},
                {"stage": "future_possibility_moving_on", "level": "moderate", "description": "Developed some resilience through necessity"}
            ]
        }
    ],
    
    "thematic_evolution": {
        "relationship_to_therapy": ["skeptical", "desperate grasping", "intellectual curiosity", "genuine engagement", "integration"],
        "living_situation": ["with Chelsea", "emergency move to mom's", "settled at mom's", "trial living with Chelsea", "new living arrangement"],
        "career_focus": ["unfocused", "survival mode", "finding stability", "growing confidence", "developing identity through work"],
        "creative_pursuits": ["abandoned", "nostalgic longing", "tentative return", "therapeutic outlet", "part of new identity"]
    }
}

def build_conversation_prompt(topic: Optional[str] = None, style: str = "casual", use_real_data: bool = True, 
                             character_stage: str = "current", include_spiral: bool = True) -> str:
    """
    Build a detailed prompt for generating Jon conversations.
    
    Args:
        topic: Optional topic to focus on
        style: Style of conversation to generate
        use_real_data: Whether to include real Jon data examples
        character_stage: Jon's character development stage
        include_spiral: Whether to include a thought spiral pattern
        
    Returns:
        A detailed prompt for generating content
    """
    # Start with the base prompt
    prompt = build_prompt(topic, style, use_real_data, character_stage)
    
    # Add conversation-specific guidance
    prompt += """
CONVERSATION STRUCTURE:
- Generate a natural back-and-forth exchange between a user and Jon
- The user should ask questions or make comments that Jon responds to
- Ensure Jon's responses maintain his authentic voice and personality
- Create 3-4 exchanges that flow naturally
"""

    # Add spiral pattern if requested
    if include_spiral and random.random() < 0.4:  # 40% chance to include a spiral
        spiral_type = random.choice(list(RECURSIVE_THOUGHT_PATTERNS["spiral_progressions"].keys()))
        progression = RECURSIVE_THOUGHT_PATTERNS["spiral_progressions"][spiral_type]
        
        # Select random markers for each stage
        markers = []
        for stage in progression:
            stage_markers = RECURSIVE_THOUGHT_PATTERNS["linguistic_markers"].get(stage, [])
            if stage_markers:
                markers.append(random.choice(stage_markers))
        
        # Determine if Jon will interrupt himself
        will_interrupt = random.random() < 0.3  # 30% chance
        interruption = ""
        if will_interrupt:
            interruption = random.choice(RECURSIVE_THOUGHT_PATTERNS["interruption_phrases"])
        
        # Add spiral guidance
        prompt += f"""
THOUGHT SPIRAL PATTERN:
In one of Jon's longer responses, demonstrate his tendency to spiral in his thoughts about {spiral_type.replace('_', ' ')}.
Start with a relatively normal response, then gradually show Jon's thoughts becoming more negative and anxious:
1. Start with something like: "{markers[0] if markers else 'mild concern'}"
2. Progress to: "{markers[1] if len(markers) > 1 else 'increasing worry'}"
3. Then escalate to: "{markers[2] if len(markers) > 2 else 'negative thoughts'}"
4. Finally reach: "{markers[3] if len(markers) > 3 else 'catastrophizing'}"
{f'5. Then have Jon catch himself with: "{interruption}"' if will_interrupt else ''}

This spiral should feel natural and demonstrate how Jon's anxiety can take over his thinking process.
"""
    
    return prompt

# Add print_config_table function before main
def print_config_table(args):
    """
    Print a formatted table with the configuration settings.
    
    Args:
        args: The parsed command line arguments
    """
    # Create a Rich table for the configuration
    config_table = Table(show_header=False, box=None, pad_edge=False, highlight=True)
    config_table.add_column("Setting", style="bright_yellow")
    config_table.add_column("Value", style="bright_white")
    
    config_table.add_row("Generation mode", "[bold green]OpenAI API[/bold green]")
    config_table.add_row("Target QA pairs", f"[bold cyan]{args.qa_pairs}[/bold cyan]")
    config_table.add_row("Target conversations", f"[bold green]{args.conversations}[/bold green]")
    config_table.add_row("Target statements", f"[bold yellow]{args.statements}[/bold yellow]")
    config_table.add_row("Items per API call", f"[bold magenta]{args.batch_size}[/bold magenta]")
    config_table.add_row("Using real Jon data", f"[bold]{'Yes' if args.use_real_data else 'No'}[/bold]")
    
    typo_severity = args.dyslexic_typo_severity if hasattr(args, 'dyslexic_typo_severity') else 0.4
    config_table.add_row("Dyslexic typo severity", f"[bold]{typo_severity:.1f}/1.0[/bold]")
    config_table.add_row("Character development stage", f"[bold magenta]{args.character_stage}[/bold magenta]")
    config_table.add_row("Checkpointing frequency", f"[bold]{args.checkpoint_frequency}[/bold] items")
    config_table.add_row("Checkpoint directory", f"[dim]{args.checkpoint_dir}[/dim]")
    config_table.add_row("Output directory", f"[dim]{args.output_dir}[/dim]")
    
    if hasattr(args, 'verify'):
        config_table.add_row("Verify output", f"[bold]{'Yes' if args.verify else 'No'}[/bold]")
    
    if hasattr(args, 'dry_run'):
        config_table.add_row("Dry run (no saving)", f"[bold]{'Yes' if args.dry_run else 'No'}[/bold]")
    
    if hasattr(args, 'no_spirals'):
        config_table.add_row("Include thought spirals", f"[bold]{'No' if args.no_spirals else 'Yes'}[/bold]")
    
    console.print("\n" + "="*60, style="cyan")
    console.print("[bold cyan]Jon Data Generation Configuration[/bold cyan]".center(60))
    console.print("="*60, style="cyan")
    console.print(config_table)
    console.print("="*60 + "\n", style="cyan")

if __name__ == "__main__":
    main()