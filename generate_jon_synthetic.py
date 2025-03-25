#!/usr/bin/env python3
import json
import os
import random
import re
import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional
import argparse
from tqdm import tqdm
import numpy as np
from collections import Counter
import sys  # Add missing import

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINE_TUNED_MODEL = os.getenv("CURRENT_MODEL_ID")

# Global cache for style analysis
STYLE_ANALYSIS_CACHE = None
# Default batch size for parallel API calls
DEFAULT_BATCH_SIZE = 5
# Default parallel thread count
MAX_WORKERS = 4
# Rate limit protection - requests per minute
MAX_REQUESTS_PER_MINUTE = 60

# Global authenticity configuration
AUTHENTICITY_CONFIG = {
    "style_similarity": 0.55,  # Lowered from 0.7 to be more realistic
    "rhythm_required": False,
    "min_typos": 1,
    "grammar_required": False,
    "min_weighted_score": 0.6
}

"""
Jon's Texting Style Analysis:
Based on actual analysis of Jon's text messages in jon_training_data_from_convo.jsonl

Key characteristics:
1. Uses "idk" and "lol" frequently, but rarely uses "fr" or other modern slang
2. Often uses "man" as a filler word when talking to friends
3. Tends to use lowercase most of the time, but not always
4. Missing apostrophes is very common in his texts
5. Uses ellipsis occasionally but not excessively (only about 15% of messages)
6. Often repeats filler words like "like" or "just" multiple times in the same message
7. Frequently starts messages with "I mean", "I think", "I feel", or similar phrases
8. Uses "honestly" more often than synthetic examples were capturing
9. Typos are more common than initially estimated
"""

# Jon's target style vector (will be refined by analyze_jon_style)
TARGET_STYLE_VECTOR = {
    "all_lowercase_rate": 0.7, 
    "missing_apostrophes_rate": 0.8,
    "no_punctuation_rate": 0.5, 
    "run_on_sentences_rate": 0.4,
    "ellipsis_rate": 0.02,  # Reduced to match Jon's actual frequency
    "filler_words_rate": 0.3,
    "typo_words_rate": 0.4
}

# Common Jon typo words and fillers - based on actual usage patterns
JON_TYPO_WORDS = ["ur", "u", "ya", "im", "thats", "dont", "cant", "didnt", "isnt", "wouldnt", "ive", "youre", 
                  "doesnt", "whats", "wont", "couldnt", "id", "theyre", "whos", "wasnt", "havent",
                  "its", "theres", "shouldnt", "arent", "werent", "hasnt"]

# Jon's filler words (actual usage patterns from training data)
JON_FILLER_WORDS = ["like", "idk", "like", "man", "just", "honestly", "really", 
                    "actually", "basically", "literally", "whatever", "you know", 
                    "anyway", "so", "kinda", "haha", "lol"]
                    
# Note: Removed redundant words and focused on ones Jon actually uses frequently

# Threshold for style similarity
STYLE_SIMILARITY_THRESHOLD = 0.85
# Whether rhythm patterns are required
RHYTHM_MATCH_REQUIRED = True
# Minimum typo count in authentic messages
MIN_TYPO_COUNT = 1

# Jon's rhythm patterns - derived from actual message analysis
JON_RHYTHM_PATTERNS = [
    r'\blike\b.*\blike\b',  # Repeating "like" in the same message - common pattern
    r'i mean.*(?:idk|lol)', # "I mean" followed by "idk" or "lol" - very frequent
    r'idk.*idk',            # Repeating "idk" in the same message
    r'\bjust\b.*\bjust\b',  # Repeating "just" in the same message - common for Jon
    r'^i (?:mean|think|feel|know).*', # Starting with "I mean/think/feel/know" - frequent opener
    r'(?:lol|haha).*(?:lol|haha)', # Multiple "lol" or "haha" in the same message
    r'.*\bman\b.*',         # Using "man" as a filler - extremely common for Jon when talking to friends
    r'.*\bhonestly\b.*',    # Using "honestly" - Jon uses this frequently when expressing feelings
    r'.*\byeah\b.*\blike\b.*', # "yeah" followed by "like" - common transition
    r'.*\bso\b.*\blike\b.*' # "so" followed by "like" - common transition
]

# Jon's grammar patterns - capturing his distinct sentence structures
JON_GRAMMAR_PATTERNS = [
    # Starting patterns
    {"pattern": r"^i (?:just|feel|think|mean)\b", "desc": "Starting with 'I just/feel/think/mean'", "weight": 0.25},
    {"pattern": r"^(?:yeah|no|honestly|like)\b", "desc": "Starting with filler (yeah/no/honestly/like)", "weight": 0.2},
    
    # Mid-sentence structures
    {"pattern": r"\bbut (?:like|yeah|honestly)\b", "desc": "Contradiction followed by filler", "weight": 0.15},
    {"pattern": r"\bso i (?:just|kinda|basically)\b", "desc": "Consequence with minimizer", "weight": 0.15},
    
    # Trailing off patterns
    {"pattern": r"\bjust\.{3}$", "desc": "Trailing off with 'just...'", "weight": 0.1},
    {"pattern": r"\bknow\b.{0,10}$", "desc": "Ending with 'know' or 'you know'", "weight": 0.1},
    
    # Question formations
    {"pattern": r"\bdo you (?:think|feel)\b", "desc": "Asking with 'do you think/feel'", "weight": 0.1},
    {"pattern": r"\bis that (?:weird|crazy|normal)\b", "desc": "Asking for validation", "weight": 0.1},
    
    # Clause ordering
    {"pattern": r"\b(?:like|honestly|actually), (?:i|it's)\b", "desc": "Filler, then subject", "weight": 0.15},
    {"pattern": r"\bi (?:mean|think), (?:like|but|it's)\b", "desc": "Opinion, then connector", "weight": 0.15},
    
    # Sentence fragments
    {"pattern": r"\b(?:not sure|kinda|sorta) (?:about|if|how)\b", "desc": "Uncertainty fragment", "weight": 0.2},
    {"pattern": r"\bjust (?:tired|stressed|confused|excited)\b", "desc": "Simple emotional state", "weight": 0.2}
]

def prepare_generation_prompt(style_analysis):
    """Prepare a system prompt for generation that guides the model to write like Jon based on actual style metrics"""
    examples = style_analysis.get("examples", [])
    example_text = "\n".join([f"Example: \"{ex}\"" for ex in examples[:3]])
    
    # Get actual metrics for more precise guidance
    metrics = style_analysis.get("avg_metrics", TARGET_STYLE_VECTOR)
    
    # Get top filler words from analysis if available
    common_words = style_analysis.get("common_words", {})
    top_fillers = [word for word in common_words.keys() if word.lower() in JON_FILLER_WORDS][:5]
    top_fillers_str = ", ".join([f'"{word}"' for word in top_fillers]) if top_fillers else '"man", "like", "just", "idk", "haha"'
    
    # Extract pattern frequencies
    rhythm_counts = style_analysis.get("rhythm_counts", {})
    top_patterns = []
    if rhythm_counts:
        # Find the most common patterns
        pattern_indices = sorted([(int(k.split('_')[1]), v) for k, v in rhythm_counts.items() if k.startswith('pattern_')], 
                                key=lambda x: x[1], reverse=True)
        for idx, _ in pattern_indices[:3]:  # Get top 3 pattern indices
            if idx < len(JON_RHYTHM_PATTERNS):
                pattern_desc = ""
                if idx == 0:
                    pattern_desc = "repeating 'like' in the same message"
                elif idx == 1:
                    pattern_desc = "using 'I mean' followed by 'idk' or 'lol'"
                elif idx == 2:
                    pattern_desc = "repeating 'idk' in the same message"
                elif idx == 3:
                    pattern_desc = "repeating 'just' in the same message"
                elif idx == 4:
                    pattern_desc = "starting messages with 'I mean/think/feel/know'"
                elif idx == 5:
                    pattern_desc = "using multiple 'lol' or 'haha' in the same message"
                elif idx == 6:
                    pattern_desc = "using 'man' as a filler word"
                elif idx == 7:
                    pattern_desc = "using 'honestly' when expressing feelings"
                elif idx == 8:
                    pattern_desc = "using 'yeah' followed by 'like'"
                elif idx == 9:
                    pattern_desc = "using 'so' followed by 'like'"
                
                if pattern_desc:
                    top_patterns.append(pattern_desc)
    
    top_patterns_str = "\n- " + "\n- ".join(top_patterns) if top_patterns else "- using 'man' occasionally\n- starting sentences with 'I mean' or 'I feel like'\n- using 'haha' or 'lol' at the end"
    
    # Extract grammar pattern frequencies
    grammar_counts = style_analysis.get("grammar_patterns", {})
    top_grammar_patterns = []
    if grammar_counts:
        # Find the most common grammar patterns
        grammar_indices = sorted([(int(k.split('_')[1]), v) for k, v in grammar_counts.items() if k.startswith('pattern_')], 
                                key=lambda x: x[1], reverse=True)
        for idx, _ in grammar_indices[:4]:  # Get top 4 grammar pattern indices
            if idx < len(JON_GRAMMAR_PATTERNS):
                grammar_desc = JON_GRAMMAR_PATTERNS[idx]["desc"]
                if grammar_desc:
                    top_grammar_patterns.append(grammar_desc)
    
    top_grammar_str = "\n- " + "\n- ".join(top_grammar_patterns) if top_grammar_patterns else "- starting with 'I just' or 'I feel like'\n- using sentence fragments\n- creating run-on sentences with 'and' or 'but'\n- trailing off with '...'"
    
    # Get grammar statistics
    avg_grammar_stats = style_analysis.get("avg_grammar_stats", {})
    fragment_rate = avg_grammar_stats.get("fragment_rate", 0.3)
    run_on_rate = avg_grammar_stats.get("run_on_rate", 0.2)
    
    # Adjust description based on actual metrics
    lowercase_desc = "Almost always uses lowercase" if metrics.get("all_lowercase_rate", 0) > 0.8 else "Usually writes in lowercase" if metrics.get("all_lowercase_rate", 0) > 0.5 else "Often writes in lowercase"
    apostrophes_desc = "Very frequently omits apostrophes" if metrics.get("missing_apostrophes_rate", 0) > 0.7 else "Often skips apostrophes" if metrics.get("missing_apostrophes_rate", 0) > 0.4 else "Sometimes skips apostrophes"
    ellipsis_desc = "Occasionally uses ellipsis (...)" if metrics.get("ellipsis_rate", 0) > 0.15 else "Rarely uses ellipsis (...)" if metrics.get("ellipsis_rate", 0) > 0.05 else "Seldom uses ellipsis (...)"
    filler_desc = "Uses occasional filler words" if metrics.get("filler_words_rate", 0) > 0.2 else "Uses some filler words" if metrics.get("filler_words_rate", 0) > 0.1 else "Uses filler words sparingly"
    fragment_desc = "Sometimes uses sentence fragments" if fragment_rate > 0.3 else "Occasionally uses sentence fragments"
    run_on_desc = "Creates run-on sentences occasionally" if run_on_rate > 0.3 else "Rarely creates run-on sentences"
    
    prompt = f"""You are simulating Jon's casual texting style based on analysis of Jon's actual messaging behavior. He writes with these specific characteristics:

1. {lowercase_desc}, especially for common words
2. {apostrophes_desc} (dont, cant, im, etc.)
3. {filler_desc} like {top_fillers_str}, but doesn't overuse them
4. {ellipsis_desc} - don't overuse them!
5. Uses short, direct sentences with occasional fragments
6. Has natural speech patterns:
{top_patterns_str}
7. Uses these grammar structures naturally:
{top_grammar_str}
8. His messages are to-the-point and don't ramble
9. Often uses "man" when talking to friends
10. Uses "haha" or "lol" to respond to humor

IMPORTANT: Jon's style is natural and doesn't force his patterns into every message. Don't overdo ellipses, and place "man" and filler words in natural positions.

{example_text}

Respond in Jon's authentic texting style - capturing his direct, casual tone without overusing stylistic elements. Make it sound exactly like Jon typed it himself."""

    return prompt

def extract_training_data(input_file="jon_training_data_from_convo.jsonl"):
    """Extract training data from the JSONL file"""
    examples = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to parse as JSON
                    data = json.loads(line)
                    
                    # Handle different formats
                    if isinstance(data, dict) and "messages" in data:
                        # Standard OpenAI fine-tuning format
                        examples.append(data)
                    elif isinstance(data, list) and len(data) >= 2:
                        # Simple [user_message, assistant_message] format
                        examples.append(data)
                    elif isinstance(data, dict) and "prompt" in data and "completion" in data:
                        # Convert to standard format
                        examples.append({
                            "messages": [
                                {"role": "user", "content": data["prompt"]},
                                {"role": "assistant", "content": data["completion"]}
                            ]
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
                except Exception as e:
                    print(f"Warning: Error processing line: {str(e)}")
    except FileNotFoundError:
        print(f"Warning: Training data file not found: {input_file}")
        return []
    
    print(f"Extracted {len(examples)} training examples")
    return examples

def compute_style_metrics(text: str) -> Dict[str, float]:
    """Compute Jon's style metrics for a given text"""
    # Convert to lowercase for consistent processing
    text_lower = text.lower()
    
    # Count words and sentences
    word_count = len(re.findall(r'\b\w+\b', text_lower))
    if word_count == 0:
        return {k: 0.0 for k in TARGET_STYLE_VECTOR.keys()}
    
    sentences = re.split(r'[.!?]+', text_lower)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))
    
    # Calculate metrics
    all_lowercase = text == text_lower
    apostrophe_words = ['dont', 'cant', 'im', 'youre', 'thats', 'ive', 'theres', 'wont', 'didnt', 'isnt', 'wasnt', 'couldnt', 'wouldnt', 'shouldnt']
    missing_apostrophes = sum(1 for word in apostrophe_words if word in re.findall(r'\b\w+\b', text_lower))
    
    # Check for run-on sentences (sentences with 'and' or commas)
    run_on_count = sum(1 for s in sentences if len(re.findall(r'\band\b|,', s)) > 1)
    
    # Count ellipsis
    ellipsis_count = len(re.findall(r'\.\.\.', text))
    
    # Count filler words
    filler_count = sum(1 for word in JON_FILLER_WORDS if word in re.findall(r'\b\w+\b', text_lower))
    
    # Count typo words
    typo_count = sum(1 for word in JON_TYPO_WORDS if word in re.findall(r'\b\w+\b', text_lower))
    
    # Calculate improper spacing
    improper_spacing = len(re.findall(r'[.!?][A-Za-z]|[A-Za-z],[A-Za-z]', text))
    
    # Normalize metrics
    metrics = {
        "all_lowercase_rate": 1.0 if all_lowercase else 0.0,
        "missing_apostrophes_rate": missing_apostrophes / max(1, word_count * 0.1),  # Assume ~10% of words might have apostrophes
        "no_punctuation_rate": 1.0 if len(re.findall(r'[.!?,]', text)) == 0 else 0.0,
        "run_on_sentences_rate": run_on_count / sentence_count,
        "ellipsis_rate": ellipsis_count / sentence_count,
        "improper_spacing_rate": improper_spacing / max(1, len(text) * 0.01),
        "filler_words_rate": filler_count / word_count,
        "typo_words_rate": typo_count / word_count
    }
    
    # Cap rates at 1.0
    return {k: min(v, 1.0) for k, v in metrics.items()}

def compute_style_similarity(text: str, target_style: Dict[str, float]) -> float:
    """Compute cosine similarity between text style and target style"""
    # Get style metrics for the text
    text_style = compute_style_metrics(text)
    
    # Extract vectors ensuring same keys in same order
    keys = sorted(set(target_style.keys()) & set(text_style.keys()))
    vec_a = np.array([text_style[k] for k in keys])
    vec_b = np.array([target_style[k] for k in keys])
    
    # Compute cosine similarity
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def matches_jon_rhythm(text: str) -> Tuple[bool, List[int], float]:
    """Check if the text matches Jon's rhythm patterns and return pattern detail"""
    text_lower = text.lower()
    matched_indices = []
    
    for i, pattern in enumerate(JON_RHYTHM_PATTERNS):
        if re.search(pattern, text_lower):
            matched_indices.append(i)
    
    has_rhythm = len(matched_indices) > 0
    
    # Calculate a rhythm score based on number of matched patterns
    # Scale from 0.0 (no matches) to 1.0 (matched all patterns)
    rhythm_score = min(1.0, len(matched_indices) / len(JON_RHYTHM_PATTERNS))
    
    # Add a weight multiplier based on text length - longer texts should match more patterns
    words = len(text_lower.split())
    if words > 15:
        # For longer messages, we expect more pattern matches
        rhythm_score = rhythm_score * (len(matched_indices) / max(1, min(5, words / 10)))
    
    # Normalize score to 0-1 range
    rhythm_score = min(1.0, max(0.1, rhythm_score))
    
    return has_rhythm, matched_indices, rhythm_score

def count_typo_words(text: str) -> int:
    """Count Jon's typical typo words in the text"""
    text_lower = text.lower()
    return sum(1 for word in JON_TYPO_WORDS + JON_FILLER_WORDS if word in re.findall(r'\b\w+\b', text_lower))

def classify_mood_and_topics(text: str) -> Tuple[str, List[str]]:
    """Classify Jon's mood and tag possible topics"""
    lower = text.lower()
    mood = "neutral"
    topics = []

    if any(word in lower for word in ["fuck", "shit", "wtf", "damn", "omg", "crap"]):
        mood = "spiral"
    elif any(word in lower for word in ["idk", "whatever", "lol", "meh", "i guess"]):
        mood = "numb"
    elif any(word in lower for word in ["working", "lifting", "writing", "gym", "trying", "effort"]):
        mood = "trying"
    elif any(word in lower for word in ["tired", "sad", "lonely", "depressed", "sorry"]):
        mood = "down"
    elif any(word in lower for word in ["yo", "sup", "cool", "nice", "awesome", "great"]):
        mood = "okay"

    # Topic detection
    if any(word in lower for word in ["chelsea", "girlfriend", "her", "dating", "relationship", "tinder"]):
        topics.append("relationship")
    if any(word in lower for word in ["gym", "workout", "lifting", "fitness", "exercise"]):
        topics.append("fitness")
    if any(word in lower for word in ["therapist", "therapy", "session", "anxiety", "depression"]):
        topics.append("mental_health")
    if any(word in lower for word in ["writing", "book", "story", "novel", "creative"]):
        topics.append("creative")
    if any(word in lower for word in ["drinking", "sober", "alcohol", "beer", "drunk"]):
        topics.append("recovery")
    if any(word in lower for word in ["work", "job", "boss", "office", "project"]):
        topics.append("work")
    if any(word in lower for word in ["code", "coding", "programming", "developer", "software"]):
        topics.append("coding")
    if any(word in lower for word in ["friend", "friends", "party", "hang", "hangout"]):
        topics.append("social")

    return mood, topics

def is_authentic_jon(text: str, target_style: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate if the text is authentically Jon using the 3-layer approach
    Returns (is_authentic, diagnostic_info)
    """
    # Skip very short messages, they're hard to analyze
    if len(text.split()) < 3:
        return True, {"reason": "too_short_to_analyze"}
    
    # Layer 1: Style similarity
    style_similarity = compute_style_similarity(text, target_style)
    
    # Layer 2: Rhythm matching
    has_rhythm, matched_patterns, rhythm_score = matches_jon_rhythm(text)
    
    # Layer 3: Typo count
    typo_count = count_typo_words(text)
    
    # Layer 4: Grammar pattern analysis
    grammar_stats = analyze_grammar_patterns(text)
    grammar_score = grammar_stats["grammar_match_score"]
    has_grammar_patterns = len(grammar_stats["matched_patterns"]) > 0
    
    # Combine validation with weighted scoring
    style_weight = 0.4
    rhythm_weight = 0.3
    typo_weight = 0.2
    grammar_weight = 0.3
    
    # Calculate weighted score (0-1)
    weighted_score = (
        (style_similarity * style_weight) +
        (rhythm_score * rhythm_weight) +
        (min(1.0, typo_count / max(1, thresholds.get("min_typos", AUTHENTICITY_CONFIG["min_typos"]))) * typo_weight) +
        (grammar_score * grammar_weight)
    ) / (style_weight + rhythm_weight + typo_weight + grammar_weight)
    
    # Determine if authentic based on weighted score and required thresholds
    min_weighted_score = thresholds.get("min_weighted_score", 0.6)
    is_authentic = (
        weighted_score >= min_weighted_score and
        style_similarity >= thresholds.get("style_similarity", AUTHENTICITY_CONFIG["style_similarity"]) and
        (not thresholds.get("rhythm_required", AUTHENTICITY_CONFIG["rhythm_required"]) or has_rhythm) and
        typo_count >= thresholds.get("min_typos", AUTHENTICITY_CONFIG["min_typos"]) and
        (not thresholds.get("grammar_required", AUTHENTICITY_CONFIG.get("grammar_required", False)) or has_grammar_patterns)
    )
    
    # Return detailed diagnostic info
    return is_authentic, {
        "style_similarity": style_similarity,
        "has_rhythm": has_rhythm,
        "rhythm_score": rhythm_score,
        "matched_patterns": matched_patterns,
        "typo_count": typo_count,
        "style_metrics": compute_style_metrics(text),
        "grammar_score": grammar_score,
        "has_grammar_patterns": has_grammar_patterns,
        "grammar_patterns": grammar_stats["matched_patterns"],
        "weighted_score": weighted_score,
        "is_authentic": is_authentic
    }

def analyze_grammar_patterns(text: str) -> Dict[str, Any]:
    """
    Analyze text for Jon's grammar patterns
    Returns a dictionary of matched patterns and statistics
    """
    text_lower = text.lower()
    
    # Count matches for each pattern
    matches = []
    for i, pattern_info in enumerate(JON_GRAMMAR_PATTERNS):
        pattern = pattern_info["pattern"]
        if re.search(pattern, text_lower):
            matches.append({
                "index": i,
                "desc": pattern_info["desc"],
                "pattern": pattern
            })
    
    # Count sentence types
    sentences = re.split(r'[.!?]+', text_lower)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Count fragments (short sentences without a clear structure)
    fragments = [s for s in sentences if len(s.split()) < 5 and not re.search(r'\b(?:is|am|are|was|were)\b', s)]
    
    # Count run-ons (long sentences with multiple main clauses)
    run_ons = [s for s in sentences if len(s.split()) > 10 and s.count('and') + s.count('but') > 1]
    
    # Count sentences starting with conjunction (but, and, so, or)
    conj_starts = [s for s in sentences if re.match(r'\b(?:but|and|so|or)\b', s)]
    
    # Calculate statistics
    total_sentences = max(1, len(sentences))
    
    grammar_stats = {
        "matched_patterns": [m["index"] for m in matches],
        "pattern_descriptions": [m["desc"] for m in matches],
        "fragment_rate": len(fragments) / total_sentences,
        "run_on_rate": len(run_ons) / total_sentences,
        "conj_start_rate": len(conj_starts) / total_sentences,
        "grammar_match_score": min(1.0, len(matches) / 3.0)  # Normalize to 0-1
    }
    
    return grammar_stats

def apply_jon_grammar(text: str, style_analysis=None) -> str:
    """
    Apply Jon's grammar patterns to the text
    This should be called before apply_typos_and_style for best results
    """
    if not style_analysis:
        style_analysis = analyze_jon_style()
    
    # Extract common grammar patterns
    grammar_patterns = {}
    if "grammar_patterns" in style_analysis:
        grammar_patterns = style_analysis["grammar_patterns"]
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Process each sentence to apply Jon's grammar
    for i in range(len(sentences)):
        # Skip very short sentences
        if len(sentences[i].split()) < 4:
            continue
        
        # 15% chance to apply a transformation (much lower than before)
        if random.random() < 0.15:
            sentence = sentences[i]
            
            # 1. Small chance to start with a filler - Jon doesn't overdo this
            if random.random() < 0.2 and not re.match(r'^(?:i|you|we|they|it)\b', sentence.lower()):
                starter = random.choice(["like ", "honestly ", "i think ", "i mean "])
                sentence = starter + sentence[0].lower() + sentence[1:]
            
            # 2. Occasional run-on by replacing period with 'and' or 'but'
            # Jon uses these sometimes but not excessively
            if i < len(sentences) - 1 and random.random() < 0.15:
                connector = random.choice([" and ", " but ", " so "])
                next_sentence = sentences[i+1]
                if re.search(r'[.!?]$', sentence):
                    sentence = sentence[:-1] + connector + next_sentence[0].lower() + next_sentence[1:]
                    sentences[i+1] = ""  # Mark for removal
            
            # 3. Jon uses occasional fragments but not constantly
            if len(sentence.split()) > 8 and random.random() < 0.12:
                parts = sentence.split()
                mid = len(parts) // 2
                
                # Jon doesn't use ellipses frequently - will use a comma instead
                for j in range(mid-1, mid+2):
                    if j < len(parts) and not parts[j].lower() in ["the", "a", "an", "and", "but", "or"]:
                        # Use comma more often than ellipsis
                        if random.random() < 0.7:
                            parts.insert(j+1, ",")
                        else:
                            parts.insert(j+1, "...")
                        break
                
                sentence = " ".join(parts)
            
            # 4. Replace phrases with Jon's style
            if random.random() < 0.3:
                # Jon says "i feel like" rather than "I believe/think"
                sentence = re.sub(r'\bI (?:believe|think)\b', 'i feel like', sentence, flags=re.IGNORECASE)
            
            # Update the sentence
            sentences[i] = sentence
    
    # Remove marked sentences and join
    sentences = [s for s in sentences if s]
    result = " ".join(sentences)
    
    # Final transformations for Jon's casual style
    
    # Replace formal phases with more casual ones
    casual_replacements = {
        r'\bI am going to\b': 'im gonna',
        r'\bI am\b': 'im',
        r'\byou are\b': 'youre',
        r'\bhave to\b': 'gotta',
        r'\bwant to\b': 'wanna',
        r'\bprobably\b': 'prob',
        r'\bdefinitely\b': 'def',
        r'\b(?:very|really) good\b': 'pretty good'
    }
    
    for pattern, replacement in casual_replacements.items():
        if random.random() < 0.4:  # Only sometimes, not every occurrence
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def analyze_jon_style(input_file=None):
    """Analyze Jon's style from training data"""
    global STYLE_ANALYSIS_CACHE, TARGET_STYLE_VECTOR
    
    if STYLE_ANALYSIS_CACHE:
        return STYLE_ANALYSIS_CACHE
    
    # Use default file if not specified
    if not input_file:
        input_file = "jon_training_data_from_convo.jsonl"
    
    # Load training data
    examples = extract_training_data(input_file)
    print(f"Extracted {len(examples)} training examples")
    
    # Extract Jon's responses and analyze
    responses = []
    for example in examples:
        # Handle both possible formats of training data
        if isinstance(example, dict) and len(example.get("messages", [])) >= 2:
            response = example["messages"][1].get("content", "")
            responses.append(response)
        elif isinstance(example, list) and len(example) >= 2:
            # Handle list format [user_message, assistant_message]
            response = example[1]
            responses.append(response)
    
    # Compute average style metrics
    all_metrics = []
    for response in responses:
        if response and len(response.split()) >= 3:  # Skip very short responses
            metrics = compute_style_metrics(response)
            all_metrics.append(metrics)
    
    # Calculate averages
    avg_metrics = {}
    for key in TARGET_STYLE_VECTOR.keys():
        values = [m.get(key, 0.0) for m in all_metrics if key in m]
        avg_metrics[key] = sum(values) / max(1, len(values))
    
    # Count types of rhythm patterns
    rhythm_counts = Counter()
    for response in responses:
        # Cache lowercased response
        response_lower = response.lower()
        for i, pattern in enumerate(JON_RHYTHM_PATTERNS):
            if re.search(pattern, response_lower):
                rhythm_counts[f"pattern_{i}"] += 1
    
    # Analyze grammar patterns
    grammar_counts = Counter()
    grammar_stats = []
    for response in responses:
        if len(response.split()) >= 5:  # Skip very short responses
            stats = analyze_grammar_patterns(response)
            grammar_stats.append(stats)
            for idx in stats["matched_patterns"]:
                grammar_counts[f"pattern_{idx}"] += 1
    
    # Calculate average grammar stats
    avg_grammar_stats = {}
    if grammar_stats:
        for key in ["fragment_rate", "run_on_rate", "conj_start_rate", "grammar_match_score"]:
            values = [stats.get(key, 0.0) for stats in grammar_stats]
            avg_grammar_stats[key] = sum(values) / len(values)
    
    # Analyze specific words and patterns
    word_counter = Counter()
    for response in responses:
        # Cache lowercased response
        response_lower = response.lower()
        words = re.findall(r'\b\w+\b', response_lower)
        word_counter.update(words)
    
    # Create top examples of Jon's style
    top_examples = []
    for response in responses:
        if len(response.split()) >= 5:  # Skip very short responses
            # Cache lowercased response
            response_lower = response.lower()
            metrics = compute_style_metrics(response)
            sim = compute_style_similarity(response, avg_metrics)
            # Check rhythm using lowercased version
            has_rhythm, matched_patterns, rhythm_score = matches_jon_rhythm(response_lower)
            # Check grammar patterns
            grammar_stats = analyze_grammar_patterns(response)
            
            # Combined quality score
            quality_score = (sim * 0.5) + (rhythm_score * 0.3) + (grammar_stats["grammar_match_score"] * 0.2)
            
            if quality_score > 0.7:
                top_examples.append(response)
    
    # Update global target style vector
    TARGET_STYLE_VECTOR = avg_metrics
    
    # Create result
    result = {
        "avg_metrics": avg_metrics,
        "rhythm_counts": dict(rhythm_counts),
        "grammar_patterns": dict(grammar_counts),
        "avg_grammar_stats": avg_grammar_stats,
        "common_words": dict(word_counter.most_common(50)),
        "examples": random.sample(top_examples, min(5, len(top_examples))),
        "total_examples": len(responses)
    }
    
    STYLE_ANALYSIS_CACHE = result
    return result

def apply_typos_and_style(text, style_analysis=None, seed=None):
    """Apply Jon's style to the response (now enhanced with authenticity validation)"""
    if not text or not text.strip():
        return text
    
    # Set random seed if provided for reproducible results
    if seed is not None:
        random_state = random.getstate()
        random.seed(seed)
    
    if not style_analysis:
        style_analysis = analyze_jon_style()
    
    # Extract actual metrics from style analysis
    style_metrics = style_analysis.get("avg_metrics", TARGET_STYLE_VECTOR)
    
    # Get actual rates from analyzed data, with fallbacks to defaults
    all_lowercase_rate = style_metrics.get("all_lowercase_rate", 0.7)
    ellipsis_rate = style_metrics.get("ellipsis_rate", 0.08)  # Reduced from 0.15
    filler_rate = style_metrics.get("filler_words_rate", 0.3)
    typo_rate = style_metrics.get("typo_words_rate", 0.4)
    missing_apostrophes_rate = style_metrics.get("missing_apostrophes_rate", 0.8)
    
    # Original style application
    text = text.strip()
    
    # Lowercase based on actual Jon frequency
    if random.random() < all_lowercase_rate:
        text = text.lower()
    
    # Apply filler words at starts of sentences based on actual data
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i in range(len(sentences)):
        if random.random() < filler_rate * 0.7 and len(sentences[i].split()) > 3:  # Scale for sentence starts
            filler = random.choice(["yeah ", "like ", "honestly ", "i mean ", "man ", ""])
            sentences[i] = filler + sentences[i][0].lower() + sentences[i][1:]
    
    # Rejoin sentences
    text = " ".join(sentences)
    
    # Apply typos and Jon's speech patterns
    # Apply missing apostrophes based on real frequency
    replacements = {
        "I'm": "im",
        "I am": "im",
        "don't": "dont",
        "can't": "cant",
        "you're": "youre",
        "that's": "thats",
        "it's": "its",
        "I've": "ive",
        "there's": "theres",
        "what's": "whats",
        "won't": "wont",
        "didn't": "didnt",
        "isn't": "isnt",
        "wasn't": "wasnt",
        "I'll": "ill",
        "I'd": "id",
    }
    
    # Apply apostrophe removals based on actual frequency
    words = text.split()
    for i, word in enumerate(words):
        for original, typo in replacements.items():
            if word.lower() == original.lower() and random.random() < missing_apostrophes_rate:
                words[i] = typo
    
    text = " ".join(words)
    
    # Random ellipsis based on Jon's actual usage rate - MUCH LESS FREQUENT
    if random.random() < ellipsis_rate:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences) - 2)
            sentences[idx] = sentences[idx].rstrip('.!?') + "..."
            text = " ".join(sentences)
    
    # Add filler words more naturally - not random placement
    words = text.split()
    positions = []
    
    # Only add fillers at natural positions (after conjunctions, at end of thoughts)
    for i in range(1, len(words) - 1):
        prev_word = words[i-1].lower()
        # Only add after certain transition words or at pauses
        if prev_word in ['but', 'and', 'so', 'or', 'because', 'like'] or words[i-1].endswith(','):
            positions.append(i)
    
    # Add at most 1-2 fillers per message, not littering everywhere
    filler_count = min(min(2, len(positions)), max(1, int(len(words) * filler_rate * 0.25)))
    
    if positions and random.random() < filler_rate * 0.5:  # Reduced chance
        for _ in range(filler_count):
            if not positions:
                break
            pos_idx = random.randrange(len(positions))
            pos = positions.pop(pos_idx)
            
            # Choose fillers that work after transitions
            filler = random.choice(["like", "you know", "honestly", "just"])
            words.insert(pos, filler)
    
    text = " ".join(words)
    
    # Only inject typos on longer words, not randomly
    words = text.split()
    candidates = []
    
    # Find good candidates for typos (longer words, not already typos/fillers)
    for i, word in enumerate(words):
        if (len(word) > 3 and 
            word.lower() not in JON_TYPO_WORDS and 
            word.lower() not in JON_FILLER_WORDS and
            not re.match(r'^\W+$', word)):  # Not just punctuation
            candidates.append(i)
    
    # Apply a small number of typos
    typo_count = min(2, max(1, int(len(words) * typo_rate * 0.2)))
    
    if candidates and random.random() < typo_rate * 0.5:  # Reduced chance
        for _ in range(min(typo_count, len(candidates))):
            idx = random.choice(candidates)
            candidates.remove(idx)
            
            word = words[idx]
            # Apply realistic typos:
            # 1. Drop a letter
            if len(word) > 4 and random.random() < 0.5:
                drop_idx = random.randint(1, len(word) - 2)  # Don't drop first or last letter
                words[idx] = word[:drop_idx] + word[drop_idx+1:]
            # 2. Letter swap
            elif len(word) > 3:
                swap_idx = random.randint(0, len(word) - 2)
                chars = list(word)
                chars[swap_idx], chars[swap_idx+1] = chars[swap_idx+1], chars[swap_idx]
                words[idx] = ''.join(chars)
    
    text = " ".join(words)
    
    # Check if the result is authentic Jon - but DON'T force authenticity
    is_authentic, diagnostics = is_authentic_jon(
        text, 
        TARGET_STYLE_VECTOR,
        {
            "style_similarity": AUTHENTICITY_CONFIG["style_similarity"],
            "rhythm_required": AUTHENTICITY_CONFIG["rhythm_required"],
            "min_typos": AUTHENTICITY_CONFIG["min_typos"]
        }
    )
    
    # If not authentic, consider adding specific patterns that Jon uses
    if not is_authentic and diagnostics.get("weighted_score", 0) < 0.4:
        # Add a characteristic Jon pattern, but don't force it into every message
        if random.random() < 0.3:
            pattern = random.choice([
                "not bad",
                "its all good",
                "been busy with",
                "just working on",
                "trying to",
                "not sure yet"
            ])
            
            # Insert in a natural position, not randomly tacked on
            words = text.split()
            if len(words) > 6:
                pos = len(words) // 2  # Middle of message
                words.insert(pos, pattern)
                text = " ".join(words)
    
    # Restore random state if seed was provided
    if seed is not None:
        random.setstate(random_state)
    
    return text

def create_chat_completion(messages, model=None, temperature=0.9, max_tokens=150):
    """Create a chat completion with retry logic"""
    if not model:
        model = FINE_TUNED_MODEL or "gpt-4o"
    
    retries = 3
    backoff = 2
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = backoff ** attempt
                print(f"API error: {str(e)}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {retries} attempts: {str(e)}")
                return None

def create_variations_worker(args):
    """Worker function for parallel variation generation"""
    index, example_response, count, style_prompt = args
    
    try:
        messages = [
            {"role": "system", "content": style_prompt},
            {"role": "user", "content": f"Generate {count} variations that sound like Jon would say them but are different enough from the original. Include his casual spelling, typos, and grammatical patterns."}
        ]
        
        content = create_chat_completion(messages, max_tokens=200)
        
        if not content:
            return index, [example_response]  # fallback to original
        
        # Try to separate variations
        pattern = r'\d+\.\s+(.*?)(?=\d+\.|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            variations = [match.strip() for match in matches]
        else:
            # Fallback to line splitting
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            variations = [line for line in lines if not line.startswith('Variation') and not line.isdigit()]
        
        if not variations:
            variations = [content.strip()]
            
        # Return the index and variations for proper ordering
        return index, variations[:count]
    
    except Exception as e:
        print(f"Error in variation worker {index}: {str(e)}")
        return index, [example_response]  # fallback to original

def parallel_generate_variations(examples, style_analysis, variations_per_example=2):
    """Generate variations in parallel"""
    if not examples:
        return []
    
    results = []
    
    # Determine batch size
    batch_size = min(DEFAULT_BATCH_SIZE, len(examples))
    
    # Create batches
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    
    # Create a prompt to guide the model
    style_prompt = prepare_generation_prompt(style_analysis)
    
    # Add avg length info if available, otherwise calculate it
    if 'avg_length' not in style_analysis:
        # Calculate average length from examples
        total_words = 0
        count = 0
        for example in examples:
            if isinstance(example, dict) and len(example.get("messages", [])) >= 2:
                response = example["messages"][1].get("content", "")
                total_words += len(response.split())
                count += 1
            elif isinstance(example, list) and len(example) >= 2:
                response = example[1]
                total_words += len(response.split())
                count += 1
        
        avg_length = total_words / max(1, count)
        style_analysis['avg_length'] = avg_length
    
    # Now safely use avg_length
    if 'avg_length' in style_analysis:
        style_prompt += f"His messages typically have around {int(style_analysis['avg_length'])} words. "
    
    # Define worker for variation generation
    def process_batch(batch):
        """Process a batch of examples to generate variations"""
        batch_results = []
        
        for example in batch:
            # Extract prompt and response
            if isinstance(example, dict) and len(example.get("messages", [])) >= 2:
                prompt = example["messages"][0].get("content", "")
                original_response = example["messages"][1].get("content", "")
            elif isinstance(example, list) and len(example) >= 2:
                prompt = example[0]
                original_response = example[1]
            else:
                continue
            
            # Skip if missing either prompt or response
            if not prompt or not original_response:
                continue
            
            # Create variations
            variations = []
            for _ in range(variations_per_example):
                try:
                    messages = [
                        {"role": "system", "content": style_prompt},
                        {"role": "user", "content": f"Original prompt: {prompt}\n\nOriginal response: {original_response}\n\nCreate a variation of the original response that captures the same meaning but phrases it differently, while maintaining Jon's casual style."}
                    ]
                    
                    response = create_chat_completion(messages)
                    if response:
                        variations.append(response)
                except Exception as e:
                    print(f"Error generating variation: {str(e)}")
            
            # Add to batch results
            if variations:
                batch_results.append(({"prompt": prompt, "response": original_response}, variations))
        
        return batch_results
    
    # Process batches using ThreadPoolExecutor
    variations_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        # Collect results
        with tqdm(total=len(batches), desc="Generating variations") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    variations_results.extend(batch_results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Batch processing failed: {str(e)}")
    
    # Extract variations and build result data
    for (example, variations) in variations_results:
        for variation in variations:
            variation = apply_typos_and_style(variation, style_analysis)
            if variation and variation.strip():
                # Validate authenticity
                is_authentic, diagnostics = is_authentic_jon(
                    variation,
                    TARGET_STYLE_VECTOR,
                    {
                        "style_similarity": AUTHENTICITY_CONFIG["style_similarity"],
                        "rhythm_required": AUTHENTICITY_CONFIG["rhythm_required"],
                        "min_typos": AUTHENTICITY_CONFIG["min_typos"]
                    }
                )
                
                # Get emotional classification
                mood, topics = classify_mood_and_topics(variation)
                
                # Add rhythm information
                _, matched_patterns, rhythm_score = matches_jon_rhythm(variation)
                
                # Create result with metadata
                results.append({
                    "messages": [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": variation}
                    ],
                    "metadata": {
                        "synthetic": True,
                        "mood": mood,
                        "topics": topics,
                        "source": "jon_synthetic_generator",
                        "variation": True,
                        "timestamp": time.time(),
                        "authenticity": {
                            "style_similarity": diagnostics.get("style_similarity", 0),
                            "has_rhythm": diagnostics.get("has_rhythm", False),
                            "rhythm_score": rhythm_score,
                            "matched_patterns": matched_patterns,
                            "typo_count": diagnostics.get("typo_count", 0),
                            "is_authentic": is_authentic
                        }
                    }
                })
    
    return results

def parallel_process_prompts(prompts, style_analysis, max_fidelity=False):
    """Process prompts in parallel and generate responses"""
    # Setup common prompt style
    system_prompt = prepare_generation_prompt(style_analysis)
    
    # Calculate average length of responses if not available
    if 'avg_length' not in style_analysis:
        # Extract examples and calculate average length
        total_words = 0
        count = 0
        for example in style_analysis.get("examples", []):
            total_words += len(example.split())
            count += 1
        
        if count > 0:
            avg_length = total_words / count
            style_analysis['avg_length'] = avg_length
    
    # Ensure we don't process too many at once for API rate limits
    batch_size = min(DEFAULT_BATCH_SIZE, len(prompts))
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    all_results = []
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        batch_results = []
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(tqdm(batch, desc=f"Processing batch {batch_idx+1}/{len(batches)}")):
            try:
                user_content = prompt
                if isinstance(prompt, dict):
                    user_content = prompt.get("content", "")
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
                # Generate response
                response = create_chat_completion(messages)
                
                if response:
                    # First apply Jon's grammar structure
                    grammar_response = apply_jon_grammar(response, style_analysis)
                    
                    # Then apply Jon's style (typos, fillers, etc.)
                    styled_response = apply_typos_and_style(grammar_response, style_analysis)
                    
                    # Validate authenticity
                    is_authentic, diagnostics = is_authentic_jon(
                        styled_response,
                        TARGET_STYLE_VECTOR,
                        {
                            "style_similarity": AUTHENTICITY_CONFIG["style_similarity"],
                            "rhythm_required": AUTHENTICITY_CONFIG["rhythm_required"],
                            "min_typos": AUTHENTICITY_CONFIG["min_typos"]
                        }
                    )
                    
                    # Get mood and topics
                    mood, topics = classify_mood_and_topics(styled_response)
                    
                    # Get rhythm information
                    _, matched_patterns, rhythm_score = matches_jon_rhythm(styled_response)
                    
                    # Get grammar information
                    grammar_stats = analyze_grammar_patterns(styled_response)
                    
                    # Ensure all values are JSON serializable
                    is_authentic_json = bool(is_authentic)
                    style_similarity = float(diagnostics.get("style_similarity", 0))
                    has_rhythm = bool(diagnostics.get("has_rhythm", False))
                    rhythm_score = float(rhythm_score)
                    matched_patterns = [int(idx) for idx in matched_patterns]
                    typo_count = int(diagnostics.get("typo_count", 0))
                    
                    # Create result with metadata
                    result = {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": styled_response}
                        ],
                        "metadata": {
                            "synthetic": True,
                            "mood": str(mood),
                            "topics": [str(topic) for topic in topics],
                            "source": "jon_synthetic_generator",
                            "variation": False,
                            "timestamp": float(time.time()),
                            "authenticity": {
                                "style_similarity": style_similarity,
                                "has_rhythm": has_rhythm,
                                "rhythm_score": rhythm_score,
                                "matched_patterns": matched_patterns,
                                "typo_count": typo_count,
                                "is_authentic": is_authentic_json,
                                "grammar_stats": {
                                    "fragment_rate": float(grammar_stats.get("fragment_rate", 0)),
                                    "run_on_rate": float(grammar_stats.get("run_on_rate", 0)),
                                    "grammar_match_score": float(grammar_stats.get("grammar_match_score", 0)),
                                    "matched_grammar_patterns": [int(idx) for idx in grammar_stats.get("matched_patterns", [])]
                                }
                            }
                        }
                    }
                    
                    # Add to batch results
                    batch_results.append(result)
            except Exception as e:
                print(f"Error processing prompt {prompt_idx}: {str(e)}")
        
        # Add batch results to all results
        all_results.extend(batch_results)
    
    return all_results

def generate_dynamic_prompt():
    """Use GPT to dynamically generate a Jon-style prompt"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use gpt-3.5-turbo if you're optimizing for cost
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are helping simulate emotional texting prompts from a friend to Jon — someone who is tired, moody, overthinks, "
                        "and spirals a lot. Generate one emotionally honest, lowercase, casual prompt like a real friend texting him. "
                        "It should sound like something from a real chat. Keep it under 25 words. Examples: "
                        "\"yo u good or nah?\", \"she hit u back?\", \"you still spiralin?\""
                    )
                },
                {"role": "user", "content": "Give me one prompt only."}
            ],
            temperature=1.1,
            max_tokens=60
        )
        prompt = response.choices[0].message.content.strip()
        return prompt
    except Exception as e:
        print(f"Error generating dynamic prompt: {str(e)}")
        return None

def generate_synthetic_prompts(style_analysis=None, count=10, use_dynamic=False):
    """Generate synthetic prompts that would be asked to Jon"""
    if count <= 0:
        return []
    
    # Use dynamic GPT generated prompts if requested
    if use_dynamic:
        prompts = []
        with tqdm(total=count, desc="Generating dynamic prompts") as pbar:
            while len(prompts) < count:
                prompt = generate_dynamic_prompt()
                if prompt and prompt not in prompts:
                    prompts.append(prompt)
                    pbar.update(1)

        return prompts

    # Define prompt templates based on common conversation topics with Jon
    templates = {
        "relationship": [
            "How are things going with Chelsea?",
            "Have you and Chelsea talked recently?",
            "Any progress with your relationship?",
            "How's married life?",
            "Everything okay at home?",
            "Have you two figured things out?",
            "Are you and Chelsea working things out?",
            "How's the communication going with Chelsea?"
        ],
        "mental_health": [
            "How was your therapy session?",
            "Are you feeling better these days?",
            "Have you been taking care of yourself?",
            "How's your mental health journey going?",
            "Is therapy helping?",
            "Are you still feeling down?",
            "Have you been managing your anxiety?",
            "Are you in a better headspace now?"
        ],
        "work": [
            "How's the new job going?",
            "Any luck with the job search?",
            "Did you hear back from that interview?",
            "Are you still looking for work?",
            "How's work treating you?",
            "Found any good job opportunities?",
            "Are you happy at your current job?",
            "Any progress on the career front?"
        ],
        "creative": [
            "How's the writing coming along?",
            "Made any progress on your book?",
            "Still working on your stories?",
            "Have you been writing lately?",
            "Any new creative projects?",
            "How's the creative work going?",
            "Written anything new?",
            "Still doing the writing thing?"
        ],
        "fitness": [
            "Have you been working out?",
            "Still going to the gym?",
            "How's the fitness journey?",
            "Making progress with your health goals?",
            "Been exercising lately?",
            "Are you still lifting?",
            "How's the workout routine going?",
            "Taking care of your physical health?"
        ],
        "social": [
            "Want to hang out this weekend?",
            "Are you free to get together?",
            "Want to grab food sometime?",
            "Up for gaming tonight?",
            "Want to do something?",
            "Free to catch up soon?",
            "Want to meet up?",
            "Should we plan something?"
        ],
        "check_in": [
            "How have you been?",
            "Everything okay?",
            "Haven't heard from you in a while - how are you?",
            "Just checking in - how are things?",
            "You doing alright?",
            "How's everything going?",
            "What's new with you?",
            "How are you holding up?"
        ],
        "support": [
            "Do you want to talk about it?",
            "Is there anything I can do to help?",
            "Want some company?",
            "Need someone to listen?",
            "Should I come by?",
            "Need to get your mind off things?",
            "Want to get out of the house?",
            "Need anything?"
        ],
        "future": [
            "What are your plans going forward?",
            "Have you thought about what's next?",
            "Where do you see things going?",
            "What do you want to do?",
            "Have you figured out your next steps?",
            "What's your plan?",
            "What are you thinking of doing?",
            "Have you made any decisions?"
        ]
    }

    # Generate prompts with good distribution across categories
    prompts = []
    categories = list(templates.keys())
    
    while len(prompts) < count:
        # Cycle through categories to ensure distribution
        category = categories[len(prompts) % len(categories)]
        
        # Select a random prompt from the category
        category_prompts = templates[category]
        prompt = random.choice(category_prompts)
        
        # Add some natural variation
        if random.random() < 0.2:  # 20% chance to add a follow-up
            followup = random.choice([
                "I've been worried about you.",
                "Just wanted to check in.",
                "No pressure.",
                "If you want to talk.",
                "When you're ready.",
                "I'm here if you need anything.",
                "Take your time.",
                "Let me know."
            ])
            prompt = f"{prompt} {followup}"
        
        if prompt not in prompts:  # Avoid duplicates
            prompts.append(prompt)
    
    return prompts

def summarize_emotionally(text: str) -> Optional[str]:
    """Summarize the emotional theme of a message in a short phrase"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the emotional theme of this message in a sentence or phrase, like a thought Jon might say to himself. Keep it under 20 words."},
                {"role": "user", "content": text}
            ],
            temperature=0.6,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing emotional theme: {str(e)}")
        return None

def get_embedding_worker(args):
    """Worker function for parallel embedding generation"""
    idx, text, user_prompt = args
    try:
        # Combine user prompt and Jon's response for richer context
        embedding_input = f"[User]: {user_prompt}\n[Jon]: {text}"
        response = client.embeddings.create(
            input=embedding_input,
            model="text-embedding-3-small"
        )
        # Convert numpy array to list for JSON serialization
        embedding = response.data[0].embedding
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return idx, embedding
    except Exception as e:
        print(f"Error getting embedding {idx}: {str(e)}")
        return idx, None

def parallel_convert_to_vector_store(data, output_file):
    """Convert generated data to vector store format using parallel processing"""
    # Extract data to embed
    items_to_embed = []
    for i, item in enumerate(data):
        if "messages" in item and len(item["messages"]) >= 2:
            user_prompt = item["messages"][0].get("content", "")
            jon_response = item["messages"][1].get("content", "")
            
            if jon_response and jon_response.strip():
                items_to_embed.append((i, user_prompt, jon_response))
    
    # Prepare worker tasks for embeddings
    tasks = [(i, item[2], item[1]) for i, item in enumerate(items_to_embed)]
    
    # Process embeddings in parallel
    embeddings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
        futures = [executor.submit(get_embedding_worker, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating embeddings"):
            try:
                idx, embedding = future.result()
                if embedding:
                    embeddings[idx] = embedding
            except Exception as e:
                print(f"Embedding worker failed: {str(e)}")
    
    # Create vector items with the retrieved embeddings
    vector_items = []
    
    # Process each embedding and create vector items
    for i, (idx, user_prompt, jon_response) in enumerate(items_to_embed):
        if idx in embeddings:
            # Determine mood and topics
            mood, topics = classify_mood_and_topics(jon_response)
            
            # Generate emotional summary
            summary = summarize_emotionally(jon_response)
            
            # Check authenticity
            is_authentic, diagnostics = is_authentic_jon(
                jon_response,
                TARGET_STYLE_VECTOR,
                {
                    "style_similarity": AUTHENTICITY_CONFIG["style_similarity"],
                    "rhythm_required": AUTHENTICITY_CONFIG["rhythm_required"],
                    "min_typos": AUTHENTICITY_CONFIG["min_typos"]
                }
            )
            
            # Add rhythm information
            _, matched_patterns, rhythm_score = matches_jon_rhythm(jon_response)
            
            # Ensure all values are JSON serializable
            is_authentic_json = bool(is_authentic)
            style_similarity = float(diagnostics.get("style_similarity", 0))
            has_rhythm = bool(diagnostics.get("has_rhythm", False))
            rhythm_score = float(rhythm_score)
            matched_patterns = [int(idx) for idx in matched_patterns]
            typo_count = int(diagnostics.get("typo_count", 0))
            
            # Check if we should fragment long messages
            if len(jon_response.split()) > 25:
                # Split at punctuation
                segments = re.split(r'(?<=[.!?]) +', jon_response)
                for j, segment in enumerate(segments):
                    if not segment.strip():
                        continue
                    
                    # Get embedding for this segment
                    embedding_input = f"[User]: {user_prompt}\n[Jon]: {segment.strip()}"
                    
                    # Use the existing embedding function
                    try:
                        segment_response = client.embeddings.create(
                            input=embedding_input,
                            model="text-embedding-3-small"
                        )
                        segment_embedding = segment_response.data[0].embedding
                        # Convert numpy array to list for JSON serialization
                        if isinstance(segment_embedding, np.ndarray):
                            segment_embedding = segment_embedding.tolist()
                    except Exception:
                        continue
                    
                    # Check segment authenticity
                    segment_authentic, segment_diagnostics = is_authentic_jon(
                        segment, 
                        TARGET_STYLE_VECTOR,
                        {"style_similarity": 0.65, "rhythm_required": False, "min_typos": 0}
                    )
                    
                    # Add rhythm information for segment
                    _, segment_matched_patterns, segment_rhythm_score = matches_jon_rhythm(segment)
                    
                    # Ensure JSON serializable for segment
                    segment_authentic_json = bool(segment_authentic)
                    segment_similarity = float(segment_diagnostics.get("style_similarity", 0))
                    segment_has_rhythm = bool(segment_diagnostics.get("has_rhythm", False))
                    segment_typo_count = int(segment_diagnostics.get("typo_count", 0))
                    
                    # Classify mood and topics for this segment
                    segment_mood, segment_topics = classify_mood_and_topics(segment)
                    
                    # Generate summary for this segment
                    segment_summary = summarize_emotionally(segment)
                    
                    # Create the vector store item for this segment
                    item_id = f"jon-synth-{i}-{j}"
                    vector_item = {
                        "id": item_id,
                        "values": segment_embedding,
                        "metadata": {
                            "text": segment.strip(),
                            "context": user_prompt,
                            "mood": segment_mood,
                            "topics": segment_topics,
                            "summary": segment_summary,
                            "source": "synthetic_fragment",
                            "synthetic": True,
                            "fragment": True,
                            "timestamp": time.time(),
                            "authenticity": {
                                "style_similarity": segment_similarity,
                                "has_rhythm": segment_has_rhythm,
                                "rhythm_score": segment_rhythm_score,
                                "matched_patterns": segment_matched_patterns,
                                "typo_count": segment_typo_count,
                                "is_authentic": segment_authentic_json
                            }
                        }
                    }
                    vector_items.append(vector_item)
                
                # Still include the full message but with a different tag
                vector_item = {
                    "id": f"jon-synth-{i}-full",
                    "values": embeddings[idx],
                    "metadata": {
                        "text": jon_response,
                        "context": user_prompt,
                        "mood": mood,
                        "topics": topics,
                        "summary": summary,
                        "source": "synthetic_full",
                        "synthetic": True,
                        "fragment": False,
                        "timestamp": time.time(),
                        "authenticity": {
                            "style_similarity": style_similarity,
                            "has_rhythm": has_rhythm,
                            "rhythm_score": rhythm_score,
                            "matched_patterns": matched_patterns,
                            "typo_count": typo_count,
                            "is_authentic": is_authentic_json
                        }
                    }
                }
                vector_items.append(vector_item)
            else:
                # Create the vector store item for shorter messages
                vector_item = {
                    "id": f"jon-synth-{i}",
                    "values": embeddings[idx],
                    "metadata": {
                        "text": jon_response,
                        "context": user_prompt,
                        "mood": mood,
                        "topics": topics,
                        "summary": summary,
                        "source": "synthetic",
                        "synthetic": True,
                        "fragment": False,
                        "timestamp": time.time(),
                        "authenticity": {
                            "style_similarity": style_similarity,
                            "has_rhythm": has_rhythm,
                            "rhythm_score": rhythm_score,
                            "matched_patterns": matched_patterns,
                            "typo_count": typo_count,
                            "is_authentic": is_authentic_json
                        }
                    }
                }
                vector_items.append(vector_item)
    
    # Write to file
    if vector_items:
        with open(output_file, "w") as f:
            for item in vector_items:
                f.write(json.dumps(item) + "\n")
        
        print(f"Converted {len(vector_items)} items to vector store format: {output_file}")
        print(f"  - Generated fragments for longer messages")
        print(f"  - Added mood classification: {set(item['metadata']['mood'] for item in vector_items)}")
        print(f"  - Added topic detection: {set(topic for item in vector_items for topic in item['metadata'].get('topics', []))}")
        print(f"  - Added emotional summaries")
        print(f"  - Added authenticity scoring with:")
        print(f"    - Style similarity scoring")
        print(f"    - Rhythm pattern detection")
        print(f"    - Typo/filler word analysis")
        
        # Show authenticity stats
        authentic_count = sum(1 for item in vector_items if item['metadata'].get('authenticity', {}).get('is_authentic', False))
        print(f"    - {authentic_count}/{len(vector_items)} items ({authentic_count/len(vector_items)*100:.1f}%) passed authenticity check")
        
        return True
    else:
        print("No items converted")
        return False

def filter_and_regenerate(responses, style_analysis, max_attempts=2):
    """Filter low-quality responses and attempt to regenerate them for better quality"""
    
    # Extract thresholds for validation
    min_style_similarity = AUTHENTICITY_CONFIG["style_similarity"]
    rhythm_required = AUTHENTICITY_CONFIG["rhythm_required"]
    min_typos = AUTHENTICITY_CONFIG["min_typos"]
    
    # Track metrics for reporting
    total_responses = len(responses)
    regeneration_attempts = 0
    improved_responses = 0
    
    # Prepare enhanced prompt for regeneration
    enhanced_prompt = prepare_generation_prompt(style_analysis)
    enhanced_prompt += "\n\nIMPORTANT: The previous generations were not authentic enough. Please make this more authentically Jon by:"
    enhanced_prompt += "\n- Using more of Jon's speech patterns and fillers"
    enhanced_prompt += "\n- Including his typical typos and lowercase style"
    enhanced_prompt += "\n- Adding characteristic rhythm patterns (like repeating 'idk' or using 'man' as a filler)"
    
    # Process each response
    for i, response in enumerate(responses):
        if not isinstance(response, dict) or "messages" not in response or "metadata" not in response:
            continue
            
        metadata = response["metadata"]
        authenticity = metadata.get("authenticity", {})
        
        # Check if this response needs improvement
        if authenticity.get("is_authentic", False):
            continue  # Skip already authentic responses
            
        # Get current metrics
        style_similarity = float(authenticity.get("style_similarity", 0))
        has_rhythm = bool(authenticity.get("has_rhythm", False))
        typo_count = int(authenticity.get("typo_count", 0))
        
        # Identify specific issues
        issues = []
        if style_similarity < min_style_similarity:
            issues.append(f"style similarity ({style_similarity:.2f}) below threshold ({min_style_similarity:.2f})")
        if rhythm_required and not has_rhythm:
            issues.append("missing Jon's rhythm patterns")
        if typo_count < min_typos:
            issues.append(f"not enough Jon-style typos/fillers ({typo_count})")
        
        # If there are issues, attempt regeneration
        for attempt in range(max_attempts):
            if not issues:
                break
                
            regeneration_attempts += 1
            
            # Get the original prompt
            user_content = response["messages"][0]["content"]
            original_response = response["messages"][1]["content"]
            
            # Create specific guidance for this regeneration
            issues_guidance = "\n".join([f"- {issue}" for issue in issues])
            regeneration_prompt = f"{enhanced_prompt}\n\nSpecific issues to fix:\n{issues_guidance}\n\nOriginal prompt: {user_content}\n\nCurrent response: {original_response}\n\nImproved Jon-authentic response:"
            
            try:
                # Generate improved response
                messages = [
                    {"role": "system", "content": regeneration_prompt},
                    {"role": "user", "content": f"Rewrite this in a more authentically Jon style"}
                ]
                
                new_response = create_chat_completion(messages)
                
                if new_response and len(new_response.strip()) > 0:
                    # Apply Jon style to ensure consistency
                    styled_response = apply_typos_and_style(new_response, style_analysis)
                    
                    # Validate the new response
                    is_authentic, diagnostics = is_authentic_jon(
                        styled_response,
                        TARGET_STYLE_VECTOR,
                        {
                            "style_similarity": min_style_similarity,
                            "rhythm_required": rhythm_required,
                            "min_typos": min_typos
                        }
                    )
                    
                    # Check if it's an improvement
                    new_style_similarity = float(diagnostics.get("style_similarity", 0))
                    new_has_rhythm = bool(diagnostics.get("has_rhythm", False))
                    new_typo_count = int(diagnostics.get("typo_count", 0))
                    
                    # Determine if this is better
                    is_improvement = (
                        new_style_similarity > style_similarity or
                        (not has_rhythm and new_has_rhythm) or
                        (typo_count < min_typos and new_typo_count >= min_typos)
                    )
                    
                    if is_improvement:
                        # Update the response with the improved version
                        response["messages"][1]["content"] = styled_response
                        
                        # Ensure all values are JSON serializable
                        matched_patterns = []
                        if "matched_patterns" in diagnostics:
                            matched_patterns = [int(idx) for idx in diagnostics["matched_patterns"]]
                            
                        response["metadata"]["authenticity"] = {
                            "style_similarity": float(new_style_similarity),
                            "has_rhythm": bool(new_has_rhythm),
                            "rhythm_score": float(diagnostics.get("rhythm_score", 0)),
                            "matched_patterns": matched_patterns,
                            "typo_count": int(new_typo_count),
                            "is_authentic": bool(is_authentic),
                            "regenerated": True,
                            "regeneration_attempt": int(attempt + 1)
                        }
                        
                        improved_responses += 1
                        
                        # Update issues for next attempt if needed
                        issues = []
                        if new_style_similarity < min_style_similarity:
                            issues.append(f"style similarity ({new_style_similarity:.2f}) still below threshold ({min_style_similarity:.2f})")
                        if rhythm_required and not new_has_rhythm:
                            issues.append("still missing Jon's rhythm patterns")
                        if new_typo_count < min_typos:
                            issues.append(f"still not enough Jon-style typos/fillers ({new_typo_count})")
                        
                        # If authentic or no more issues, we're done with this response
                        if bool(is_authentic) or not issues:
                            break
            except Exception as e:
                print(f"Error during regeneration: {str(e)}")
                # Continue to the next attempt
    
    # Report regeneration statistics
    if regeneration_attempts > 0:
        print(f"Regeneration: improved {improved_responses}/{total_responses} responses ({improved_responses/total_responses*100:.1f}%) with {regeneration_attempts} attempts")
    
    # Return the potentially improved responses
    return responses

def generate_synthetic_data(count, style_analysis=None, use_variations=False, variation_count=2, use_dynamic_prompts=False, max_fidelity=False, seed=None):
    """Generate synthetic data with Jon's style"""
    if not style_analysis:
        style_analysis = analyze_jon_style()
    
    # Generate synthetic prompts if not using variations
    examples = []
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Check if we have existing prompts
    prompts_file = "synthetic_prompts.jsonl"
    existing_prompts = []
    
    try:
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r') as f:
                for line in f:
                    try:
                        prompt = json.loads(line.strip())
                        existing_prompts.append(prompt)
                    except:
                        pass
        
        if existing_prompts:
            print(f"Found {len(existing_prompts)} existing prompts to generate responses for")
        else:
            print("No existing prompts to generate responses for")
    except Exception as e:
        print(f"Error reading prompts file: {str(e)}")
    
    # Generate new prompts if needed
    synthetic_prompts = []
    if len(existing_prompts) < count and not use_variations:
        prompt_count = count - len(existing_prompts)
        synthetic_prompts = generate_synthetic_prompts(style_analysis, prompt_count, use_dynamic=use_dynamic_prompts)
        
        # Save new prompts
        try:
            with open(prompts_file, 'a') as f:
                for prompt in synthetic_prompts:
                    f.write(json.dumps(prompt) + "\n")
        except Exception as e:
            print(f"Error saving prompts: {str(e)}")
    
    # Combine existing and new prompts
    prompts_to_use = existing_prompts + synthetic_prompts
    
    # Shuffle and limit to desired count
    random.shuffle(prompts_to_use)
    prompts_to_use = prompts_to_use[:count]
    
    # Generate responses to prompts
    if prompts_to_use:
        responses = parallel_process_prompts(prompts_to_use, style_analysis, max_fidelity=max_fidelity)
        examples.extend(responses)
        print(f"Generated {len(responses)} responses to synthetic prompts")
    
    # Generate variations if requested
    if use_variations:
        # Extract training data
        training_data = extract_training_data()
        # Sample examples for variation
        sample_size = min(count // variation_count, len(training_data))
        samples = random.sample(training_data, sample_size)
        # Generate variations
        variations = parallel_generate_variations(samples, style_analysis, variations_per_example=variation_count)
        examples.extend(variations)
        print(f"Generated {len(variations)} variations")
    
    # Apply the filter and regeneration to improve quality
    if max_fidelity:
        print("Applying quality filter and regeneration for maximum fidelity...")
        examples = filter_and_regenerate(examples, style_analysis)
    
    return examples

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate synthetic Jon data')
    parser.add_argument('--count', type=int, default=2, help='Number of synthetic examples to generate')
    parser.add_argument('--output', type=str, default='synthetic_prompts.jsonl', help='Output file path')
    parser.add_argument('--only-prompts', action='store_true', help='Only generate prompts without responses')
    parser.add_argument('--variations', action='store_true', help='Generate variations of existing examples')
    parser.add_argument('--variation-count', type=int, default=2, help='Number of variations per example')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--vectors', action='store_true', help='Convert to vector store format')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--max-fidelity', action='store_true', help='Use maximum fidelity mode')
    parser.add_argument('--dynamic-prompts', action='store_true', help='Use dynamically generated prompts')
    parser.add_argument('--similarity-threshold', type=float, help='Style similarity threshold')
    parser.add_argument('--rhythm-required', action='store_true', help='Require rhythm pattern match')
    parser.add_argument('--min-typos', type=int, help='Minimum typo count required')
    parser.add_argument('--test-input', type=str, help='Test a specific input for Jon authenticity')
    
    args = parser.parse_args()
    
    # If only generating prompts
    if args.only_prompts:
        print("Generating prompts...")
        prompts = generate_synthetic_prompts(count=args.count, use_dynamic=args.dynamic_prompts)
        print(f"Generated {len(prompts)} prompts")
        
        try:
            # Use 'with' statement with explicit flush for reliable output
            with open(args.output, 'w') as f:
                for i, prompt in enumerate(prompts):
                    f.write(f'"{prompt}"\n')
                    # Periodically flush to ensure data is written
                    if i % 100 == 0:
                        f.flush()
            print(f"Successfully saved {len(prompts)} prompts to {args.output}")
        except Exception as e:
            print(f"Error saving prompts: {str(e)}")
        return

    # Rest of the main function...
    MAX_WORKERS = args.threads
    
    # Update authenticity configuration if specified
    if args.similarity_threshold is not None:
        STYLE_SIMILARITY_THRESHOLD = args.similarity_threshold
        AUTHENTICITY_CONFIG.update({
            "style_similarity": args.similarity_threshold
        })
    
    if args.rhythm_required:
        RHYTHM_MATCH_REQUIRED = True
        AUTHENTICITY_CONFIG.update({
            "rhythm_required": True
        })
    
    if args.min_typos is not None:
        MIN_TYPO_COUNT = args.min_typos
        AUTHENTICITY_CONFIG.update({
            "min_typos": args.min_typos
        })
    
    # Test a specific input if provided
    if args.test_input:
        print(f"Testing input: {args.test_input}")
        is_authentic, diagnostics = is_authentic_jon(
            args.test_input,
            TARGET_STYLE_VECTOR,
            AUTHENTICITY_CONFIG
        )
        
        print(f"Jon Authenticity Analysis:")
        print(f"Style Similarity: {diagnostics['style_similarity']:.2f} (threshold: {AUTHENTICITY_CONFIG['style_similarity']})")
        print(f"Has Rhythm Pattern: {'✅' if diagnostics['has_rhythm'] else '❌'} (required: {AUTHENTICITY_CONFIG['rhythm_required']})")
        print(f"Matched Patterns: {diagnostics.get('matched_patterns', [])}")
        print(f"Typo/Filler Count: {diagnostics['typo_count']} (minimum: {AUTHENTICITY_CONFIG['min_typos']})")
        
        # Add grammar pattern information
        print(f"Grammar Score: {diagnostics.get('grammar_score', 0):.2f}")
        print(f"Has Grammar Patterns: {'✅' if diagnostics.get('has_grammar_patterns', False) else '❌'}")
        
        # Add weighted score
        weighted_score = diagnostics.get('weighted_score', 0)
        min_weighted_score = AUTHENTICITY_CONFIG.get('min_weighted_score', 0.6)
        print(f"Weighted Authenticity Score: {weighted_score:.2f} (threshold: {min_weighted_score:.2f})")
        
        print(f"Result: {'✅ AUTHENTIC JON' if is_authentic else '❌ NOT AUTHENTIC'}")
        
        if 'style_metrics' in diagnostics:
            print("\nStyle Metrics:")
            for k, v in diagnostics['style_metrics'].items():
                print(f"  {k}: {v:.2f}")
        
        # Check which rhythms it matches
        if diagnostics.get('matched_patterns'):
            print("\nMatched Rhythm Patterns:")
            for idx in diagnostics['matched_patterns']:
                if idx < len(JON_RHYTHM_PATTERNS):
                    print(f"  Pattern {idx}: {JON_RHYTHM_PATTERNS[idx]}")
        
        # Check which grammar patterns it matches
        if diagnostics.get('grammar_patterns'):
            print("\nMatched Grammar Patterns:")
            for idx in diagnostics['grammar_patterns']:
                if idx < len(JON_GRAMMAR_PATTERNS):
                    print(f"  {JON_GRAMMAR_PATTERNS[idx]['desc']}: {JON_GRAMMAR_PATTERNS[idx]['pattern']}")
        
        sys.exit(0)
        
    # Generate synthetic data
    if args.max_fidelity:
        print("🧠 Maximum Jon Fidelity Mode™ activated!")
        print("- Embedding prompt+response for better context")
        print("- Adding mood and topic classification")
        print("- Fragmenting long messages into emotional segments")
        print("- Including emotional summaries")
        print("- Adding rich metadata for future filtering")
        print("- 3-Layer Jon Authenticity System")
        print("  - Layer 1: Structural Style Fidelity")
        print("  - Layer 2: Rhythmic Authenticity")
        print("  - Layer 3: Typo-Weighted Style Distance Scoring")
    
    # Generate the synthetic data
    examples = generate_synthetic_data(
        args.count, 
        use_variations=args.variations,
        variation_count=args.variation_count,
        use_dynamic_prompts=args.dynamic_prompts,
        max_fidelity=args.max_fidelity,
        seed=args.seed
    )
    
    # Save the generated examples
    with open(args.output, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Generated {len(examples)} synthetic examples and saved to {args.output}")
    
    # Convert to vector store format if requested
    if args.vectors:
        vector_filename = os.path.splitext(args.output)[0] + '_vectors.jsonl'
        parallel_convert_to_vector_store(examples, vector_filename)
        
        print(f"Converted {len(examples)} items to vector store format: {vector_filename}")
        
        # Count authentic examples
        authentic_count = sum(1 for ex in examples if ex.get("metadata", {}).get("authenticity", {}).get("is_authentic", False))
        authentic_percent = authentic_count / max(1, len(examples)) * 100
        
        # Get unique moods and topics
        moods = set()
        topics = set()
        for ex in examples:
            metadata = ex.get("metadata", {})
            if "mood" in metadata:
                moods.add(metadata["mood"])
            if "topics" in metadata:
                topics.update(metadata.get("topics", []))
        
        print(f"  - Generated fragments for longer messages")
        print(f"  - Added mood classification: {moods}")
        print(f"  - Added topic detection: {topics}")
        print(f"  - Added emotional summaries")
        print(f"  - Added authenticity scoring with:")
        print(f"    - Style similarity scoring")
        print(f"    - Rhythm pattern detection")
        print(f"    - Typo/filler word analysis")
        print(f"    - {authentic_count}/{len(examples)} items ({authentic_percent:.1f}%) passed authenticity check")
        
        print("\nTo upload to OpenAI Vector Store, run:")
        print(f"./upload_vectors.py {vector_filename}")
        
        print("\nYou can use this synthetic data to:")
        print("1. Add to your training data for fine-tuning")
        print("2. Convert to vector embeddings with the convert_vectors.py script")
        print("3. Test the model with new conversation scenarios")
        
        print("\nVector store format is ready for upload to OpenAI Vector Store!")
        print("With Maximum Jon Fidelity™, you can now query by:")
        print("- Emotional state: 'spiral', 'down', 'trying', 'okay', 'numb'")
        print("- Topics: 'relationship', 'fitness', 'mental_health', 'work', etc.")
        print("- Semantic meaning through embedded summaries")
        print("- Full or fragmented responses for granular matches")
        print("- Rhythm score for measuring tonal richness")
        print("- Authenticity level for maximum Jon fidelity")

if __name__ == "__main__":
    main() 