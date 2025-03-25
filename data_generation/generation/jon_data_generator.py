#!/usr/bin/env python
"""
Jon Data Generator

This script generates synthetic data that captures Jon's personality and style
for use in various parts of SchultzGPT:
- Vector store entries (Q&A format)
- Fine-tuning examples (conversations)
- Embeddings data (statements, facts, opinions)

Optimized for OpenAI embeddings with:
- Semantic clustering for better retrieval
- Length-optimized chunks for embedding efficiency
- Contextual variations for diverse retrieval
- Rich metadata for filtering
- Contrastive examples for nuanced understanding

Run with: python -m data_generation.jon_data_generator --help
"""

import os
import sys
import json
import random
import argparse
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from src.config.config import Config
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Constants for data generation
TOPICS = [
    # Personal topics
    "work stress", "favorite books", "weekend plans", "dating life",
    "family relationships", "personal goals", "apartment living",
    "exercise routine", "sleep habits", "hobbies", "travel experiences",
    
    # Interests
    "science fiction", "literary fiction", "technology news", "programming",
    "creative writing", "indie games", "films", "music", "philosophy",
    "psychology", "history", "politics", "social media",
    
    # Situational
    "giving advice", "reacting to news", "responding to a joke",
    "providing support", "discussing a problem", "debating ideas",
    "sharing an opinion", "making plans", "catching up", "reminiscing",
    
    # Emotional
    "feeling anxious", "feeling excited", "feeling cynical", "feeling thoughtful",
    "feeling disappointed", "feeling amused", "feeling motivated", "feeling tired"
]

# Group topics into semantic clusters for better embedding performance
TOPIC_CLUSTERS = {
    "personal_life": ["work stress", "dating life", "family relationships", "personal goals", 
                     "apartment living", "exercise routine", "sleep habits"],
    "entertainment": ["favorite books", "science fiction", "literary fiction", "indie games", 
                     "films", "music"],
    "intellectual": ["philosophy", "psychology", "history", "politics", "technology news", 
                    "programming", "creative writing"],
    "social": ["weekend plans", "hobbies", "travel experiences", "social media", 
              "giving advice", "making plans", "catching up"],
    "emotional_states": ["feeling anxious", "feeling excited", "feeling cynical", 
                        "feeling thoughtful", "feeling disappointed", "feeling amused", 
                        "feeling motivated", "feeling tired"]
}

MOODS = [
    "neutral", "sarcastic", "thoughtful", "cynical", "supportive", 
    "amused", "irritated", "relaxed", "tired", "energetic", "philosophical"
]

# Entities that Jon might discuss - for metadata enrichment
ENTITIES = {
    "authors": ["david foster wallace", "kurt vonnegut", "george saunders", "zadie smith", 
               "joan didion", "haruki murakami", "margaret atwood", "philip k dick"],
    "books": ["infinite jest", "slaughterhouse five", "10th of december", "kafka on the shore", 
             "neuromancer", "the corrections", "dune", "the road"],
    "tech_companies": ["google", "apple", "microsoft", "amazon", "meta", "tesla"],
    "places": ["coffee shop", "bookstore", "apartment", "office", "gym", "bar", "park"],
    "tech_concepts": ["ai", "machine learning", "programming", "software development", 
                     "algorithms", "data science", "blockchain"]
}

JON_STYLE_ELEMENTS = [
    "uses lowercase extensively",
    "skips apostrophes in contractions",
    "uses minimal punctuation",
    "adds 'lol' or 'haha' for lighter moments",
    "makes dry observations",
    "uses brief sentences",
    "references books or authors occasionally",
    "shows flashes of insight between casual language",
    "employs self-deprecating humor",
    "writes concisely, rarely more than 3 sentences",
    "uses text abbreviations like 'u' for 'you' and 'ur' for 'your' sometimes",
    "demonstrates intelligence without showing off",
    "balances cynicism with genuine care",
    "occasionally uses ellipses... to trail off thoughts"
]

JON_FACTS = [
    "works in tech but has creative writing aspirations",
    "reads a lot, especially literary fiction and philosophy",
    "is somewhat cynical about politics and social media",
    "values authenticity and dislikes phoniness",
    "is deeply loyal to friends despite sarcastic exterior",
    "can get into 'spiraling' moods where he's more negative and raw",
    "has strong opinions about books, movies, and technology",
    "exercises occasionally but isn't obsessive about it",
    "enjoys craft beer and knows a lot about it",
    "grew up in a small town before moving to the city",
    "has an older sister he respects but doesn't talk to often enough",
    "keeps his apartment minimal but has many books",
    "prefers texting to calling",
    "has a dry, sometimes self-deprecating sense of humor",
    "enjoys independent films and has strong opinions about mainstream movies"
]

# OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Thread-local storage for tracking API calls
thread_local = threading.local()

# Global tracking
api_calls_lock = threading.Lock()
api_calls = {
    "qa_pairs": 0,
    "conversations": 0,
    "statements": 0,
    "variations": 0,
    "total_tokens": 0,
    "total_cost": 0.0,
    "batched_calls": 0,
    "individual_calls": 0
}

def get_token_estimate(text: str) -> int:
    """Roughly estimate token count for text sizing"""
    # Approximation: average English word is ~1.3 tokens
    return int(len(text.split()) * 1.3)

def track_api_call(call_type, tokens_used=0, batch_size=1):
    """Track API usage statistics"""
    with api_calls_lock:
        api_calls[call_type] = api_calls.get(call_type, 0) + 1
        api_calls["total_tokens"] += tokens_used
        # Using GPT-4 pricing as an approximation ($0.03/1K input, $0.06/1K output)
        estimated_cost = tokens_used * 0.00005  # Simplified average cost per token
        api_calls["total_cost"] += estimated_cost
        
        if batch_size > 1:
            api_calls["batched_calls"] += 1
        else:
            api_calls["individual_calls"] += 1
            
        return tokens_used, estimated_cost

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
    }
    
    all_text = ""
    
    # Extract text based on item type
    for item in data_items:
        if item_type == "qa":
            text = item.get("question", "") + " " + item.get("answer", "")
            topics = [item.get("metadata", {}).get("topic", "unknown")]
            sentiment = item.get("metadata", {}).get("sentiment", "neutral")
            entities = item.get("metadata", {}).get("entities", [])
        elif item_type == "conversation":
            text = " ".join([m.get("content", "") for m in item.get("messages", [])])
            topics = [item.get("topic", "unknown")]
            sentiment = "mixed"  # Conversations typically have mixed sentiment
            entities = []  # No entities tracked for conversations yet
        elif item_type == "statement":
            text = item.get("statement", "")
            topics = [item.get("metadata", {}).get("topic", "unknown")]
            sentiment = item.get("metadata", {}).get("sentiment", "neutral")
            entities = []  # No entities tracked for statements yet
        
        # Track token counts
        token_count = get_token_estimate(text)
        metrics["token_counts"].append(token_count)
        
        # Track topic distribution
        for topic in topics:
            if topic:
                metrics["topic_distribution"][topic] = metrics["topic_distribution"].get(topic, 0) + 1
                
        # Track sentiment distribution
        metrics["sentiment_distribution"][sentiment] = metrics["sentiment_distribution"].get(sentiment, 0) + 1
        
        # Track unique entities
        metrics["unique_entities"].update(entities)
        
        all_text += " " + text
    
    # Calculate vocabulary richness (unique words / total words)
    words = all_text.lower().split()
    if words:
        unique_words = len(set(words))
        total_words = len(words)
        metrics["vocabulary_richness"] = unique_words / total_words
    
    # Detect redundancy (simplified approach)
    if data_items:
        # Compare each item with others to detect similarities
        similarities = []
        sample_size = min(len(data_items), 20)  # Limit sample size for performance
        sample_items = random.sample(data_items, sample_size) if len(data_items) > 20 else data_items
        
        for i, item1 in enumerate(sample_items):
            for j, item2 in enumerate(sample_items):
                if i < j:  # Only compare unique pairs
                    similarity = calculate_text_similarity(
                        extract_text(item1, item_type),
                        extract_text(item2, item_type)
                    )
                    similarities.append(similarity)
        
        if similarities:
            metrics["redundancy_score"] = sum(similarities) / len(similarities)
    
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

def generate_vector_qa_pair(topic: Optional[str] = None, 
                          entities: Optional[List[str]] = None,
                          target_length: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a Q&A pair about Jon for vector store indexing
    
    Args:
        topic: Optional topic to focus on
        entities: Optional entities to include
        target_length: Target token length (250-500 recommended for embeddings)
    """
    topic = topic or random.choice(TOPICS)
    
    # Find which cluster this topic belongs to
    topic_cluster = None
    for cluster, topics in TOPIC_CLUSTERS.items():
        if topic in topics:
            topic_cluster = cluster
            break
    
    # Select entities if not provided
    if not entities:
        # Select 1-2 random entity categories
        entity_categories = random.sample(list(ENTITIES.keys()), random.randint(1, 2))
        entities = []
        for category in entity_categories:
            # Select 1-3 entities from each category
            entities.extend(random.sample(ENTITIES[category], random.randint(1, min(3, len(ENTITIES[category])))))
    
    # Length optimization
    length_instruction = ""
    if target_length:
        length_instruction = f"The answer should be approximately {target_length} tokens long."
    else:
        # Default to embedding-optimized length
        length_instruction = "The answer should be between 250-500 tokens for optimal embedding performance."
    
    entity_mentions = ""
    if entities:
        entity_mentions = f"Include references to these entities if natural: {', '.join(entities)}."
    
    prompt = f"""
    Generate a question and answer pair about Jon for a knowledge base.
    
    Jon's persona:
    {Config.PERSONA}
    
    Additional Jon facts:
    {random.choice(JON_FACTS)}
    {random.choice(JON_FACTS)}
    
    Topic area: {topic}
    {entity_mentions}
    
    The question should be something a user might ask about Jon.
    The answer should be in Jon's authentic voice and style.
    {length_instruction}
    
    Format your response as a JSON object with:
    - "question": The user's question
    - "answer": Jon's response in his authentic style
    - "topic": The specific topic this covers
    - "entities": List of entities mentioned
    - "sentiment": The emotional tone (positive, negative, neutral, mixed)
    - "topic_cluster": "{topic_cluster}" if referenced
    
    Ensure the answer showcases these Jon style elements:
    - {random.choice(JON_STYLE_ELEMENTS)}
    - {random.choice(JON_STYLE_ELEMENTS)}
    
    Return ONLY valid JSON without explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata for better filtering
        result["metadata"] = {
            "topic": topic,
            "topic_cluster": topic_cluster,
            "entities": result.get("entities", []),
            "sentiment": result.get("sentiment", "neutral"),
            "token_estimate": get_token_estimate(result["answer"])
        }
        
        return result
    except Exception as e:
        print(f"Error generating QA pair: {e}")
        return {"question": "", "answer": "", "error": str(e)}

def generate_contextual_variations(base_qa: Dict[str, Any], variations: int = 2) -> List[Dict[str, Any]]:
    """
    Generate contextual variations of a base Q&A pair
    
    This creates similar content framed in different contexts
    for better embedding retrieval performance
    """
    results = [base_qa]
    topic = base_qa.get("topic", "") or base_qa.get("metadata", {}).get("topic", "")
    
    prompt = f"""
    Create a variation of this Q&A pair about Jon.
    Keep the core information similar but change the context, framing, or specific details.
    
    Original question: "{base_qa['question']}"
    Original answer: "{base_qa['answer']}"
    Topic: {topic}
    
    Create a new question and answer that covers similar information but:
    1. Is asked from a different perspective
    2. Focuses on a slightly different aspect of the same topic
    3. Still feels natural and conversational
    
    Format response as JSON with:
    - "question": The new variation question 
    - "answer": Jon's response in his authentic style
    - "relation_to_original": Brief explanation of how this varies
    
    Return ONLY valid JSON.
    """
    
    for i in range(variations):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.85
            )
            
            variation = json.loads(response.choices[0].message.content)
            
            # Copy metadata with variation
            variation["metadata"] = base_qa.get("metadata", {}).copy()
            variation["metadata"]["variation_of"] = base_qa.get("question", "")
            variation["metadata"]["variation_type"] = variation.get("relation_to_original", "contextual")
            variation["metadata"]["token_estimate"] = get_token_estimate(variation["answer"])
            
            results.append(variation)
            time.sleep(0.5)  # Avoid rate limits
        except Exception as e:
            print(f"Error generating variation: {e}")
    
    return results

def generate_contrastive_pair(base_qa: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a contrastive example to the base Q&A
    
    This creates related but distinctly different content
    to help embedding models better understand nuanced differences
    """
    topic = base_qa.get("topic", "") or base_qa.get("metadata", {}).get("topic", "")
    
    prompt = f"""
    Create a contrastive Q&A pair related to but distinctly different from the original.
    
    Original question: "{base_qa['question']}"
    Original answer: "{base_qa['answer']}"
    Topic: {topic}
    
    Create a new question and answer that:
    1. Relates to the same general topic area
    2. But represents a DIFFERENT perspective, opinion, or information
    3. Would help an embedding model understand nuanced differences
    
    Format response as JSON with:
    - "question": The contrastive question 
    - "answer": Jon's response in his authentic style
    - "contrast_dimension": How specifically this contrasts with the original
    
    Return ONLY valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        contrast = json.loads(response.choices[0].message.content)
        
        # Add contrast-specific metadata
        contrast["metadata"] = base_qa.get("metadata", {}).copy()
        contrast["metadata"]["contrast_to"] = base_qa.get("question", "")
        contrast["metadata"]["contrast_dimension"] = contrast.get("contrast_dimension", "semantic")
        contrast["metadata"]["is_contrastive"] = True
        contrast["metadata"]["token_estimate"] = get_token_estimate(contrast["answer"])
        
        return contrast
    except Exception as e:
        print(f"Error generating contrastive pair: {e}")
        return {"question": "", "answer": "", "error": str(e)}

def generate_conversation_turn(context: List[Dict[str, str]], topic: str, 
                              mood: Optional[str] = None) -> Dict[str, str]:
    """Generate the next turn in a conversation given the context"""
    mood = mood or random.choice(MOODS)
    
    # Format previous messages for context
    conversation_history = ""
    for msg in context:
        role = "User" if msg["role"] == "user" else "Jon"
        conversation_history += f"{role}: {msg['content']}\n"
    
    style_elements = [
        random.choice(JON_STYLE_ELEMENTS),
        random.choice(JON_STYLE_ELEMENTS)
    ]
    
    prompt = f"""
    Generate the next message in this conversation between User and Jon.
    
    Jon's persona:
    {Config.PERSONA}
    
    Jon's current mood: {mood}
    
    Conversation topic: {topic}
    
    Style requirements for Jon's response:
    - {style_elements[0]}
    - {style_elements[1]}
    
    Previous conversation:
    {conversation_history}
    
    The last message was from {'Jon' if context[-1]['role'] == 'assistant' else 'User'}.
    Now generate a message from {'User' if context[-1]['role'] == 'assistant' else 'Jon'}.
    
    Format response as JSON with:
    - "role": "{'user' if context[-1]['role'] == 'assistant' else 'assistant'}"
    - "content": the message text
    - "mood": "{mood}" (only for Jon's messages)
    - "entities": list of any specific entities mentioned
    - "sentiment": emotional tone of the message
    
    Return ONLY valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error generating conversation turn: {e}")
        return {"role": "user" if context[-1]["role"] == "assistant" else "assistant", 
                "content": "", "error": str(e)}

def generate_full_conversation(turns: int = 6, topic: Optional[str] = None,
                              initial_mood: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate a complete conversation between User and Jon"""
    topic = topic or random.choice(TOPICS)
    mood = initial_mood or random.choice(MOODS)
    
    # Start with a user message
    prompt = f"""
    Generate the first user message to start a conversation with Jon about {topic}.
    The message should be natural and conversational, as if texting a friend.
    
    Format response as JSON with:
    - "role": "user"
    - "content": the message text
    - "entities": list of any specific entities mentioned
    - "intent": the conversational intent (question, greeting, statement, etc.)
    
    Return ONLY valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        first_message = json.loads(response.choices[0].message.content)
        conversation = [first_message]
        
        # Generate remaining turns
        for i in range(turns - 1):
            # Occasionally change Jon's mood
            if conversation[-1]["role"] == "user" and random.random() < 0.2:
                mood = random.choice(MOODS)
                
            # Add small delay to avoid rate limits
            time.sleep(0.5)
            
            next_turn = generate_conversation_turn(conversation, topic, mood)
            conversation.append(next_turn)
            
        # Add conversation-level metadata
        convo_metadata = {
            "topic": topic,
            "turns": turns,
            "initial_mood": mood,
            "entities": [],
            "segment_token_counts": []
        }
        
        # Extract entities and calculate token counts for segments
        current_segment = []
        for msg in conversation:
            if "entities" in msg and msg["entities"]:
                convo_metadata["entities"].extend(msg["entities"])
            
            current_segment.append(msg["content"])
            
            # Create segments of optimal size for embeddings (300-500 tokens)
            segment_text = " ".join(current_segment)
            token_estimate = get_token_estimate(segment_text)
            
            if token_estimate >= 300:
                convo_metadata["segment_token_counts"].append(token_estimate)
                current_segment = []
        
        # Add any remaining content
        if current_segment:
            segment_text = " ".join(current_segment)
            convo_metadata["segment_token_counts"].append(get_token_estimate(segment_text))
        
        # Deduplicate entities
        convo_metadata["entities"] = list(set(convo_metadata["entities"]))
        
        # Add metadata to conversation
        for msg in conversation:
            msg["metadata"] = {
                "topic": topic,
                "token_estimate": get_token_estimate(msg["content"])
            }
            if msg["role"] == "assistant" and "mood" in msg:
                msg["metadata"]["mood"] = msg["mood"]
        
        return {
            "messages": conversation,
            "metadata": convo_metadata
        }
    except Exception as e:
        print(f"Error generating conversation: {e}")
        return {"messages": [{"role": "user", "content": "", "error": str(e)}], 
                "metadata": {"error": str(e)}}

def generate_jon_statements(count: int = 5, topic: Optional[str] = None, 
                          target_length: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate standalone Jon statements/opinions for embeddings
    
    Args:
        count: Number of statements to generate
        topic: Optional topic to focus on
        target_length: Target token length for optimal embedding
    """
    topic = topic or random.choice(TOPICS)
    
    # Find cluster
    topic_cluster = None
    for cluster, topics in TOPIC_CLUSTERS.items():
        if topic in topics:
            topic_cluster = cluster
            break
    
    # Length optimization
    length_instruction = ""
    if target_length:
        length_instruction = f"Each statement should be approximately {target_length} tokens long."
    else:
        # Default to embedding-optimized length
        length_instruction = "Each statement should be between 100-300 tokens for optimal embedding performance."
    
    prompt = f"""
    Generate {count} standalone statements or opinions from Jon about {topic}.
    
    Jon's persona:
    {Config.PERSONA}
    
    {length_instruction}
    
    Include these Jon style elements in your statements:
    - {random.choice(JON_STYLE_ELEMENTS)}
    - {random.choice(JON_STYLE_ELEMENTS)}
    
    Each statement should include:
    1. Jon's authentic voice and writing style
    2. A clear opinion or thought on the topic
    3. Personal insight that reveals his character
    
    Format response as a JSON array where each object has:
    - "statement": Jon's standalone statement
    - "subtopic": Specific aspect of the main topic addressed
    - "entities": List of any specific entities mentioned
    - "sentiment": Emotional tone (positive, negative, neutral, mixed)
    
    Return ONLY valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata for better retrieval
        statements_with_metadata = []
        if isinstance(result, list):
            for statement in result:
                statement["metadata"] = {
                    "topic": topic,
                    "topic_cluster": topic_cluster,
                    "subtopic": statement.get("subtopic", ""),
                    "entities": statement.get("entities", []),
                    "sentiment": statement.get("sentiment", "neutral"),
                    "token_estimate": get_token_estimate(statement["statement"])
                }
                statements_with_metadata.append(statement)
        else:
            # Handle case where result isn't a list
            for i in range(count):
                if f"statement{i+1}" in result:
                    statement = {
                        "statement": result[f"statement{i+1}"],
                        "metadata": {
                            "topic": topic,
                            "topic_cluster": topic_cluster,
                            "token_estimate": get_token_estimate(result[f"statement{i+1}"])
                        }
                    }
                    statements_with_metadata.append(statement)
        
        return statements_with_metadata
    except Exception as e:
        print(f"Error generating Jon statements: {e}")
        return [{"statement": "", "error": str(e)}]

def format_for_vector_store(qa_data: List[Dict[str, Any]]) -> str:
    """Format Q&A pairs for vector store (JSONL format)"""
    jsonl_data = []
    
    for qa in qa_data:
        if "question" in qa and "answer" in qa and not qa.get("error"):
            # Extract metadata
            metadata = qa.get("metadata", {})
            
            # Create vector store entry
            entry = {
                "text": f"Question: {qa['question']}\nAnswer: {qa['answer']}",
                "metadata": {
                    "type": "qa_pair",
                    "question": qa["question"],
                    "topic": metadata.get("topic", ""),
                    "topic_cluster": metadata.get("topic_cluster", ""),
                    "entities": metadata.get("entities", []),
                    "sentiment": metadata.get("sentiment", "neutral"),
                    "token_estimate": metadata.get("token_estimate", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
            jsonl_data.append(json.dumps(entry))
    
    return "\n".join(jsonl_data)

def format_for_retrieval_store(qa_data: List[Dict[str, Any]]) -> str:
    """Format Q&A pairs for the OpenAI Retrieval API (JSONL format)"""
    jsonl_data = []
    
    for qa in qa_data:
        if "question" in qa and "answer" in qa and not qa.get("error"):
            # Extract metadata
            metadata = qa.get("metadata", {})
            
            # Create retrieval store entry
            entry = {
                "text": f"Question: {qa['question']}\nAnswer: {qa['answer']}",
                "metadata": {
                    "type": "qa_pair",
                    "question": qa["question"],
                    "topic": metadata.get("topic", ""),
                    "topic_cluster": metadata.get("topic_cluster", ""),
                    "entities": metadata.get("entities", []),
                    "sentiment": metadata.get("sentiment", "neutral"),
                    "token_estimate": metadata.get("token_estimate", 0),
                    "source": "jon_data_generator",
                    "timestamp": datetime.now().isoformat()
                }
            }
            jsonl_data.append(json.dumps(entry))
    
    return "\n".join(jsonl_data)

def format_for_fine_tuning(conversations_data: List[Dict[str, Any]]) -> str:
    """Format conversations for fine-tuning (JSONL format)"""
    jsonl_data = []
    
    for convo_data in conversations_data:
        conversation = convo_data.get("messages", [])
        if conversation and not any(msg.get("error") for msg in conversation):
            # Extract conversation system message
            system_msg = f"You are Jon Schultz. {Config.PERSONA[:200]}..."
            
            # Format messages for fine-tuning
            messages = [{"role": "system", "content": system_msg}]
            
            for msg in conversation:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add to JSONL if valid conversation (need at least system + 2 messages)
            if len(messages) >= 3:
                ft_example = {"messages": messages}
                jsonl_data.append(json.dumps(ft_example))
    
    return "\n".join(jsonl_data)

def format_for_embeddings(statements_data: List[Dict[str, Any]]) -> str:
    """Format statements for embeddings (JSONL format)"""
    jsonl_data = []
    
    for statement_data in statements_data:
        if "statement" in statement_data and not statement_data.get("error"):
            # Extract metadata
            metadata = statement_data.get("metadata", {})
            
            # Create embedding entry
            entry = {
                "text": statement_data["statement"],
                "metadata": {
                    "type": "jon_statement",
                    "topic": metadata.get("topic", ""),
                    "topic_cluster": metadata.get("topic_cluster", ""),
                    "subtopic": metadata.get("subtopic", ""),
                    "entities": metadata.get("entities", []),
                    "sentiment": metadata.get("sentiment", "neutral"),
                    "token_estimate": metadata.get("token_estimate", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
            jsonl_data.append(json.dumps(entry))
    
    return "\n".join(jsonl_data)

def optimize_chunks(data_list: List[Dict[str, Any]], 
                  target_token_range: Tuple[int, int] = (300, 500)) -> List[Dict[str, Any]]:
    """
    Optimize chunks for embedding performance
    
    Args:
        data_list: List of data items with text content
        target_token_range: Target token count range (min, max)
    
    Returns:
        List of optimized chunks
    """
    min_tokens, max_tokens = target_token_range
    optimized_chunks = []
    
    current_chunk = {"text": "", "sources": [], "metadata": {}}
    current_token_count = 0
    
    for item in data_list:
        # Extract text based on item type
        if "statement" in item:
            text = item["statement"]
            item_type = "statement"
        elif "question" in item and "answer" in item:
            text = f"Q: {item['question']}\nA: {item['answer']}"
            item_type = "qa_pair"
        elif "content" in item:
            text = item["content"]
            item_type = "message"
        else:
            continue  # Skip invalid items
        
        # Get metadata
        metadata = item.get("metadata", {})
        
        # Estimate tokens
        token_estimate = metadata.get("token_estimate", get_token_estimate(text))
        
        # If item alone exceeds max tokens, split it
        if token_estimate > max_tokens:
            # Simple splitting by sentences
            sentences = text.split(". ")
            sentence_chunks = []
            temp_chunk = ""
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                sentence_with_period = sentence + "." if not sentence.endswith(".") else sentence
                temp_token_count = get_token_estimate(temp_chunk + " " + sentence_with_period)
                
                if temp_token_count <= max_tokens:
                    temp_chunk += " " + sentence_with_period
                else:
                    if temp_chunk:
                        sentence_chunks.append(temp_chunk.strip())
                    temp_chunk = sentence_with_period
            
            if temp_chunk:
                sentence_chunks.append(temp_chunk.strip())
            
            # Add each chunk
            for chunk in sentence_chunks:
                chunk_item = {
                    "text": chunk,
                    "metadata": metadata.copy(),
                    "sources": [{"id": id(item), "type": item_type}]
                }
                chunk_item["metadata"]["is_split"] = True
                chunk_item["metadata"]["token_estimate"] = get_token_estimate(chunk)
                optimized_chunks.append(chunk_item)
        
        # If adding to current chunk would exceed max, finalize current and start new
        elif current_token_count + token_estimate > max_tokens:
            if current_token_count >= min_tokens:
                optimized_chunks.append(current_chunk)
            
            # Start new chunk with this item
            current_chunk = {
                "text": text,
                "sources": [{"id": id(item), "type": item_type}],
                "metadata": metadata.copy()
            }
            current_token_count = token_estimate
        
        # Otherwise add to current chunk
        else:
            if current_chunk["text"]:
                current_chunk["text"] += "\n\n" + text
            else:
                current_chunk["text"] = text
            
            current_chunk["sources"].append({"id": id(item), "type": item_type})
            
            # Merge metadata (simplistic approach)
            if "topic" in metadata and "topic" not in current_chunk["metadata"]:
                current_chunk["metadata"]["topic"] = metadata["topic"]
            if "entities" in metadata:
                if "entities" not in current_chunk["metadata"]:
                    current_chunk["metadata"]["entities"] = []
                current_chunk["metadata"]["entities"].extend(metadata["entities"])
            
            current_token_count += token_estimate
            
            # If we've reached a good chunk size, finalize it
            if min_tokens <= current_token_count <= max_tokens:
                current_chunk["metadata"]["token_estimate"] = current_token_count
                optimized_chunks.append(current_chunk)
                current_chunk = {"text": "", "sources": [], "metadata": {}}
                current_token_count = 0
    
    # Add any remaining chunk
    if current_chunk["text"] and current_token_count > 0:
        current_chunk["metadata"]["token_estimate"] = current_token_count
        optimized_chunks.append(current_chunk)
    
    return optimized_chunks

def generate_bulk_qa_pairs(count: int, topic_cluster: str = None) -> List[Dict[str, Any]]:
    """
    Generate multiple Q&A pairs in a single API call for efficiency
    
    Args:
        count: Number of QA pairs to generate in this batch
        topic_cluster: Optional topic cluster to focus on
    """
    # Select topic cluster if not provided
    if not topic_cluster:
        topic_cluster = random.choice(list(TOPIC_CLUSTERS.keys()))
    
    # Get topics from this cluster
    topics = TOPIC_CLUSTERS[topic_cluster]
    
    # Select a sample of entities to include
    entity_categories = random.sample(list(ENTITIES.keys()), min(3, len(ENTITIES.keys())))
    selected_entities = []
    for category in entity_categories:
        selected_entities.extend(random.sample(ENTITIES[category], min(3, len(ENTITIES[category]))))
    
    # Style elements to include
    style_elements = random.sample(JON_STYLE_ELEMENTS, min(4, len(JON_STYLE_ELEMENTS)))
    facts = random.sample(JON_FACTS, min(3, len(JON_FACTS)))
    
    prompt = f"""
    Generate {count} question-answer pairs about Jon for a knowledge base.
    
    Jon's persona:
    {Config.PERSONA}
    
    Additional Jon facts:
    {facts[0]}
    {facts[1] if len(facts) > 1 else ""}
    {facts[2] if len(facts) > 2 else ""}
    
    Topic cluster: {topic_cluster}
    Topics to cover: {', '.join(topics)}
    
    Entities that could be mentioned if relevant: {', '.join(selected_entities)}
    
    Each question should be something a user might ask about Jon.
    Each answer should be in Jon's authentic voice and style.
    Answers should be between 100-300 tokens for optimal performance.
    
    Jon's style elements to incorporate:
    {' '.join([f"- {element}" for element in style_elements])}
    
    Format your response as a JSON array of objects, each with:
    - "question": The user's question
    - "answer": Jon's response in his authentic style
    - "topic": The specific topic this covers
    - "entities": List of entities mentioned
    - "sentiment": The emotional tone (positive, negative, neutral, mixed)
    - "topic_cluster": "{topic_cluster}"
    
    Your response should be ONLY a valid JSON array without explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85
        )
        
        # Track API usage
        tokens_used = response.usage.total_tokens
        track_api_call("qa_pairs", tokens_used, count)
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Check if result is already an array or if it's wrapped in an object
        if isinstance(result_json, dict) and "pairs" in result_json:
            qa_pairs = result_json["pairs"]
        elif isinstance(result_json, list):
            qa_pairs = result_json
        else:
            # Try to find an array in the response
            for key in result_json:
                if isinstance(result_json[key], list):
                    qa_pairs = result_json[key]
                    break
            else:
                raise ValueError("Could not find QA pairs array in response")
        
        # Add metadata to each pair
        for qa in qa_pairs:
            qa["metadata"] = {
                "topic": qa.get("topic", ""),
                "topic_cluster": topic_cluster,
                "entities": qa.get("entities", []),
                "sentiment": qa.get("sentiment", "neutral"),
                "token_estimate": get_token_estimate(qa["answer"])
            }
        
        return qa_pairs
    except Exception as e:
        print(f"Error generating bulk QA pairs: {e}")
        # Return empty array with specified count to avoid breaking the loop
        return [{"question": "", "answer": "", "error": str(e), "metadata": {}} for _ in range(count)]

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
            model="gpt-4o",
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
            model="gpt-4o",
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

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Generate Jon data for vector store, fine-tuning, and embeddings")
    parser.add_argument("--qa-pairs", type=int, default=100, help="Number of Q&A pairs to generate")
    parser.add_argument("--conversations", type=int, default=20, help="Number of conversations to generate")
    parser.add_argument("--statements", type=int, default=50, help="Number of standalone Jon statements to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of items per API call")
    parser.add_argument("--variations", type=int, default=2, help="Number of contextual variations per Q&A pair")
    parser.add_argument("--contrastive", action="store_true", help="Generate contrastive examples")
    parser.add_argument("--output-dir", type=str, default="data_generation/output", help="Output directory")
    parser.add_argument("--optimize-chunks", action="store_true", help="Optimize chunk sizes for embedding performance")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing for generation")
    parser.add_argument("--dynamic-batching", action="store_true", help="Dynamically adjust batch sizes")
    parser.add_argument("--enrich-metadata", action="store_true", help="Add enhanced metadata to generated items")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine batch sizes based on strategy
    if args.dynamic_batching:
        qa_batch_size = calculate_optimal_batch_size("qa", complexity_factor=1.0)
        conversation_batch_size = calculate_optimal_batch_size("conversation", complexity_factor=1.5)
        statement_batch_size = calculate_optimal_batch_size("statement", complexity_factor=0.5)
        print(f"Dynamic batch sizes: QA={qa_batch_size}, Conversations={conversation_batch_size}, Statements={statement_batch_size}")
    else:
        # Fixed batch sizes (capped at reasonable limits)
        qa_batch_size = min(args.batch_size, 10)
        conversation_batch_size = min(args.batch_size // 2, 5)  # Smaller batch for conversations
        statement_batch_size = min(args.batch_size, 10)
    
    # Results container for metrics
    generation_results = {
        "qa_metrics": {},
        "conversation_metrics": {},
        "statement_metrics": {}
    }
    
    # Generate Q&A pairs for vector store (in batches)
    print(f"Generating {args.qa_pairs} Q&A pairs...")
    qa_data = []
    
    if args.parallel and args.qa_pairs >= qa_batch_size:
        # Parallel generation
        qa_data = generate_parallel(
            generate_bulk_qa_pairs, 
            args.qa_pairs, 
            qa_batch_size
        )
    else:
        # Sequential generation
        # Process in batches
        batches = args.qa_pairs // qa_batch_size
        remainder = args.qa_pairs % qa_batch_size
        
        for i in tqdm(range(batches)):
            # Rotate through topic clusters for variety
            topic_cluster = list(TOPIC_CLUSTERS.keys())[i % len(TOPIC_CLUSTERS.keys())]
            
            # Generate batch of QA pairs
            batch_data = generate_bulk_qa_pairs(qa_batch_size, topic_cluster)
            qa_data.extend(batch_data)
            
            # Avoid rate limits
            time.sleep(1.0)
        
        # Process remainder if any
        if remainder > 0:
            topic_cluster = random.choice(list(TOPIC_CLUSTERS.keys()))
            batch_data = generate_bulk_qa_pairs(remainder, topic_cluster)
            qa_data.extend(batch_data)
    
    # Analyze QA data quality
    generation_results["qa_metrics"] = analyze_data_quality(qa_data, "qa")
    
    # Generate variations and contrastive examples if needed
    if args.variations > 0 or args.contrastive:
        print("Generating variations and contrastive examples...")
        # Use existing methods but only for a sample of the generated QA pairs
        variation_sample = random.sample(qa_data, min(len(qa_data) // 5, 20))
        
        for base_qa in tqdm(variation_sample):
            # Generate contextual variations if requested
            if args.variations > 0:
                variations = generate_contextual_variations(base_qa, args.variations)
                qa_data.extend(variations[1:])  # Skip the first one as it's the same as base_qa
            
            # Generate contrastive examples if requested
            if args.contrastive:
                contrastive = generate_contrastive_pair(base_qa)
                qa_data.append(contrastive)
            
            # Avoid rate limits
            time.sleep(0.5)
    
    # Optimize chunks if requested
    if args.optimize_chunks:
        print("Optimizing chunk sizes for embedding performance...")
        qa_data = optimize_chunks(qa_data)
    
    # Generate conversations for fine-tuning (in batches)
    print(f"\nGenerating {args.conversations} conversations...")
    conversation_data = []
    
    if args.parallel and args.conversations >= conversation_batch_size:
        # Parallel generation
        conversation_data = generate_parallel(
            generate_bulk_conversations, 
            args.conversations, 
            conversation_batch_size
        )
    else:
        # Sequential generation
        # Process in batches
        batches = args.conversations // conversation_batch_size
        remainder = args.conversations % conversation_batch_size
        
        for i in tqdm(range(batches)):
            # Generate batch of conversations
            batch_data = generate_bulk_conversations(conversation_batch_size)
            conversation_data.extend(batch_data)
            
            # Avoid rate limits
            time.sleep(1.5)
        
        # Process remainder if any
        if remainder > 0:
            batch_data = generate_bulk_conversations(remainder)
            conversation_data.extend(batch_data)
    
    # Analyze conversation data quality
    generation_results["conversation_metrics"] = analyze_data_quality(conversation_data, "conversation")
    
    # Generate standalone statements for embeddings (in batches)
    print(f"\nGenerating {args.statements} Jon statements...")
    statement_data = []
    
    if args.parallel and args.statements >= statement_batch_size:
        # Parallel generation
        statement_data = generate_parallel(
            generate_bulk_statements, 
            args.statements, 
            statement_batch_size
        )
    else:
        # Sequential generation
        # Distribute statements across topic clusters
        statements_per_cluster = max(1, args.statements // len(TOPIC_CLUSTERS))
        remaining_statements = args.statements - (statements_per_cluster * len(TOPIC_CLUSTERS))
        
        for cluster, topics in tqdm(TOPIC_CLUSTERS.items()):
            # Select random topic from this cluster
            topic = random.choice(topics)
            
            # Calculate batches for this cluster
            cluster_batches = statements_per_cluster // statement_batch_size
            cluster_remainder = statements_per_cluster % statement_batch_size
            
            for j in range(cluster_batches):
                batch_data = generate_bulk_statements(statement_batch_size, topic)
                statement_data.extend(batch_data)
                
                # Avoid rate limits
                time.sleep(1.0)
            
            # Process remainder for this cluster
            if cluster_remainder > 0:
                batch_data = generate_bulk_statements(cluster_remainder, topic)
                statement_data.extend(batch_data)
        
        # Generate any remaining statements with random topics
        if remaining_statements > 0:
            batches = remaining_statements // statement_batch_size
            remainder = remaining_statements % statement_batch_size
            
            for i in range(batches):
                topic = random.choice(TOPICS)
                batch_data = generate_bulk_statements(statement_batch_size, topic)
                statement_data.extend(batch_data)
                
                # Avoid rate limits
                time.sleep(1.0)
            
            # Process remainder if any
            if remainder > 0:
                topic = random.choice(TOPICS)
                batch_data = generate_bulk_statements(remainder, topic)
                statement_data.extend(batch_data)
    
    # Analyze statement data quality
    generation_results["statement_metrics"] = analyze_data_quality(statement_data, "statement")
    
    # Add enhanced metadata if requested
    if args.enrich_metadata:
        print("\nEnriching metadata...")
        
        # Enrich QA pairs
        for i in tqdm(range(len(qa_data)), desc="Enriching QA pairs"):
            qa_data[i] = enrich_metadata(qa_data[i], "qa", generation_results["qa_metrics"])
            
        # Enrich conversations
        for i in tqdm(range(len(conversation_data)), desc="Enriching conversations"):
            conversation_data[i] = enrich_metadata(conversation_data[i], "conversation", generation_results["conversation_metrics"])
            
        # Enrich statements
        for i in tqdm(range(len(statement_data)), desc="Enriching statements"):
            statement_data[i] = enrich_metadata(statement_data[i], "statement", generation_results["statement_metrics"])
    
    # Format and save data
    print("\nFormatting and saving data...")
    
    # Vector store data (Q&A pairs) - for backward compatibility
    vector_data = format_for_vector_store(qa_data)
    vector_file = os.path.join(args.output_dir, f"jon_vector_data_{timestamp}.jsonl")
    with open(vector_file, "w") as f:
        f.write(vector_data)
    print(f"Vector store data saved to {vector_file}")
    
    # Retrieval store data (Q&A pairs) - for OpenAI Retrieval API
    retrieval_data = format_for_retrieval_store(qa_data)
    retrieval_file = os.path.join(args.output_dir, f"jon_retrieval_data_{timestamp}.jsonl")
    with open(retrieval_file, "w") as f:
        f.write(retrieval_data)
    print(f"Retrieval store data saved to {retrieval_file}")
    
    # Fine-tuning data (conversations)
    fine_tuning_data = format_for_fine_tuning(conversation_data)
    fine_tuning_file = os.path.join(args.output_dir, f"jon_fine_tuning_{timestamp}.jsonl")
    with open(fine_tuning_file, "w") as f:
        f.write(fine_tuning_data)
    print(f"Fine-tuning data saved to {fine_tuning_file}")
    
    # Embeddings data (statements)
    embeddings_data = format_for_embeddings(statement_data)
    embeddings_file = os.path.join(args.output_dir, f"jon_embeddings_{timestamp}.jsonl")
    with open(embeddings_file, "w") as f:
        f.write(embeddings_data)
    print(f"Embeddings data saved to {embeddings_file}")
    
    # Save raw data for reference
    raw_file = os.path.join(args.output_dir, f"jon_raw_data_{timestamp}.json")
    with open(raw_file, "w") as f:
        json.dump({
            "qa_data": qa_data,
            "conversation_data": conversation_data,
            "statement_data": statement_data
        }, f, indent=2)
    print(f"Raw data saved to {raw_file}")
    
    # Save metrics data
    metrics_file = os.path.join(args.output_dir, f"jon_metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "generation_metrics": generation_results,
            "api_usage": api_calls
        }, f, indent=2)
    print(f"Metrics data saved to {metrics_file}")
    
    # Print API usage stats
    old_approach_calls = args.qa_pairs + (args.conversations * 5) + args.statements
    new_approach_calls = api_calls["batched_calls"] + api_calls["individual_calls"]
    savings_percent = ((old_approach_calls - new_approach_calls) / old_approach_calls) * 100
    
    # Calculate token savings
    estimated_old_tokens = old_approach_calls * 2000  # Rough estimate
    token_savings_percent = ((estimated_old_tokens - api_calls["total_tokens"]) / estimated_old_tokens) * 100
    
    # Data quality summary
    qa_quality = generation_results["qa_metrics"]
    topic_count = len(qa_quality.get("topic_distribution", {}))
    vocab_richness = qa_quality.get("vocabulary_richness", 0)
    redundancy = qa_quality.get("redundancy_score", 0)
    
    # Display stats in table format
    print("\n" + "="*60)
    print("Jon Data Generation Summary".center(60))
    print("="*60)
    
    # API usage table
    api_table = [
        ["API Call Type", "Count"],
        ["QA Pairs", api_calls.get("qa_pairs", 0)],
        ["Conversations", api_calls.get("conversations", 0)],
        ["Statements", api_calls.get("statements", 0)],
        ["Variations", api_calls.get("variations", 0)],
        ["Batched Calls", api_calls.get("batched_calls", 0)],
        ["Individual Calls", api_calls.get("individual_calls", 0)],
        ["Total Tokens", api_calls.get("total_tokens", 0)],
        ["Estimated Cost", f"${api_calls.get('total_cost', 0):.2f}"]
    ]
    print("\nAPI Usage:")
    print(tabulate(api_table, headers="firstrow", tablefmt="grid"))
    
    # Efficiency table
    efficiency_table = [
        ["Metric", "Value"],
        ["Traditional API Calls", old_approach_calls],
        ["Bulk Generation API Calls", new_approach_calls],
        ["API Call Reduction", f"{int(savings_percent)}%"],
        ["Token Usage Reduction", f"{int(token_savings_percent)}%"],
        ["Cost Savings", f"${(estimated_old_tokens/1000*0.06 - api_calls.get('total_cost', 0)):.2f}"]
    ]
    print("\nEfficiency Metrics:")
    print(tabulate(efficiency_table, headers="firstrow", tablefmt="grid"))
    
    # Data quality table
    quality_table = [
        ["Quality Metric", "Value"],
        ["Topic Diversity", f"{topic_count} unique topics"],
        ["Vocabulary Richness", f"{vocab_richness:.2f}"],
        ["Content Redundancy", f"{redundancy:.2f}"],
        ["Total Generated Items", len(qa_data) + len(conversation_data) + len(statement_data)]
    ]
    print("\nData Quality Metrics:")
    print(tabulate(quality_table, headers="firstrow", tablefmt="grid"))
    
    print("\nData generation complete!")
    print(f"Detailed metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main() 