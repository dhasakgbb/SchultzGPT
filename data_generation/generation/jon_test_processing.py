#!/usr/bin/env python3
"""
Simple test processing for Jon data generation
"""

import os
import json
import re
import random
import traceback
from openai import OpenAI
from dotenv import load_dotenv

def apply_jon_texting_style(text, typo_severity=0.4):
    """
    Apply Jon's texting style and dyslexic typos to text.
    
    Args:
        text: Text to modify
        typo_severity: How severe the typos should be (0.0-1.0)
        
    Returns:
        Modified text with Jon's style
    """
    if not text or typo_severity <= 0:
        return text
        
    # Define common misspellings for Jon
    common_misspellings = {
        'because': ['becuase', 'becuz', 'cuz'],
        'really': ['realy', 'relly'],
        'with': ['wiht', 'wit'],
        'just': ['jsut', 'jst'],
        'think': ['tink', 'thnik'],
        'though': ['tho', 'thou'],
    }
    
    # Lowercase transformation (Jon rarely uses uppercase)
    modified_text = text.lower()
    
    # Simple modifications for testing
    words = modified_text.split()
    for i, word in enumerate(words):
        # Check if this is a word Jon commonly misspells
        lower_word = word.lower()
        if lower_word in common_misspellings and random.random() < typo_severity:
            words[i] = random.choice(common_misspellings[lower_word])
        
    modified_text = ' '.join(words)
    
    # Add 'haha' or 'lol' occasionally
    if random.random() < typo_severity * 0.3:
        if not modified_text.endswith(('haha', 'lol')):
            if random.random() < 0.5:
                modified_text += ' haha'
            else:
                modified_text += ' lol'
    
    return modified_text

def process_qa_content(content, typo_severity=0.4):
    """Extract QA pair from content"""
    # Extract QA pair using regex
    qa_pattern = r"\*\*User:\*\*\s*(.*?)\s*\*\*Jon:\*\*\s*(.*)"
    match = re.search(qa_pattern, content, re.DOTALL)
    
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        
        # Apply Jon's texting style
        answer_with_typos = apply_jon_texting_style(answer, typo_severity)
        
        # Create a QA item
        return {
            "question": question,
            "answer": answer_with_typos,
            "metadata": {}
        }
    else:
        print(f"Warning: Failed to extract QA pair from: '{content[:100]}...'")
        # Fallback
        parts = content.split('\n\n', 1)
        if len(parts) >= 2:
            return {
                "question": parts[0],
                "answer": apply_jon_texting_style(parts[1], typo_severity),
                "metadata": {"fallback": True}
            }
        return None

def process_conversation_content(content, typo_severity=0.4):
    """Extract conversation from content"""
    # Extract conversation messages using regex
    message_pattern = r"\*\*(User|Jon):\*\*\s*(.*?)(?=\n\*\*(?:User|Jon):|$)"
    matches = re.findall(message_pattern, content, re.DOTALL)
    
    messages = []
    for role, text in matches:
        # Map roles to OpenAI format
        role_mapped = "user" if role.lower() == "user" else "assistant"
        
        # Apply Jon's texting style to his messages
        message_text = text.strip()
        if role_mapped == "assistant":
            message_text = apply_jon_texting_style(message_text, typo_severity)
        
        messages.append({
            "role": role_mapped,
            "content": message_text
        })
    
    if not messages:
        # Fallback extraction
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": line})
    
    # Ensure conversation starts with user
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Hey Jon, how's it going?"})
    
    if not messages:
        print("Warning: Failed to extract conversation messages")
        return None
    
    return {
        "messages": messages,
        "metadata": {}
    }

def process_statement_content(content, typo_severity=0.4):
    """Process statement content"""
    # Apply Jon's texting style
    statement = apply_jon_texting_style(content.strip(), typo_severity)
    
    return {
        "statement": statement,
        "metadata": {}
    }

def test_api_call():
    """Test an API call with a simple prompt"""
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG_ID"),
        default_headers={
            "OpenAI-Organization": os.environ.get("OPENAI_ORG_ID")
        }
    )
    
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    prompt = "Generate a short conversation between a user and Jon about his cats."
    
    try:
        print(f"Making API call to {OPENAI_MODEL}...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "text"}
        )
        
        print("\nAPI Response:")
        print(f"Number of choices: {len(response.choices)}")
        
        # Extract and process content
        content = response.choices[0].message.content
        print(f"\nResponse content (first 200 chars):")
        print(f"{content[:200]}...")
        
        # Process as QA
        qa_result = process_qa_content(content)
        if qa_result:
            print("\nProcessed as QA:")
            print(f"Question: {qa_result['question'][:50]}...")
            print(f"Answer: {qa_result['answer'][:50]}...")
        else:
            print("\nFailed to process as QA")
        
        # Process as conversation
        conv_result = process_conversation_content(content)
        if conv_result:
            print("\nProcessed as conversation:")
            print(f"Number of messages: {len(conv_result['messages'])}")
            for i, msg in enumerate(conv_result['messages']):
                print(f"Message {i+1} ({msg['role']}): {msg['content'][:30]}...")
        
        # Process as statement
        stmt_result = process_statement_content(content)
        print("\nProcessed as statement:")
        print(f"Statement: {stmt_result['statement'][:50]}...")
        
        return True
    except Exception as e:
        print(f"Error making API call: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_call()
    print(f"\nTest {'succeeded' if success else 'failed'}") 