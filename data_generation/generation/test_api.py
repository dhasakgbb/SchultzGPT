#!/usr/bin/env python3
"""
Test script for API call and response processing
"""

import os
import json
import time
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

def process_qa_content(content, typo_severity=0.4):
    """Extract QA pair from content"""
    # Simple string extraction for testing
    qa_pattern = r"\*\*User:\*\*\s*(.*?)\s*\*\*Jon:\*\*\s*(.*)"
    import re
    match = re.search(qa_pattern, content, re.DOTALL)
    
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        
        # Create a QA item
        return {
            "question": question,
            "answer": answer,
            "metadata": {"processed": True}
        }
    else:
        print(f"Warning: Failed to extract QA pair from: '{content[:100]}...'")
        return None

def process_conversation_content(content, typo_severity=0.4):
    """Extract conversation from content"""
    # Just return a simple structure for testing
    return {
        "messages": [{"role": "user", "content": "Test message"}],
        "metadata": {"processed": True}
    }

def process_statement_content(content, typo_severity=0.4):
    """Process statement content"""
    # Just return the content as a statement
    return {
        "statement": content.strip(),
        "metadata": {"processed": True}
    }

def test_api_call():
    """Test an API call with a simple prompt"""
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