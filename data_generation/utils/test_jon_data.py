#!/usr/bin/env python
"""
Test Jon Data

This script allows you to test and explore the synthetic Jon data generated 
by jon_data_generator.py. Supports both vector store and retrieval API formats.

Usage:
    python -m data_generation.test_jon_data --file path/to/data.jsonl
"""

import os
import sys
import json
import argparse
import random
from pprint import pprint
import time

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from openai import OpenAI

# Optional import for retrieval testing
try:
    from src.services.retrieval_store import RetrievalStore
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False

# Load environment variables
load_dotenv()

def display_qa_pair(qa_pair):
    """Format and display a Q&A pair"""
    print("\n" + "=" * 50)
    print(f"[USER QUESTION]")
    
    # Handle different formats
    if isinstance(qa_pair, dict):
        if "question" in qa_pair and "answer" in qa_pair:
            # Direct QA format
            print(f"{qa_pair['question']}")
            print("\n[JON'S ANSWER]")
            print(f"{qa_pair['answer']}")
        elif "text" in qa_pair and "metadata" in qa_pair:
            # Retrieval/vector store format
            text = qa_pair["text"]
            if text.startswith("Question:") and "Answer:" in text:
                # Split the text into question and answer
                parts = text.split("Answer:", 1)
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip() if len(parts) > 1 else "No answer provided"
                
                print(f"{question}")
                print("\n[JON'S ANSWER]")
                print(f"{answer}")
                
                # Display metadata if verbose
                metadata = qa_pair.get("metadata", {})
                if metadata:
                    print("\n[METADATA]")
                    for key, value in metadata.items():
                        if key != "question" and key != "answer":
                            print(f"{key}: {value}")
            else:
                # Just display the text
                print(f"{text}")
    print("=" * 50)

def display_conversation(conversation):
    """Format and display a conversation"""
    print("\n" + "=" * 50)
    print("CONVERSATION")
    print("=" * 50)
    
    for msg in conversation:
        role = "User" if msg["role"] == "user" else "Jon"
        mood = f" [{msg.get('mood', '')}]" if "mood" in msg and msg["role"] == "assistant" else ""
        print(f"\n{role}{mood}:")
        print(f"{msg['content']}")
    
    print("=" * 50)

def display_fine_tuning_example(example):
    """Format and display a fine-tuning example"""
    print("\n" + "=" * 50)
    print("FINE-TUNING EXAMPLE")
    print("=" * 50)
    
    messages = example["messages"]
    print("[SYSTEM]")
    print(messages[0]["content"][:100] + "..." if len(messages[0]["content"]) > 100 else messages[0]["content"])
    
    print("\n[USER]")
    print(messages[1]["content"])
    
    print("\n[ASSISTANT]")
    print(messages[2]["content"])
    
    print("=" * 50)

def display_statement(statement):
    """Format and display a Jon statement"""
    print("\n" + "=" * 50)
    print(f"JON STATEMENT:")
    print(f"{statement}")
    print("=" * 50)

def test_with_model(client, fine_tuned_model, user_message):
    """Test a user message with the fine-tuned model"""
    if not fine_tuned_model:
        print("No fine-tuned model specified. Use --model parameter.")
        return
    
    print(f"\nTesting with model: {fine_tuned_model}")
    print(f"User message: {user_message}")
    print("Generating response...")
    
    try:
        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": "You are simulating a conversation as Jon with the user. Jon is a witty, intelligent but sometimes cynical friend. He texts in a casual style, often using lowercase and minimal punctuation."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8
        )
        
        print("\nJon's response:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error generating response: {e}")

def test_with_retrieval(client, assistant_id, user_message):
    """Test a user message with the retrieval-backed assistant"""
    if not assistant_id:
        print("No assistant ID specified. Use --assistant-id parameter.")
        return
    
    print(f"\nTesting with assistant: {assistant_id}")
    print(f"User message: {user_message}")
    print("Generating response...")
    
    try:
        # Create a thread
        thread = client.beta.threads.create()
        
        # Add the user message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )
        
        # Run the assistant on the thread
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="You are simulating a conversation as Jon with the user. Jon is a witty, intelligent but sometimes cynical friend. He texts in a casual style, often using lowercase and minimal punctuation."
        )
        
        # Wait for the run to complete
        print("Waiting for assistant response...")
        while run.status in ["queued", "in_progress"]:
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
        # Get the response
        if run.status == "completed":
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Print the assistant's response
            for msg in messages.data:
                if msg.role == "assistant":
                    print("\nJon's response:")
                    for content in msg.content:
                        if content.type == "text":
                            print(content.text.value)
        else:
            print(f"Run failed with status: {run.status}")
            if run.status == "failed":
                print(f"Failure reason: {run.last_error}")
    
    except Exception as e:
        print(f"Error generating response: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Jon synthetic data")
    parser.add_argument("--file", type=str, required=True,
                       help="Path to the data file to test (JSONL or JSON)")
    parser.add_argument("--type", type=str, choices=["qa", "conversation", "fine-tuning", "statement", "retrieval"],
                       help="Type of data to test (auto-detected if not specified)")
    parser.add_argument("--count", type=int, default=5,
                       help="Number of examples to display")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode to test with messages")
    parser.add_argument("--model", type=str, default=None,
                       help="Fine-tuned model to test with")
    parser.add_argument("--assistant-id", type=str, default=None,
                       help="OpenAI Assistant ID to test with (for retrieval testing)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Determine file type if not specified
    if not args.type:
        # Auto-detect from file contents
        with open(args.file, 'r') as f:
            first_line = f.readline().strip()
            try:
                data = json.loads(first_line)
                
                if isinstance(data, list):
                    args.type = "conversation"
                elif "messages" in data:
                    args.type = "fine-tuning"
                elif "question" in data and "answer" in data:
                    args.type = "qa"
                elif "text" in data and "metadata" in data:
                    # Could be retrieval or vector store format
                    if "source" in data.get("metadata", {}) and data["metadata"].get("source") == "jon_data_generator":
                        args.type = "retrieval"
                    else:
                        args.type = "qa"  # Default to qa for vector store format
                elif isinstance(data, str):
                    args.type = "statement"
                else:
                    print(f"Could not determine data type. Please specify with --type")
                    sys.exit(1)
                    
            except json.JSONDecodeError:
                print(f"File does not contain valid JSON. Please check format.")
                sys.exit(1)
                
        print(f"Detected data type: {args.type}")
    
    # Load data based on type
    data = []
    
    if args.file.endswith(".jsonl"):
        with open(args.file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    pass
    else:
        with open(args.file, 'r') as f:
            data = json.load(f)
    
    # Display sample data
    if not data:
        print("No valid data found in file")
        sys.exit(1)
        
    print(f"Found {len(data)} items in file")
    
    # Show sample based on type
    samples = random.sample(data, min(args.count, len(data)))
    
    for sample in samples:
        if args.type == "qa":
            display_qa_pair(sample)
        elif args.type == "retrieval":
            display_qa_pair(sample)  # Use same display function but formats differently
        elif args.type == "conversation":
            display_conversation(sample)
        elif args.type == "fine-tuning":
            display_fine_tuning_example(sample)
        elif args.type == "statement":
            if isinstance(sample, str):
                display_statement(sample)
            elif "text" in sample:
                display_statement(sample["text"])
    
    # Interactive mode
    if args.interactive:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Handle different interactive modes
        if args.assistant_id:
            # Test with Retrieval API assistant
            print(f"\nEntering interactive mode with Assistant {args.assistant_id}. Type 'exit' to quit.")
            while True:
                user_message = input("\nYou: ")
                if user_message.lower() in ["exit", "quit"]:
                    break
                    
                test_with_retrieval(client, args.assistant_id, user_message)
                
        elif args.model:
            # Test with fine-tuned model
            print(f"\nEntering interactive mode with model {args.model}. Type 'exit' to quit.")
            while True:
                user_message = input("\nYou: ")
                if user_message.lower() in ["exit", "quit"]:
                    break
                    
                test_with_model(client, args.model, user_message)
        else:
            print("For interactive mode, specify either --model or --assistant-id.")

if __name__ == "__main__":
    main() 