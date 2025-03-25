#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI's API"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def convert_training_data(input_file: str, output_file: str) -> bool:
    """Convert conversation training data to OpenAI Vector Store format"""
    try:
        if not os.path.exists(input_file):
            print(f"Error: Training data file not found: {input_file}")
            return False
            
        # Read the training data file
        training_data = []
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    training_data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {i+1}")
        
        if not training_data:
            print("Error: No valid training data found in file.")
            return False
            
        print(f"Loaded {len(training_data)} training examples")
        
        # Process the training data to vector store format
        converted_items = []
        for i, item in enumerate(tqdm(training_data, desc="Processing items")):
            # Check if the item has the expected structure
            if "messages" in item and isinstance(item["messages"], list):
                # Find assistant messages in the conversation
                for j, msg in enumerate(item["messages"]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        # Create a unique ID for each message
                        message_id = f"jon-convo-{i}-{j}"
                        
                        # Get embedding for the text
                        message_text = msg["content"]
                        message_embedding = get_embedding(message_text)
                        
                        if not message_embedding:
                            print(f"Warning: Failed to get embedding for message {i}-{j}")
                            continue
                            
                        # Determine mood if possible (basic heuristic)
                        text = message_text.lower()
                        mood = "neutral"
                        if any(word in text for word in ["fuck", "shit", "damn", "wtf", "omg"]):
                            mood = "spiral"
                        elif any(word in text for word in ["sorry", "sad", "upset"]):
                            mood = "disappointed"
                        elif any(word in text for word in ["cool", "awesome", "great"]):
                            mood = "excited"
                        
                        # Add context from previous user message if available
                        context = ""
                        if j > 0 and item["messages"][j-1].get("role") == "user":
                            context = item["messages"][j-1].get("content", "")
                        
                        # Create the vector store item
                        vector_item = {
                            "id": message_id,
                            "values": message_embedding,
                            "metadata": {
                                "text": message_text,
                                "mood": mood,
                                "context": context,
                                "source": os.path.basename(input_file)
                            }
                        }
                        converted_items.append(vector_item)
            # Check if the item is already in vector format but needs to be converted
            elif all(key in item for key in ["text", "embedding"]):
                # Create a unique ID for each item
                item_id = f"jon-memory-{i}"
                
                # Determine mood if possible (basic heuristic)
                text = item["text"].lower()
                mood = "neutral"
                if any(word in text for word in ["fuck", "shit", "damn", "wtf", "omg"]):
                    mood = "spiral"
                elif any(word in text for word in ["sorry", "sad", "upset"]):
                    mood = "disappointed"
                elif any(word in text for word in ["cool", "awesome", "great"]):
                    mood = "excited"
                
                # Create the vector store item
                vector_item = {
                    "id": item_id,
                    "values": item["embedding"],
                    "metadata": {
                        "text": item["text"],
                        "mood": mood,
                        "source": os.path.basename(input_file)
                    }
                }
                converted_items.append(vector_item)
        
        # Save to file
        with open(output_file, "w") as f:
            for item in converted_items:
                f.write(json.dumps(item) + "\n")
                
        print(f"\nConverted {len(converted_items)} items to {output_file}")
        print("\nTo upload to OpenAI Vector Store, run:")
        print(f"./upload_vectors.py {output_file}")
        return True
        
    except Exception as e:
        print(f"Error converting data: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert training data or embeddings to OpenAI Vector Store format")
    parser.add_argument("input", help="Input JSONL file (training data or embeddings)")
    parser.add_argument("output", help="Output file for vector store format")
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to Vector Store format...")
    success = convert_training_data(args.input, args.output)
    
    if success:
        print("Conversion successful!")
    else:
        print("Conversion failed. Check the error messages above.")
        exit(1)

if __name__ == "__main__":
    main() 