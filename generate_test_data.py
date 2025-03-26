#!/usr/bin/env python
"""
Test Data Generator Wrapper

This script sets up the Python path correctly and generates a small amount of test data
to verify the functionality of the generation system.
"""

import os
import sys
import json
import random
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure OpenAI API
from dotenv import load_dotenv
load_dotenv()

# Now import the jon_data_generator module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_generation', 'generation'))
import jon_data_generator

def main():
    """Generate a minimal set of test data"""
    print("Generating test data...")
    
    # Set parameters for minimal data generation
    num_qa_pairs = 3
    num_conversations = 1
    num_statements = 2
    
    # Generate QA pairs
    print(f"Generating {num_qa_pairs} QA pairs...")
    qa_pairs = []
    for i in range(num_qa_pairs):
        topic = random.choice(jon_data_generator.TOPICS)
        qa_pair = jon_data_generator.generate_qa_pair(topic=topic)
        qa_pairs.append(qa_pair)
    
    # Generate a conversation
    print(f"Generating {num_conversations} conversation...")
    conversations = []
    for i in range(num_conversations):
        topic = random.choice(jon_data_generator.TOPICS)
        conversation = jon_data_generator.generate_full_conversation(turns=4, topic=topic)
        conversations.append(conversation)
    
    # Generate statements
    print(f"Generating {num_statements} statements...")
    statements = []
    for i in range(num_statements):
        topic = random.choice(jon_data_generator.TOPICS)
        batch = jon_data_generator.generate_jon_statements(count=1, topic=topic)
        statements.extend(batch)
    
    # Save to output directory
    output_dir = os.path.join("data_generation", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save QA pairs
    qa_path = os.path.join(output_dir, f"test_qa_pairs_{timestamp}.json")
    with open(qa_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    # Save conversations
    conv_path = os.path.join(output_dir, f"test_conversations_{timestamp}.json")
    with open(conv_path, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    # Save statements
    stmt_path = os.path.join(output_dir, f"test_statements_{timestamp}.json")
    with open(stmt_path, 'w') as f:
        json.dump(statements, f, indent=2)
    
    print(f"Data saved to {output_dir}:")
    print(f"- QA pairs: {qa_path}")
    print(f"- Conversations: {conv_path}")
    print(f"- Statements: {stmt_path}")

if __name__ == "__main__":
    main() 