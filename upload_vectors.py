#!/usr/bin/env python3
import json
import os
import sys
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def create_vector_store():
    """Create a new vector store"""
    try:
        vector_store = client.vector_stores.create(
            name="Jon Memory Store",
            description="Jon SchultzGPT's memories and typical responses"
        )
        print(f"Created vector store with ID: {vector_store.id}")
        return vector_store.id
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

def add_vectors_to_store(vector_store_id, jsonl_file="jon_vector_store.jsonl"):
    """Add vectors from a JSONL file to the vector store"""
    try:
        if not os.path.exists(jsonl_file):
            print(f"Error: Input file not found: {jsonl_file}")
            return False
            
        total_vectors = 0
        batch_size = 100
        
        with open(jsonl_file, "r") as f:
            batch = []
            for line in f:
                try:
                    item = json.loads(line)
                    # Format for the API
                    vector = {
                        "id": item["id"],
                        "values": item["values"],
                        "metadata": item["metadata"]
                    }
                    batch.append(vector)
                    
                    if len(batch) >= batch_size:
                        # Using files API for batch upload
                        batch_file = f"batch_{total_vectors}.json"
                        with open(batch_file, "w") as bf:
                            json.dump(batch, bf)
                        
                        # Create a file batch
                        file_batch = client.vector_stores.file_batches.create(
                            vector_store_id=vector_store_id,
                            file=batch_file
                        )
                        print(f"Created file batch with ID: {file_batch.id} for {len(batch)} vectors")
                        
                        # Clean up the temp file
                        os.remove(batch_file)
                        
                        total_vectors += len(batch)
                        batch = []
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in file: {e}")
                    continue
            
            # Process any remaining vectors
            if batch:
                batch_file = f"batch_final.json"
                with open(batch_file, "w") as bf:
                    json.dump(batch, bf)
                
                file_batch = client.vector_stores.file_batches.create(
                    vector_store_id=vector_store_id,
                    file=batch_file
                )
                print(f"Created final file batch with ID: {file_batch.id} for {len(batch)} vectors")
                
                # Clean up the temp file
                os.remove(batch_file)
                
                total_vectors += len(batch)
        
        print(f"Total vectors uploaded: {total_vectors}")
        return True
    except Exception as e:
        print(f"Error adding vectors: {str(e)}")
        return False

if __name__ == "__main__":
    # Check for input file from command line
    input_file = "jon_vector_store.jsonl"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Using input file: {input_file}")
    
    # Create a new vector store or use an existing one
    existing_id = os.environ.get("VECTOR_STORE_ID")
    
    if existing_id:
        print(f"Using existing vector store with ID: {existing_id}")
        vector_store_id = existing_id
    else:
        print("Creating a new vector store...")
        vector_store_id = create_vector_store()
        
    if vector_store_id:
        print(f"Adding vectors to store: {vector_store_id}")
        if add_vectors_to_store(vector_store_id, input_file):
            print("\nTo use this vector store in your app, add this to your .env file:")
            print(f"VECTOR_STORE_ID={vector_store_id}")
        else:
            print("Failed to upload vectors.") 