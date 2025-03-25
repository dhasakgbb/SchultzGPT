"""
Retrieval API service for SchultzGPT.
Uses OpenAI's Retrieval API for storing and retrieving conversation context.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime

from openai import OpenAI
from config.config import Config

class RetrievalStore:
    """OpenAI Retrieval API integration for SchultzGPT's memory system."""
    
    def __init__(self, 
                 assistant_id: Optional[str] = None,
                 file_ids: Optional[List[str]] = None,
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the retrieval store.
        
        Args:
            assistant_id: Optional assistant ID to use (if None, will create a new one)
            file_ids: Optional list of file IDs already uploaded to OpenAI
            embedding_model: Model to use for embeddings
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.assistant_id = assistant_id or os.environ.get("OPENAI_ASSISTANT_ID")
        self.file_ids = file_ids or []
        self.available = False
        
        # Initialize the API connection
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the connection and verify availability."""
        try:
            # If we don't have an assistant ID, create a new assistant
            if not self.assistant_id:
                self._create_assistant()
            else:
                # Verify the assistant exists
                try:
                    self.client.beta.assistants.retrieve(self.assistant_id)
                except Exception:
                    self._create_assistant()
            
            # Check if we have files
            if self.file_ids:
                for file_id in self.file_ids:
                    try:
                        self.client.files.retrieve(file_id)
                    except Exception as e:
                        print(f"File ID {file_id} not found: {e}")
                        self.file_ids.remove(file_id)
            
            # If we have an assistant and at least one file, we're available
            self.available = bool(self.assistant_id and self.file_ids)
            
        except Exception as e:
            print(f"Error initializing retrieval store: {e}")
            self.available = False
    
    def _create_assistant(self) -> None:
        """Create a new assistant for retrievals."""
        try:
            assistant = self.client.beta.assistants.create(
                name="SchultzGPT Memory",
                description="Assistant for handling SchultzGPT's memory",
                model=Config.FINE_TUNED_MODEL,
                tools=[{"type": "retrieval"}]
            )
            self.assistant_id = assistant.id
            print(f"Created new assistant with ID: {self.assistant_id}")
            
            # Save assistant ID to .env file if possible
            self._save_assistant_id_to_env()
        except Exception as e:
            print(f"Error creating assistant: {e}")
            self.assistant_id = None
    
    def _save_assistant_id_to_env(self) -> None:
        """Save the assistant ID to .env file if it exists."""
        if not self.assistant_id:
            return
            
        try:
            # Check if .env file exists
            if os.path.exists(".env"):
                # Read current .env file
                with open(".env", "r") as f:
                    env_content = f.read()
                
                # Check if OPENAI_ASSISTANT_ID is already in the file
                if "OPENAI_ASSISTANT_ID=" in env_content:
                    # Replace the existing assistant ID
                    new_env_content = []
                    for line in env_content.split("\n"):
                        if line.startswith("OPENAI_ASSISTANT_ID="):
                            new_env_content.append(f"OPENAI_ASSISTANT_ID={self.assistant_id}")
                        else:
                            new_env_content.append(line)
                    
                    # Write updated content
                    with open(".env", "w") as f:
                        f.write("\n".join(new_env_content))
                else:
                    # Append the assistant ID to the file
                    with open(".env", "a") as f:
                        f.write(f"\n# Added automatically by SchultzGPT\nOPENAI_ASSISTANT_ID={self.assistant_id}\n")
                
                print(f"Saved Assistant ID to .env file: {self.assistant_id}")
        except Exception as e:
            print(f"Error saving Assistant ID to .env file: {e}")
    
    def save_store(self) -> bool:
        """
        Save the current state of the retrieval store.
        
        For the Retrieval API implementation, this saves the assistant ID
        and file IDs to a local JSON file for reference and backup.
        
        Returns:
            Success status
        """
        try:
            # Create a state object with assistant and file information
            state = {
                "assistant_id": self.assistant_id,
                "file_ids": self.file_ids,
                "embedding_model": self.embedding_model,
                "last_updated": datetime.now().isoformat(),
                "available": self.available
            }
            
            # Ensure the directory exists
            os.makedirs(".retrieval_store", exist_ok=True)
            
            # Save to a JSON file
            with open(".retrieval_store/state.json", "w") as f:
                json.dump(state, f, indent=2)
                
            # Also make sure the assistant ID is saved in .env
            self._save_assistant_id_to_env()
                
            return True
        except Exception as e:
            print(f"Error saving retrieval store state: {e}")
            return False
    
    def add_text(self, 
                text: str, 
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a text and its metadata to the retrieval store.
        
        Args:
            text: The text to store
            metadata: Optional metadata to store with the text
            
        Returns:
            Success status
        """
        try:
            # Create a temporary JSON file with the text and metadata
            timestamp = datetime.now().isoformat()
            temp_file_path = f".temp_retrieval_{int(time.time())}.json"
            
            # Prepare content with metadata
            content = {
                "text": text,
                "metadata": metadata or {},
                "timestamp": timestamp
            }
            
            # Write to temp file
            with open(temp_file_path, "w") as f:
                json.dump(content, f)
            
            # Upload the file to OpenAI
            with open(temp_file_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            # Add the file to the assistant
            if self.assistant_id:
                self.client.beta.assistants.update(
                    assistant_id=self.assistant_id,
                    file_ids=[*self.file_ids, file.id]
                )
            
            # Add to our file IDs
            self.file_ids.append(file.id)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # Set available to true since we have files and an assistant
            self.available = bool(self.assistant_id and self.file_ids)
            
            return True
        except Exception as e:
            print(f"Error adding text to retrieval store: {e}")
            return False
    
    def add_batch(self, 
                 texts: List[str], 
                 metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add a batch of texts to the retrieval store.
        
        Args:
            texts: List of texts to store
            metadatas: Optional list of metadata dicts
            
        Returns:
            Success status
        """
        if not texts:
            return True
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        try:
            # Create a temporary JSON file with all texts and metadata
            timestamp = datetime.now().isoformat()
            temp_file_path = f".temp_retrieval_batch_{int(time.time())}.jsonl"
            
            # Write each item as a JSON line
            with open(temp_file_path, "w") as f:
                for i, text in enumerate(texts):
                    content = {
                        "text": text,
                        "metadata": metadatas[i],
                        "timestamp": timestamp
                    }
                    f.write(json.dumps(content) + "\n")
            
            # Upload the file to OpenAI
            with open(temp_file_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            # Add the file to the assistant
            if self.assistant_id:
                self.client.beta.assistants.update(
                    assistant_id=self.assistant_id,
                    file_ids=[*self.file_ids, file.id]
                )
            
            # Add to our file IDs
            self.file_ids.append(file.id)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # Set available to true since we have files and an assistant
            self.available = bool(self.assistant_id and self.file_ids)
            
            return True
        except Exception as e:
            print(f"Error adding batch to retrieval store: {e}")
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 5,
              filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        Search the retrieval store for texts similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of results with text, metadata, and similarity score
        """
        if not self.available or not self.assistant_id or not self.file_ids:
            return []
            
        try:
            # Create a thread
            thread = self.client.beta.threads.create()
            
            # Add the query message to the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Search for context related to: {query}"
            )
            
            # Run the assistant on the thread
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id,
                instructions=f"Search for information related to: {query}. Return results as JSON with text and metadata."
            )
            
            # Wait for the run to complete
            while run.status in ["queued", "in_progress"]:
                time.sleep(0.5)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            # If the run completed, get the messages
            if run.status == "completed":
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                # Extract the results from the messages
                results = []
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if content.type == "text":
                                # Try to parse as JSON, or just use the text
                                try:
                                    text = content.text.value
                                    
                                    # Apply filter if provided
                                    result = {
                                        "text": text,
                                        "metadata": {},
                                        "score": 1.0  # Default score since OpenAI doesn't provide scores
                                    }
                                    
                                    if filter_fn is None or filter_fn(result):
                                        results.append(result)
                                except Exception as e:
                                    print(f"Error parsing message: {e}")
                
                # Limit to top_k
                return results[:top_k]
            
            return []
        except Exception as e:
            print(f"Error searching retrieval store: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval store."""
        stats = {
            "available": self.available,
            "assistant_id": self.assistant_id,
            "file_count": len(self.file_ids),
            "embedding_model": self.embedding_model
        }
        return stats
    
    def clear_store(self) -> bool:
        """Clear all files from the retrieval store."""
        try:
            # Delete each file
            for file_id in self.file_ids:
                try:
                    self.client.files.delete(file_id)
                except Exception as e:
                    print(f"Error deleting file {file_id}: {e}")
            
            # Clear our file list
            self.file_ids = []
            
            # Update the assistant
            if self.assistant_id:
                self.client.beta.assistants.update(
                    assistant_id=self.assistant_id,
                    file_ids=[]
                )
            
            self.available = bool(self.assistant_id)
            
            return True
        except Exception as e:
            print(f"Error clearing retrieval store: {e}")
            return False
    
    def load_from_jsonl(self, file_path: str) -> bool:
        """
        Load data from a JSONL file into the retrieval store.
        
        Each line should be a JSON object with 'text' and optional 'metadata'.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Success status
        """
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            
            # Upload the file directly to OpenAI
            with open(file_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            # Add the file to the assistant
            if self.assistant_id:
                self.client.beta.assistants.update(
                    assistant_id=self.assistant_id,
                    file_ids=[*self.file_ids, file.id]
                )
            
            # Add to our file IDs
            self.file_ids.append(file.id)
            
            # Set available to true since we have files and an assistant
            self.available = bool(self.assistant_id and self.file_ids)
            
            return True
        except Exception as e:
            print(f"Error loading data from JSONL: {e}")
            return False 