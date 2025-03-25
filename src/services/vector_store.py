"""
Vector store service for SchultzGPT.
Handles storing and retrieving embeddings and associated metadata.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

import numpy as np
from openai.types.chat import ChatCompletion

from models.state import ResponseCache
from services.openai import embeddings_create, async_embeddings_create, run_async


class VectorStore:
    """Vector storage and retrieval for message embeddings."""
    
    def __init__(self, 
                 store_dir: str = ".vector_store", 
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the vector store.
        
        Args:
            store_dir: Directory to store vector data
            embedding_model: Model to use for embeddings
        """
        self.store_dir = store_dir
        self.embedding_model = embedding_model
        self.vectors = []
        self.metadata = []
        self.texts = []
        self.available = False
        self._ensure_store_dir()
        self.load_store()
        
    def _ensure_store_dir(self) -> None:
        """Ensure the vector store directory exists."""
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir, exist_ok=True)
    
    def load_store(self) -> bool:
        """Load the vector store from disk."""
        store_path = os.path.join(self.store_dir, "vector_store.json")
        
        if not os.path.exists(store_path):
            self.available = False
            return False
            
        try:
            with open(store_path, 'r') as f:
                data = json.load(f)
                
            self.vectors = data.get('vectors', [])
            self.metadata = data.get('metadata', [])
            self.texts = data.get('texts', [])
            
            self.available = len(self.vectors) > 0
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.available = False
            return False
    
    def save_store(self) -> bool:
        """Save the vector store to disk."""
        store_path = os.path.join(self.store_dir, "vector_store.json")
        
        try:
            data = {
                'vectors': self.vectors,
                'metadata': self.metadata,
                'texts': self.texts
            }
            
            with open(store_path, 'w') as f:
                json.dump(data, f)
                
            return True
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            return False
    
    def add_text(self, 
                text: str, 
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a text and its embedding to the vector store.
        
        Args:
            text: The text to embed and store
            metadata: Optional metadata to store with the embedding
            
        Returns:
            Success status
        """
        try:
            # Create embedding
            embedding = embeddings_create(text, model=self.embedding_model)[0]
            
            # Add to store
            self.vectors.append(embedding)
            self.metadata.append(metadata or {})
            self.texts.append(text)
            
            # Update availability
            self.available = True
            
            # Save periodically (every 5 additions)
            if len(self.vectors) % 5 == 0:
                self.save_store()
                
            return True
        except Exception as e:
            print(f"Error adding text to vector store: {str(e)}")
            return False
    
    async def add_text_async(self, 
                           text: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a text and its embedding to the vector store asynchronously.
        
        Args:
            text: The text to embed and store
            metadata: Optional metadata to store with the embedding
            
        Returns:
            Success status
        """
        try:
            # Create embedding asynchronously
            embedding = (await async_embeddings_create(text, model=self.embedding_model))[0]
            
            # Add to store
            self.vectors.append(embedding)
            self.metadata.append(metadata or {})
            self.texts.append(text)
            
            # Update availability
            self.available = True
            
            # Save periodically (every 5 additions)
            if len(self.vectors) % 5 == 0:
                self.save_store()
                
            return True
        except Exception as e:
            print(f"Error adding text to vector store asynchronously: {str(e)}")
            return False
    
    def add_batch(self, 
                 texts: List[str], 
                 metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add a batch of texts and their embeddings to the vector store.
        
        Args:
            texts: List of texts to embed and store
            metadatas: Optional list of metadata dicts
            
        Returns:
            Success status
        """
        if not texts:
            return True
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        try:
            # Create embeddings
            embeddings = embeddings_create(texts, model=self.embedding_model)
            
            # Add to store
            for i, embedding in enumerate(embeddings):
                self.vectors.append(embedding)
                self.metadata.append(metadatas[i])
                self.texts.append(texts[i])
            
            # Update availability
            self.available = True
            
            # Save store
            self.save_store()
                
            return True
        except Exception as e:
            print(f"Error adding batch to vector store: {str(e)}")
            return False
    
    @run_async
    async def add_batch_async(self, 
                            texts: List[str], 
                            metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add a batch of texts and their embeddings to the vector store asynchronously.
        
        Args:
            texts: List of texts to embed and store
            metadatas: Optional list of metadata dicts
            
        Returns:
            Success status
        """
        if not texts:
            return True
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        try:
            # Create embeddings asynchronously
            embeddings = await async_embeddings_create(texts, model=self.embedding_model)
            
            # Add to store
            for i, embedding in enumerate(embeddings):
                self.vectors.append(embedding)
                self.metadata.append(metadatas[i])
                self.texts.append(texts[i])
            
            # Update availability
            self.available = True
            
            # Save store
            self.save_store()
                
            return True
        except Exception as e:
            print(f"Error adding batch to vector store asynchronously: {str(e)}")
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 5,
              filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store for texts similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of results with text, metadata, and similarity score
        """
        if not self.available or not self.vectors:
            return []
            
        try:
            # Create query embedding
            query_embedding = embeddings_create(query, model=self.embedding_model)[0]
            
            # Compute similarities and filter
            similarities = self._compute_similarities(query_embedding)
            results = self._process_search_results(similarities, top_k, filter_fn)
            
            return results
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []
    
    @run_async
    async def search_async(self, 
                         query: str, 
                         top_k: int = 5,
                         filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store for texts similar to the query asynchronously.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of results with text, metadata, and similarity score
        """
        if not self.available or not self.vectors:
            return []
            
        try:
            # Create query embedding asynchronously
            query_embedding = (await async_embeddings_create(query, model=self.embedding_model))[0]
            
            # Compute similarities and filter
            similarities = self._compute_similarities(query_embedding)
            results = self._process_search_results(similarities, top_k, filter_fn)
            
            return results
        except Exception as e:
            print(f"Error searching vector store asynchronously: {str(e)}")
            return []
    
    def _compute_similarities(self, query_embedding: List[float]) -> List[float]:
        """Compute cosine similarities between query embedding and all vectors."""
        # Convert query embedding to numpy array
        query_array = np.array(query_embedding)
        
        # Compute similarities
        similarities = []
        for vec in self.vectors:
            # Compute cosine similarity
            vec_array = np.array(vec)
            dot_product = np.dot(query_array, vec_array)
            norm_product = np.linalg.norm(query_array) * np.linalg.norm(vec_array)
            similarity = dot_product / norm_product if norm_product != 0 else 0
            similarities.append(similarity)
            
        return similarities
    
    def _process_search_results(self, 
                               similarities: List[float],
                               top_k: int,
                               filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """Process similarities and metadata to create search results."""
        # Create indices and zip with similarities
        indices = list(range(len(similarities)))
        results = list(zip(indices, similarities))
        
        # Apply filter if provided
        if filter_fn is not None:
            results = [(i, score) for i, score in results if filter_fn(self.metadata[i])]
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        results = results[:top_k]
        
        # Format results
        formatted_results = []
        for i, score in results:
            formatted_results.append({
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "score": score
            })
            
        return formatted_results
    
    def clear(self) -> bool:
        """Clear the vector store."""
        try:
            self.vectors = []
            self.metadata = []
            self.texts = []
            self.available = False
            
            # Save empty store
            self.save_store()
            
            return True
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")
            return False
    
    def delete_by_filter(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> int:
        """
        Delete entries based on a filter function.
        
        Args:
            filter_fn: Function that returns True for entries to delete
            
        Returns:
            Number of deleted entries
        """
        if not self.available:
            return 0
            
        try:
            # Find indices to keep
            keep_indices = [i for i, meta in enumerate(self.metadata) if not filter_fn(meta)]
            
            # Count deleted
            deleted_count = len(self.vectors) - len(keep_indices)
            
            # Keep only filtered entries
            self.vectors = [self.vectors[i] for i in keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.texts = [self.texts[i] for i in keep_indices]
            
            # Update availability
            self.available = len(self.vectors) > 0
            
            # Save store
            self.save_store()
            
            return deleted_count
        except Exception as e:
            print(f"Error deleting from vector store: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "available": self.available,
            "entry_count": len(self.vectors),
            "embedding_model": self.embedding_model
        } 