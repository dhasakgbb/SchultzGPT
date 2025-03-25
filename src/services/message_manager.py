"""
Message Manager service for SchultzGPT.
Manages conversation history, vector storage, context retrieval, and conversation segmentation.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json

from models.message import Message, MessageHistory
from models.state import SchultzState
from services.vector_store import VectorStore
from services.openai import chat_completion, async_chat_completion, run_async


class ConversationSegment:
    """Represents a segment of conversation history."""
    
    def __init__(self, start_time: datetime, end_time: Optional[datetime] = None):
        self.start_time = start_time
        self.end_time = end_time or start_time
        self.messages: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.topics: List[str] = []
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to this segment."""
        self.messages.append(message)
        self.end_time = datetime.now()
        
    def set_summary(self, summary: str) -> None:
        """Set the summary for this segment."""
        self.summary = summary
        
    def set_topics(self, topics: List[str]) -> None:
        """Set the topics for this segment."""
        self.topics = topics
        
    @property
    def duration(self) -> timedelta:
        """Get the duration of this segment."""
        return self.end_time - self.start_time
        
    @property
    def message_count(self) -> int:
        """Get the number of messages in this segment."""
        return len(self.messages)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert this segment to a dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "message_count": self.message_count,
            "summary": self.summary,
            "topics": self.topics,
            "messages": self.messages
        }


class MessageManager:
    """Manager for conversation messages, context retrieval, and conversation segmentation."""
    
    def __init__(self, 
                state: SchultzState,
                system_message: str = "",
                vector_store: Optional[VectorStore] = None,
                max_history: int = 100,
                max_active_segments: int = 3,
                segment_message_threshold: int = 10,
                segment_time_threshold: int = 30):  # minutes
        """
        Initialize the message manager.
        
        Args:
            state: Application state
            system_message: System message for conversations
            vector_store: Optional vector store for semantic retrieval
            max_history: Maximum messages to keep in memory
            max_active_segments: Maximum segments to keep active
            segment_message_threshold: Messages before creating a new segment
            segment_time_threshold: Minutes before creating a new segment
        """
        self.state = state
        self.message_history = MessageHistory(system_message)
        self.vector_store = vector_store or VectorStore()
        self.max_history = max_history
        self.vector_store_available = self.vector_store.available
        
        # Conversation segmentation parameters
        self.max_active_segments = max_active_segments
        self.segment_message_threshold = segment_message_threshold
        self.segment_time_threshold = segment_time_threshold
        self.segments: List[ConversationSegment] = []
        self.current_segment: Optional[ConversationSegment] = None
        self._ensure_current_segment()
        
    def _ensure_current_segment(self) -> None:
        """Ensure there is a current segment."""
        if not self.current_segment:
            self.current_segment = ConversationSegment(datetime.now())
            self.segments.append(self.current_segment)
            
    def _should_create_new_segment(self) -> bool:
        """Determine if a new segment should be created."""
        if not self.current_segment:
            return True
            
        # Check if message threshold exceeded
        if self.current_segment.message_count >= self.segment_message_threshold:
            return True
            
        # Check if time threshold exceeded
        time_diff = datetime.now() - self.current_segment.end_time
        if time_diff.total_seconds() / 60 >= self.segment_time_threshold:
            return True
            
        return False
        
    def _create_new_segment(self) -> None:
        """Create a new segment and summarize the previous one."""
        if self.current_segment and self.current_segment.message_count > 0:
            # Summarize the current segment before creating a new one
            self.summarize_segment(self.current_segment)
            
        # Create new segment
        self.current_segment = ConversationSegment(datetime.now())
        self.segments.append(self.current_segment)
        
        # Prune old segments if needed
        self._prune_old_segments()
        
    def _prune_old_segments(self) -> None:
        """Prune old segments to stay within the max active segments limit."""
        if len(self.segments) <= self.max_active_segments:
            return
            
        # Keep the most recent segments
        excess_count = len(self.segments) - self.max_active_segments
        self.segments = self.segments[excess_count:]
        
    def add_user_message(self, content: str) -> Message:
        """
        Add a user message to the conversation history.
        
        Args:
            content: Message content
            
        Returns:
            The created message
        """
        # Add to message history
        message = self.message_history.add_user_message(content)
        
        # Add to vector store if available
        if self.vector_store_available and self.state.vector_store_enabled:
            metadata = {
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            self.vector_store.add_text(content, metadata)
            
        # Check if we need a new segment for conversation management
        if self._should_create_new_segment():
            self._create_new_segment()
        
        # Add to current segment
        if self.current_segment:
            self.current_segment.add_message(message.to_dict())
            
        return message
        
    def add_assistant_message(self, 
                             content: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: Message content
            metadata: Optional metadata (mood, topics, etc.)
            
        Returns:
            The created message
        """
        # Add to message history
        message = self.message_history.add_assistant_message(content, metadata)
        
        # Add to vector store if available
        if self.vector_store_available and self.state.vector_store_enabled:
            vector_metadata = {
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            
            # Add additional metadata if provided
            if metadata:
                vector_metadata.update(metadata)
                
            self.vector_store.add_text(content, vector_metadata)
        
        # Add to current segment
        if self.current_segment:
            msg_dict = message.to_dict()
            if metadata:
                msg_dict.update(metadata)
            self.current_segment.add_message(msg_dict)
            
        return message
    
    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add messages to the conversation.
        
        Args:
            messages: List of messages to add
        """
        # Check if we need a new segment
        if self._should_create_new_segment():
            self._create_new_segment()
            
        # Add messages to the current segment
        for message in messages:
            if self.current_segment:
                self.current_segment.add_message(message)
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get the most recent messages in OpenAI format.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of messages in OpenAI format
        """
        # Get recent messages
        recent_messages = self.message_history.get_last_n_messages(count + 1)  # +1 for system message
        
        # Convert to OpenAI format
        return [msg.to_dict() for msg in recent_messages]
    
    def get_context(self, 
                   query: str, 
                   count: int = 5, 
                   semantic_weight: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get conversation context based on semantic similarity.
        
        Args:
            query: The user's query to find relevant context for
            count: Number of context items to retrieve
            semantic_weight: Weight for semantic vs. recency
            
        Returns:
            List of context items as dictionaries
        """
        if not self.vector_store_available or not self.state.vector_store_enabled:
            # Fallback to recent messages if vector store is unavailable
            recent_messages = self.message_history.get_last_n_messages(count)
            recent_messages = [msg for msg in recent_messages if msg.role != "system"]
            return [
                {"text": msg.content, "metadata": msg.metadata, "score": 1.0}
                for msg in recent_messages
            ]
        
        # Search vector store for similar messages
        results = self.vector_store.search(query, top_k=count)
        
        if not results:
            # Fallback if no results found
            recent_messages = self.message_history.get_last_n_messages(count)
            recent_messages = [msg for msg in recent_messages if msg.role != "system"]
            return [
                {"text": msg.content, "metadata": msg.metadata, "score": 1.0}
                for msg in recent_messages
            ]
            
        return results
    
    async def get_context_async(self, 
                              query: str, 
                              count: int = 5, 
                              semantic_weight: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get conversation context based on semantic similarity asynchronously.
        
        Args:
            query: The user's query to find relevant context for
            count: Number of context items to retrieve
            semantic_weight: Weight for semantic vs. recency
            
        Returns:
            List of context items as dictionaries
        """
        if not self.vector_store_available or not self.state.vector_store_enabled:
            # Fallback to recent messages if vector store is unavailable
            recent_messages = self.message_history.get_last_n_messages(count)
            recent_messages = [msg for msg in recent_messages if msg.role != "system"]
            return [
                {"text": msg.content, "metadata": msg.metadata, "score": 1.0}
                for msg in recent_messages
            ]
        
        # Search vector store for similar messages
        results = await self.vector_store.search_async(query, top_k=count)
        
        if not results:
            # Fallback if no results found
            recent_messages = self.message_history.get_last_n_messages(count)
            recent_messages = [msg for msg in recent_messages if msg.role != "system"]
            return [
                {"text": msg.content, "metadata": msg.metadata, "score": 1.0}
                for msg in recent_messages
            ]
            
        return results
    
    def build_prompt_with_context(self, 
                                 user_query: str, 
                                 instruction: str,
                                 context_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build a prompt with retrieved context items.
        
        Args:
            user_query: The user's query
            instruction: Instruction for the model on how to use context
            context_items: List of context items
            
        Returns:
            List of messages in OpenAI format
        """
        # Start with system message
        messages = [{"role": "system", "content": self.message_history.messages[0].content}]
        
        # Format context
        context_str = "\n\n".join([
            f"CONTEXT ITEM {i+1}:\n{item['text']}"
            for i, item in enumerate(context_items)
        ])
        
        # Add context message
        if context_items:
            messages.append({
                "role": "system",
                "content": f"{instruction}\n\nRELEVANT CONTEXT:\n{context_str}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def clear_history(self) -> None:
        """Clear conversation history and segments."""
        # Clear message history
        self.message_history.clear()
        
        # Clear segments
        self.segments = []
        self.current_segment = None
        self._ensure_current_segment()
    
    def reindex_vector_store(self) -> bool:
        """Rebuild the vector store from message history."""
        if not self.vector_store:
            return False
            
        try:
            # Clear the vector store
            self.vector_store.clear()
            
            # Get all messages except system messages
            messages = [msg for msg in self.message_history.messages if msg.role != "system"]
            
            # Prepare texts and metadata
            texts = [msg.content for msg in messages]
            metadatas = [
                {
                    "role": msg.role,
                    "timestamp": msg.timestamp.isoformat(),
                    "type": "message",
                    **msg.metadata
                }
                for msg in messages
            ]
            
            # Add batch to vector store
            success = self.vector_store.add_batch(texts, metadatas)
            
            # Update availability
            self.vector_store_available = self.vector_store.available
            
            return success
        except Exception as e:
            print(f"Error reindexing vector store: {str(e)}")
            return False
    
    @run_async
    async def reindex_vector_store_async(self) -> bool:
        """Rebuild the vector store from message history asynchronously."""
        if not self.vector_store:
            return False
            
        try:
            # Clear the vector store
            self.vector_store.clear()
            
            # Get all messages except system messages
            messages = [msg for msg in self.message_history.messages if msg.role != "system"]
            
            # Prepare texts and metadata
            texts = [msg.content for msg in messages]
            metadatas = [
                {
                    "role": msg.role,
                    "timestamp": msg.timestamp.isoformat(),
                    "type": "message",
                    **msg.metadata
                }
                for msg in messages
            ]
            
            # Add batch to vector store asynchronously
            success = await self.vector_store.add_batch_async(texts, metadatas)
            
            # Update availability
            self.vector_store_available = self.vector_store.available
            
            return success
        except Exception as e:
            print(f"Error reindexing vector store asynchronously: {str(e)}")
            return False
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"available": False, "entry_count": 0}
            
        return self.vector_store.get_stats()
    
    def summarize_segment(self, segment: ConversationSegment) -> str:
        """
        Summarize a conversation segment.
        
        Args:
            segment: The segment to summarize
            
        Returns:
            Summary text
        """
        if not segment.messages:
            segment.set_summary("")
            return ""
            
        try:
            # Extract just the messages in OpenAI format
            messages_to_summarize = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in segment.messages
                if "content" in msg
            ]
            
            # Generate summary
            summary = self.generate_summary(messages_to_summarize)
            
            # Store the summary
            segment.set_summary(summary)
            
            # Extract topics (simple implementation)
            topics = self._extract_topics(summary)
            segment.set_topics(topics)
            
            return summary
        except Exception as e:
            print(f"Error summarizing segment: {str(e)}")
            segment.set_summary("Error generating summary.")
            return "Error generating summary."
            
    async def summarize_segment_async(self, segment: ConversationSegment) -> str:
        """
        Summarize a conversation segment asynchronously.
        
        Args:
            segment: The segment to summarize
            
        Returns:
            Summary text
        """
        if not segment.messages:
            segment.set_summary("")
            return ""
            
        try:
            # Extract just the messages in OpenAI format
            messages_to_summarize = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in segment.messages
                if "content" in msg
            ]
            
            # Generate summary asynchronously
            summary = await self.generate_summary_async(messages_to_summarize)
            
            # Store the summary
            segment.set_summary(summary)
            
            # Extract topics (simple implementation)
            topics = self._extract_topics(summary)
            segment.set_topics(topics)
            
            return summary
        except Exception as e:
            print(f"Error summarizing segment asynchronously: {str(e)}")
            segment.set_summary("Error generating summary.")
            return "Error generating summary."
            
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text (simple implementation).
        
        Args:
            text: Text to extract topics from
            
        Returns:
            List of topics
        """
        # Simple implementation - split by commas and filter out common words
        words = text.lower().replace(",", " ").replace(".", " ").split()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        topics = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Remove duplicates and limit to 5 topics
        unique_topics = list(set(topics))
        return unique_topics[:5]
        
    def summarize_conversation(self) -> str:
        """
        Summarize the entire conversation.
        
        Returns:
            Summary text
        """
        # First ensure all segments are summarized
        for segment in self.segments:
            if not segment.summary and segment.message_count > 0:
                self.summarize_segment(segment)
                
        # Combine segment summaries
        combined_summary = "\n\n".join([
            f"Segment {i+1}: {segment.summary}"
            for i, segment in enumerate(self.segments)
            if segment.summary
        ])
        
        if not combined_summary:
            return "No conversation to summarize."
            
        return combined_summary
        
    async def summarize_conversation_async(self) -> str:
        """
        Summarize the entire conversation asynchronously.
        
        Returns:
            Summary text
        """
        # First ensure all segments are summarized
        for segment in self.segments:
            if not segment.summary and segment.message_count > 0:
                await self.summarize_segment_async(segment)
                
        # Combine segment summaries
        combined_summary = "\n\n".join([
            f"Segment {i+1}: {segment.summary}"
            for i, segment in enumerate(self.segments)
            if segment.summary
        ])
        
        if not combined_summary:
            return "No conversation to summarize."
            
        return combined_summary
        
    def get_active_topics(self) -> List[str]:
        """
        Get the active topics from recent segments.
        
        Returns:
            List of active topics
        """
        all_topics = []
        for segment in self.segments:
            all_topics.extend(segment.topics)
            
        # Count topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
        # Sort by count
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top topics
        return [topic for topic, count in sorted_topics[:10]]
        
    def get_segment_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics about conversation segments.
        
        Returns:
            List of segment statistics
        """
        return [
            {
                "index": i,
                "message_count": segment.message_count,
                "duration_minutes": segment.duration.total_seconds() / 60,
                "start_time": segment.start_time.isoformat(),
                "end_time": segment.end_time.isoformat(),
                "has_summary": bool(segment.summary),
                "topics": segment.topics
            }
            for i, segment in enumerate(self.segments)
        ]
        
    def generate_summary(self, 
                        messages: List[Dict[str, str]],
                        model: str = "gpt-3.5-turbo") -> str:
        """
        Generate a summary of the provided messages.
        
        Args:
            messages: List of messages to summarize
            model: Model to use for summarization
            
        Returns:
            Summary text
        """
        if not messages:
            return ""
            
        try:
            # Prepare messages for summary
            summary_prompt = [
                {"role": "system", "content": "Summarize the following conversation concisely:"}
            ]
            
            # Add messages to summarize
            summary_prompt.extend(messages)
            
            # Get summary
            response = chat_completion(
                messages=summary_prompt,
                model=model,
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error generating summary."
    
    async def generate_summary_async(self, 
                                   messages: List[Dict[str, str]],
                                   model: str = "gpt-3.5-turbo") -> str:
        """
        Generate a summary of the provided messages asynchronously.
        
        Args:
            messages: List of messages to summarize
            model: Model to use for summarization
            
        Returns:
            Summary text
        """
        if not messages:
            return ""
            
        try:
            # Prepare messages for summary
            summary_prompt = [
                {"role": "system", "content": "Summarize the following conversation concisely:"}
            ]
            
            # Add messages to summarize
            summary_prompt.extend(messages)
            
            # Get summary asynchronously
            response = await async_chat_completion(
                messages=summary_prompt,
                model=model,
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary asynchronously: {str(e)}")
            return "Error generating summary." 