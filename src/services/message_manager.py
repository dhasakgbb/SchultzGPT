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
from services.retrieval_store import RetrievalStore
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
                retrieval_store: Optional[RetrievalStore] = None,
                max_history: int = 100,
                max_active_segments: int = 3,
                segment_message_threshold: int = 10,
                segment_time_threshold: int = 30):  # minutes
        """
        Initialize the message manager.
        
        Args:
            state: Application state
            system_message: System message for conversations
            retrieval_store: Optional retrieval store for semantic search
            max_history: Maximum messages to keep in memory
            max_active_segments: Maximum segments to keep active
            segment_message_threshold: Messages before creating a new segment
            segment_time_threshold: Minutes before creating a new segment
        """
        self.state = state
        self.message_history = MessageHistory(system_message)
        self.retrieval_store = retrieval_store or RetrievalStore()
        self.max_history = max_history
        self.retrieval_store_available = self.retrieval_store.available
        
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
        
        # Add to retrieval store if available
        if self.retrieval_store_available and self.state.retrieval_store_enabled:
            metadata = {
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            self.retrieval_store.add_text(content, metadata)
            
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
        
        # Add to retrieval store if available
        if self.retrieval_store_available and self.state.retrieval_store_enabled:
            retrieval_metadata = {
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            
            # Add additional metadata if provided
            if metadata:
                retrieval_metadata.update(metadata)
                
            self.retrieval_store.add_text(content, retrieval_metadata)
            
        # Add to current segment
        if self.current_segment:
            self.current_segment.add_message(message.to_dict())
            
        return message
        
    def get_context(self, 
                   query: str, 
                   count: int = 3, 
                   threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query from the retrieval store.
        
        Args:
            query: The query to find context for
            count: Number of context items to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of context items (text and metadata)
        """
        if not (self.retrieval_store_available and self.state.retrieval_store_enabled):
            return []
            
        try:
            # Search the retrieval store
            results = self.retrieval_store.search(query, top_k=count)
            
            # Format for use in prompts
            context_items = []
            for result in results:
                # Skip if below threshold (although retrieval API doesn't provide scores)
                if "score" in result and result["score"] < threshold:
                    continue
                    
                context_items.append({
                    "text": result["text"],
                    "metadata": result.get("metadata", {})
                })
                
            return context_items
        except Exception as e:
            print(f"Error getting context: {e}")
            return []
            
    @run_async
    async def get_context_async(self, 
                              query: str, 
                              count: int = 3, 
                              threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query asynchronously.
        
        A wrapper around get_context that runs asynchronously.
        
        Args:
            query: The query to find context for
            count: Number of context items to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of context items (text and metadata)
        """
        return self.get_context(query, count, threshold)
            
    def build_prompt_with_context(self, 
                                 user_query: str, 
                                 instruction: str, 
                                 context_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build a prompt that includes context from the retrieval store.
        
        Args:
            user_query: The user's query
            instruction: Instruction for using the context
            context_items: Context items from get_context
            
        Returns:
            List of messages for the prompt
        """
        # Get basic messages from history
        messages = self.message_history.get_messages()
        
        # If no context, return basic messages
        if not context_items:
            return messages
            
        # Insert context before the last message (the user query)
        context_text = instruction + "\n\n"
        
        for i, item in enumerate(context_items):
            context_text += f"Context item {i+1}:\n{item['text']}\n\n"
            
        # Replace the system message with one that includes context
        system_content = messages[0]["content"]
        new_system_content = system_content + "\n\n" + context_text
        
        context_messages = [{"role": "system", "content": new_system_content}]
        context_messages.extend(messages[1:])
        
        return context_messages
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.message_history.clear()
        self.segments = []
        self.current_segment = None
        self._ensure_current_segment()
        
    def get_messages(self, count: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation messages.
        
        Args:
            count: Optional limit on number of messages
            
        Returns:
            List of message dictionaries
        """
        return self.message_history.get_messages(count)
        
    def get_all_messages(self) -> List[Message]:
        """Get all messages as Message objects."""
        return self.message_history.messages
        
    def summarize_conversation(self) -> str:
        """
        Summarize the current conversation.
        
        Returns:
            Conversation summary
        """
        # If no messages, return empty string
        if not self.message_history.messages:
            return ""
            
        # Format messages for the summary prompt
        message_texts = []
        for msg in self.message_history.messages[1:]:  # Skip system message
            role = "User" if msg.role == "user" else "Jon"
            message_texts.append(f"{role}: {msg.content}")
            
        message_history = "\n".join(message_texts)
        
        # Create the summary prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant who summarizes conversations concisely."},
            {"role": "user", "content": f"Please summarize this conversation in a paragraph. Focus on the main topics and key points.\n\nConversation:\n{message_history}"}
        ]
        
        try:
            # Get the summary
            response = chat_completion(
                messages=prompt,
                model="gpt-4o",
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            
            # Add to retrieval store with special metadata
            if self.retrieval_store_available and self.state.retrieval_store_enabled:
                metadata = {
                    "type": "conversation_summary",
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(self.message_history.messages) - 1  # Exclude system
                }
                self.retrieval_store.add_text(summary, metadata)
                
            return summary
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return "Error generating summary."
            
    def summarize_segment(self, segment: ConversationSegment) -> None:
        """
        Summarize a conversation segment and extract topics.
        
        Args:
            segment: The conversation segment to summarize
        """
        if not segment.messages:
            return
            
        # Format messages for the summary prompt
        message_texts = []
        for msg in segment.messages:
            role = "User" if msg["role"] == "user" else "Jon"
            message_texts.append(f"{role}: {msg['content']}")
            
        message_history = "\n".join(message_texts)
        
        # Create the summary prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant who summarizes conversations and identifies main topics."},
            {"role": "user", "content": f"""Please analyze this conversation segment and provide:
1. A concise summary (1-2 sentences)
2. A list of 3-5 main topics discussed

Format as JSON with "summary" and "topics" keys.

Conversation:
{message_history}"""}
        ]
        
        try:
            # Get the summary
            response = chat_completion(
                messages=prompt,
                model="gpt-4o",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Update segment with summary and topics
            summary = result.get("summary", "")
            topics = result.get("topics", [])
            
            segment.set_summary(summary)
            segment.set_topics(topics)
            
            # Add to retrieval store with special metadata
            if self.retrieval_store_available and self.state.retrieval_store_enabled:
                metadata = {
                    "type": "segment_summary",
                    "timestamp": segment.end_time.isoformat(),
                    "start_time": segment.start_time.isoformat(),
                    "message_count": segment.message_count,
                    "topics": topics
                }
                self.retrieval_store.add_text(summary, metadata)
                
        except Exception as e:
            print(f"Error summarizing segment: {e}")
            segment.set_summary("Error generating summary.")
            
    def get_conversation_topics(self) -> List[str]:
        """
        Extract topics from the conversation.
        
        Returns:
            List of conversation topics
        """
        # If no messages, return empty list
        if not self.message_history.messages or len(self.message_history.messages) < 3:
            return []
            
        # Format messages for the topic prompt
        message_texts = []
        for msg in self.message_history.messages[1:]:  # Skip system message
            role = "User" if msg.role == "user" else "Jon"
            message_texts.append(f"{role}: {msg.content}")
            
        message_history = "\n".join(message_texts)
        
        # Create the topics prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant who identifies the main topics in conversations."},
            {"role": "user", "content": f"Please list the 3-5 main topics in this conversation. Return the topics as a JSON array of strings.\n\nConversation:\n{message_history}"}
        ]
        
        try:
            # Get the topics
            response = chat_completion(
                messages=prompt,
                model="gpt-4o",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle different response formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "topics" in result:
                return result["topics"]
            else:
                return []
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return [] 