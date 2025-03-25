#!/usr/bin/env python
"""
Visualize Jon's Memory

This script will visualize Jon's memory structure from the OpenAI Retrieval API.
It provides a graphical representation of topics, connections, and knowledge clusters.

Usage:
    python -m data_generation.visualize_memory --assistant-id asst_abc123
    python -m data_generation.visualize_memory --input data_generation/output/jon_raw_data_20230526_123045.json
"""

import os
import sys
import json
import argparse
import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Load environment variables
load_dotenv()

def retrieve_assistant_data(client: OpenAI, assistant_id: str) -> Dict[str, Any]:
    """
    Retrieve all file data from an assistant for analysis.
    """
    print(f"Retrieving data from assistant {assistant_id}...")
    
    try:
        # Get the assistant details
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        print(f"Found assistant: {assistant.name}")
        
        # Get all files associated with the assistant
        file_ids = assistant.file_ids if hasattr(assistant, 'file_ids') else []
        
        if not file_ids:
            print("No files found for this assistant. Add files before visualizing.")
            return {"status": "no_files", "files": []}
        
        # Get content from each file
        files_data = []
        for file_id in file_ids:
            try:
                # Get file info
                file_info = client.files.retrieve(file_id=file_id)
                
                # Download file content as bytes (to be parsed later)
                content_response = client.files.content(file_id=file_id)
                content = content_response.read().decode('utf-8')
                
                # Attempt to parse JSON content
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # Handle JSONL files
                    if file_info.filename.endswith('.jsonl'):
                        parsed_content = []
                        for line in content.splitlines():
                            if line.strip():
                                try:
                                    parsed_content.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    else:
                        # Not JSON or JSONL, use text content
                        parsed_content = {"text": content}
                
                files_data.append({
                    "file_id": file_id,
                    "filename": file_info.filename,
                    "created_at": file_info.created_at,
                    "content": parsed_content
                })
                
                print(f"Retrieved content from file: {file_info.filename}")
                
            except Exception as e:
                print(f"Error retrieving file {file_id}: {str(e)}")
        
        return {
            "status": "success", 
            "assistant": {
                "id": assistant_id,
                "name": assistant.name,
                "model": assistant.model
            },
            "files": files_data
        }
        
    except Exception as e:
        print(f"Error retrieving assistant data: {str(e)}")
        return {"status": "error", "error": str(e)}

def load_local_data(input_file: str) -> Dict[str, Any]:
    """
    Load data from a local JSON or JSONL file.
    """
    print(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return {"status": "error", "error": "File not found"}
    
    try:
        if input_file.endswith('.jsonl'):
            # Parse JSONL file
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            return {
                "status": "success",
                "files": [{
                    "file_id": hashlib.md5(input_file.encode()).hexdigest(),
                    "filename": os.path.basename(input_file),
                    "created_at": os.path.getmtime(input_file),
                    "content": data
                }]
            }
        else:
            # Parse JSON file
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            return {
                "status": "success",
                "files": [{
                    "file_id": hashlib.md5(input_file.encode()).hexdigest(),
                    "filename": os.path.basename(input_file),
                    "created_at": os.path.getmtime(input_file),
                    "content": data
                }]
            }
    
    except Exception as e:
        print(f"Error loading data from file: {str(e)}")
        return {"status": "error", "error": str(e)}

def extract_qa_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Q&A pairs from the memory data.
    """
    qa_pairs = []
    
    # Process each file in the data
    for file in data.get("files", []):
        content = file.get("content", [])
        
        # If content is a list, assume it's already in a structured format
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Check for QA pair structure
                    if "question" in item and "answer" in item:
                        qa_pairs.append({
                            "question": item["question"],
                            "answer": item["answer"],
                            "metadata": item.get("metadata", {}),
                            "file": file["filename"]
                        })
                    # Check for JSONL format used in retrieval API
                    elif "text" in item and ("metadata" in item or "title" in item):
                        # This looks like a retrieval store format
                        text = item["text"]
                        # Try to extract question and answer
                        if "Question:" in text and "Answer:" in text:
                            parts = text.split("Answer:", 1)
                            question_part = parts[0].replace("Question:", "").strip()
                            answer = parts[1].strip()
                            qa_pairs.append({
                                "question": question_part,
                                "answer": answer,
                                "metadata": item.get("metadata", {}),
                                "file": file["filename"]
                            })
        
        # If content is a dict, it might be raw data JSON with nested QA pairs
        elif isinstance(content, dict):
            # Check for common structures in the raw data JSON
            if "qa_data" in content and isinstance(content["qa_data"], list):
                for item in content["qa_data"]:
                    if "question" in item and "answer" in item:
                        qa_pairs.append({
                            "question": item["question"],
                            "answer": item["answer"],
                            "metadata": item.get("metadata", {}),
                            "file": file["filename"]
                        })
    
    print(f"Extracted {len(qa_pairs)} Q&A pairs from memory data")
    return qa_pairs

def extract_conversations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract conversations from the memory data.
    """
    conversations = []
    
    # Process each file in the data
    for file in data.get("files", []):
        content = file.get("content", [])
        
        # If content is a list, check each item
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "messages" in item:
                    conversations.append({
                        "topic": item.get("topic", "Unknown"),
                        "messages": item["messages"],
                        "file": file["filename"]
                    })
        
        # If content is a dict, it might be raw data JSON
        elif isinstance(content, dict):
            if "conversation_data" in content and isinstance(content["conversation_data"], list):
                for item in content["conversation_data"]:
                    if isinstance(item, dict) and "messages" in item:
                        conversations.append({
                            "topic": item.get("topic", "Unknown"),
                            "messages": item["messages"],
                            "file": file["filename"]
                        })
    
    print(f"Extracted {len(conversations)} conversations from memory data")
    return conversations

def extract_statements(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract standalone statements from the memory data.
    """
    statements = []
    
    # Process each file in the data
    for file in data.get("files", []):
        content = file.get("content", [])
        
        # If content is a list, check each item
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "statement" in item:
                    statements.append({
                        "statement": item["statement"],
                        "metadata": item.get("metadata", {}),
                        "file": file["filename"]
                    })
        
        # If content is a dict, it might be raw data JSON
        elif isinstance(content, dict):
            if "statement_data" in content and isinstance(content["statement_data"], list):
                for item in content["statement_data"]:
                    if isinstance(item, dict) and "statement" in item:
                        statements.append({
                            "statement": item["statement"],
                            "metadata": item.get("metadata", {}),
                            "file": file["filename"]
                        })
    
    print(f"Extracted {len(statements)} statements from memory data")
    return statements

def extract_topics(qa_pairs: List[Dict[str, Any]], statements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract main topics from the memory data.
    """
    topics = defaultdict(list)
    
    # Extract topics from QA pairs
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        topic = metadata.get("topic")
        topic_cluster = metadata.get("topic_cluster")
        
        if topic and topic_cluster:
            topics[topic_cluster].append(topic)
        elif topic:
            topics["uncategorized"].append(topic)
    
    # Extract topics from statements
    for statement in statements:
        metadata = statement.get("metadata", {})
        topic = metadata.get("topic")
        topic_cluster = metadata.get("topic_cluster")
        
        if topic and topic_cluster:
            topics[topic_cluster].append(topic)
        elif topic:
            topics["uncategorized"].append(topic)
    
    # Make unique topics per cluster
    for cluster in topics:
        topics[cluster] = list(set(topics[cluster]))
    
    print(f"Extracted {sum(len(topics[c]) for c in topics)} topics across {len(topics)} clusters")
    return dict(topics)

def build_knowledge_graph(qa_pairs: List[Dict[str, Any]], topics: Dict[str, List[str]]) -> nx.Graph:
    """
    Build a knowledge graph representation of Jon's memory.
    """
    G = nx.Graph()
    
    # Add topic clusters as the main nodes
    for cluster, subtopics in topics.items():
        G.add_node(cluster, type="cluster", size=len(subtopics) * 5, topics=subtopics)
        
        # Add topics as nodes connected to their cluster
        for topic in subtopics:
            G.add_node(topic, type="topic", size=3)
            G.add_edge(cluster, topic, weight=1)
    
    # Add connections between topics based on QA pairs
    topic_connections = defaultdict(int)
    
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        topic = metadata.get("topic")
        entities = metadata.get("entities", [])
        
        if topic:
            # Connect topic to entities
            for entity in entities:
                # Add entity as node if it doesn't exist
                if not G.has_node(entity):
                    G.add_node(entity, type="entity", size=2)
                
                # Add or strengthen connection between topic and entity
                if G.has_edge(topic, entity):
                    G[topic][entity]["weight"] += 1
                else:
                    G.add_edge(topic, entity, weight=1)
            
            # Find related topics in the same QA pair
            related_topics = []
            question = qa.get("question", "").lower()
            answer = qa.get("answer", "").lower()
            
            for other_topic in G.nodes():
                if G.nodes[other_topic].get("type") == "topic" and other_topic != topic:
                    # Check if other topic is mentioned in this QA pair
                    if other_topic.lower() in question or other_topic.lower() in answer:
                        related_topics.append(other_topic)
            
            # Connect related topics
            for related_topic in related_topics:
                key = tuple(sorted([topic, related_topic]))
                topic_connections[key] += 1
    
    # Add edges for topic connections
    for (topic1, topic2), weight in topic_connections.items():
        if weight > 0:  # Only add connections that occur at least once
            G.add_edge(topic1, topic2, weight=weight)
    
    print(f"Built knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def create_topic_visualization(topics: Dict[str, List[str]], qa_pairs: List[Dict[str, Any]]) -> str:
    """
    Create a visualization of topic clusters and their relationships.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is required for interactive topic visualization. Install with pip install plotly")
        return None
    
    # Count QA pairs per topic
    topic_counts = defaultdict(int)
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        topic = metadata.get("topic")
        if topic:
            topic_counts[topic] += 1
    
    # Create data for treemap
    treemap_data = []
    for cluster, subtopics in topics.items():
        # Add cluster node
        treemap_data.append({
            "id": cluster,
            "parent": "",
            "value": sum(topic_counts.get(topic, 1) for topic in subtopics)
        })
        
        # Add topic nodes
        for topic in subtopics:
            treemap_data.append({
                "id": topic,
                "parent": cluster,
                "value": topic_counts.get(topic, 1)
            })
    
    # Create treemap
    df = pd.DataFrame(treemap_data)
    fig = px.treemap(
        df, 
        ids='id', 
        parents='parent',
        values='value',
        title="Jon's Knowledge Structure",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Customize layout
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        font=dict(size=14)
    )
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data_generation/output/jon_topic_treemap_{timestamp}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    fig.write_html(output_path)
    print(f"Topic visualization saved to {output_path}")
    
    return output_path

def create_knowledge_graph_visualization(graph: nx.Graph) -> str:
    """
    Create a visualization of the knowledge graph.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is required for interactive graph visualization. Install with pip install plotly")
        return None
    
    # Create a spring layout for better visualization
    pos = nx.spring_layout(graph, k=0.15, iterations=50)
    
    # Get node types
    node_types = {}
    node_sizes = {}
    for node, attrs in graph.nodes(data=True):
        node_types[node] = attrs.get("type", "unknown")
        node_sizes[node] = attrs.get("size", 5)
    
    # Create a color map for different node types
    color_map = {
        "cluster": "#ff9999",  # Light red
        "topic": "#66b3ff",    # Light blue
        "entity": "#99ff99"    # Light green
    }
    
    # Calculate edge weights for width
    edge_weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
    
    # Create traces for different node types
    node_trace_dict = {}
    for node_type in set(node_types.values()):
        node_trace_dict[node_type] = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=color_map.get(node_type, "#cccccc"),
                size=[],
                line=dict(width=2)
            ),
            name=node_type.capitalize()
        )
    
    # Add node positions and properties to traces
    for node in graph.nodes():
        x, y = pos[node]
        node_type = node_types.get(node, "unknown")
        node_trace = node_trace_dict[node_type]
        node_trace['x'] = node_trace['x'] + (x,)
        node_trace['y'] = node_trace['y'] + (y,)
        node_trace['text'] = node_trace['text'] + (node,)
        node_trace['marker']['size'] = node_trace['marker']['size'] + (node_sizes.get(node, 5),)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Add edges to the trace
    for i, (u, v) in enumerate(graph.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace] + list(node_trace_dict.values()),
        layout=go.Layout(
            title='Jon\'s Knowledge Graph',
            titlefont=dict(size=16),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper"
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data_generation/output/jon_knowledge_graph_{timestamp}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    fig.write_html(output_path)
    print(f"Knowledge graph visualization saved to {output_path}")
    
    return output_path

def create_sentiment_visualization(qa_pairs: List[Dict[str, Any]], statements: List[Dict[str, Any]]) -> str:
    """
    Create a visualization of sentiment across topics.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is required for interactive sentiment visualization. Install with pip install plotly")
        return None
    
    # Collect sentiment data
    sentiment_data = []
    
    # Process QA pairs
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        topic = metadata.get("topic")
        topic_cluster = metadata.get("topic_cluster", "unknown")
        sentiment = metadata.get("sentiment")
        
        if topic and sentiment:
            sentiment_data.append({
                "topic": topic,
                "cluster": topic_cluster,
                "sentiment": sentiment,
                "type": "QA"
            })
    
    # Process statements
    for statement in statements:
        metadata = statement.get("metadata", {})
        topic = metadata.get("topic")
        topic_cluster = metadata.get("topic_cluster", "unknown")
        sentiment = metadata.get("sentiment")
        
        if topic and sentiment:
            sentiment_data.append({
                "topic": topic,
                "cluster": topic_cluster,
                "sentiment": sentiment,
                "type": "Statement"
            })
    
    if not sentiment_data:
        print("No sentiment data found")
        return None
    
    # Create dataframe
    df = pd.DataFrame(sentiment_data)
    
    # Count sentiment distribution per topic
    sentiment_counts = df.groupby(['cluster', 'sentiment']).size().unstack(fill_value=0)
    
    # Create heatmap
    fig = px.imshow(
        sentiment_counts,
        labels=dict(x="Sentiment", y="Topic Cluster", color="Count"),
        x=sentiment_counts.columns,
        y=sentiment_counts.index,
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Sentiment Distribution Across Topic Clusters"
    )
    
    # Add text annotations
    for i, cluster in enumerate(sentiment_counts.index):
        for j, sentiment in enumerate(sentiment_counts.columns):
            count = sentiment_counts.iloc[i, j]
            fig.add_annotation(
                x=sentiment,
                y=cluster,
                text=str(count),
                showarrow=False,
                font=dict(color="white" if count > sentiment_counts.values.max()/2 else "black")
            )
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data_generation/output/jon_sentiment_heatmap_{timestamp}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    fig.write_html(output_path)
    print(f"Sentiment visualization saved to {output_path}")
    
    return output_path

def create_dashboard(data_sources: Dict[str, Any], output_path: str = None) -> str:
    """
    Create a comprehensive dashboard with multiple visualizations.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is required for interactive dashboard. Install with pip install plotly")
        return None
    
    qa_pairs = data_sources.get("qa_pairs", [])
    topics = data_sources.get("topics", {})
    statements = data_sources.get("statements", [])
    conversations = data_sources.get("conversations", [])
    
    # Create subplots for dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Topic Distribution", 
            "Sentiment Analysis",
            "Content Types", 
            "Top Entities"
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "pie"}, {"type": "bar"}]
        ]
    )
    
    # 1. Topic Distribution (top 10 topics by count)
    topic_counts = defaultdict(int)
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        topic = metadata.get("topic")
        if topic:
            topic_counts[topic] += 1
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    topic_names = [t[0] for t in top_topics]
    topic_values = [t[1] for t in top_topics]
    
    fig.add_trace(
        go.Bar(
            x=topic_names, 
            y=topic_values,
            name="Topics",
            marker_color="lightblue"
        ),
        row=1, col=1
    )
    
    # 2. Sentiment Analysis
    sentiment_counts = defaultdict(int)
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        sentiment = metadata.get("sentiment")
        if sentiment:
            sentiment_counts[sentiment] += 1
    
    for statement in statements:
        metadata = statement.get("metadata", {})
        sentiment = metadata.get("sentiment")
        if sentiment:
            sentiment_counts[sentiment] += 1
    
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_values = list(sentiment_counts.values())
    
    fig.add_trace(
        go.Pie(
            labels=sentiment_labels, 
            values=sentiment_values,
            name="Sentiment",
            marker=dict(colors=px.colors.sequential.Plasma)
        ),
        row=1, col=2
    )
    
    # 3. Content Types
    content_types = {
        "Q&A Pairs": len(qa_pairs),
        "Statements": len(statements),
        "Conversations": len(conversations)
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(content_types.keys()), 
            values=list(content_types.values()),
            name="Content Types",
            marker=dict(colors=px.colors.qualitative.Set3)
        ),
        row=2, col=1
    )
    
    # 4. Top Entities
    entity_counts = defaultdict(int)
    for qa in qa_pairs:
        metadata = qa.get("metadata", {})
        entities = metadata.get("entities", [])
        for entity in entities:
            entity_counts[entity] += 1
    
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    entity_names = [e[0] for e in top_entities]
    entity_values = [e[1] for e in top_entities]
    
    fig.add_trace(
        go.Bar(
            x=entity_names, 
            y=entity_values,
            name="Entities",
            marker_color="lightgreen"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Jon's Memory Dashboard",
        height=800,
        showlegend=False
    )
    
    # Create output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data_generation/output/jon_memory_dashboard_{timestamp}.html"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    fig.write_html(output_path)
    print(f"Dashboard saved to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize Jon's memory structure")
    parser.add_argument("--assistant-id", type=str, help="Assistant ID to visualize")
    parser.add_argument("--input", "-i", type=str, help="Input file with memory data (JSON/JSONL)")
    parser.add_argument("--output", "-o", type=str, help="Output directory for visualizations")
    parser.add_argument("--format", type=str, choices=["html", "png", "json"], 
                       default="html", help="Output format")
    parser.add_argument("--type", type=str, choices=["graph", "topics", "sentiment", "dashboard", "all"],
                       default="all", help="Type of visualization to generate")
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = args.output or "data_generation/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from assistant or file
    if args.assistant_id:
        # Get assistant ID from args or environment
        assistant_id = args.assistant_id or os.environ.get("OPENAI_ASSISTANT_ID")
        
        if not assistant_id:
            print("Error: No assistant ID provided. Use --assistant-id or set OPENAI_ASSISTANT_ID in .env")
            sys.exit(1)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        data = retrieve_assistant_data(client, assistant_id)
    elif args.input:
        data = load_local_data(args.input)
    else:
        print("Error: Either --assistant-id or --input must be provided")
        sys.exit(1)
    
    if data["status"] != "success":
        print(f"Error loading data: {data.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("\nJon Memory Visualization Tool")
    print("==============================")
    
    # Extract data
    qa_pairs = extract_qa_data(data)
    statements = extract_statements(data)
    conversations = extract_conversations(data)
    topics = extract_topics(qa_pairs, statements)
    
    # Build knowledge graph
    knowledge_graph = build_knowledge_graph(qa_pairs, topics)
    
    # Track output files
    output_files = {}
    
    # Generate visualizations based on type
    vis_types = ["graph", "topics", "sentiment", "dashboard"] if args.type == "all" else [args.type]
    
    data_sources = {
        "qa_pairs": qa_pairs,
        "statements": statements,
        "conversations": conversations,
        "topics": topics,
        "knowledge_graph": knowledge_graph
    }
    
    for vis_type in vis_types:
        if vis_type == "graph":
            output_files["graph"] = create_knowledge_graph_visualization(knowledge_graph)
        elif vis_type == "topics":
            output_files["topics"] = create_topic_visualization(topics, qa_pairs)
        elif vis_type == "sentiment":
            output_files["sentiment"] = create_sentiment_visualization(qa_pairs, statements)
        elif vis_type == "dashboard":
            output_files["dashboard"] = create_dashboard(data_sources)
    
    # Print summary
    print("\nMemory Visualization Summary")
    print("----------------------------")
    print(f"Q&A Pairs: {len(qa_pairs)}")
    print(f"Statements: {len(statements)}")
    print(f"Conversations: {len(conversations)}")
    print(f"Topic Clusters: {len(topics)}")
    print(f"Total Topics: {sum(len(t) for t in topics.values())}")
    
    # Print file locations
    print("\nGenerated Visualizations:")
    for vis_type, file_path in output_files.items():
        if file_path:
            print(f"- {vis_type.capitalize()}: {file_path}")

if __name__ == "__main__":
    main() 