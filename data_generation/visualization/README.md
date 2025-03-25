# Jon Memory Visualization

This directory contains tools for visualizing Jon's memory structure and knowledge organization:

## Scripts

- `visualize_memory.py`: Creates interactive visualizations of Jon's memory structure

## Usage Examples

```bash
# Visualize an assistant's memory
python -m data_generation.visualization.visualize_memory --assistant-id asst_abc123

# Visualize from local data file
python -m data_generation.visualization.visualize_memory --input path/to/jon_raw_data.json

# Generate specific types of visualizations
python -m data_generation.visualization.visualize_memory --type graph --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type topics --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type sentiment --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type dashboard --assistant-id asst_abc123
```

## Visualization Types

The visualization tool can generate several types of interactive visualizations:

1. **Knowledge Graph**: Shows connections between topics, entities, and concepts
2. **Topic Treemap**: Displays hierarchical organization of Jon's knowledge by topic clusters
3. **Sentiment Analysis**: Visualizes emotional tone across different topics
4. **Memory Dashboard**: Comprehensive overview with multiple visualizations

## Dependencies

The visualization tools require additional Python packages:

```bash
pip install matplotlib networkx plotly seaborn pandas wordcloud scikit-learn nltk
```

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 