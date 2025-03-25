# Jon Data Generation

This directory contains the scripts responsible for generating synthetic Jon data for various purposes including:

- Question & Answer pairs for the vector store (legacy)
- Q&A pairs for the OpenAI Retrieval API
- Complete conversations for fine-tuning
- Standalone statements for embeddings

## Main Script

The main script is `jon_data_generator.py`, which handles all data generation tasks with various options:

```bash
# Generate data with default settings
python -m data_generation.generation.jon_data_generator

# Generate specific amounts of data
python -m data_generation.generation.jon_data_generator --qa-pairs 200 --conversations 50 --statements 300
```

## Advanced Features

- **Bulk Generation**: Reduces API calls by generating multiple items per prompt
- **Parallel Processing**: Generates data in parallel for faster completion
- **Dynamic Batch Sizing**: Optimizes batch sizes based on content complexity
- **Enhanced Metadata**: Enriches data with detailed metadata for better retrieval
- **Data Quality Metrics**: Analyzes and reports on the quality of generated data

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 