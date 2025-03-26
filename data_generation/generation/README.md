# Generation

Core data generation code for SchultzGPT. This module generates synthetic Jon data in various formats for use with the OpenAI Retrieval API and fine-tuning.

## Features

- Q&A pair generation with rich metadata
- Conversation generation for fine-tuning
- Standalone statement generation
- Automatic checkpointing and recovery
- Output verification
- Real data integration for improved quality

## Usage

```bash
# Generate Q&A pairs
python jon_data_generator.py --qa-pairs 10

# Generate conversations
python jon_data_generator.py --conversations 5

# Generate statements
python jon_data_generator.py --statements 20

# Use real data examples
python jon_data_generator.py --qa-pairs 10 --use-real-data

# Set checkpoint frequency
python jon_data_generator.py --qa-pairs 100 --checkpoint-interval 10

# Verify output
python jon_data_generator.py --qa-pairs 10 --verify
```

## Output Formats

1. Raw JSON
   - Complete data with metadata
   - Used for analysis and debugging

2. Retrieval JSONL
   - Formatted for OpenAI Retrieval API
   - Optimized for semantic search

3. Fine-tuning JSONL
   - Formatted for model training
   - Includes conversation context

## Quality Assurance

The generator ensures high-quality data through:

1. Rich metadata generation
2. Topic clustering
3. Entity tracking
4. Sentiment analysis
5. Style consistency checks

## Configuration

Key settings in `.env`:

```
OUTPUT_DIR=data_generation/output
CHECKPOINT_FREQUENCY=10
CHECKPOINT_DIR=checkpoints
``` 