# Data Generation

This module handles the generation of synthetic Jon data in various formats:

1. Q&A pairs for the OpenAI Retrieval API
2. Conversation data for fine-tuning
3. Standalone statements for the Retrieval API

## Directory Structure

- `generation/`: Core data generation code
- `loading/`: Retrieval API data loading utilities
- `models/`: Data models and schemas
- `output/`: Generated data output
- `training/`: Fine-tuning data preparation
- `utils/`: Test utilities and helpers

## Usage

### Generate Q&A Pairs

```bash
python -m data_generation.generation.jon_data_generator --qa-pairs 10 --output-dir output
```

### Generate Conversations

```bash
python -m data_generation.generation.jon_data_generator --conversations 5 --output-dir output
```

### Generate Statements

```bash
python -m data_generation.generation.jon_data_generator --statements 20 --output-dir output
```

### Test Generated Data

```bash
python -m data_generation.utils.test_jon_data --file path/to/data.jsonl
```

### Load Data into Retrieval API

```bash
python -m data_generation.loading.load_jon_retrieval --file path/to/data.jsonl
```

## Configuration

Key settings in `.env`:

```
OUTPUT_DIR=data_generation/output
CHECKPOINT_FREQUENCY=10
CHECKPOINT_DIR=checkpoints
```

## Data Quality

The generator ensures high-quality data through:

1. Rich metadata generation
2. Topic clustering
3. Entity tracking
4. Sentiment analysis
5. Style consistency checks

## Development

When adding new features:

1. Update the appropriate generator in `generation/`
2. Add tests in `utils/test_jon_data.py`
3. Update loading utilities if needed
4. Document changes in this README 