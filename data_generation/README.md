# Jon Data Generation Toolkit

This toolkit provides tools for generating synthetic Jon data for SchultzGPT. It enables you to create large volumes of data that capture Jon's personality and distinctive conversational style for use in:

1. **Vector store**: Question-answer pairs about Jon (legacy)
2. **Retrieval API**: Question-answer pairs for OpenAI's Retrieval API 
3. **Fine-tuning**: Complete conversations between users and Jon 
4. **Embeddings**: Standalone Jon statements and opinions

## Directory Structure

The toolkit is organized into several subdirectories for better organization:

```
data_generation/
  ├── jon_tools.py             # Main unified CLI entry point
  ├── generation/              # Data generation scripts
  │   └── jon_data_generator.py  # Main data generator
  ├── loading/                 # Data loading utilities
  │   ├── load_jon_data.py       # Vector store loader (legacy)
  │   ├── load_jon_retrieval.py  # Retrieval API loader
  │   └── manage_assistants.py   # Assistant management tools
  ├── training/                # Fine-tuning utilities
  │   ├── prepare_fine_tuning.py # Data preparation for fine-tuning
  │   └── monitor_jobs.py        # Fine-tuning job management
  ├── utils/                   # Shared utilities
  │   └── test_jon_data.py       # Data testing and exploration
  ├── visualization/           # Memory visualization tools
  │   └── visualize_memory.py    # Memory structure visualizer
  └── migration/               # Migration tools
      └── migrate_vector_to_retrieval.py # Vector store to retrieval migrator
```

## Setup

The toolkit requires the following dependencies:

```bash
pip install openai tqdm tabulate dotenv
```

Make sure your `.env` file has the necessary OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

## Generating Data

You can use either the unified CLI tool or the individual scripts:

### Using the Unified CLI

The `jon_tools.py` provides a consolidated interface to all data tools:

```bash
# Generate data with default settings
python -m data_generation.jon_tools generate

# Generate specific amounts of data
python -m data_generation.jon_tools generate --qa-pairs 200 --conversations 50 --statements 300

# Use bulk generation with specific batch size
python -m data_generation.jon_tools generate --qa-pairs 500 --batch-size 20 --parallel

# Enable dynamic batch sizing and metadata enrichment
python -m data_generation.jon_tools generate --dynamic-batching --enrich-metadata
```

### Using Individual Scripts

If you prefer using the original scripts directly:

```bash
# Generate data with default settings
python -m data_generation.generation.jon_data_generator

# Generate specific amounts of data
python -m data_generation.generation.jon_data_generator --qa-pairs 200 --conversations 50 --statements 300

# Use bulk generation with specific batch size (more efficient)
python -m data_generation.generation.jon_data_generator --qa-pairs 500 --batch-size 20

# Enable dynamic batch sizing for optimal efficiency
python -m data_generation.generation.jon_data_generator --qa-pairs 200 --dynamic-batching

# Use parallel processing for faster generation
python -m data_generation.generation.jon_data_generator --qa-pairs 300 --parallel

# Generate with enriched metadata tagging
python -m data_generation.generation.jon_data_generator --qa-pairs 100 --enrich-metadata

# Change output directory
python -m data_generation.generation.jon_data_generator --output-dir custom/path
```

The generator creates four types of output in the specified directory, timestamped for tracking:

- `jon_vector_data_{timestamp}.jsonl`: Q&A pairs for the legacy vector store
- `jon_retrieval_data_{timestamp}.jsonl`: Q&A pairs for the OpenAI Retrieval API
- `jon_fine_tuning_{timestamp}.jsonl`: Formatted examples for OpenAI fine-tuning
- `jon_embeddings_{timestamp}.jsonl`: Jon statements for embeddings
- `jon_raw_data_{timestamp}.json`: Complete raw data for reference
- `jon_metrics_{timestamp}.json`: Data quality metrics and API usage statistics

### Bulk Generation Efficiency

The generator now uses a bulk generation approach that significantly reduces API calls:

```bash
# Compare API usage stats
python -m data_generation.generation.jon_data_generator --qa-pairs 100 --conversations 20 --statements 50 --batch-size 10
```

Benefits of bulk generation:
- Reduces API calls by up to 90% (e.g., 170 items can be generated with just 17 API calls)
- Lowers generation costs significantly
- Maintains output quality and variety
- Configurable batch sizes (default is 10, adjust based on needs)
- Automatically handles remainder items in partial batches

Default batch sizes:
- QA pairs: 10 per batch
- Conversations: 5 per batch (more complex, so smaller batch size)
- Statements: 10 per batch

### Advanced Generation Features

The generator includes several advanced features to improve efficiency and data quality:

#### Dynamic Batch Sizing
Automatically calculates optimal batch sizes based on content complexity and token limits:
```bash
python -m data_generation.generation.jon_data_generator --dynamic-batching
```

#### Parallel Processing
Utilizes multiple threads to generate data in parallel for faster completion:
```bash
python -m data_generation.generation.jon_data_generator --parallel
```

#### Enhanced Metadata
Enriches generated data with detailed metadata for better retrieval and analysis:
```bash
python -m data_generation.generation.jon_data_generator --enrich-metadata
```

Added metadata includes:
- Generation timestamps
- Complexity scores
- Uniqueness scores
- Token estimates
- Version tracking

#### Data Quality Metrics
Automatically analyzes generated data quality and provides detailed metrics:
- Topic diversity
- Vocabulary richness
- Content redundancy
- Sentiment distribution
- Entity distribution

The metrics are displayed in the console and saved to a JSON file for further analysis.

## Loading Data Into Vector Store (Legacy)

### Using the Unified CLI:

```bash
# Load data into vector store
python -m data_generation.jon_tools load --file data/jon_vector_data.jsonl --target vector

# Validate data without loading
python -m data_generation.jon_tools load --file data/jon_vector_data.jsonl --target vector --dry-run

# Specify vector store directory
python -m data_generation.jon_tools load --file data/jon_vector_data.jsonl --target vector --store-dir custom/vector_store
```

### Using the Original Script:

```bash
# Validate data without loading
python -m data_generation.loading.load_jon_data --file data_generation/output/jon_vector_data_20230526_123045.jsonl --dry-run

# Load data into vector store
python -m data_generation.loading.load_jon_data --file data_generation/output/jon_vector_data_20230526_123045.jsonl

# Specify vector store directory
python -m data_generation.loading.load_jon_data --file path/to/data.jsonl --store-dir custom/vector_store
```

## Loading Data Into Retrieval Store

### Using the Unified CLI:

```bash
# Load data into retrieval API
python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl

# Validate data without loading
python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl --dry-run

# Use a specific Assistant ID
python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl --assistant-id asst_abc123
```

### Using the Original Script:

```bash
# Validate data without loading
python -m data_generation.loading.load_jon_retrieval --file data_generation/output/jon_retrieval_data_20230526_123045.jsonl --dry-run

# Load data into the retrieval store
python -m data_generation.loading.load_jon_retrieval --file data_generation/output/jon_retrieval_data_20230526_123045.jsonl

# Use a specific Assistant ID
python -m data_generation.loading.load_jon_retrieval --file path/to/data.jsonl --assistant-id asst_abc123

# Configure batch size for performance
python -m data_generation.loading.load_jon_retrieval --file path/to/data.jsonl --batch-size 30
```

## Managing OpenAI Assistants

Use the Assistant Management tools to list, create, and manage assistants for the Retrieval API:

```bash
# List all assistants
python -m data_generation.jon_tools assistants list

# Get detailed information about an assistant
python -m data_generation.jon_tools assistants details --id asst_abc123

# Create a new assistant
python -m data_generation.jon_tools assistants create --name "Jon Memory" --save

# Clean up old files from an assistant
python -m data_generation.jon_tools assistants clean --id asst_abc123
```

Or use the individual script:

```bash
# List all assistants
python -m data_generation.loading.manage_assistants list

# Get detailed information about an assistant
python -m data_generation.loading.manage_assistants details --id asst_abc123

# Create a new assistant
python -m data_generation.loading.manage_assistants create --name "Jon Memory" --save

# Clean up old files from an assistant
python -m data_generation.loading.manage_assistants clean --id asst_abc123
```

## Memory Visualization

Use the Memory Visualization tools to explore Jon's memory structure:

```bash
# Visualize an assistant's memory
python -m data_generation.jon_tools visualize --assistant-id asst_abc123

# Visualize from local data file
python -m data_generation.jon_tools visualize --input data_generation/output/jon_raw_data_20230526_123045.json

# Generate a specific type of visualization
python -m data_generation.jon_tools visualize --type graph --assistant-id asst_abc123
```

Or use the individual script:

```bash
# Visualize an assistant's memory
python -m data_generation.visualization.visualize_memory --assistant-id asst_abc123

# Visualize from local data file
python -m data_generation.visualization.visualize_memory --input data_generation/output/jon_raw_data_20230526_123045.json

# Generate a specific type of visualization
python -m data_generation.visualization.visualize_memory --type graph --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type topics --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type sentiment --assistant-id asst_abc123
python -m data_generation.visualization.visualize_memory --type dashboard --assistant-id asst_abc123
```

This tool generates interactive visualizations of Jon's memory structure:

1. **Knowledge Graph**: Shows connections between topics, entities, and concepts
2. **Topic Treemap**: Displays hierarchical organization of Jon's knowledge by topic clusters
3. **Sentiment Analysis**: Visualizes emotional tone across different topics
4. **Memory Dashboard**: Comprehensive overview with multiple visualizations

Visualizations are saved as interactive HTML files in the `data_generation/output` directory.

Requirements:
```bash
pip install matplotlib networkx plotly seaborn pandas wordcloud scikit-learn nltk
```

## Migrating from Vector Store to Retrieval API

If you have existing data in the legacy vector store, you can migrate it to the new Retrieval API:

```bash
# Check what would be migrated without uploading
python -m data_generation.jon_tools migrate --vector-dir vector_store --dry-run

# Migrate data from vector store to retrieval API
python -m data_generation.jon_tools migrate --vector-dir vector_store

# Specify an existing assistant to add data to
python -m data_generation.jon_tools migrate --vector-dir vector_store --assistant-id asst_abc123
```

Or use the individual script:

```bash
# Check what would be migrated without uploading
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store --dry-run

# Migrate data from vector store to retrieval API
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store

# Specify an existing assistant to add data to
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store --assistant-id asst_abc123
```

## Preparing Fine-Tuning Data

Use the Fine-Tuning tools to validate and prepare data:

```bash
# Prepare data for fine-tuning
python -m data_generation.jon_tools train prepare --file data_generation/output/jon_fine_tuning_20230526_123045.jsonl

# Specify output file and test split
python -m data_generation.jon_tools train prepare --file path/to/data.jsonl --output prepared_data.jsonl --test-split 0.2

# Start fine-tuning job directly
python -m data_generation.jon_tools train prepare --file path/to/data.jsonl --fine-tune --model gpt-3.5-turbo
```

Or use the individual script:

```bash
# Prepare data for fine-tuning
python -m data_generation.training.prepare_fine_tuning --file data_generation/output/jon_fine_tuning_20230526_123045.jsonl

# Specify output file and test split
python -m data_generation.training.prepare_fine_tuning --file path/to/data.jsonl --output prepared_data.jsonl --test-split 0.2

# Start fine-tuning job directly
python -m data_generation.training.prepare_fine_tuning --file path/to/data.jsonl --fine-tune --model gpt-3.5-turbo
```

## Managing Fine-Tuning Jobs

Use the Fine-Tuning job management tools:

```bash
# List all fine-tuning jobs
python -m data_generation.jon_tools train list

# Check status of a specific job
python -m data_generation.jon_tools train status ft-abc123

# Register a completed model in your .env file
python -m data_generation.jon_tools train status ft-abc123 --register

# Cancel a running job
python -m data_generation.jon_tools train cancel ft-abc123
```

Or use the individual script:

```bash
# List all fine-tuning jobs
python -m data_generation.training.monitor_jobs list

# Check status of a specific job
python -m data_generation.training.monitor_jobs status ft-abc123

# Register a completed model in your .env file
python -m data_generation.training.monitor_jobs status ft-abc123 --register

# Cancel a running job
python -m data_generation.training.monitor_jobs cancel ft-abc123
```

## Testing Generated Data

Use the data testing tools to explore and test generated data:

```bash
# View sample Q&A pairs
python -m data_generation.jon_tools test --file path/to/vector_data.jsonl --type qa

# View retrieval format data
python -m data_generation.jon_tools test --file path/to/retrieval_data.jsonl --type retrieval

# View conversations
python -m data_generation.jon_tools test --file path/to/raw_conversations.json --type conversation

# Test interactively
python -m data_generation.jon_tools test --file path/to/data.jsonl --interactive
```

Or use the individual script:

```bash
# View sample Q&A pairs
python -m data_generation.utils.test_jon_data --file path/to/vector_data.jsonl --type qa

# View retrieval format data
python -m data_generation.utils.test_jon_data --file path/to/retrieval_data.jsonl --type retrieval

# View conversations
python -m data_generation.utils.test_jon_data --file path/to/raw_conversations.json --type conversation

# Test with your fine-tuned model interactively
python -m data_generation.utils.test_jon_data --file path/to/data.jsonl --interactive --model ft:gpt-3.5-turbo-0613:personal:jon-v1

# Test with retrieval-backed assistant interactively
python -m data_generation.utils.test_jon_data --file path/to/retrieval_data.jsonl --interactive --assistant-id asst_abc123
```

## Workflow Example

Here's a complete workflow example using the unified CLI tool:

1. **Generate data**:
   ```bash
   python -m data_generation.jon_tools generate --qa-pairs 500 --conversations 100 --parallel
   ```

2. **Create a dedicated assistant** (optional):
   ```bash
   python -m data_generation.jon_tools assistants create --name "Jon Memory" --save
   ```

3. **Load data into retrieval store**:
   ```bash
   python -m data_generation.jon_tools load --file data_generation/output/jon_retrieval_data_*.jsonl
   ```

4. **Prepare and start fine-tuning**:
   ```bash
   python -m data_generation.jon_tools train prepare --file data_generation/output/jon_fine_tuning_*.jsonl --fine-tune
   ```

5. **Monitor job status**:
   ```bash
   python -m data_generation.jon_tools train status ft-abc123 --register
   ```

6. **Test the implementation**:
   ```bash
   # Test fine-tuned model
   python -m data_generation.jon_tools test --interactive --model ft:gpt-3.5-turbo-0613:personal:jon-v1
   
   # Test retrieval-backed memory
   python -m data_generation.jon_tools test --interactive --assistant-id asst_abc123
   ```

7. **Visualize Jon's memory structure**:
   ```bash
   python -m data_generation.jon_tools visualize --assistant-id asst_abc123 --type dashboard
   ```

This workflow provides a complete pipeline for generating, preparing, fine-tuning, and using Jon data in your SchultzGPT application with the Retrieval API. 