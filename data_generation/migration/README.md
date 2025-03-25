# Jon Data Migration Tools

This directory contains tools for migrating data from legacy storage systems to newer implementations:

## Scripts

- `migrate_vector_to_retrieval.py`: Migrates data from legacy vector store to OpenAI Retrieval API

## Usage Examples

```bash
# Check what would be migrated without uploading (dry run)
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store --dry-run

# Migrate data from vector store to retrieval API (creates new assistant)
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store

# Specify an existing assistant to add data to
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store --assistant-id asst_abc123

# Configure batch size for uploads
python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir vector_store --batch-size 30
```

## Migration Process

The migration tool performs the following steps:

1. Loads data from the vector store directory (`data.json` and `embeddings.jsonl`)
2. Converts the data to a format compatible with the OpenAI Retrieval API
3. Creates a new assistant or uses an existing one (if assistant-id is provided)
4. Uploads the converted data in batches to the OpenAI Retrieval API
5. Saves the assistant ID to your `.env` file (if requested)

This allows for a smooth transition from the legacy vector store to the OpenAI Retrieval API while preserving all your data.

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 