# QUACK 🦆: Query Answering with User Feedback

## Usage
The main script provides two commands:

1. `embed`: Embeds entities based on their textual descriptions.
2. `generate`: Clusters query results based on precomputed embeddings.

### Embedding Entities

This command processes textual descriptions of entities and generates vector embeddings using a pre-trained sentence transformer. The computed embeddings, along with the entity-to-row mappings and descriptions, are saved in a .pt file under the `datasets/<dataset_name>/mapping/` directory. These embeddings are later used for clustering. To embed entity descriptions:

```bash
python main.py embed --dataset <dataset_name> --embedding_model <model_name>
```

#### Options:

- `--dataset`: The name of the dataset to process (default: fb15k237).
- `--embedding_model`: Sentence transformer model to use for embedding (default: `dunzhang/stella_en_400M_v5`).
- `--batch_size`: Batch size for embedding (default: 64).
- `--num_workers`: Number of workers for data loading (default: 0).

#### Example:

```bash
python main.py embed --dataset fb15k237 --embedding_model all-MiniLM-L6-v2
```

### Generating Clusters
To cluster answers to queries:

```bash
python main.py generate --dataset <dataset_name> --num_answers_threshold <threshold>
```

#### Options:

- `--dataset`: The name of the dataset to process (default: fb15k237).
- `--num_answers_threshold`: Minimum number of answers required to process a query (default: 10).
- `--train_data`, `--valid_data`, `--test_data`: Data partitions for query generation.

#### Example:

```bash
python main.py generate --dataset fb15k237 --num_answers_threshold 15
```
