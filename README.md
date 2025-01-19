# QUACK 🦆: Query Answering with User Feedback

This repository contains an implementation for training and models for approximate query answering on knowledge graphs, that can capture preferences over entities during query answering. These preferences are specified as a selection of entities that can occur in variables of the query, and the selection can indicate both positive and negative preferences.

We assume a given knowledge graph and a dataset of queries and their answers. The process for training and evaluating the model is as follows:

1. **Automatic preference generation**: Since it is impractical to manually specify preferences for queries at a large scale, we automatically generate preferences for a subset of complex queries. We use a clustering algorithm to group similar answers and select the most frequent entities in each cluster as preferences.
2. **Training QA model**: We make use of QTO, a state-of-the-art model for complex query answering, to obtain rankings of entities for variables in the query. This step provides answers to a query **without** considering the preferences.
3. **Training and evaluating the preference model**: Using the preferences generated in step 1, we train a re-ranking model for modifying the predictions of QTO (obtained in step 2) to align with the preferences. We evaluate the model on a held-out set of queries.

## 1. Automatic Preference Generation

Current datasets for approximate query answering on knowledge graphs contain pairs of queries and their answers. Each query has at least one variable (the target variable), but it can also contain intermediate variables. We extend these datasets by enumerating the possible values of variables, and grouping them according to a proxy of similarity (such as cosine similarity based on an embedding of entitiy descriptions). These groups model **soft** preferences over the entities that can occur in the variables of the query.

We implement this in two steps: (1) **embedding** entities based on their textual descriptions, and (2) **clustering** variable assignments using entity embeddings. The `main.py` script provides two commands for this:

1. `embed`: Embeds entities based on their textual descriptions.
2. `generate`: Clusters query results based on precomputed embeddings.

### Embedding Entities

This command processes textual descriptions of entities and generates vector embeddings using a pre-trained sentence transformer. The computed embeddings, along with the entity-to-row mappings and descriptions, are saved in a .pt file under the `datasets/<dataset_name>/mapping/` directory. These embeddings are later used for clustering. To embed entity descriptions:

```bash
python -m quack embed <data_path> --embedding_model <model_name>
```

#### Arguments:

- `data_path`: The path to the data, containing triples and entity descriptions.
- `--embedding_model`: Sentence transformer model to use for embedding (default: `dunzhang/stella_en_400M_v5`).
- `--batch_size`: Batch size for embedding (default: 64).
- `--num_workers`: Number of workers for data loading (default: 0).

#### Example:

```bash
python -m quack embed data/fb15k-betae --embedding_model all-MiniLM-L6-v2
```

### Generating Sessions
Given the answer set of a query, we can generate a partition consisting of *positive* examples indicating a preference, and *negative* examples consisting of the rest of the entities in the answer set. We generate this partition using the embeddings from the previous step and a hierarchical clustering algorithm.
A partition can be used to simulate *sessions*, where one item at a time from the positive set is provided to a re-ranking model to modify the answers to a query. Use the `generate` command to generate the session data for complex queries:

```bash
python -m quack generate <data_path>
```

#### Arguments:

- `data_path`: The path to the data, also containing the embeddings from the previous step.
- `--tasks`: Types of queries used to generate data, with tasks separated by a dot.
  (default: `1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u-DNF.up-DNF`)
- `--min_answer_threshold`: Minimum number of answers required to process a query (default: 10).
- `--max_answer_threshold`: Maximum number of answers required to process a query (default: 100).
- `--max_num_sessions`: Maximum number of sessions to generate per query (default: 5).

#### Example:

```bash
python -m quack generate data/fb15k-betae --min_answer_threshold 10 ---max_answer_threshold 100
```

### Getting statistics

Once the `embed` and `generate` commands have been run, we can inspect the statistics of the generated data with the `describe` command. The output is a table of query and session count for each query type in the dataset.

#### Arguments

- `data_path`: The path to the data, containing the generated query session data.

#### Example:

```bash
python -m quack describe data/fb15k-betae
```


## 2. Training QA Model

In this step we train QTO, a state-of-the-art model for complex query answering on knowledge graphs. QTO answers complex queries by relying on ComplEx, a model for 1-hop link prediction. Therefore, we first train ComplEx, and then reuse it with QTO to answer queries.

### 2.1 Data Preparation

Use the `quack.qto.complex` module with the `preprocess` command to preprocess the data for training a ComplEx link prediction model:

```bash
python -m quack.qto.complex preprocess --data_path data/fb15k-betae
python -m quack.qto.complex preprocess --data_path data/fb15k237-betae
python -m quack.qto.complex preprocess --data_path data/nell-betae
```

### 2.2 Training link prediction model

Use the `quack.qto.complex` module and the `train` command to train a ComplEx model for link prediction:

```bash
python -m quack.qto.complex train --data_path data/fb15k-betae --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 100
python -m quack.qto.complex train --data_path data/fb15k237-betae --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
python -m quack.qto.complex train --data_path data/nell-betae --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100 
```

### 2.3 Evaluating QTO with the trained ComplEx model

```bash
python -m quack.qto.query
```


