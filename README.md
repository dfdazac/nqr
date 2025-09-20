# SCORE: Similarity-Constrained Reweighting for Query Answering

This repository implements **SCORE** (Similarity-Constrained Reweighting), a novel method for preference-aware complex query answering on knowledge graphs. SCORE leverages entity embeddings from knowledge graph completion models to rerank query results based on user preferences, using similarity-based evidence aggregation in the logit space.

## Overview

SCORE addresses the problem of incorporating user preferences into complex query answering on knowledge graphs. Given a complex logical query and user feedback indicating preferred (positive) and non-preferred (negative) entities, SCORE reweights the initial query scores to better align with user preferences while preserving the overall ranking quality.

### Key Features

- **Similarity-based reweighting**: Uses complex-valued entity embeddings from ComplEx to compute similarities between candidate answers and preference examples
- **Logit-space evidence aggregation**: Operates in logit space to maintain score interpretability and enable principled combination of evidence
- **Support for complex queries**: Handles various query types including conjunctions, disjunctions, and negations
- **Flexible preference modes**: Supports both target-only and full variable preference incorporation


## Usage

### Basic Evaluation

Run SCORE on a dataset with the following command:

```bash
python -m quack.qto.query \
    --data_path data/fb15k237-betae \
    --kbc_path checkpoints/fb15k237-betae/fb15k237-betae_ComplEx_Rank1000_RegN3_Lmbda0.05_W4.0/model.pt \
    --fraction 10 \
    --thrshd 0.0002 \
    --neg_scale 3 \
    --do_test \
    --wp 1.0 \
    --wn 0.5 \
    --reranker score \
    --preference mixed \
    --preference_mode full
```

### Key Parameters

- `--reranker score`: Activates the SCORE reweighting method
- `--wp`, `--wn`: Weights for positive and negative preference evidence (default: 1.0 each)
- `--preference`: Type of preferences to use (`positive`, `negative`, `mixed`, or `none`)
- `--preference_mode`: Whether to apply preferences to `target` variable only or `full` query variables
- `--fraction`: Memory optimization parameter for large knowledge graphs
- `--thrshd`: Threshold for neural adjacency matrix construction

### Prerequisites

Before running SCORE, you need:

1. **Preprocessed dataset**: Knowledge graph data in the expected format
2. **Trained ComplEx model**: A ComplEx knowledge graph completion model
3. **Preference sessions**: Generated preference data (positive/negative entity examples)

### Data Preparation

1. **Preprocess the knowledge graph**:
```bash
python -m quack.qto.complex preprocess --data_path data/fb15k237-betae
```

2. **Train ComplEx model**:
```bash
python -m quack.qto.complex train \
    --data_path data/fb15k237-betae \
    --score_rel True \
    --model ComplEx \
    --rank 1000 \
    --learning_rate 0.1 \
    --batch_size 1000 \
    --lmbda 0.05 \
    --w_rel 4 \
    --max_epochs 100
```

3. **Generate preference sessions** (if not already available):
```bash
python -m quack embed data/fb15k237-betae
python -m quack generate data/fb15k237-betae
```
