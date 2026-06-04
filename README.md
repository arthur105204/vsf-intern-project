# Production-Inspired Two-Tower Retrieval System for Ecommerce Recommendation

This repository contains a production-inspired learning prototype for ecommerce candidate retrieval using the RetailRocket Ecommerce Dataset from Kaggle.

The goal is to demonstrate the core applied ML workflow behind a two-stage recommendation system: data preprocessing, baseline retrieval, two-tower training, top-K offline evaluation, precomputed item embeddings, vector search retrieval, and a simple demo/API.

This project is production-inspired, not production-ready.

## What It Shows

- two-stage recommendation with candidate generation and ranking separation
- two-tower architecture for retrieval
- embeddings and dot product similarity
- in-batch negatives and LogQ correction
- BOW history features for user/session context
- offline item embedding generation
- ANN/vector search with ScaNN or HNSW-style retrieval
- retrieval metrics such as Recall@K, NDCG@K, and HitRate@K
- logged feedback bias awareness and retrieval-ranking interaction

## Dataset

Dataset: RetailRocket Ecommerce Dataset from Kaggle.

Raw data is not committed to GitHub. Download it locally into:

```text
data/raw/retailrocket/
```

Download command:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

Expected raw files:

```text
data/raw/retailrocket/events.csv
data/raw/retailrocket/item_properties.csv
data/raw/retailrocket/category_tree.csv
```

## Project Scope

In scope:

- preprocessing implicit feedback events
- baseline retrieval models
- two-tower retrieval training
- offline evaluation with top-K metrics
- offline item embedding export
- vector search or brute-force retrieval
- simple demo or API

Out of scope:

- real GenRec
- distributed training
- DeepSpeed or Ray
- real Lucene Plus
- full real-time pipeline
- full ranking system
- real-time feature streaming
- production A/B testing
- full MLOps

## Recommended Repo Layout

```text
README.md
REPORT/
  PRD.md
  DATASET.md
  docs/
    README.md
    TECHNICAL_DESIGN.md
    EXPERIMENT_PLAN.md
```

## How to Run

1. Create a Python environment and install the project dependencies.
2. Download the RetailRocket dataset into `data/raw/retailrocket/`.
3. Run preprocessing to build implicit feedback splits and feature tables.
4. Train the baselines and the two-tower retrieval model.
5. Export item embeddings and build the ANN or brute-force retrieval index.
6. Start the demo or API and query recommendations for a visitor/session.

## Verified Phase 1 Commands

The following commands were run successfully in this repository:

```bash
# download raw data
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip

# inventory raw files
python scripts/inventory_data.py --input data/raw/retailrocket --out data/inventory.json

# prepare dataset
python scripts/prepare_dataset.py --raw data/raw --out data/processed/dataset.parquet

# generate EDA summary
python scripts/eda.py --input data/processed/dataset.parquet --out reports/eda_summary.md

# create chronological splits
python scripts/split_dataset.py --input data/processed/dataset.parquet --out_dir data/processed/splits --min_gap_seconds 0
```

Typical commands will look like this:

```bash
# train retrieval model
python -m src.train_two_tower

# evaluate top-K retrieval
python -m src.evaluate_retrieval

# export item embeddings and build vector index
python -m src.export_item_embeddings
python -m src.build_vector_index

# start demo or API
python -m src.api
```

Adjust the exact commands to match the implementation in this repository.

## Expected Deliverables

- preprocessing and split logic for implicit feedback
- baseline retrieval comparison
- two-tower training setup
- offline metrics report
- item embedding export
- vector search retrieval demo/API

## Documentation

- [Docs Index](REPORT/docs/README.md)
- [Technical Design](REPORT/docs/TECHNICAL_DESIGN.md)
- [Experiment Plan](REPORT/docs/EXPERIMENT_PLAN.md)
- [Dataset Notes](REPORT/DATASET.md)
- [Project PRD](REPORT/PRD.md)

## Notes

- Keep the raw dataset out of version control.
- Focus on retrieval, not a full ranking stack.
- Keep the implementation small, readable, and suitable for mentor review.
