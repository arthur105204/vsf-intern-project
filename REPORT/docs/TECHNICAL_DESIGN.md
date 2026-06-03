# Technical Design

This document explains how the retrieval system works. For the experiment sequence and metrics plan, see [Experiment Plan](EXPERIMENT_PLAN.md). For dataset specifics, see [Dataset Notes](../DATASET.md).

## 1. Overview

This project implements a production-inspired two-tower retrieval system for ecommerce recommendation. The system is designed as a learning prototype for candidate generation, not a full production recommender.

The core idea is to map a user/session context and an item into the same embedding space, then rank items by dot product similarity.

## 2. System Goals

- model implicit feedback from RetailRocket ecommerce events
- support candidate generation for top-K recommendation
- precompute item embeddings offline
- serve online queries with fast vector search or brute-force retrieval
- evaluate retrieval quality with top-K metrics

## 3. Architecture Summary

The prototype follows a simple offline/online split:

- offline preprocessing builds implicit-feedback training data and item metadata
- offline training learns user and item embeddings
- offline indexing precomputes item vectors and builds a retrieval index
- online serving embeds the query context and returns top-K candidates

## 4. Data Pipeline

### 4.1 Raw Data

Raw data is downloaded locally to:

```text
data/raw/retailrocket/
```

Expected files:

```text
events.csv
item_properties.csv
category_tree.csv
```

### 4.2 Preprocessing

The preprocessing pipeline should:

1. clean and sort interaction logs by time
2. filter invalid or low-signal records
3. convert events into implicit feedback labels or weights
4. build user/session histories
5. construct train/validation/test splits by time
6. generate item metadata tables
7. create training examples for retrieval learning

### 4.3 Feature Views

The model can use two feature views:

- user/query features: visitor ID, recent history, and BOW history features
- item features: item ID, category, and simple item metadata

The design should keep feature construction lightweight so the prototype stays reproducible.

## 5. Model Architecture

### 5.1 Two-Tower Retrieval

The model has two encoders:

- user tower: encodes visitor or session context into a user embedding
- item tower: encodes item identity and metadata into an item embedding

The towers output vectors in a shared space. Retrieval uses dot product similarity:

```text
score(user, item) = user_embedding · item_embedding
```

### 5.2 User Tower

The user tower can start simple and grow incrementally:

- visitor ID embedding
- pooled history embedding
- BOW history features over recently viewed categories or items

This keeps the prototype aligned with common retrieval systems while remaining easy to debug.

### 5.3 Item Tower

The item tower can include:

- item ID embedding
- category embedding
- optional item property features

### 5.4 Training Objective

Recommended training options:

- in-batch negative sampling
- retrieval loss with softmax over candidate items
- optional LogQ correction to reduce sampled-popularity bias

LogQ correction is useful because logged feedback is biased toward popular items and exposure effects.

## 6. Retrieval Serving Design

### 6.1 Offline Indexing

After training, the item tower is run over the full item catalog to precompute item embeddings.

Artifacts produced offline:

- item embedding table
- item ID to vector mapping
- ANN index or brute-force lookup structure

### 6.2 Online Query Flow

The online path should be simple:

1. receive visitor/session input
2. build the query embedding with the user tower
3. search the item embedding space
4. return top-K candidate items with scores

### 6.3 Vector Search

The prototype can use either:

- brute-force dot-product search for simplicity
- ANN/vector search for faster retrieval at larger catalog sizes

If ANN is used, ScaNN or HNSW-style indexing is a good fit for the project scope.

## 7. Retrieval-Ranking Boundary

This project focuses on candidate generation only.

The retriever returns a compact candidate set that could later be passed to a ranking model. The ranking stage itself is out of scope.

The main learning objective is to understand retrieval-ranking interaction, not to implement the full downstream stack.

## 8. Evaluation

Retrieval quality should be evaluated offline with:

- Recall@K
- NDCG@K
- HitRate@K

Additional analysis can include:

- coverage
- popularity bias
- long-tail item performance
- example recommendations by user history type

## 9. Limitations

This prototype intentionally avoids several production concerns:

- no real-time feature streaming
- no distributed training
- no full ranking system
- no production A/B testing
- no full MLOps pipeline
- no real GenRec
- no Lucene Plus implementation

The design is sufficient for a mentor review and portfolio demonstration, but it is not production-ready.