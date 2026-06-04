 # PROJECT — Production-Inspired Two-Tower Retrieval System

Created: 2026-06-03

## Purpose
Build a production-inspired learning prototype for ecommerce candidate retrieval using the RetailRocket dataset. The focus is two-stage recommendation, two-tower retrieval, offline item embedding export, and vector-search-based candidate generation.

## Scope
- Main scope: two-tower candidate retrieval on RetailRocket
- Ranking: downstream context only, not implemented in the MVP
- Evaluation: offline top-K retrieval metrics
- Serving: simple demo/API, not production deployment

## Data
- Raw data path: `data/raw/retailrocket/`
- Raw data must not be committed
- Download command:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

## Out of Scope
- No full MLOps pipeline
- No distributed training
- No GenRec
- No reranking system
- No production deployment

## Current Phase Status
- Phase 1 completed: inventory, preprocessing, EDA, chronological split
- Next phase: baseline/model work after cleanup commit
# PROJECT — Production-Inspired Two-Tower Retrieval System

Short name: two-tower-retailrocket

Created: 2026-06-03

## Purpose
Build a production-inspired learning prototype for candidate retrieval in ecommerce using the RetailRocket dataset. Focus on two-stage recommendation (retrieval → ranking), two-tower retrieval, offline item embedding export, and ANN/brute-force retrieval serving.

## Data
Raw dataset path (local): data/raw/retailrocket/
Kaggle download command:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

Raw data MUST NOT be committed to Git.

## High-level goals
- reproducible preprocessing and time-based splits
- baselines (popularity, co-occurrence)
- two-tower retrieval models with incremental features
- offline item embedding export and ANN index
- offline evaluation with Recall@K, NDCG@K, HitRate@K
- a minimal demo/API to show retrieval

## Out of scope
- distributed training, real-time streaming, full ranking stack, GenRec, DeepSpeed/Ray, production A/B testing

## Contacts
Owner: Internship project
Reviewer: Mentor
