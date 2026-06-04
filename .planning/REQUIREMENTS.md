# REQUIREMENTS

## Functional Requirements

1. Preprocessing
   - ingest `events.csv`, `item_properties.csv`, `category_tree.csv`
   - time-based train/validation/test split
   - produce `data/processed/` feature tables (do not commit raw files)

2. Baselines
   - popularity baseline
   - item co-occurrence baseline (optional)

3. Two-tower model
   - user tower (visitor ID + pooled history/BOW)
   - item tower (item ID + optional metadata)
   - train with in-batch negatives and softmax retrieval loss
   - optional LogQ correction toggle

4. Offline indexing
   - export item embeddings
   - build vector index (brute-force and optional ANN)

5. Evaluation & reporting
   - compute Recall@K, NDCG@K, HitRate@K for K in `config.json`
   - produce a single results table and short analysis

6. Demo/API
   - simple endpoint or Streamlit demo to query top-K

## Non-functional Requirements
- reproducible scripts or notebooks
- clear README and docs
- small, focused deliverables for mentor review

## Acceptance Criteria (UAT)
- able to reproduce train/val/test split from instructions
- baselines and two-tower results reported in table
- item embeddings exported and index can retrieve candidates
- demo/API returns plausible top-K for a sample visitor
