# PRD — Production-Inspired Two-Tower Retrieval System for Ecommerce Recommendation

## 1. Project Overview

This project builds a production-inspired recommendation retrieval system using the RetailRocket Ecommerce Dataset. The goal is to understand and implement the applied ML workflow behind modern recommender systems, especially two-tower candidate retrieval.

The project focuses on:

* ecommerce implicit feedback
* sparse user-item interactions
* long-tail item distribution
* popularity bias
* two-tower retrieval modeling
* top-K recommendation evaluation
* offline item embedding generation
* vector search / ANN-based retrieval
* simple serving demo or API

This is not a full production recommender platform. It is a learning prototype designed to demonstrate applied ML engineering skills.

---

## 2. Problem Statement

Ecommerce platforms often contain many products, but each user only interacts with a small subset of them. Some products receive many clicks, while many products receive few or no interactions. This creates a sparse and long-tail recommendation problem.

The project aims to answer:

> Given a user or session interaction history, can we retrieve a relevant set of top-K product candidates efficiently?

The focus is on the **candidate generation / retrieval stage**, not the final ranking stage.

---

## 3. Dataset

### Main Dataset

**RetailRocket Ecommerce Dataset**

Source:

* Kaggle: Retailrocket recommender system dataset

Expected files:

```text
events.csv
item_properties.csv
category_tree.csv
```

Dataset description:

* `events.csv`: visitor-item behavior data
* `item_properties.csv`: item property metadata
* `category_tree.csv`: item category hierarchy

The dataset is based on real-world ecommerce behavior data. Values are hashed for confidentiality.

---

## 4. Dataset Acquisition

### Recommended way to download

Use the Kaggle dataset page manually or Kaggle API.

Manual option:

1. Go to Kaggle.
2. Search for `Retailrocket recommender system dataset`.
3. Download the dataset.
4. Extract it into the local project folder.

Kaggle API option:

```bash
pip install kaggle
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

Before using the Kaggle API, configure your Kaggle token:

```text
~/.kaggle/kaggle.json
```

The dataset should be stored locally in:

```text
data/raw/retailrocket/
```

Expected local structure:

```text
data/
  raw/
    retailrocket/
      events.csv
      item_properties.csv
      category_tree.csv
```

---

## 5. Dataset Storage Policy

Raw dataset files should not be committed to GitHub.

Reason:

* raw datasets can be large
* Kaggle datasets may have usage/license constraints
* repository should remain lightweight
* data can be reproduced by download instructions

Add this to `.gitignore`:

```gitignore
data/raw/
data/processed/
*.zip
*.csv
*.parquet
```

Recommended GitHub contents:

```text
README.md
docs/
notebooks/
src/
configs/
tests/
requirements.txt
.gitignore
```

Do not commit:

```text
events.csv
item_properties.csv
category_tree.csv
large processed datasets
model checkpoints
ANN index files
```

Optional small files allowed:

```text
data/sample/
  sample_events.csv
  sample_items.csv
```

Only add sample files if they are small and used for demo/testing.

---

## 6. Target Users

This project is mainly for:

* mentor review
* internship learning
* portfolio demonstration
* ML Engineer / Applied AI Engineer interview discussion

The end user of the demo is:

> A reviewer who wants to understand how the recommendation system retrieves candidate products for a user/session.

---

## 7. Goals

### Primary Goals

1. Understand ecommerce implicit feedback data.
2. Build simple recommendation baselines.
3. Implement a two-tower retrieval model.
4. Evaluate recommendations using top-K retrieval metrics.
5. Precompute item embeddings offline.
6. Retrieve top-K items using brute-force or vector search.
7. Build a simple demo/API to visualize recommendations.
8. Connect the implementation to production patterns from Uber/Google recommender systems.

### Learning Goals

The project should demonstrate understanding of:

* content-based vs collaborative filtering vs two-tower retrieval
* implicit feedback
* long-tail distribution
* popularity bias
* in-batch negatives
* false negatives
* LogQ correction
* BOW user/history features
* Recall@K / NDCG@K / HitRate@K
* ANN/vector search
* offline vs online serving architecture
* retrieval-ranking separation

---

## 8. Non-goals / Out of Scope

To keep the project focused and feasible, the following are out of scope:

* No need to implement real GenRec / generative recommender models.
* No need for distributed training.
* No need for DeepSpeed, Ray, or large-scale ML infrastructure.
* No need to implement real Lucene Plus or a production search platform.
* No need to build a full production real-time recommendation pipeline.
* No need for a full ranking system after retrieval.
* No need for near-real-time UserContext or streaming features.
* No need for production A/B testing.
* No need for full MLOps platform.
* No need for production-grade monitoring.

This project focuses on two-tower retrieval, not the full recommender stack.

---

## 9. Product Scope

### Input

The system should accept:

* user ID or visitor ID
* optional session history
* optional event history
* optional top-K value

Example:

```text
visitor_id = 12345
k = 10
```

### Output

The system should return:

* top-K recommended item IDs
* similarity scores
* optional item category/property information
* optional explanation based on user history/category overlap
* baseline comparison if available

Example output:

```json
{
  "visitor_id": "12345",
  "recommendations": [
    {
      "item_id": "987",
      "score": 0.82,
      "reason": "similar to previously viewed product categories"
    }
  ]
}
```

---

## 10. System Workflow

### Offline Training Workflow

```text
RetailRocket raw data
→ data cleaning
→ interaction filtering
→ implicit feedback construction
→ train/validation/test split
→ baseline models
→ two-tower model training
→ offline evaluation
→ export trained model
```

### Offline Indexing Workflow

```text
trained item tower
→ compute all item embeddings
→ save item embedding table
→ build vector search index
```

### Online Retrieval Workflow

```text
visitor/session input
→ user/query tower
→ query embedding
→ vector search / brute-force search
→ top-K product candidates
→ optional filtering
→ return recommendations
```

---

## 11. Model Scope

### Baselines

The project should include at least:

1. Popularity baseline
2. Item co-occurrence or simple collaborative filtering baseline if time allows

### Main Model

Two-tower retrieval model:

```text
User tower:
visitor_id embedding
optional user history / BOW features

Item tower:
item_id embedding
optional item category/property features

Similarity:
dot product or cosine similarity

Loss:
cross entropy / retrieval loss / contrastive loss
```

### Improvements to Compare

Potential improvements:

* event weighting
* item category features
* BOW user history features
* in-batch negatives
* LogQ correction
* false-negative awareness
* vector search retrieval

---

## 12. Evaluation Metrics

The project should evaluate recommendation retrieval using:

* Recall@K
* NDCG@K
* HitRate@K
* coverage
* popularity bias analysis

Possible K values:

```text
K = 5, 10, 20, 50
```

Evaluation should compare:

| Model                       | Metrics                     |
| --------------------------- | --------------------------- |
| Popularity baseline         | Recall@K, NDCG@K, HitRate@K |
| Two-tower ID-only           | Recall@K, NDCG@K, HitRate@K |
| Two-tower + item metadata   | Recall@K, NDCG@K, HitRate@K |
| Two-tower + BOW history     | Recall@K, NDCG@K, HitRate@K |
| Two-tower + LogQ correction | Recall@K, NDCG@K, HitRate@K |

---

## 13. Demo / API Requirements

The final project should include either a simple API or a lightweight demo.

### Option A — FastAPI

Endpoints:

```text
GET /health
GET /recommend?visitor_id={id}&k=10
GET /similar-items?item_id={id}&k=10
```

### Option B — Streamlit Demo

Features:

* select visitor ID
* show user interaction history
* click “Recommend”
* display top-K items
* show scores
* compare with popularity baseline

Recommended first version:

> Start with Streamlit for faster visualization. Add FastAPI later if time allows.

---

## 14. Repository Structure

Recommended repo structure:

```text
two-tower-retailrocket/
  README.md
  .gitignore
  requirements.txt

  docs/
    PRD.md
    TECHNICAL_DESIGN.md
    EXPERIMENT_PLAN.md
    REPORT.md

  data/
    raw/
      retailrocket/
        .gitkeep
    processed/
      .gitkeep
    sample/
      sample_events.csv

  notebooks/
    01_eda_retailrocket.ipynb
    02_baselines.ipynb
    03_two_tower_experiments.ipynb

  src/
    data/
      preprocess.py
      split.py

    features/
      build_features.py

    models/
      baselines.py
      two_tower.py
      losses.py

    training/
      train.py
      evaluate.py

    retrieval/
      build_embeddings.py
      build_index.py
      search.py

    api/
      main.py

    demo/
      app.py

  configs/
    retailrocket.yaml

  tests/
    test_metrics.py
    test_preprocess.py
```

---

## 15. Milestones

### Milestone 1 — Dataset Understanding

Deliverables:

* EDA notebook
* dataset statistics
* event distribution
* user/item sparsity analysis
* long-tail item analysis

### Milestone 2 — Baseline Models

Deliverables:

* popularity baseline
* optional co-occurrence baseline
* baseline metric table

### Milestone 3 — Two-Tower Model

Deliverables:

* ID-only two-tower model
* training script or notebook
* Recall@K / NDCG@K / HitRate@K evaluation

### Milestone 4 — Model Improvements

Deliverables:

* event weighting
* item metadata/category features
* BOW user history features
* LogQ correction if feasible
* experiment comparison table

### Milestone 5 — Retrieval Serving

Deliverables:

* precomputed item embeddings
* brute-force or FAISS retrieval
* top-K search script
* simple API or Streamlit demo

### Milestone 6 — Final Report

Deliverables:

* technical report
* project README
* architecture diagram
* experiment summary
* limitations and future work

---

## 16. Success Criteria

The project is successful if it demonstrates:

1. Clear understanding of RetailRocket ecommerce behavior data.
2. Clean preprocessing and train/test split.
3. At least one baseline recommender.
4. A working two-tower retrieval model.
5. Top-K evaluation metrics.
6. Offline item embedding generation.
7. Working top-K retrieval using brute-force or vector search.
8. Simple demo/API.
9. Clear documentation connecting the project to production recommender patterns.

---

## 17. Risks and Mitigations

### Risk 1 — Dataset is sparse and noisy

Mitigation:

* filter users/items with too few interactions
* start with a smaller subset
* use simple baselines first

### Risk 2 — Two-tower model underperforms popularity baseline

Mitigation:

* check data split
* check negative sampling
* use event weighting
* add item category/history features
* analyze popularity bias

### Risk 3 — Vector search adds complexity

Mitigation:

* start with brute-force retrieval
* add FAISS only after model works

### Risk 4 — Scope becomes too large

Mitigation:

* keep GenRec, distributed training, real-time ranking, and production MLOps out of scope
* focus on retrieval only

---

## 18. Final Project Statement

This project implements a production-inspired two-tower candidate retrieval system for ecommerce recommendation using the RetailRocket dataset. It covers the applied ML workflow from raw implicit feedback preprocessing, baseline recommendation, two-tower model training, top-K evaluation, offline item embedding generation, and vector-search-based retrieval to a simple demo/API. The goal is to demonstrate practical understanding of recommender systems and connect the prototype to real production patterns such as retrieval-ranking separation, ANN search, and model/index serving architecture.
