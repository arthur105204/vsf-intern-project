# Code Walkthrough

This document explains the current RetailRocket two-tower retrieval project for mentor review. It is intentionally focused on what exists today, not future features.

## Project scope

The project builds an offline candidate retrieval prototype for ecommerce recommendations. The main task is to retrieve likely relevant item candidates from RetailRocket interaction data using:

- a weighted popularity baseline
- an ID-only two-tower model
- a history-aware two-tower model
- an event-weighted history tower
- a category-aware history item tower

Out of scope for the current stage:

- reranking
- FAISS / ANN serving
- production API
- MLOps
- LogQ or advanced objective corrections

## Data flow

| Stage | File | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| Inventory | `scripts/inventory_data.py` | Raw RetailRocket CSVs | Inventory report | Check available files and schemas |
| Preprocess | `scripts/prepare_dataset.py` | Raw events | Processed interactions | Normalize event data for modeling |
| Split | `scripts/split_dataset.py` | Processed interactions | Train/val/test parquet files | Chronological offline evaluation |
| EDA | `scripts/eda.py` | Processed data | EDA reports | Understand sparsity, events, popularity |
| Baseline | `scripts/run_popularity_baseline.py` | Train/val/test splits | Metrics/report | Establish a strong simple reference |
| ID-only train/eval | `scripts/train_two_tower.py`, `scripts/evaluate_two_tower.py` | Splits and checkpoint | Model, metrics, report | Basic two-tower learning check |
| History train/eval | `scripts/train_history_tower.py`, `scripts/evaluate_history_tower.py` | Train-only histories | Model, metrics, report | Use behavior history as query signal |
| Category train/eval | `scripts/train_category_history_tower.py`, `scripts/evaluate_category_history_tower.py` | Item metadata and train split | Model, metrics, report | Add category embedding to item tower |

## Core source files

### `src/evaluation/metrics.py`

Defines retrieval metrics:

- `recall_at_k`
- `hit_rate_at_k`
- `ndcg_at_k`
- `evaluate_recommendations`

The metric implementation is clean and tested. It deduplicates predictions, skips users with empty ground truth, and treats missing predictions as empty recommendations.

### `src/models/baselines.py`

Defines the weighted popularity baseline and evaluation helpers:

- `DEFAULT_EVENT_WEIGHTS`
- `build_popularity_model`
- `recommend_for_user`
- `build_ground_truth`
- `build_seen_items_for_users`

The popularity model uses train-only item popularity and excludes train-seen items during recommendation.

### `src/models/two_tower.py`

This is the main model module. It currently contains:

- config and vocab classes
- ID-only two-tower model
- history-aware query model
- category-aware item tower support
- train-frame preparation
- vocabulary construction
- history construction
- category metadata mapping
- training loops
- recommendation helpers
- checkpoint save/load logic

This file is functional but too broad. It should not keep growing before a future cleanup.

## Current model behavior

| Model | Query representation | Item representation | Notes |
| --- | --- | --- | --- |
| ID-only two-tower | `visitor_id` embedding | `item_id` embedding | Weak on sparse users |
| History tower | Average of prior train-history item embeddings | `item_id` embedding | Improves over ID-only |
| Event-weighted history | Weighted average of prior item embeddings | `item_id` embedding | Small, inconsistent gain |
| Category-aware history | Weighted history item embeddings | `item_id + category_id` embedding | Category is item-side only |

## Evaluation protocol

All models are trained on the train split only. Validation/test ground truth is built from held-out interactions.

The learned model evaluators usually use:

- `eval_user_limit = 10000`
- first N evaluable users from the ground-truth dictionary
- train-only seen-item filtering
- top-K metrics at K = 5, 10, 20, 50

The popularity baseline now supports the same capped protocol via:

```bash
python scripts/run_popularity_baseline.py --eval_user_limit 10000
```

Important caveat: earlier popularity metrics were full validation/test metrics, while learned model metrics were capped. The new capped popularity report should be used for matched comparison.

## Known limitations

- The current history protocol uses train-only history for validation/test users. This prevents leakage, but it is not a session-prefix next-item prediction setup.
- Category metadata is timestamped and should be handled with a train-time cutoff when used for modeling.
- Category embedding is currently added to the item tower only. The query tower does not explicitly aggregate category embeddings.
- In-batch negatives are a reasonable prototype objective, but they can include false negatives.
- One-epoch CPU training is useful for gates, not final model quality.
- `two_tower.py` should be split into smaller modules before substantial additional model complexity.

