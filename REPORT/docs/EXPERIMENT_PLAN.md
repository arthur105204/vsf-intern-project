# Experiment Plan

This document describes the evaluation sequence for the retrieval prototype. For architecture details, see [Technical Design](TECHNICAL_DESIGN.md). For dataset notes, see [Dataset Notes](../DATASET.md).

## 1. Objective

Measure how well a two-tower retrieval model improves candidate generation over simple baselines on the RetailRocket Ecommerce Dataset.

The experiments should show the effect of preprocessing choices, feature choices, negative sampling, and LogQ correction on top-K retrieval quality.

## 2. Evaluation Setup

### Splits

- time-based train/validation/test split
- hold out the most recent interactions for validation and testing
- avoid leakage across splits

### Metrics

- Recall@K
- NDCG@K
- HitRate@K
- coverage
- popularity bias analysis

Suggested K values:

```text
K = 5, 10, 20, 50
```

## 3. Baseline Sequence

Run the experiments in this order so the results build on each other:

1. Popularity baseline
2. Item co-occurrence baseline
3. Two-tower ID-only
4. Two-tower + BOW history
5. Two-tower + item metadata
6. In-batch negatives vs sampled negatives
7. LogQ correction on/off
8. Brute-force vs ANN retrieval

## 4. Experiments

| ID | Experiment | Purpose | Expected Signal |
| --- | --- | --- | --- |
| E1 | Popularity baseline | Establish a strong non-personalized reference | Good short-term signal, weak personalization |
| E2 | Item co-occurrence baseline | Check simple collaborative retrieval | Better than popularity for repeated patterns |
| E3 | Two-tower ID-only | Validate the retrieval architecture | Improvement over baselines if training works |
| E4 | Two-tower + BOW history | Test user/session context features | Better recall for session-aware behavior |
| E5 | Two-tower + item metadata | Test category/property features | Better long-tail generalization |
| E6 | In-batch negatives vs sampled negatives | Compare training signal quality | In-batch negatives should help retrieval learning |
| E7 | LogQ correction on/off | Measure logged feedback bias correction | More stable ranking under popularity bias |
| E8 | Brute-force vs ANN retrieval | Compare serving tradeoffs | ANN should reduce latency at some recall cost |

## 5. Comparison Table Template

| Model | Features | Negatives | LogQ | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | None | None | No | TBD | TBD | TBD | TBD |
| Co-occurrence | History co-clicks | None | No | TBD | TBD | TBD | TBD |
| Two-tower ID-only | Visitor ID, item ID | In-batch | No | TBD | TBD | TBD | TBD |
| Two-tower + BOW | ID + history BOW | In-batch | No | TBD | TBD | TBD | TBD |
| Two-tower + metadata | ID + item metadata | In-batch | Optional | TBD | TBD | TBD | TBD |

## 6. Expected Learnings

- whether two-tower retrieval beats popularity and simple co-occurrence baselines
- how much user/session history helps candidate generation
- whether item metadata improves generalization for sparse items
- whether LogQ correction reduces popularity bias in logged feedback
- how much ANN retrieval changes serving behavior compared with exact search
- how retrieval quality changes when the candidate set is built from precomputed item embeddings

## 7. Risks

- sparse interactions may make offline gains noisy
- popularity bias may dominate results if the dataset is not filtered carefully
- false negatives can reduce training signal quality
- time-based splits may leave limited validation examples for rare items
- ANN settings may trade off recall against latency more than expected

## 8. Fallback Plans

If an experiment does not behave as expected:

1. fall back to a simpler baseline and verify the evaluation pipeline
2. reduce model complexity to ID-only towers
3. simplify the history representation to BOW features
4. remove ANN and evaluate exact brute-force retrieval first
5. disable LogQ correction and compare the baseline learning curve

## 9. Success Criteria

The project is successful if it demonstrates:

- a working candidate retrieval pipeline
- reproducible preprocessing and offline evaluation
- at least one two-tower model that improves over popularity
- offline item embedding export
- a simple retrieval demo or API
- clear discussion of limitations and bias effects

## 10. Reporting Output

Each experiment should report:

- dataset split details
- training configuration
- metric table for Recall@K, NDCG@K, and HitRate@K
- qualitative examples of recommended items
- notes on popularity bias and retrieval-ranking interaction
