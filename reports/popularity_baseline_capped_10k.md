# Popularity Baseline

This baseline uses weighted item popularity from the train split only.

Event weights: view=1.0, addtocart=3.0, transaction=5.0.

Evaluation protocol: capped to the first 10000 evaluable users per split, matching the learned-model capped evaluation protocol.

## Validation metrics

- K=5: Recall@K=0.006635, HitRate@K=0.007200, NDCG@K=0.003523
- K=10: Recall@K=0.008974, HitRate@K=0.010300, NDCG@K=0.004287
- K=20: Recall@K=0.013631, HitRate@K=0.017500, NDCG@K=0.005597
- K=50: Recall@K=0.022049, HitRate@K=0.028700, NDCG@K=0.007361

## Test metrics

- K=5: Recall@K=0.004884, HitRate@K=0.005500, NDCG@K=0.002884
- K=10: Recall@K=0.006010, HitRate@K=0.007400, NDCG@K=0.003288
- K=20: Recall@K=0.009975, HitRate@K=0.013200, NDCG@K=0.004339
- K=50: Recall@K=0.020517, HitRate@K=0.028300, NDCG@K=0.006575
