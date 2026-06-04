# Popularity Baseline

This baseline uses weighted item popularity from the train split only.

Event weights: view=1.0, addtocart=3.0, transaction=5.0.

## Validation metrics

- K=5: Recall@K=0.005981, HitRate@K=0.006512, NDCG@K=0.003217
- K=10: Recall@K=0.007537, HitRate@K=0.008264, NDCG@K=0.003732
- K=20: Recall@K=0.012040, HitRate@K=0.014457, NDCG@K=0.004942
- K=50: Recall@K=0.022231, HitRate@K=0.027091, NDCG@K=0.007049

## Test metrics

- K=5: Recall@K=0.005499, HitRate@K=0.006094, NDCG@K=0.003226
- K=10: Recall@K=0.007064, HitRate@K=0.007841, NDCG@K=0.003740
- K=20: Recall@K=0.010670, HitRate@K=0.012777, NDCG@K=0.004695
- K=50: Recall@K=0.019197, HitRate@K=0.023870, NDCG@K=0.006448
