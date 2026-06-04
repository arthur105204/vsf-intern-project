# Two-Tower ID-Only Retrieval

Minimal user/item embedding model trained with in-batch negatives on the train split only.

## Training diagnostics

- Train rows used: 1355957
- Unique train users used: 319045
- Unique train items used: 100648
- Model users in vocab: 319045
- Model items in vocab: 100648
- Vocab from effective train: True
- Min user interactions: 2
- Min item interactions: 2
- Training loss history: [6.125339900290754]

## Validation coverage diagnostics

- Known-user coverage: 1088/10000 (10.88%)
- Known-item coverage: 23519 eval rows in vocab (85.07%)
- Ground-truth items in vocab: 11161/14387 (77.58%)

## Validation metrics

- K=5: Recall@K=0.000458 (baseline 0.005981), HitRate@K=0.001500 (baseline 0.006512), NDCG@K=0.000475 (baseline 0.003217)
- K=10: Recall@K=0.000771 (baseline 0.007537), HitRate@K=0.002400 (baseline 0.008264), NDCG@K=0.000614 (baseline 0.003732)
- K=20: Recall@K=0.001331 (baseline 0.012040), HitRate@K=0.003500 (baseline 0.014457), NDCG@K=0.000790 (baseline 0.004942)
- K=50: Recall@K=0.002251 (baseline 0.022231), HitRate@K=0.005900 (baseline 0.027091), NDCG@K=0.000984 (baseline 0.007049)

## Test coverage diagnostics

- Known-user coverage: 628/10000 (6.28%)
- Known-item coverage: 22173 eval rows in vocab (80.33%)
- Ground-truth items in vocab: 10702/14564 (73.48%)

## Test metrics

- K=5: Recall@K=0.000102 (baseline 0.005499), HitRate@K=0.000300 (baseline 0.006094), NDCG@K=0.000173 (baseline 0.003226)
- K=10: Recall@K=0.000306 (baseline 0.007064), HitRate@K=0.000700 (baseline 0.007841), NDCG@K=0.000221 (baseline 0.003740)
- K=20: Recall@K=0.000627 (baseline 0.010670), HitRate@K=0.001600 (baseline 0.012777), NDCG@K=0.000301 (baseline 0.004695)
- K=50: Recall@K=0.001055 (baseline 0.019197), HitRate@K=0.002900 (baseline 0.023870), NDCG@K=0.000385 (baseline 0.006448)

