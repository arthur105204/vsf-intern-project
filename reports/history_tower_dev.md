# Two-Tower History Query Retrieval

History-aware query tower using train-only item interaction history with average pooling.

## Training diagnostics

- Train rows used: 1355957
- Unique train users used: 319045
- Unique train items used: 100648
- Users with non-empty train history: 305553
- Model items in vocab: 100648
- Vocab from effective train: True
- Min user interactions: 2
- Min item interactions: 2
- Max history length: 20
- Training loss history: [5.575522424230579]

## Validation coverage diagnostics

- Eval users with usable history: 1088/10000 (10.88%)
- Known-item coverage: 23519 eval rows in vocab (85.07%)
- Ground-truth items in vocab: 11161/14387 (77.58%)

## Validation metrics

- K=5: Recall@K=0.001707 (popularity 0.005981, id-only 0.000458), HitRate@K=0.004500 (popularity 0.006512, id-only 0.001500), NDCG@K=0.001870 (popularity 0.003217, id-only 0.000475)
- K=10: Recall@K=0.002427 (popularity 0.007537, id-only 0.000771), HitRate@K=0.006000 (popularity 0.008264, id-only 0.002400), NDCG@K=0.002022 (popularity 0.003732, id-only 0.000614)
- K=20: Recall@K=0.003569 (popularity 0.012040, id-only 0.001331), HitRate@K=0.008700 (popularity 0.014457, id-only 0.003500), NDCG@K=0.002339 (popularity 0.004942, id-only 0.000790)
- K=50: Recall@K=0.005407 (popularity 0.022231, id-only 0.002251), HitRate@K=0.012100 (popularity 0.027091, id-only 0.005900), NDCG@K=0.002777 (popularity 0.007049, id-only 0.000984)

## Test coverage diagnostics

- Eval users with usable history: 628/10000 (6.28%)
- Known-item coverage: 22173 eval rows in vocab (80.33%)
- Ground-truth items in vocab: 10702/14564 (73.48%)

## Test metrics

- K=5: Recall@K=0.000378 (popularity 0.005499, id-only 0.000102), HitRate@K=0.001200 (popularity 0.006094, id-only 0.000300), NDCG@K=0.000491 (popularity 0.003226, id-only 0.000173)
- K=10: Recall@K=0.000837 (popularity 0.007064, id-only 0.000306), HitRate@K=0.002300 (popularity 0.007841, id-only 0.000700), NDCG@K=0.000637 (popularity 0.003740, id-only 0.000221)
- K=20: Recall@K=0.001325 (popularity 0.010670, id-only 0.000627), HitRate@K=0.003600 (popularity 0.012777, id-only 0.001600), NDCG@K=0.000781 (popularity 0.004695, id-only 0.000301)
- K=50: Recall@K=0.002140 (popularity 0.019197, id-only 0.001055), HitRate@K=0.005500 (popularity 0.023870, id-only 0.002900), NDCG@K=0.000950 (popularity 0.006448, id-only 0.000385)

## Conclusion

History features improve over the ID-only tower on both validation and test, so the query-side history signal is worth carrying forward.

The history tower is still below the weighted popularity baseline, so the next step should focus on richer history/category signal rather than more ID-only retraining.
