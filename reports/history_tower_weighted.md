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
- Use event weights in history: True
- Training loss history: [5.583500058932681]

## Validation coverage diagnostics

- Eval users with usable history: 1088/10000 (10.88%)
- Known-item coverage: 23519 eval rows in vocab (85.07%)
- Ground-truth items in vocab: 11161/14387 (77.58%)

## Validation metrics

- K=5: Recall@K=0.001755 (popularity 0.005981, id-only 0.000458, unweighted history 0.001707), HitRate@K=0.003900 (popularity 0.006512, id-only 0.001500, unweighted history 0.004500), NDCG@K=0.001702 (popularity 0.003217, id-only 0.000475, unweighted history 0.001870)
- K=10: Recall@K=0.002359 (popularity 0.007537, id-only 0.000771, unweighted history 0.002427), HitRate@K=0.005500 (popularity 0.008264, id-only 0.002400, unweighted history 0.006000), NDCG@K=0.001895 (popularity 0.003732, id-only 0.000614, unweighted history 0.002022)
- K=20: Recall@K=0.003544 (popularity 0.012040, id-only 0.001331, unweighted history 0.003569), HitRate@K=0.008000 (popularity 0.014457, id-only 0.003500, unweighted history 0.008700), NDCG@K=0.002232 (popularity 0.004942, id-only 0.000790, unweighted history 0.002339)
- K=50: Recall@K=0.005290 (popularity 0.022231, id-only 0.002251, unweighted history 0.005407), HitRate@K=0.011400 (popularity 0.027091, id-only 0.005900, unweighted history 0.012100), NDCG@K=0.002653 (popularity 0.007049, id-only 0.000984, unweighted history 0.002777)

## Test coverage diagnostics

- Eval users with usable history: 628/10000 (6.28%)
- Known-item coverage: 22173 eval rows in vocab (80.33%)
- Ground-truth items in vocab: 10702/14564 (73.48%)

## Test metrics

- K=5: Recall@K=0.000520 (popularity 0.005499, id-only 0.000102, unweighted history 0.000378), HitRate@K=0.001800 (popularity 0.006094, id-only 0.000300, unweighted history 0.001200), NDCG@K=0.000653 (popularity 0.003226, id-only 0.000173, unweighted history 0.000491)
- K=10: Recall@K=0.000926 (popularity 0.007064, id-only 0.000306, unweighted history 0.000837), HitRate@K=0.002800 (popularity 0.007841, id-only 0.000700, unweighted history 0.002300), NDCG@K=0.000768 (popularity 0.003740, id-only 0.000221, unweighted history 0.000637)
- K=20: Recall@K=0.001413 (popularity 0.010670, id-only 0.000627, unweighted history 0.001325), HitRate@K=0.003700 (popularity 0.012777, id-only 0.001600, unweighted history 0.003600), NDCG@K=0.000867 (popularity 0.004695, id-only 0.000301, unweighted history 0.000781)
- K=50: Recall@K=0.002281 (popularity 0.019197, id-only 0.001055, unweighted history 0.002140), HitRate@K=0.005700 (popularity 0.023870, id-only 0.002900, unweighted history 0.005500), NDCG@K=0.001041 (popularity 0.006448, id-only 0.000385, unweighted history 0.000950)

## Conclusion

History features improve over the ID-only tower on both validation and test, so the query-side history signal is worth carrying forward.

Event-weighted history pooling improves over the unweighted history tower on the capped test slice, but not consistently on validation.

The history tower is still below the weighted popularity baseline, so the next step should focus on richer history/category signal rather than more ID-only retraining.
