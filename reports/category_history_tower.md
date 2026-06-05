# Category-Aware History Two-Tower

History-aware query tower with event-weighted pooling and category-aware item tower.

## Training diagnostics

- Train rows used: 1355957
- Unique train users used: 319045
- Unique train items used: 100648
- Users with non-empty train history: 305553
- Model items in vocab: 100648
- Model vocab items with category: 88605
- Min user interactions: 2
- Min item interactions: 2
- Max history length: 20
- Use event weights in history: True
- Use category embeddings: True
- Training loss history: [4.698086177236227]

## Validation coverage diagnostics

- Eval users with usable history: 1088/10000 (10.88%)
- Known-item coverage: 23519 eval rows in vocab (85.07%)
- Ground-truth items in vocab: 11161/14387 (77.58%)
- Ground-truth items with category: 10332/14387 (71.81%)

## Test coverage diagnostics

- Eval users with usable history: 628/10000 (6.28%)
- Known-item coverage: 22173 eval rows in vocab (80.33%)
- Ground-truth items in vocab: 10702/14564 (73.48%)
- Ground-truth items with category: 9771/14564 (67.09%)

## Comparison at K=50

| Model | Validation Recall@50 | Validation HitRate@50 | Validation NDCG@50 | Test Recall@50 | Test HitRate@50 | Test NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline | 0.022231 | 0.027091 | 0.007049 | 0.019197 | 0.023870 | 0.006448 |
| ID-only two-tower | 0.002251 | 0.005900 | 0.000984 | 0.001055 | 0.002900 | 0.000385 |
| Unweighted history tower | 0.005407 | 0.012100 | 0.002777 | 0.002140 | 0.005500 | 0.000950 |
| Event-weighted history tower | 0.005290 | 0.011400 | 0.002653 | 0.002281 | 0.005700 | 0.001041 |
| Category-aware history tower | 0.006034 | 0.012500 | 0.002782 | 0.002143 | 0.004500 | 0.000872 |

## Conclusion

Category embedding does not improve over the history-only tower consistently on the capped test slice, so the current category formulation is not yet a clear win.

The weighted popularity baseline remains stronger, so the next step should focus on richer session/history construction or deeper item/category feature modeling rather than infrastructure work.
