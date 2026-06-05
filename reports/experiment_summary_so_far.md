# Experiment Summary So Far

## Project scope

This project is focused on **two-tower candidate retrieval on RetailRocket**. The goal is to establish a credible retrieval baseline, understand which signals matter, and decide what feature/model improvements are worth pursuing before any later-stage retrieval or production concerns.

Current scope completed:

- weighted popularity baseline
- ID-only two-tower
- history-aware query tower
- event-weighted history pooling
- category-aware history item tower

Not in scope yet:

- reranking
- FAISS / ANN retrieval
- API / serving
- MLOps / deployment

## Dataset and split summary

The project uses processed RetailRocket interactions with explicit train, validation, and test parquet splits. All learned models are trained on the train split only and evaluated on held-out validation/test interactions with:

- `Recall@K`
- `HitRate@K`
- `NDCG@K`

The strongest learned runs so far use the same filtered train setup:

- `min_user_interactions = 2`
- `min_item_interactions = 2`
- `embedding_dim = 32`
- `epochs = 1`
- `batch_size = 512`
- `max_history_length = 20` for history-based models

That filtered train set contains:

- `1,355,957` train rows
- `319,045` unique train users
- `100,648` unique train items

## Weighted popularity baseline

The baseline uses weighted train-only popularity with:

- `view = 1.0`
- `addtocart = 3.0`
- `transaction = 5.0`

This remains the strongest completed experiment.

Two popularity evaluations now exist:

- Full validation/test evaluation: `outputs/baselines/popularity_metrics.json`
- Capped evaluation with `eval_user_limit = 10000`: `outputs/baselines/popularity_metrics_capped_10k.json`

The learned model reports below were produced with capped evaluation. Earlier summaries compared learned capped metrics against the full popularity baseline, which is directionally useful but not a perfectly matched protocol. The capped popularity run uses the same "first N evaluable users per split" protocol as the learned evaluators.

## ID-only two-tower

The ID-only model trains and evaluates successfully, but it is weak on sparse RetailRocket users. Even after moving from tiny dev runs to filtered full training, it stays far below popularity on both validation and test.

## Unweighted history tower

Replacing the query tower with simple average pooling over train-only item history improves clearly over ID-only on both validation and test. This shows that behavior-based query construction is more useful than relying on `visitor_id` alone.

## Event-weighted history tower

Weighting history pooling by event strength improves the test slice slightly over unweighted history, but it is not consistently better on validation. It looks directionally reasonable, but not decisive.

## Category-aware history tower

Adding category embedding to the item tower improves validation over the history-only variants, but it does not improve the capped test slice over the event-weighted history tower. So the current category formulation is not yet a clear win.

Important implementation caveat: the category embedding is currently added on the item/candidate side only. The history query tower still averages history item ID embeddings; it does not explicitly pool category embeddings into the query representation.

## Coverage diagnostics

Coverage improved a lot after filtered full training, especially for items, but user/query coverage is still modest.

Common filtered-full item coverage:

- validation known-item coverage: `85.07%`
- validation ground-truth item vocab coverage: `77.58%`
- test known-item coverage: `80.33%`
- test ground-truth item vocab coverage: `73.48%`

User/query coverage:

- ID-only validation known-user coverage: `10.88%`
- ID-only test known-user coverage: `6.28%`
- history models validation usable-history coverage: `10.88%`
- history models test usable-history coverage: `6.28%`

Category-aware coverage:

- model vocab items with category: `88,605 / 100,648`
- validation ground-truth items with category: `10,332 / 14,387` = `71.81%`
- test ground-truth items with category: `9,771 / 14,564` = `67.09%`

Main interpretation: item/category coverage is usable, but sparse user history remains the harder problem.

## Artifact and runtime notes

The pipeline is stable, but CPU runtime is non-trivial.

Representative filtered-full runtimes:

- ID-only train: about `212s`
- ID-only capped eval (`10k` val + `10k` test): about `243s`
- unweighted history train: about `118s`
- unweighted history capped eval: about `289s`
- event-weighted history train: about `150s`
- event-weighted history capped eval: about `329s`
- category-aware history train: about `3459s`
- category-aware history capped eval: about `428s`

Representative model artifact sizes:

- ID-only filtered full `model.pt`: `62.3 MB`
- unweighted history `model.pt`: `21.5 MB`
- event-weighted history `model.pt`: `21.5 MB`
- category-aware history `model.pt`: `22.7 MB`

The category-aware gate is much slower mainly because metadata preparation is part of the training workflow.

## Matched capped comparison table

This table uses capped evaluation where available. Popularity is capped to `10,000` evaluable validation users and `10,000` evaluable test users, matching the learned-model protocol.

| Model | Val Recall@50 | Val HitRate@50 | Val NDCG@50 | Test Recall@50 | Test HitRate@50 | Test NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline, capped 10k | 0.022049 | 0.028700 | 0.007361 | 0.020517 | 0.028300 | 0.006575 |
| ID-only two-tower | 0.002251 | 0.005900 | 0.000984 | 0.001055 | 0.002900 | 0.000385 |
| Unweighted history tower | 0.005407 | 0.012100 | 0.002777 | 0.002140 | 0.005500 | 0.000950 |
| Event-weighted history tower | 0.005290 | 0.011400 | 0.002653 | 0.002281 | 0.005700 | 0.001041 |
| Category-aware history tower | 0.006034 | 0.012500 | 0.002782 | 0.002143 | 0.004500 | 0.000872 |

For reference, the original full popularity baseline was:

| Model | Val Recall@50 | Val HitRate@50 | Val NDCG@50 | Test Recall@50 | Test HitRate@50 | Test NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline, full eval | 0.022231 | 0.027091 | 0.007049 | 0.019197 | 0.023870 | 0.006448 |

## Main conclusions

- ID-only is weak on sparse RetailRocket data.
- History features clearly improve over ID-only.
- Event weighting gives small but inconsistent gains.
- Category embedding improves validation but not capped test.
- Popularity remains strongest.
- The main problem now is representation quality and evaluation realism, not pipeline correctness.

## Known limitations

- Learned model evaluation is capped for runtime. Capped evaluation is useful for iteration but should not be treated as final benchmark quality.
- History evaluation uses train-only user history for validation/test users. This avoids leakage, but it is not a session-prefix next-item prediction protocol.
- Category metadata is timestamped. The category-aware training path uses a train-time cutoff when building the category map, but metadata handling should stay documented carefully in future experiments.
- Category features are currently item-side only. The query tower does not explicitly aggregate category embeddings from user history.
- `src/models/two_tower.py` now contains model definitions, data preparation, history construction, category metadata preparation, training loops, and checkpoint helpers. It works, but it should be refactored before substantially more model complexity is added.
- Training scripts now set `random.seed`, NumPy seed, and `torch.manual_seed`, but prior completed experiment artifacts may have been produced before that stabilization change.

## Recommended next step

The next step should be **richer session/history modeling or a stronger training/evaluation setup**, not reranking, MLOps, or FAISS yet.

Most likely productive directions:

- stronger session-prefix or temporally tighter history construction
- richer history encoders than plain average pooling
- deeper item/category feature modeling beyond a simple additive category embedding
- better train/eval setup for sparse-user generalization

What should not be prioritized yet:

- more ID-only training
- retrieval infrastructure work
- productionization layers
