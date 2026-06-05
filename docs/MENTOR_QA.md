# Mentor Q&A

## 1. What is this project trying to build?

It is building an offline two-tower candidate retrieval prototype for ecommerce recommendations on RetailRocket. The goal is to retrieve likely relevant item candidates before any later ranking stage.

## 2. What is candidate retrieval?

Candidate retrieval is the first stage of a recommender system. It quickly narrows a large item catalog down to a smaller set of candidate items that a later ranker could sort more carefully.

## 3. What does two-tower mean here?

One tower produces a query/user embedding. The other tower produces item embeddings. Items are scored by dot product between the query embedding and item embeddings.

## 4. Why does popularity beat the learned models?

RetailRocket is sparse and ecommerce behavior is heavily popularity-driven. A simple train-only popularity prior is very strong when many users have little usable history.

## 5. Why is the ID-only two-tower weak?

The ID-only model learns one embedding per visitor ID. Many users are sparse or unseen in validation/test, so the model has little useful signal and weak generalization.

## 6. Why does history help?

History uses prior interacted items to represent the user's intent. This is more informative than relying only on a sparse `visitor_id`.

## 7. Why is event weighting not a clear win?

Event weighting gives stronger importance to `addtocart` and `transaction` events. That is reasonable, but the observed gains are small and inconsistent across validation and test.

## 8. Why is category embedding not a clear test win?

The current category feature is simple and item-side only. It improves validation but not capped test, so the result is not strong enough to claim a reliable improvement.

## 9. What does Recall@50 mean?

Recall@50 measures how many of a user's held-out relevant items appear in the top 50 recommendations, averaged across evaluable users.

## 10. What does HitRate@50 mean?

HitRate@50 measures whether at least one held-out relevant item appears in the top 50 recommendations for a user.

## 11. What does NDCG@50 mean?

NDCG@50 rewards putting relevant items higher in the recommendation list. A hit at rank 1 is worth more than a hit at rank 50.

## 12. How is leakage avoided?

Models and popularity scores are trained from the train split only. Validation/test users use train-only history, not validation/test target items, when building query representations.

## 13. What is the main evaluation caveat?

Learned model evaluations are capped for runtime. The popularity baseline now has a capped 10k-user run for matched comparison, but final claims should eventually use a consistent full or carefully sampled protocol.

## 14. What is the category metadata caveat?

RetailRocket item metadata is timestamped. Category mappings used for modeling should respect a train-time cutoff so future metadata does not leak into training/evaluation.

## 15. What is still out of scope?

Reranking, FAISS/ANN retrieval, production serving, MLOps, and advanced training objectives are still out of scope for this stabilization gate.

## 16. What is the safest next step?

The safest next step is mentor review using the experiment summary, code walkthrough, and Q&A. More modeling should wait until the current pipeline and limitations are understood clearly.

