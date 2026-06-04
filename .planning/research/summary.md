# Domain Research — Two-Tower Retrieval (summary)

- Two-tower: separate encoders for user/query and item; similarity via dot product or cosine.
- Training: in-batch negatives, contrastive softmax, optional LogQ correction for logged feedback bias.
- Features: ID embeddings, pooled history, BOW history features, item metadata (category/property).
- Serving: precompute item embeddings offline; perform brute-force or ANN search (ScaNN/HNSW).
- Evaluation: Recall@K, NDCG@K, HitRate@K; analyze coverage and popularity bias.

References: standard retrieval literature, ScaNN/HNSW docs, LogQ correction notes.
