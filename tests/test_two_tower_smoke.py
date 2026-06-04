import pandas as pd
import torch

from src.models.two_tower import (
    HistoryQueryTwoTowerRetrievalModel,
    TwoTowerRetrievalModel,
    build_eval_history_tensor,
    build_history_training_loader,
    build_item_embedding_matrix,
    build_train_history_map,
    build_training_loader,
    build_vocabularies,
    recommend_top_k_for_user,
    recommend_top_k_for_history,
    train_history_two_tower,
    train_two_tower,
)


def _toy_train_frame():
    return pd.DataFrame(
        {
            "visitorid": ["u1", "u1", "u2", "u3"],
            "itemid": ["i1", "i2", "i2", "i3"],
            "event": ["view", "addtocart", "view", "transaction"],
        }
    )


def test_forward_pass_shapes():
    vocabs = build_vocabularies(_toy_train_frame())
    model = TwoTowerRetrievalModel(vocabs.num_users, vocabs.num_items, embedding_dim=8)

    users = torch.tensor([1, 2, 3], dtype=torch.long)
    items = torch.tensor([1, 2, 3], dtype=torch.long)

    pair_scores = model(users, items)
    logits = model.pairwise_logits(users, items)

    assert pair_scores.shape == (3,)
    assert logits.shape == (3, 3)


def test_training_smoke_and_recommendations():
    train_df = _toy_train_frame()
    vocabs = build_vocabularies(train_df)
    model = TwoTowerRetrievalModel(vocabs.num_users, vocabs.num_items, embedding_dim=8)
    loader = build_training_loader(train_df, vocabs, batch_size=2)

    history = train_two_tower(model, loader, epochs=2, learning_rate=5e-3)
    assert len(history) == 2
    assert all(loss >= 0.0 for loss in history)

    item_matrix = build_item_embedding_matrix(model)
    recs = recommend_top_k_for_user(
        model,
        vocabs,
        user_id="u1",
        k=2,
        item_embedding_matrix=item_matrix,
        seen_items={"i1"},
    )

    assert len(recs) == 2
    assert "i1" not in recs


def test_history_query_pooling_shape_and_empty_history_fallback():
    vocabs = build_vocabularies(_toy_train_frame())
    model = HistoryQueryTwoTowerRetrievalModel(vocabs.num_items, embedding_dim=8)

    history_tensor = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    query_vectors = model.encode_queries(history_tensor)

    assert query_vectors.shape == (3, 8)
    assert torch.allclose(query_vectors[0], torch.zeros(8))


def test_history_training_smoke_and_recommendations():
    train_df = _toy_train_frame()
    vocabs = build_vocabularies(train_df)
    model = HistoryQueryTwoTowerRetrievalModel(vocabs.num_items, embedding_dim=8)
    loader = build_history_training_loader(train_df, vocabs, batch_size=2, max_history_length=3)

    history = train_history_two_tower(model, loader, epochs=2, learning_rate=5e-3)
    assert len(history) == 2
    assert all(loss >= 0.0 for loss in history)

    history_map = build_train_history_map(train_df, vocabs, max_history_length=3)
    history_tensor = build_eval_history_tensor(["u1"], history_map, max_history_length=3)
    item_matrix = build_item_embedding_matrix(model)
    recs = recommend_top_k_for_history(
        model,
        vocabs,
        history_item_indices=history_tensor[0],
        k=2,
        item_embedding_matrix=item_matrix,
        seen_items={"i1"},
    )

    assert len(recs) == 2
    assert "i1" not in recs
