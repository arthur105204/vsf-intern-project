import pandas as pd
import torch

from src.models.two_tower import (
    TwoTowerRetrievalModel,
    build_item_embedding_matrix,
    build_training_loader,
    build_vocabularies,
    recommend_top_k_for_user,
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
