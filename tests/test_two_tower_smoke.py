import tempfile
from pathlib import Path

import pandas as pd
import torch

from src.models.two_tower import (
    HistoryQueryTwoTowerRetrievalModel,
    TwoTowerRetrievalModel,
    build_category_vocab,
    build_eval_history_tensors,
    build_history_training_loader,
    build_item_embedding_matrix,
    build_item_category_index_tensor,
    build_item_category_map,
    build_train_history_maps,
    build_training_loader,
    build_vocabularies,
    category_tree_ids_from_csv,
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


def test_history_query_weighted_pooling_respects_event_strength():
    vocabs = build_vocabularies(_toy_train_frame())
    model = HistoryQueryTwoTowerRetrievalModel(vocabs.num_items, embedding_dim=4)

    with torch.no_grad():
        model.item_embedding.weight.zero_()
        model.item_embedding.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.item_embedding.weight[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    history_tensor = torch.tensor([[1, 2]], dtype=torch.long)
    weighted = model.encode_queries(history_tensor, history_event_weights=torch.tensor([[1.0, 3.0]], dtype=torch.float32))
    unweighted = model.encode_queries(history_tensor, history_event_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32))

    assert weighted.shape == (1, 4)
    assert weighted[0, 1] > weighted[0, 0]
    assert torch.allclose(unweighted[0, :2], torch.tensor([0.5, 0.5]), atol=1e-6)


def test_history_training_smoke_and_recommendations():
    train_df = _toy_train_frame()
    vocabs = build_vocabularies(train_df)
    model = HistoryQueryTwoTowerRetrievalModel(vocabs.num_items, embedding_dim=8)
    loader = build_history_training_loader(train_df, vocabs, batch_size=2, max_history_length=3, use_event_weights=True)

    history = train_history_two_tower(model, loader, epochs=2, learning_rate=5e-3)
    assert len(history) == 2
    assert all(loss >= 0.0 for loss in history)

    history_map, history_weight_map = build_train_history_maps(
        train_df,
        vocabs,
        max_history_length=3,
        use_event_weights=True,
    )
    history_tensor, history_weight_tensor = build_eval_history_tensors(
        ["u1"],
        history_map,
        history_weight_map,
        max_history_length=3,
    )
    item_matrix = build_item_embedding_matrix(model)
    recs = recommend_top_k_for_history(
        model,
        vocabs,
        history_item_indices=history_tensor[0],
        history_event_weights=history_weight_tensor[0],
        k=2,
        item_embedding_matrix=item_matrix,
        seen_items={"i1"},
    )

    assert len(recs) == 2
    assert "i1" not in recs


def test_category_mapping_handles_missing_category():
    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        temp_path = Path(temp_dir)
        item_properties_path = temp_path / "item_properties.csv"
        category_tree_path = temp_path / "category_tree.csv"

        with item_properties_path.open("w", encoding="utf-8") as f:
            f.write(
                "timestamp,itemid,property,value\n"
                "1000,1,categoryid,10\n"
                "1001,2,available,1\n"
                "1002,3,categoryid,999\n"
                "1003,1,categoryid,20\n"
            )
        with category_tree_path.open("w", encoding="utf-8") as f:
            f.write(
                "categoryid,parentid\n"
                "10,\n"
                "20,10\n"
            )

        category_ids = category_tree_ids_from_csv(category_tree_path)
        mapping = build_item_category_map(
            [item_properties_path],
            item_ids={"1", "2", "3"},
            category_ids=category_ids,
            max_timestamp_ms=1003,
            chunksize=10,
        )

    assert mapping["1"] == "20"
    assert "2" not in mapping
    assert "3" not in mapping


def test_category_item_tower_forward_shape_and_missing_category_fallback():
    vocabs = build_vocabularies(_toy_train_frame())
    item_category_map = {"i1": "c1"}
    category_to_idx = build_category_vocab(item_category_map)
    item_category_indices = build_item_category_index_tensor(vocabs, item_category_map, category_to_idx)
    model = HistoryQueryTwoTowerRetrievalModel(
        vocabs.num_items,
        embedding_dim=4,
        num_categories=max(category_to_idx.values(), default=0) + 1,
        item_category_indices=item_category_indices,
        category_to_idx=category_to_idx,
    )

    with torch.no_grad():
        model.item_embedding.weight.zero_()
        model.category_embedding.weight.zero_()
        model.category_embedding.weight[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    item_vectors = model.encode_items(torch.tensor([1, 2], dtype=torch.long))
    logits = model.pairwise_logits(
        torch.tensor([[1, 0], [2, 0]], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
        history_event_weights=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
    )

    assert item_vectors.shape == (2, 4)
    assert logits.shape == (2, 2)
    assert torch.allclose(item_vectors[0], torch.ones(4))
    assert torch.allclose(item_vectors[1], torch.zeros(4))
