import pandas as pd

from src.models.baselines import build_ground_truth, build_popularity_model, recommend_for_user


def test_popularity_order_is_correct():
    train_df = pd.DataFrame(
        {
            "visitorid": ["u1", "u2", "u3", "u4"],
            "itemid": ["i1", "i2", "i2", "i3"],
            "event": ["view", "transaction", "view", "addtocart"],
        }
    )
    model = build_popularity_model(train_df)

    ranked = [item_id for item_id, _ in model.item_scores]
    assert ranked[0] == "i2"
    assert set(ranked[:3]) == {"i1", "i2", "i3"}


def test_seen_items_can_be_excluded():
    train_df = pd.DataFrame(
        {
            "visitorid": ["u1", "u1", "u2"],
            "itemid": ["i1", "i2", "i3"],
            "event": ["view", "view", "view"],
        }
    )
    model = build_popularity_model(train_df)
    recs = recommend_for_user(model, "u1", k=2)
    assert "i1" not in recs
    assert "i2" not in recs


def test_predictions_return_k_items_when_enough_items_exist():
    train_df = pd.DataFrame(
        {
            "visitorid": ["u1", "u2", "u3", "u4"],
            "itemid": ["i1", "i2", "i3", "i4"],
            "event": ["view", "view", "view", "view"],
        }
    )
    model = build_popularity_model(train_df)
    recs = recommend_for_user(model, "new_user", k=3)
    assert len(recs) == 3


def test_build_ground_truth_uses_positive_interactions():
    split_df = pd.DataFrame(
        {
            "visitorid": ["u1", "u1", "u2"],
            "itemid": ["i1", "i2", "i3"],
            "event": ["view", "transaction", "addtocart"],
        }
    )
    truth = build_ground_truth(split_df, positive_events=("view", "addtocart", "transaction"))
    assert truth["u1"] == {"i1", "i2"}
    assert truth["u2"] == {"i3"}
