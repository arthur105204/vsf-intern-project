"""Simple retrieval baselines for ecommerce recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_EVENT_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


@dataclass(frozen=True)
class PopularityModel:
    item_scores: List[Tuple[str, float]]
    seen_items_by_user: Dict[str, Set[str]]
    event_weights: Dict[str, float]


def _standardize_id(value) -> str:
    return str(value)


def build_popularity_model(
    train_df: pd.DataFrame,
    event_weights: Optional[Mapping[str, float]] = None,
    build_seen_items: bool = True,
) -> PopularityModel:
    """Build a popularity baseline from train data only.

    If event_weights is None, a weighted setup is used with:
    view=1.0, addtocart=3.0, transaction=5.0.
    """

    weights = dict(event_weights or DEFAULT_EVENT_WEIGHTS)

    df = train_df.copy()
    if "itemid" not in df.columns:
        raise ValueError("train_df must contain itemid")
    if "visitorid" not in df.columns:
        raise ValueError("train_df must contain visitorid")

    df["itemid"] = df["itemid"].map(_standardize_id)
    df["visitorid"] = df["visitorid"].map(_standardize_id)

    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.lower()
        df["weight"] = df["event"].map(weights).fillna(1.0)
    else:
        df["weight"] = 1.0

    popularity = (
        df.groupby("itemid", sort=False)["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    item_scores = [(item_id, float(score)) for item_id, score in popularity.items()]

    seen_items_by_user: Dict[str, Set[str]] = {}
    if build_seen_items:
        seen_items_by_user = (
            df.groupby("visitorid", sort=False)["itemid"]
            .apply(lambda s: set(s.astype(str).tolist()))
            .to_dict()
        )

    return PopularityModel(
        item_scores=item_scores,
        seen_items_by_user=seen_items_by_user,
        event_weights=weights,
    )


def get_top_popular_items(
    model: PopularityModel,
    exclude_items: Optional[Set[str]] = None,
    k: Optional[int] = None,
) -> List[str]:
    exclude = set(map(str, exclude_items or set()))
    if not exclude and k is not None:
        return [item_id for item_id, _ in model.item_scores[:k]]

    ranked_items: List[str] = []
    for item_id, _ in model.item_scores:
        if item_id in exclude:
            continue
        ranked_items.append(item_id)
        if k is not None and len(ranked_items) >= k:
            break
    return ranked_items


def recommend_for_user(
    model: PopularityModel,
    user_id,
    k: int,
    seen_items: Optional[Iterable] = None,
) -> List[str]:
    if k <= 0:
        raise ValueError("k must be positive")

    user_key = _standardize_id(user_id)
    user_seen = set(map(_standardize_id, seen_items or model.seen_items_by_user.get(user_key, set())))
    return get_top_popular_items(model, exclude_items=user_seen, k=k)


def build_ground_truth(
    split_df: pd.DataFrame,
    positive_events: Optional[Sequence[str]] = None,
) -> Dict[str, Set[str]]:
    """Build ground truth item sets per user from validation/test interactions."""

    df = split_df.copy()
    if "visitorid" not in df.columns or "itemid" not in df.columns:
        raise ValueError("split_df must contain visitorid and itemid")

    df["visitorid"] = df["visitorid"].map(_standardize_id)
    df["itemid"] = df["itemid"].map(_standardize_id)

    if positive_events is not None and "event" in df.columns:
        allowed = {str(event).lower() for event in positive_events}
        df = df[df["event"].astype(str).str.lower().isin(allowed)]

    truth = (
        df.groupby("visitorid", sort=False)["itemid"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )
    return truth


def build_seen_items_for_users(train_df: pd.DataFrame, user_ids: Iterable) -> Dict[str, Set[str]]:
    """Build seen-item sets only for the users that need evaluation-time filtering."""

    df = train_df.copy()
    if "visitorid" not in df.columns or "itemid" not in df.columns:
        raise ValueError("train_df must contain visitorid and itemid")

    user_key_set = {_standardize_id(user_id) for user_id in user_ids}
    df["visitorid"] = df["visitorid"].map(_standardize_id)
    df["itemid"] = df["itemid"].map(_standardize_id)
    df = df[df["visitorid"].isin(user_key_set)]

    if df.empty:
        return {}

    seen = (
        df.groupby("visitorid", sort=False)["itemid"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )
    return seen


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
