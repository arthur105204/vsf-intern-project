"""Recommendation retrieval metrics.

Supports dictionary inputs:
- ground_truth: dict[user_id, set[item_id]]
- predictions: dict[user_id, list[item_id]]

Users with empty or missing ground truth are skipped safely.
Missing predictions count as empty rankings.
"""

from __future__ import annotations

from math import log2
from typing import Dict, Iterable, List, Mapping, Sequence, Set, TypeVar

import numpy as np


UserId = TypeVar("UserId")
ItemId = TypeVar("ItemId")


def _normalize_prediction_list(predicted_items: Sequence[ItemId], k: int) -> List[ItemId]:
    """Return the top-k unique items while preserving ranking order."""
    unique_items: List[ItemId] = []
    seen = set()
    for item in predicted_items:
        if item in seen:
            continue
        unique_items.append(item)
        seen.add(item)
        if len(unique_items) >= k:
            break
    return unique_items


def _per_user_recall(ground_truth_items: Set[ItemId], predicted_items: Sequence[ItemId], k: int) -> float:
    if not ground_truth_items:
        return 0.0
    topk = _normalize_prediction_list(predicted_items, k)
    hits = len(set(topk).intersection(ground_truth_items))
    return hits / float(len(ground_truth_items))


def _per_user_hit_rate(ground_truth_items: Set[ItemId], predicted_items: Sequence[ItemId], k: int) -> float:
    if not ground_truth_items:
        return 0.0
    topk = _normalize_prediction_list(predicted_items, k)
    return 1.0 if set(topk).intersection(ground_truth_items) else 0.0


def _per_user_ndcg(ground_truth_items: Set[ItemId], predicted_items: Sequence[ItemId], k: int) -> float:
    if not ground_truth_items:
        return 0.0

    topk = _normalize_prediction_list(predicted_items, k)
    if not topk:
        return 0.0

    gains = []
    for rank, item in enumerate(topk, start=1):
        if item in ground_truth_items:
            gains.append(1.0 / log2(rank + 1))
        else:
            gains.append(0.0)

    dcg = float(np.sum(gains))
    ideal_hits = min(len(ground_truth_items), k)
    ideal_dcg = float(np.sum([1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1)]))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _iter_evaluable_users(
    ground_truth: Mapping[UserId, Set[ItemId]],
    predictions: Mapping[UserId, Sequence[ItemId]],
) -> Iterable[tuple[UserId, Set[ItemId], Sequence[ItemId]]]:
    for user_id, truth_items in ground_truth.items():
        if truth_items is None or len(truth_items) == 0:
            continue
        yield user_id, truth_items, predictions.get(user_id, [])


def recall_at_k(
    ground_truth: Mapping[UserId, Set[ItemId]],
    predictions: Mapping[UserId, Sequence[ItemId]],
    k: int,
) -> float:
    """Mean recall@k across users with non-empty ground truth."""
    if k <= 0:
        raise ValueError("k must be positive")

    scores = [
        _per_user_recall(truth_items, predicted_items, k)
        for _, truth_items, predicted_items in _iter_evaluable_users(ground_truth, predictions)
    ]
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(
    ground_truth: Mapping[UserId, Set[ItemId]],
    predictions: Mapping[UserId, Sequence[ItemId]],
    k: int,
) -> float:
    """Mean hit-rate@k across users with non-empty ground truth."""
    if k <= 0:
        raise ValueError("k must be positive")

    scores = [
        _per_user_hit_rate(truth_items, predicted_items, k)
        for _, truth_items, predicted_items in _iter_evaluable_users(ground_truth, predictions)
    ]
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    ground_truth: Mapping[UserId, Set[ItemId]],
    predictions: Mapping[UserId, Sequence[ItemId]],
    k: int,
) -> float:
    """Mean NDCG@k across users with non-empty ground truth."""
    if k <= 0:
        raise ValueError("k must be positive")

    scores = [
        _per_user_ndcg(truth_items, predicted_items, k)
        for _, truth_items, predicted_items in _iter_evaluable_users(ground_truth, predictions)
    ]
    return float(np.mean(scores)) if scores else 0.0
