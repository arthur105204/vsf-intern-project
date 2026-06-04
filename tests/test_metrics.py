import math

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k


def test_metrics_known_values_at_k_2():
    ground_truth = {
        "u1": {"a", "b"},
        "u2": {"c"},
        "u3": set(),
        "u4": {"d"},
    }
    predictions = {
        "u1": ["a", "x", "b"],
        "u2": ["x", "c"],
        "u4": [],
    }

    expected_recall = 0.5
    expected_hit_rate = 2.0 / 3.0
    expected_ndcg = (1.0 / (1.0 + (1.0 / math.log2(3.0))) + (1.0 / math.log2(3.0)) + 0.0) / 3.0

    assert recall_at_k(ground_truth, predictions, k=2) == expected_recall
    assert hit_rate_at_k(ground_truth, predictions, k=2) == expected_hit_rate
    assert ndcg_at_k(ground_truth, predictions, k=2) == expected_ndcg


def test_metrics_handle_missing_users_and_empty_ground_truth_safely():
    ground_truth = {
        1: {10, 11},
        2: set(),
        3: {30},
    }
    predictions = {
        1: [99, 10, 11],
        # user 2 missing and empty => safely ignored
        # user 3 missing => safely treated as empty ranking
    }

    # user 1 recall@5 = 2/2, hit=1, ndcg= (1/log2(3) + 1/log2(4)) / ideal_dcg(2)
    # user 3 recall@5 = 0, hit=0, ndcg=0
    expected_recall = 0.5
    expected_hit_rate = 0.5
    ideal_dcg = 1.0 + (1.0 / math.log2(3.0))
    expected_ndcg = ((1.0 / math.log2(3.0)) + (1.0 / math.log2(4.0))) / ideal_dcg / 2.0

    assert recall_at_k(ground_truth, predictions, k=5) == expected_recall
    assert hit_rate_at_k(ground_truth, predictions, k=5) == expected_hit_rate
    assert ndcg_at_k(ground_truth, predictions, k=5) == expected_ndcg


def test_metrics_deduplicate_predictions_within_k():
    ground_truth = {"u": {"item1", "item2"}}
    predictions = {"u": ["item1", "item1", "item2", "item2"]}

    # Deduped ranking becomes [item1, item2]
    assert recall_at_k(ground_truth, predictions, k=2) == 1.0
    assert hit_rate_at_k(ground_truth, predictions, k=2) == 1.0
    assert ndcg_at_k(ground_truth, predictions, k=2) == 1.0
