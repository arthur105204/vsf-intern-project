#!/usr/bin/env python3
"""Run and evaluate a popularity baseline on the processed splits."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# Allow direct execution from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baselines import (
    build_ground_truth,
    build_popularity_model,
    build_seen_items_for_users,
    recommend_for_user,
    write_json,
)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def _per_user_ndcg(truth_items, predicted_items):
    if not truth_items:
        return 0.0

    gains = 0.0
    ideal_hits = min(len(truth_items), len(predicted_items))
    if ideal_hits == 0:
        return 0.0

    for rank, item in enumerate(predicted_items, start=1):
        if item in truth_items:
            gains += 1.0 / math.log2(rank + 1)

    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return gains / ideal_dcg


def _subset_users(users: list[str], limit: int | None) -> list[str]:
    return users if limit is None else users[:limit]


def evaluate_split(model, split_df: pd.DataFrame, train_seen_items_by_user, ks, eval_user_limit: int | None = None):
    ground_truth = build_ground_truth(split_df, positive_events=("view", "addtocart", "transaction"))
    eval_user_ids = _subset_users(list(ground_truth.keys()), eval_user_limit)
    ground_truth = {user_id: ground_truth[user_id] for user_id in eval_user_ids if user_id in ground_truth}

    metrics = {
        str(k): {"recall_sum": 0.0, "hit_sum": 0.0, "ndcg_sum": 0.0}
        for k in ks
    }
    evaluable_users = list(ground_truth.items())
    total_users = len(evaluable_users)

    max_k = max(ks)
    for index, (user_id, truth_items) in enumerate(evaluable_users, start=1):
        predicted = recommend_for_user(
            model,
            user_id=user_id,
            k=max_k,
            seen_items=train_seen_items_by_user.get(str(user_id), set()),
        )
        for k in ks:
            topk = predicted[:k]
            hits = len(set(topk).intersection(truth_items))
            metrics[str(k)]["recall_sum"] += hits / float(len(truth_items))
            metrics[str(k)]["hit_sum"] += 1.0 if hits else 0.0
            metrics[str(k)]["ndcg_sum"] += _per_user_ndcg(truth_items, topk)

        if index % 5000 == 0:
            print(f"  evaluated {index}/{total_users} users", flush=True)

    for k in ks:
        key = str(k)
        metrics[key] = {
            "recall_at_k": metrics[key]["recall_sum"] / float(total_users) if total_users else 0.0,
            "hit_rate_at_k": metrics[key]["hit_sum"] / float(total_users) if total_users else 0.0,
            "ndcg_at_k": metrics[key]["ndcg_sum"] / float(total_users) if total_users else 0.0,
        }
    return {
        "num_users": total_users,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/train.parquet")
    parser.add_argument("--val", default="data/processed/splits/val.parquet")
    parser.add_argument("--test", default="data/processed/splits/test.parquet")
    parser.add_argument("--out_json", default="outputs/baselines/popularity_metrics.json")
    parser.add_argument("--out_report", default="reports/popularity_baseline.md")
    parser.add_argument("--k_values", nargs="*", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--eval_user_limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading splits...", flush=True)
    train_df = load_split(Path(args.train))
    val_df = load_split(Path(args.val))
    test_df = load_split(Path(args.test))
    print(
        f"Loaded train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows",
        flush=True,
    )

    eval_user_ids = set(val_df["visitorid"].astype(str).unique()) | set(test_df["visitorid"].astype(str).unique())

    print("Building popularity model...", flush=True)
    model = build_popularity_model(train_df, build_seen_items=False)
    print(f"Built popularity ranking for {len(model.item_scores)} items", flush=True)

    print(f"Building seen-item lookup for {len(eval_user_ids)} evaluation users...", flush=True)
    train_seen_items_by_user = build_seen_items_for_users(train_df, eval_user_ids)
    print(f"Built seen-item lookup for {len(train_seen_items_by_user)} users", flush=True)

    train_summary = {
        "num_rows": int(len(train_df)),
        "num_users": int(train_df["visitorid"].nunique()),
        "num_items": int(train_df["itemid"].nunique()),
        "top_items": [
            {"item_id": item_id, "score": score}
            for item_id, score in model.item_scores[:10]
        ],
    }

    print("Evaluating validation split...", flush=True)
    val_results = evaluate_split(model, val_df, train_seen_items_by_user, args.k_values, args.eval_user_limit)
    print("Evaluating test split...", flush=True)
    test_results = evaluate_split(model, test_df, train_seen_items_by_user, args.k_values, args.eval_user_limit)

    payload = {
        "baseline": "popularity",
        "weighted_events": True,
        "event_weights": model.event_weights,
        "eval_user_limit": args.eval_user_limit,
        "train_summary": train_summary,
        "validation": val_results,
        "test": test_results,
    }

    out_json = Path(args.out_json)
    write_json(out_json, payload)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        f.write("# Popularity Baseline\n\n")
        f.write("This baseline uses weighted item popularity from the train split only.\n\n")
        f.write("Event weights: view=1.0, addtocart=3.0, transaction=5.0.\n\n")
        if args.eval_user_limit is None:
            f.write("Evaluation protocol: full validation/test user set.\n\n")
        else:
            f.write(
                "Evaluation protocol: capped to the first "
                f"{args.eval_user_limit} evaluable users per split, matching the learned-model capped evaluation protocol.\n\n"
            )
        f.write("## Validation metrics\n\n")
        for k, metrics in val_results["metrics"].items():
            f.write(
                f"- K={k}: Recall@K={metrics['recall_at_k']:.6f}, "
                f"HitRate@K={metrics['hit_rate_at_k']:.6f}, NDCG@K={metrics['ndcg_at_k']:.6f}\n"
            )
        f.write("\n## Test metrics\n\n")
        for k, metrics in test_results["metrics"].items():
            f.write(
                f"- K={k}: Recall@K={metrics['recall_at_k']:.6f}, "
                f"HitRate@K={metrics['hit_rate_at_k']:.6f}, NDCG@K={metrics['ndcg_at_k']:.6f}\n"
            )

    print(f"Wrote metrics to {out_json}", flush=True)
    print(f"Wrote report to {out_report}", flush=True)


if __name__ == "__main__":
    main()
