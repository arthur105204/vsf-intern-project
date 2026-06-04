#!/usr/bin/env python3
"""Evaluate the two-tower model and compare it with the popularity baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from src.models.baselines import build_ground_truth, build_seen_items_for_users
from src.models.two_tower import build_item_embedding_matrix, load_checkpoint, recommend_top_k_for_user


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def _subset_users(users: list[str], limit: int | None) -> list[str]:
    return users if limit is None else users[:limit]


def _load_baseline_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_split(
    model,
    vocabs,
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: list[int],
    eval_user_limit: int | None,
    device: str,
):
    ground_truth = build_ground_truth(split_df, positive_events=("view", "addtocart", "transaction"))
    eval_user_ids = _subset_users(list(ground_truth.keys()), eval_user_limit)
    ground_truth = {user_id: ground_truth[user_id] for user_id in eval_user_ids if user_id in ground_truth}
    seen_items_by_user = build_seen_items_for_users(train_df, eval_user_ids)
    item_matrix = build_item_embedding_matrix(model, device=device)

    predictions = {}
    for index, user_id in enumerate(ground_truth.keys(), start=1):
        predictions[user_id] = recommend_top_k_for_user(
            model,
            vocabs,
            user_id=user_id,
            k=max(k_values),
            item_embedding_matrix=item_matrix,
            seen_items=seen_items_by_user.get(str(user_id), set()),
            device=device,
        )
        if index % 500 == 0:
            print(f"  evaluated {index}/{len(ground_truth)} users", flush=True)

    metrics = {}
    for k in k_values:
        metrics[str(k)] = {
            "recall_at_k": recall_at_k(ground_truth, predictions, k),
            "hit_rate_at_k": hit_rate_at_k(ground_truth, predictions, k),
            "ndcg_at_k": ndcg_at_k(ground_truth, predictions, k),
        }
    return {"num_users": len(ground_truth), "metrics": metrics}


def compare_against_baseline(metrics: dict, baseline: dict, split_name: str, k_values: list[int]) -> dict:
    comparison = {}
    baseline_metrics = baseline.get(split_name, {}).get("metrics", {})
    for k in k_values:
        key = str(k)
        comparison[key] = {}
        for metric_name in ["recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
            comparison[key][metric_name] = metrics["metrics"][key][metric_name] - baseline_metrics.get(key, {}).get(metric_name, 0.0)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/experiments/two_tower_id_only/model.pt")
    parser.add_argument("--train", default="data/processed/splits/train.parquet")
    parser.add_argument("--val", default="data/processed/splits/val.parquet")
    parser.add_argument("--test", default="data/processed/splits/test.parquet")
    parser.add_argument("--baseline_json", default="outputs/baselines/popularity_metrics.json")
    parser.add_argument("--out_json", default="outputs/experiments/two_tower_id_only/metrics.json")
    parser.add_argument("--out_report", default="reports/two_tower_id_only.md")
    parser.add_argument("--k_values", nargs="*", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--eval_user_limit", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_df = load_split(Path(args.train))
    val_df = load_split(Path(args.val))
    test_df = load_split(Path(args.test))
    model, vocabs, config, history = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    baseline = _load_baseline_metrics(Path(args.baseline_json))

    print("Evaluating validation split...", flush=True)
    validation = evaluate_split(model, vocabs, val_df, train_df, args.k_values, args.eval_user_limit, args.device)
    print("Evaluating test split...", flush=True)
    test = evaluate_split(model, vocabs, test_df, train_df, args.k_values, args.eval_user_limit, args.device)

    payload = {
        "experiment": "two_tower_id_only",
        "config": {
            "embedding_dim": config.embedding_dim,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "train_limit": config.train_limit,
            "eval_user_limit": args.eval_user_limit,
            "device": args.device,
        },
        "training_loss_history": history,
        "validation": validation,
        "test": test,
        "baseline_comparison": {
            "validation": compare_against_baseline(validation, baseline, "validation", args.k_values),
            "test": compare_against_baseline(test, baseline, "test", args.k_values),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        f.write("# Two-Tower ID-Only Retrieval\n\n")
        f.write("Minimal user/item embedding model trained with in-batch negatives on the train split only.\n\n")
        f.write(f"Training loss history: {history}\n\n")
        f.write("## Validation metrics\n\n")
        for k in args.k_values:
            row = validation["metrics"][str(k)]
            base = baseline.get("validation", {}).get("metrics", {}).get(str(k), {})
            f.write(
                f"- K={k}: Recall@K={row['recall_at_k']:.6f} (baseline {base.get('recall_at_k', 0.0):.6f}), "
                f"HitRate@K={row['hit_rate_at_k']:.6f} (baseline {base.get('hit_rate_at_k', 0.0):.6f}), "
                f"NDCG@K={row['ndcg_at_k']:.6f} (baseline {base.get('ndcg_at_k', 0.0):.6f})\n"
            )
        f.write("\n## Test metrics\n\n")
        for k in args.k_values:
            row = test["metrics"][str(k)]
            base = baseline.get("test", {}).get("metrics", {}).get(str(k), {})
            f.write(
                f"- K={k}: Recall@K={row['recall_at_k']:.6f} (baseline {base.get('recall_at_k', 0.0):.6f}), "
                f"HitRate@K={row['hit_rate_at_k']:.6f} (baseline {base.get('hit_rate_at_k', 0.0):.6f}), "
                f"NDCG@K={row['ndcg_at_k']:.6f} (baseline {base.get('ndcg_at_k', 0.0):.6f})\n"
            )

    print(f"Wrote metrics to {out_json}")
    print(f"Wrote report to {out_report}")


if __name__ == "__main__":
    main()
