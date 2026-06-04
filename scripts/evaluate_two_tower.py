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


def _standardize_id(value) -> str:
    return str(value)


def _load_baseline_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_eval_frame(split_df: pd.DataFrame) -> pd.DataFrame:
    required = {"visitorid", "itemid"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"split_df missing columns: {sorted(missing)}")

    df = split_df.dropna(subset=["visitorid", "itemid"]).copy()
    df["visitorid"] = df["visitorid"].map(_standardize_id)
    df["itemid"] = df["itemid"].map(_standardize_id)
    if "event" in df.columns:
        df = df[df["event"].astype(str).str.lower().isin({"view", "addtocart", "transaction"})]
    return df.reset_index(drop=True)


def _compute_coverage_diagnostics(eval_df: pd.DataFrame, ground_truth: dict[str, set[str]], vocabs) -> dict:
    user_ids = list(ground_truth.keys())
    known_user_count = sum(1 for user_id in user_ids if user_id in vocabs.user_to_idx)
    item_vocab = set(vocabs.item_to_idx.keys())

    if eval_df.empty:
        known_item_rows = 0
    else:
        known_item_rows = int(eval_df["itemid"].isin(item_vocab).sum())

    ground_truth_items = set()
    for items in ground_truth.values():
        ground_truth_items.update(items)

    known_ground_truth_items = sum(1 for item_id in ground_truth_items if item_id in item_vocab)

    return {
        "known_user_count": known_user_count,
        "known_user_coverage": (known_user_count / float(len(user_ids))) if user_ids else 0.0,
        "known_item_row_count": known_item_rows,
        "known_item_coverage": (known_item_rows / float(len(eval_df))) if len(eval_df) else 0.0,
        "ground_truth_unique_item_count": len(ground_truth_items),
        "ground_truth_items_in_vocab_count": known_ground_truth_items,
        "ground_truth_item_vocab_coverage": (
            known_ground_truth_items / float(len(ground_truth_items)) if ground_truth_items else 0.0
        ),
    }


def evaluate_split(
    model,
    vocabs,
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: list[int],
    eval_user_limit: int | None,
    device: str,
):
    eval_df = _prepare_eval_frame(split_df)
    ground_truth = build_ground_truth(eval_df, positive_events=("view", "addtocart", "transaction"))
    eval_user_ids = _subset_users(list(ground_truth.keys()), eval_user_limit)
    ground_truth = {user_id: ground_truth[user_id] for user_id in eval_user_ids if user_id in ground_truth}
    eval_df = eval_df[eval_df["visitorid"].isin(eval_user_ids)].reset_index(drop=True)
    seen_items_by_user = build_seen_items_for_users(train_df, eval_user_ids)
    item_matrix = build_item_embedding_matrix(model, device=device)
    coverage = _compute_coverage_diagnostics(eval_df, ground_truth, vocabs)

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
    return {"num_users": len(ground_truth), "metrics": metrics, "coverage": coverage}


def compare_against_baseline(metrics: dict, baseline: dict, split_name: str, k_values: list[int]) -> dict:
    comparison = {}
    baseline_metrics = baseline.get(split_name, {}).get("metrics", {})
    for k in k_values:
        key = str(k)
        comparison[key] = {}
        for metric_name in ["recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
            comparison[key][metric_name] = metrics["metrics"][key][metric_name] - baseline_metrics.get(key, {}).get(metric_name, 0.0)
    return comparison


def build_training_diagnostics(config, vocabs) -> dict:
    return {
        "train_rows_used": config.effective_train_rows,
        "unique_train_users_used": config.effective_train_users,
        "unique_train_items_used": config.effective_train_items,
        "model_users_in_vocab": len(vocabs.user_to_idx),
        "model_items_in_vocab": len(vocabs.item_to_idx),
        "vocab_from_effective_train": config.vocab_from_effective_train,
        "min_user_interactions": config.min_user_interactions,
        "min_item_interactions": config.min_item_interactions,
    }


def write_report(
    path: Path,
    training_diagnostics: dict,
    history: list[float],
    validation: dict,
    test: dict,
    baseline: dict,
    k_values: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Two-Tower ID-Only Retrieval\n\n")
        f.write("Minimal user/item embedding model trained with in-batch negatives on the train split only.\n\n")
        f.write("## Training diagnostics\n\n")
        f.write(f"- Train rows used: {training_diagnostics['train_rows_used']}\n")
        f.write(f"- Unique train users used: {training_diagnostics['unique_train_users_used']}\n")
        f.write(f"- Unique train items used: {training_diagnostics['unique_train_items_used']}\n")
        f.write(f"- Model users in vocab: {training_diagnostics['model_users_in_vocab']}\n")
        f.write(f"- Model items in vocab: {training_diagnostics['model_items_in_vocab']}\n")
        f.write(f"- Vocab from effective train: {training_diagnostics['vocab_from_effective_train']}\n")
        f.write(f"- Min user interactions: {training_diagnostics['min_user_interactions']}\n")
        f.write(f"- Min item interactions: {training_diagnostics['min_item_interactions']}\n")
        f.write(f"- Training loss history: {history}\n\n")

        for split_name, split_metrics in [("Validation", validation), ("Test", test)]:
            coverage = split_metrics["coverage"]
            f.write(f"## {split_name} coverage diagnostics\n\n")
            f.write(
                f"- Known-user coverage: {coverage['known_user_count']}/{split_metrics['num_users']} "
                f"({coverage['known_user_coverage']:.2%})\n"
            )
            f.write(
                f"- Known-item coverage: {coverage['known_item_row_count']} eval rows in vocab "
                f"({coverage['known_item_coverage']:.2%})\n"
            )
            f.write(
                f"- Ground-truth items in vocab: {coverage['ground_truth_items_in_vocab_count']}/"
                f"{coverage['ground_truth_unique_item_count']} "
                f"({coverage['ground_truth_item_vocab_coverage']:.2%})\n\n"
            )

            f.write(f"## {split_name} metrics\n\n")
            base_metrics = baseline.get(split_name.lower(), {}).get("metrics", {})
            for k in k_values:
                row = split_metrics["metrics"][str(k)]
                base = base_metrics.get(str(k), {})
                f.write(
                    f"- K={k}: Recall@K={row['recall_at_k']:.6f} (baseline {base.get('recall_at_k', 0.0):.6f}), "
                    f"HitRate@K={row['hit_rate_at_k']:.6f} (baseline {base.get('hit_rate_at_k', 0.0):.6f}), "
                    f"NDCG@K={row['ndcg_at_k']:.6f} (baseline {base.get('ndcg_at_k', 0.0):.6f})\n"
                )
            f.write("\n")


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
    training_diagnostics = build_training_diagnostics(config, vocabs)

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
            "vocab_from_effective_train": config.vocab_from_effective_train,
            "min_user_interactions": config.min_user_interactions,
            "min_item_interactions": config.min_item_interactions,
            "eval_user_limit": args.eval_user_limit,
            "device": args.device,
        },
        "training_diagnostics": training_diagnostics,
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
    write_report(out_report, training_diagnostics, history, validation, test, baseline, args.k_values)

    print(f"Wrote metrics to {out_json}")
    print(f"Wrote report to {out_report}")


if __name__ == "__main__":
    main()
