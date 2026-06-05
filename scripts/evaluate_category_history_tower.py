#!/usr/bin/env python3
"""Evaluate the category-aware history two-tower model and compare it with prior baselines."""

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
from src.models.two_tower import (
    build_eval_history_tensors,
    build_item_embedding_matrix,
    build_train_history_maps,
    load_checkpoint,
    prepare_training_frame,
    recommend_top_k_for_history,
)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def _subset_users(users: list[str], limit: int | None) -> list[str]:
    return users if limit is None else users[:limit]


def _standardize_id(value) -> str:
    return str(value)


def _load_json(path: Path) -> dict:
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


def _compute_coverage_diagnostics(
    eval_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    vocabs,
    history_map: dict[str, list[int]],
    model,
) -> dict:
    user_ids = list(ground_truth.keys())
    users_with_history = sum(1 for user_id in user_ids if len(history_map.get(user_id, [])) > 0)
    item_vocab = set(vocabs.item_to_idx.keys())
    category_item_indices = getattr(model, "item_category_indices", None)

    if eval_df.empty:
        known_item_rows = 0
        known_category_rows = 0
    else:
        known_item_mask = eval_df["itemid"].isin(item_vocab)
        known_item_rows = int(known_item_mask.sum())
        known_category_rows = 0
        if category_item_indices is not None:
            for item_id in eval_df.loc[known_item_mask, "itemid"].tolist():
                item_index = vocabs.item_to_idx.get(str(item_id), 0)
                if item_index != 0 and int(category_item_indices[item_index]) != 0:
                    known_category_rows += 1

    ground_truth_items = set()
    for items in ground_truth.values():
        ground_truth_items.update(items)

    known_ground_truth_items = 0
    ground_truth_items_with_category = 0
    if category_item_indices is not None:
        for item_id in ground_truth_items:
            item_index = vocabs.item_to_idx.get(str(item_id), 0)
            if item_index != 0:
                known_ground_truth_items += 1
                if int(category_item_indices[item_index]) != 0:
                    ground_truth_items_with_category += 1
    else:
        known_ground_truth_items = sum(1 for item_id in ground_truth_items if item_id in item_vocab)

    model_vocab_items_with_category = 0
    if category_item_indices is not None:
        model_vocab_items_with_category = int((category_item_indices != 0).sum().item())

    return {
        "eval_users_with_usable_history": users_with_history,
        "eval_user_history_coverage": (users_with_history / float(len(user_ids))) if user_ids else 0.0,
        "known_item_row_count": known_item_rows,
        "known_item_coverage": (known_item_rows / float(len(eval_df))) if len(eval_df) else 0.0,
        "known_category_row_count": known_category_rows,
        "known_category_coverage": (known_category_rows / float(len(eval_df))) if len(eval_df) else 0.0,
        "ground_truth_unique_item_count": len(ground_truth_items),
        "ground_truth_items_in_vocab_count": known_ground_truth_items,
        "ground_truth_item_vocab_coverage": (
            known_ground_truth_items / float(len(ground_truth_items)) if ground_truth_items else 0.0
        ),
        "ground_truth_items_with_category_count": ground_truth_items_with_category,
        "ground_truth_category_coverage": (
            ground_truth_items_with_category / float(len(ground_truth_items)) if ground_truth_items else 0.0
        ),
        "model_vocab_items_with_category": model_vocab_items_with_category,
    }


def evaluate_split(
    model,
    vocabs,
    split_df: pd.DataFrame,
    effective_train_df: pd.DataFrame,
    history_map: dict[str, list[int]],
    history_weight_map: dict[str, list[float]],
    k_values: list[int],
    eval_user_limit: int | None,
    max_history_length: int | None,
    device: str,
):
    eval_df = _prepare_eval_frame(split_df)
    ground_truth = build_ground_truth(eval_df, positive_events=("view", "addtocart", "transaction"))
    eval_user_ids = _subset_users(list(ground_truth.keys()), eval_user_limit)
    ground_truth = {user_id: ground_truth[user_id] for user_id in eval_user_ids if user_id in ground_truth}
    eval_df = eval_df[eval_df["visitorid"].isin(eval_user_ids)].reset_index(drop=True)
    seen_items_by_user = build_seen_items_for_users(effective_train_df, eval_user_ids)
    history_tensor, history_weight_tensor = build_eval_history_tensors(
        eval_user_ids,
        history_map,
        history_weight_map,
        max_history_length=max_history_length,
    )
    item_matrix = build_item_embedding_matrix(model, device=device)
    coverage = _compute_coverage_diagnostics(eval_df, ground_truth, vocabs, history_map, model)

    predictions = {}
    for index, user_id in enumerate(ground_truth.keys()):
        predictions[user_id] = recommend_top_k_for_history(
            model,
            vocabs,
            history_item_indices=history_tensor[index],
            history_event_weights=history_weight_tensor[index],
            k=max(k_values),
            item_embedding_matrix=item_matrix,
            seen_items=seen_items_by_user.get(str(user_id), set()),
            device=device,
        )
        if (index + 1) % 500 == 0:
            print(f"  evaluated {index + 1}/{len(ground_truth)} users", flush=True)

    metrics = {}
    for k in k_values:
        metrics[str(k)] = {
            "recall_at_k": recall_at_k(ground_truth, predictions, k),
            "hit_rate_at_k": hit_rate_at_k(ground_truth, predictions, k),
            "ndcg_at_k": ndcg_at_k(ground_truth, predictions, k),
        }
    return {"num_users": len(ground_truth), "metrics": metrics, "coverage": coverage}


def compare_against_reference(metrics: dict, reference: dict, split_name: str, k_values: list[int]) -> dict:
    comparison = {}
    reference_metrics = reference.get(split_name, {}).get("metrics", {})
    for k in k_values:
        key = str(k)
        comparison[key] = {}
        for metric_name in ["recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
            comparison[key][metric_name] = metrics["metrics"][key][metric_name] - reference_metrics.get(key, {}).get(metric_name, 0.0)
    return comparison


def build_training_diagnostics(config, vocabs, model) -> dict:
    model_vocab_items_with_category = 0
    if getattr(model, "category_embedding", None) is not None:
        model_vocab_items_with_category = int((model.item_category_indices != 0).sum().item())
    return {
        "train_rows_used": config.effective_train_rows,
        "unique_train_users_used": config.effective_train_users,
        "unique_train_items_used": config.effective_train_items,
        "users_with_non_empty_train_history": config.users_with_non_empty_history,
        "model_items_in_vocab": len(vocabs.item_to_idx),
        "model_vocab_items_with_category": model_vocab_items_with_category,
        "vocab_from_effective_train": config.vocab_from_effective_train,
        "min_user_interactions": config.min_user_interactions,
        "min_item_interactions": config.min_item_interactions,
        "max_history_length": config.max_history_length,
        "use_event_weights_in_history": config.use_event_weights_in_history,
        "use_category_embeddings": config.use_category_embeddings,
    }


def write_report(
    path: Path,
    training_diagnostics: dict,
    history: list[float],
    validation: dict,
    test: dict,
    popularity: dict,
    id_only: dict,
    unweighted_history: dict,
    weighted_history: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Category-Aware History Two-Tower\n\n")
        f.write("History-aware query tower with event-weighted pooling and category-aware item tower.\n\n")
        f.write("## Training diagnostics\n\n")
        f.write(f"- Train rows used: {training_diagnostics['train_rows_used']}\n")
        f.write(f"- Unique train users used: {training_diagnostics['unique_train_users_used']}\n")
        f.write(f"- Unique train items used: {training_diagnostics['unique_train_items_used']}\n")
        f.write(f"- Users with non-empty train history: {training_diagnostics['users_with_non_empty_train_history']}\n")
        f.write(f"- Model items in vocab: {training_diagnostics['model_items_in_vocab']}\n")
        f.write(f"- Model vocab items with category: {training_diagnostics['model_vocab_items_with_category']}\n")
        f.write(f"- Min user interactions: {training_diagnostics['min_user_interactions']}\n")
        f.write(f"- Min item interactions: {training_diagnostics['min_item_interactions']}\n")
        f.write(f"- Max history length: {training_diagnostics['max_history_length']}\n")
        f.write(f"- Use event weights in history: {training_diagnostics['use_event_weights_in_history']}\n")
        f.write(f"- Use category embeddings: {training_diagnostics['use_category_embeddings']}\n")
        f.write(f"- Training loss history: {history}\n\n")

        for split_name, split_metrics in [("Validation", validation), ("Test", test)]:
            coverage = split_metrics["coverage"]
            f.write(f"## {split_name} coverage diagnostics\n\n")
            f.write(
                f"- Eval users with usable history: {coverage['eval_users_with_usable_history']}/{split_metrics['num_users']} "
                f"({coverage['eval_user_history_coverage']:.2%})\n"
            )
            f.write(
                f"- Known-item coverage: {coverage['known_item_row_count']} eval rows in vocab "
                f"({coverage['known_item_coverage']:.2%})\n"
            )
            f.write(
                f"- Ground-truth items in vocab: {coverage['ground_truth_items_in_vocab_count']}/"
                f"{coverage['ground_truth_unique_item_count']} "
                f"({coverage['ground_truth_item_vocab_coverage']:.2%})\n"
            )
            f.write(
                f"- Ground-truth items with category: {coverage['ground_truth_items_with_category_count']}/"
                f"{coverage['ground_truth_unique_item_count']} "
                f"({coverage['ground_truth_category_coverage']:.2%})\n\n"
            )

        def row(metric_source: dict, split_key: str, metric_key: str) -> float:
            return metric_source.get(split_key, {}).get("metrics", {}).get("50", {}).get(metric_key, 0.0)

        f.write("## Comparison at K=50\n\n")
        f.write("| Model | Validation Recall@50 | Validation HitRate@50 | Validation NDCG@50 | Test Recall@50 | Test HitRate@50 | Test NDCG@50 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        rows = [
            ("Popularity baseline", popularity),
            ("ID-only two-tower", id_only),
            ("Unweighted history tower", unweighted_history),
            ("Event-weighted history tower", weighted_history),
            ("Category-aware history tower", {"validation": validation, "test": test}),
        ]
        for label, payload in rows:
            f.write(
                f"| {label} | "
                f"{row(payload, 'validation', 'recall_at_k'):.6f} | "
                f"{row(payload, 'validation', 'hit_rate_at_k'):.6f} | "
                f"{row(payload, 'validation', 'ndcg_at_k'):.6f} | "
                f"{row(payload, 'test', 'recall_at_k'):.6f} | "
                f"{row(payload, 'test', 'hit_rate_at_k'):.6f} | "
                f"{row(payload, 'test', 'ndcg_at_k'):.6f} |\n"
            )

        weighted_test = weighted_history.get("test", {}).get("metrics", {}).get("50", {})
        category_test = test["metrics"]["50"]
        improved_over_weighted = all(
            category_test.get(metric_name, 0.0) > weighted_test.get(metric_name, 0.0)
            for metric_name in ["recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
        )

        f.write("\n## Conclusion\n\n")
        if improved_over_weighted:
            f.write(
                "Category embedding improves over the history-only tower with event-weighted pooling on the capped test slice, "
                "so item-side category signal is worth carrying forward.\n"
            )
        else:
            f.write(
                "Category embedding does not improve over the history-only tower consistently on the capped test slice, "
                "so the current category formulation is not yet a clear win.\n"
            )

        popularity_test = popularity.get("test", {}).get("metrics", {}).get("50", {})
        if all(
            category_test.get(metric_name, 0.0) < popularity_test.get(metric_name, 0.0)
            for metric_name in ["recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
        ):
            f.write(
                "\nThe weighted popularity baseline remains stronger, so the next step should focus on richer session/history construction "
                "or deeper item/category feature modeling rather than infrastructure work.\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/experiments/category_history_tower/model.pt")
    parser.add_argument("--train", default="data/processed/splits/train.parquet")
    parser.add_argument("--val", default="data/processed/splits/val.parquet")
    parser.add_argument("--test", default="data/processed/splits/test.parquet")
    parser.add_argument("--baseline_json", default="outputs/baselines/popularity_metrics.json")
    parser.add_argument("--id_only_metrics_json", default="outputs/experiments/two_tower_id_only_filtered_full/metrics.json")
    parser.add_argument("--unweighted_history_metrics_json", default="outputs/experiments/history_tower_dev/metrics.json")
    parser.add_argument("--weighted_history_metrics_json", default="outputs/experiments/history_tower_weighted/metrics.json")
    parser.add_argument("--out_json", default="reports/category_history_tower_metrics.json")
    parser.add_argument("--out_report", default="reports/category_history_tower.md")
    parser.add_argument("--k_values", nargs="*", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--eval_user_limit", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_df = load_split(Path(args.train))
    val_df = load_split(Path(args.val))
    test_df = load_split(Path(args.test))
    model, vocabs, config, history = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    if config.query_tower != "history" or not config.use_category_embeddings:
        raise ValueError("checkpoint must be a category-aware history model")

    effective_train_df = prepare_training_frame(
        train_df,
        train_limit=config.train_limit,
        min_user_interactions=config.min_user_interactions,
        min_item_interactions=config.min_item_interactions,
    )
    history_map, history_weight_map = build_train_history_maps(
        effective_train_df,
        vocabs,
        max_history_length=config.max_history_length,
        use_event_weights=config.use_event_weights_in_history,
    )
    popularity = _load_json(Path(args.baseline_json))
    id_only_payload = _load_json(Path(args.id_only_metrics_json))
    unweighted_history_payload = _load_json(Path(args.unweighted_history_metrics_json))
    weighted_history_payload = _load_json(Path(args.weighted_history_metrics_json))
    training_diagnostics = build_training_diagnostics(config, vocabs, model)

    print("Evaluating validation split...", flush=True)
    validation = evaluate_split(
        model,
        vocabs,
        val_df,
        effective_train_df,
        history_map,
        history_weight_map,
        args.k_values,
        args.eval_user_limit,
        config.max_history_length,
        args.device,
    )
    print("Evaluating test split...", flush=True)
    test = evaluate_split(
        model,
        vocabs,
        test_df,
        effective_train_df,
        history_map,
        history_weight_map,
        args.k_values,
        args.eval_user_limit,
        config.max_history_length,
        args.device,
    )

    payload = {
        "experiment": "category_history_tower",
        "config": {
            "embedding_dim": config.embedding_dim,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "train_limit": config.train_limit,
            "vocab_from_effective_train": config.vocab_from_effective_train,
            "min_user_interactions": config.min_user_interactions,
            "min_item_interactions": config.min_item_interactions,
            "max_history_length": config.max_history_length,
            "use_event_weights_in_history": config.use_event_weights_in_history,
            "use_category_embeddings": config.use_category_embeddings,
            "eval_user_limit": args.eval_user_limit,
            "device": args.device,
        },
        "training_diagnostics": training_diagnostics,
        "training_loss_history": history,
        "validation": validation,
        "test": test,
        "baseline_comparison": {
            "validation": compare_against_reference(validation, popularity, "validation", args.k_values),
            "test": compare_against_reference(test, popularity, "test", args.k_values),
        },
        "id_only_comparison": {
            "validation": compare_against_reference(validation, id_only_payload, "validation", args.k_values),
            "test": compare_against_reference(test, id_only_payload, "test", args.k_values),
        },
        "unweighted_history_comparison": {
            "validation": compare_against_reference(validation, unweighted_history_payload, "validation", args.k_values),
            "test": compare_against_reference(test, unweighted_history_payload, "test", args.k_values),
        },
        "weighted_history_comparison": {
            "validation": compare_against_reference(validation, weighted_history_payload, "validation", args.k_values),
            "test": compare_against_reference(test, weighted_history_payload, "test", args.k_values),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    write_report(
        Path(args.out_report),
        training_diagnostics,
        history,
        validation,
        test,
        popularity,
        id_only_payload,
        unweighted_history_payload,
        weighted_history_payload,
    )

    print(f"Wrote metrics to {out_json}")
    print(f"Wrote report to {args.out_report}")


if __name__ == "__main__":
    main()
