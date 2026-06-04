#!/usr/bin/env python3
"""Train the minimal ID-only two-tower retrieval model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.two_tower import (
    TwoTowerConfig,
    TwoTowerRetrievalModel,
    build_training_loader,
    build_vocabularies,
    prepare_training_frame,
    save_checkpoint,
    save_vocab_json,
    summarize_training_frame,
    train_two_tower,
)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_training_report(path: Path, summary: dict, history: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Two-Tower ID-Only Training Summary\n\n")
        f.write(f"- Train rows used: {summary['train_rows_used']}\n")
        f.write(f"- Unique train users used: {summary['unique_train_users_used']}\n")
        f.write(f"- Unique train items used: {summary['unique_train_items_used']}\n")
        f.write(f"- Model users in vocab: {summary['model_users_in_vocab']}\n")
        f.write(f"- Model items in vocab: {summary['model_items_in_vocab']}\n")
        f.write(f"- Vocab from effective train: {summary['vocab_from_effective_train']}\n")
        f.write(f"- Min user interactions: {summary['min_user_interactions']}\n")
        f.write(f"- Min item interactions: {summary['min_item_interactions']}\n")
        f.write(f"- Training loss history: {history}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/train.parquet")
    parser.add_argument("--out_dir", default="outputs/experiments/two_tower_id_only")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--vocab_from_effective_train", action="store_true")
    parser.add_argument("--min_user_interactions", type=int, default=None)
    parser.add_argument("--min_item_interactions", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_df = load_split(Path(args.train))
    effective_train_df = prepare_training_frame(
        train_df,
        train_limit=args.train_limit,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    vocab_source_df = effective_train_df if args.vocab_from_effective_train else prepare_training_frame(train_df)
    vocabs = build_vocabularies(vocab_source_df)
    train_summary = summarize_training_frame(effective_train_df)
    train_summary.update(
        {
            "model_users_in_vocab": len(vocabs.user_to_idx),
            "model_items_in_vocab": len(vocabs.item_to_idx),
            "vocab_from_effective_train": bool(args.vocab_from_effective_train),
            "min_user_interactions": args.min_user_interactions,
            "min_item_interactions": args.min_item_interactions,
        }
    )
    config = TwoTowerConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_limit=args.train_limit,
        vocab_from_effective_train=bool(args.vocab_from_effective_train),
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        effective_train_rows=train_summary["train_rows_used"],
        effective_train_users=train_summary["unique_train_users_used"],
        effective_train_items=train_summary["unique_train_items_used"],
        device=args.device,
    )

    loader = build_training_loader(
        effective_train_df,
        vocabs,
        batch_size=config.batch_size,
    )
    model = TwoTowerRetrievalModel(vocabs.num_users, vocabs.num_items, embedding_dim=config.embedding_dim)
    history = train_two_tower(
        model,
        loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_dir / "model.pt", model, vocabs, config, history=history)
    save_vocab_json(out_dir / "vocab.json", vocabs)
    write_json(out_dir / "train_summary.json", train_summary)
    write_training_report(out_dir / "train_summary.md", train_summary, history)

    print(f"Saved model checkpoint to {out_dir / 'model.pt'}")
    print(f"Saved vocab to {out_dir / 'vocab.json'}")
    print(f"Saved training summary to {out_dir / 'train_summary.json'}")
    print(f"Training loss history: {history}")


if __name__ == "__main__":
    main()
