#!/usr/bin/env python3
"""Train the minimal ID-only two-tower retrieval model."""

from __future__ import annotations

import argparse
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
    save_checkpoint,
    save_vocab_json,
    train_two_tower,
)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_df = load_split(Path(args.train))
    vocabs = build_vocabularies(train_df)
    config = TwoTowerConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_limit=args.train_limit,
        device=args.device,
    )

    loader = build_training_loader(
        train_df,
        vocabs,
        batch_size=config.batch_size,
        train_limit=config.train_limit,
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

    print(f"Saved model checkpoint to {out_dir / 'model.pt'}")
    print(f"Saved vocab to {out_dir / 'vocab.json'}")
    print(f"Training loss history: {history}")


if __name__ == "__main__":
    main()
