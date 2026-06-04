"""Minimal ID-only two-tower retrieval model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TwoTowerConfig:
    embedding_dim: int = 64
    batch_size: int = 1024
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    train_limit: Optional[int] = None
    vocab_from_effective_train: bool = False
    min_user_interactions: Optional[int] = None
    min_item_interactions: Optional[int] = None
    effective_train_rows: Optional[int] = None
    effective_train_users: Optional[int] = None
    effective_train_items: Optional[int] = None
    eval_user_limit: Optional[int] = 5000
    device: str = "cpu"


@dataclass(frozen=True)
class Vocabularies:
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]

    @property
    def num_users(self) -> int:
        return max(self.user_to_idx.values(), default=0) + 1

    @property
    def num_items(self) -> int:
        return max(self.item_to_idx.values(), default=0) + 1

    @property
    def idx_to_user(self) -> List[str]:
        values = [UNK_TOKEN] * self.num_users
        for token, index in self.user_to_idx.items():
            values[index] = token
        return values

    @property
    def idx_to_item(self) -> List[str]:
        values = [UNK_TOKEN] * self.num_items
        for token, index in self.item_to_idx.items():
            values[index] = token
        return values


def _standardize_id(value) -> str:
    return str(value)


def _validate_min_interactions(name: str, value: Optional[int]) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def summarize_training_frame(train_df: pd.DataFrame) -> dict[str, int]:
    if train_df.empty:
        return {
            "train_rows_used": 0,
            "unique_train_users_used": 0,
            "unique_train_items_used": 0,
        }
    return {
        "train_rows_used": int(len(train_df)),
        "unique_train_users_used": int(train_df["visitorid"].nunique()),
        "unique_train_items_used": int(train_df["itemid"].nunique()),
    }


def build_vocabularies(train_df: pd.DataFrame) -> Vocabularies:
    if "visitorid" not in train_df.columns or "itemid" not in train_df.columns:
        raise ValueError("train_df must contain visitorid and itemid")

    users = sorted({_standardize_id(value) for value in train_df["visitorid"].dropna().tolist()})
    items = sorted({_standardize_id(value) for value in train_df["itemid"].dropna().tolist()})
    user_to_idx = {user: index + 1 for index, user in enumerate(users)}
    item_to_idx = {item: index + 1 for index, item in enumerate(items)}
    return Vocabularies(user_to_idx=user_to_idx, item_to_idx=item_to_idx)


def _encode_pairs(df: pd.DataFrame, vocabs: Vocabularies) -> Tuple[torch.Tensor, torch.Tensor]:
    user_indices = [vocabs.user_to_idx.get(_standardize_id(value), 0) for value in df["visitorid"].tolist()]
    item_indices = [vocabs.item_to_idx.get(_standardize_id(value), 0) for value in df["itemid"].tolist()]
    return torch.tensor(user_indices, dtype=torch.long), torch.tensor(item_indices, dtype=torch.long)


class InteractionDataset(Dataset):
    def __init__(self, users: torch.Tensor, items: torch.Tensor):
        self.users = users
        self.items = items

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, index: int):
        return self.users[index], self.items[index]


class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_indices)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_indices)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(user_indices, item_indices)

    def score_pairs(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return torch.sum(user_vectors * item_vectors, dim=-1)

    def pairwise_logits(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return user_vectors @ item_vectors.T


def prepare_training_frame(
    train_df: pd.DataFrame,
    train_limit: Optional[int] = None,
    min_user_interactions: Optional[int] = None,
    min_item_interactions: Optional[int] = None,
) -> pd.DataFrame:
    _validate_min_interactions("min_user_interactions", min_user_interactions)
    _validate_min_interactions("min_item_interactions", min_item_interactions)

    df = train_df.copy()
    required = {"visitorid", "itemid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_df missing columns: {sorted(missing)}")

    df = df.dropna(subset=["visitorid", "itemid"]).reset_index(drop=True)
    df["visitorid"] = df["visitorid"].map(_standardize_id)
    df["itemid"] = df["itemid"].map(_standardize_id)

    if min_user_interactions is not None and min_user_interactions > 1:
        user_counts = df["visitorid"].value_counts()
        df = df[df["visitorid"].map(user_counts) >= min_user_interactions]

    if min_item_interactions is not None and min_item_interactions > 1:
        item_counts = df["itemid"].value_counts()
        df = df[df["itemid"].map(item_counts) >= min_item_interactions]

    if train_limit is not None:
        df = df.head(int(train_limit))

    return df.reset_index(drop=True)


def build_training_loader(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    batch_size: int,
    train_limit: Optional[int] = None,
    min_user_interactions: Optional[int] = None,
    min_item_interactions: Optional[int] = None,
) -> DataLoader:
    df = prepare_training_frame(
        train_df,
        train_limit=train_limit,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    users, items = _encode_pairs(df, vocabs)
    dataset = InteractionDataset(users, items)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train_two_tower(
    model: TwoTowerRetrievalModel,
    loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> List[float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history: List[float] = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)

            logits = model.pairwise_logits(batch_users, batch_items)
            labels = torch.arange(logits.shape[0], device=device)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        history.append(running_loss / max(batches, 1))

    return history


def build_item_embedding_matrix(model: TwoTowerRetrievalModel, device: str = "cpu") -> torch.Tensor:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model.item_embedding.weight.detach().clone().to(device)


def recommend_top_k_for_user(
    model: TwoTowerRetrievalModel,
    vocabs: Vocabularies,
    user_id,
    k: int,
    item_embedding_matrix: Optional[torch.Tensor] = None,
    seen_items: Optional[Iterable] = None,
    device: str = "cpu",
) -> List[str]:
    if k <= 0:
        raise ValueError("k must be positive")

    model = model.to(device)
    model.eval()
    user_idx = vocabs.user_to_idx.get(_standardize_id(user_id), 0)
    item_embedding_matrix = item_embedding_matrix if item_embedding_matrix is not None else build_item_embedding_matrix(model, device=device)
    seen_indices = {vocabs.item_to_idx.get(_standardize_id(value), 0) for value in (seen_items or [])}
    seen_indices.discard(0)

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
        user_vector = model.encode_users(user_tensor).squeeze(0)
        scores = item_embedding_matrix @ user_vector
        scores[0] = float("-inf")
        for index in seen_indices:
            if 0 <= index < scores.shape[0]:
                scores[index] = float("-inf")

        topk = torch.topk(scores, k=min(k, scores.shape[0] - 1)).indices.tolist()
        idx_to_item = vocabs.idx_to_item
        recommendations = [idx_to_item[index] for index in topk if index < len(idx_to_item) and idx_to_item[index] != UNK_TOKEN]
        return recommendations[:k]


def save_checkpoint(
    path: Path,
    model: TwoTowerRetrievalModel,
    vocabs: Vocabularies,
    config: TwoTowerConfig,
    history: Optional[List[float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocabs": {
            "user_to_idx": vocabs.user_to_idx,
            "item_to_idx": vocabs.item_to_idx,
        },
        "config": asdict(config),
        "history": history or [],
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[TwoTowerRetrievalModel, Vocabularies, TwoTowerConfig, List[float]]:
    payload = torch.load(path, map_location=map_location)
    config = TwoTowerConfig(**payload["config"])
    vocabs = Vocabularies(
        user_to_idx={str(key): int(value) for key, value in payload["vocabs"]["user_to_idx"].items()},
        item_to_idx={str(key): int(value) for key, value in payload["vocabs"]["item_to_idx"].items()},
    )
    model = TwoTowerRetrievalModel(vocabs.num_users, vocabs.num_items, embedding_dim=config.embedding_dim)
    model.load_state_dict(payload["model_state_dict"])
    history = [float(value) for value in payload.get("history", [])]
    return model, vocabs, config, history


def save_vocab_json(path: Path, vocabs: Vocabularies) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "user_to_idx": vocabs.user_to_idx,
                "item_to_idx": vocabs.item_to_idx,
            },
            f,
            indent=2,
            sort_keys=True,
        )
