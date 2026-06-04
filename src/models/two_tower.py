"""Minimal two-tower retrieval models."""

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
    query_tower: str = "id_only"
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
    max_history_length: Optional[int] = 20
    effective_train_rows: Optional[int] = None
    effective_train_users: Optional[int] = None
    effective_train_items: Optional[int] = None
    users_with_non_empty_history: Optional[int] = None
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


def _validate_max_history_length(value: Optional[int]) -> None:
    if value is not None and value <= 0:
        raise ValueError("max_history_length must be positive when provided")


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


def _encode_item_sequence(item_ids: Iterable, vocabs: Vocabularies, max_history_length: Optional[int] = None) -> List[int]:
    encoded = [vocabs.item_to_idx.get(_standardize_id(value), 0) for value in item_ids]
    encoded = [value for value in encoded if value != 0]
    if max_history_length is not None:
        encoded = encoded[-int(max_history_length):]
    return encoded


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


class HistoryQueryTwoTowerRetrievalModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def aggregate_history(self, history_item_indices: torch.Tensor) -> torch.Tensor:
        history_embeddings = self.item_embedding(history_item_indices)
        history_mask = history_item_indices.ne(0).unsqueeze(-1)
        masked_embeddings = history_embeddings * history_mask
        counts = history_mask.sum(dim=1).clamp(min=1)
        return masked_embeddings.sum(dim=1) / counts

    def encode_queries(self, history_item_indices: torch.Tensor) -> torch.Tensor:
        return self.aggregate_history(history_item_indices)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_indices)

    def forward(self, history_item_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(history_item_indices, item_indices)

    def score_pairs(self, history_item_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        query_vectors = self.encode_queries(history_item_indices)
        item_vectors = self.encode_items(item_indices)
        return torch.sum(query_vectors * item_vectors, dim=-1)

    def pairwise_logits(self, history_item_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        query_vectors = self.encode_queries(history_item_indices)
        item_vectors = self.encode_items(item_indices)
        return query_vectors @ item_vectors.T


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


def count_users_with_non_empty_history(train_df: pd.DataFrame) -> int:
    if train_df.empty:
        return 0
    counts = train_df["visitorid"].value_counts()
    return int((counts >= 2).sum())


def build_train_history_map(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    max_history_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    _validate_max_history_length(max_history_length)

    if train_df.empty:
        return {}

    ordered_df = train_df.copy()
    if "timestamp" in ordered_df.columns:
        ordered_df = ordered_df.sort_values(["timestamp", "visitorid", "itemid"], kind="stable")

    history_map: Dict[str, List[int]] = {}
    for user_id, frame in ordered_df.groupby("visitorid", sort=False):
        history_map[str(user_id)] = _encode_item_sequence(frame["itemid"].tolist(), vocabs, max_history_length=max_history_length)
    return history_map


def _history_to_padded_tensor(history_items: List[List[int]], max_history_length: Optional[int]) -> torch.Tensor:
    _validate_max_history_length(max_history_length)

    if not history_items:
        width = int(max_history_length or 0)
        return torch.zeros((0, width), dtype=torch.long)

    effective_width = max((len(items) for items in history_items), default=0)
    if max_history_length is not None:
        effective_width = min(effective_width, int(max_history_length))

    if effective_width == 0:
        effective_width = int(max_history_length or 1)

    tensor = torch.zeros((len(history_items), effective_width), dtype=torch.long)
    for row_index, items in enumerate(history_items):
        trimmed = items[-effective_width:]
        if trimmed:
            tensor[row_index, -len(trimmed):] = torch.tensor(trimmed, dtype=torch.long)
    return tensor


def build_eval_history_tensor(
    user_ids: Iterable,
    history_map: Dict[str, List[int]],
    max_history_length: Optional[int],
) -> torch.Tensor:
    history_items = [history_map.get(_standardize_id(user_id), []) for user_id in user_ids]
    return _history_to_padded_tensor(history_items, max_history_length=max_history_length)


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


class HistoryInteractionDataset(Dataset):
    def __init__(self, histories: torch.Tensor, items: torch.Tensor):
        self.histories = histories
        self.items = items

    def __len__(self) -> int:
        return int(self.items.shape[0])

    def __getitem__(self, index: int):
        return self.histories[index], self.items[index]


def build_history_training_loader(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    batch_size: int,
    max_history_length: Optional[int] = 20,
) -> DataLoader:
    _validate_max_history_length(max_history_length)

    if train_df.empty:
        raise ValueError("train_df must not be empty")

    ordered_df = train_df.copy()
    ordered_df["_row_order"] = range(len(ordered_df))
    if "timestamp" in ordered_df.columns:
        ordered_df = ordered_df.sort_values(["timestamp", "_row_order"], kind="stable")

    running_histories: Dict[str, List[int]] = {}
    example_histories: List[List[int]] = []
    target_items: List[int] = []

    for row in ordered_df.itertuples(index=False):
        user_id = _standardize_id(row.visitorid)
        item_index = vocabs.item_to_idx.get(_standardize_id(row.itemid), 0)
        prior_history = list(running_histories.get(user_id, []))
        example_histories.append(prior_history)
        target_items.append(item_index)

        if item_index != 0:
            updated_history = prior_history + [item_index]
            if max_history_length is not None:
                updated_history = updated_history[-int(max_history_length):]
            running_histories[user_id] = updated_history
        else:
            running_histories[user_id] = prior_history

    history_tensor = _history_to_padded_tensor(example_histories, max_history_length=max_history_length)
    item_tensor = torch.tensor(target_items, dtype=torch.long)
    dataset = HistoryInteractionDataset(history_tensor, item_tensor)
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


def train_history_two_tower(
    model: HistoryQueryTwoTowerRetrievalModel,
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
        for batch_histories, batch_items in loader:
            batch_histories = batch_histories.to(device)
            batch_items = batch_items.to(device)

            logits = model.pairwise_logits(batch_histories, batch_items)
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


def recommend_top_k_for_history(
    model: HistoryQueryTwoTowerRetrievalModel,
    vocabs: Vocabularies,
    history_item_indices: torch.Tensor,
    k: int,
    item_embedding_matrix: Optional[torch.Tensor] = None,
    seen_items: Optional[Iterable] = None,
    device: str = "cpu",
) -> List[str]:
    if k <= 0:
        raise ValueError("k must be positive")

    model = model.to(device)
    model.eval()
    item_embedding_matrix = item_embedding_matrix if item_embedding_matrix is not None else build_item_embedding_matrix(model, device=device)
    seen_indices = {vocabs.item_to_idx.get(_standardize_id(value), 0) for value in (seen_items or [])}
    seen_indices.discard(0)

    with torch.no_grad():
        history_tensor = history_item_indices.to(device)
        if history_tensor.ndim == 1:
            history_tensor = history_tensor.unsqueeze(0)
        query_vector = model.encode_queries(history_tensor).squeeze(0)
        scores = item_embedding_matrix @ query_vector
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
    model: nn.Module,
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


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[nn.Module, Vocabularies, TwoTowerConfig, List[float]]:
    payload = torch.load(path, map_location=map_location)
    config = TwoTowerConfig(**payload["config"])
    vocabs = Vocabularies(
        user_to_idx={str(key): int(value) for key, value in payload["vocabs"]["user_to_idx"].items()},
        item_to_idx={str(key): int(value) for key, value in payload["vocabs"]["item_to_idx"].items()},
    )
    if config.query_tower == "history":
        model = HistoryQueryTwoTowerRetrievalModel(vocabs.num_items, embedding_dim=config.embedding_dim)
    else:
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
