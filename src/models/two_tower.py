"""Minimal two-tower retrieval models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.baselines import DEFAULT_EVENT_WEIGHTS


UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TwoTowerConfig:
    query_tower: str = "id_only"
    use_category_embeddings: bool = False
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
    use_event_weights_in_history: bool = False
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


def _event_weight_for_value(event_value, event_weights: Optional[Dict[str, float]] = None) -> float:
    weights = event_weights or DEFAULT_EVENT_WEIGHTS
    if pd.isna(event_value):
        return 1.0
    return float(weights.get(str(event_value).lower(), 1.0))


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
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_categories: Optional[int] = None,
        item_category_indices: Optional[torch.Tensor] = None,
        category_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embedding: Optional[nn.Embedding] = None
        self.category_to_idx = dict(category_to_idx or {})
        if num_categories is not None:
            self.category_embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
            if item_category_indices is None:
                raise ValueError("item_category_indices is required when num_categories is provided")
            self.register_buffer("item_category_indices", item_category_indices.to(dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        if self.category_embedding is not None:
            nn.init.normal_(self.category_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.category_embedding.weight[0].zero_()

    def aggregate_history(
        self,
        history_item_indices: torch.Tensor,
        history_event_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        history_embeddings = self.item_embedding(history_item_indices)
        history_mask = history_item_indices.ne(0)
        if history_event_weights is None:
            weights = history_mask.to(dtype=history_embeddings.dtype)
        else:
            weights = history_event_weights.to(dtype=history_embeddings.dtype) * history_mask.to(dtype=history_embeddings.dtype)
        masked_embeddings = history_embeddings * weights.unsqueeze(-1)
        counts = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked_embeddings.sum(dim=1) / counts

    def encode_queries(
        self,
        history_item_indices: torch.Tensor,
        history_event_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.aggregate_history(history_item_indices, history_event_weights=history_event_weights)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        item_vectors = self.item_embedding(item_indices)
        if self.category_embedding is None:
            return item_vectors
        category_indices = self.item_category_indices[item_indices]
        category_vectors = self.category_embedding(category_indices)
        return item_vectors + category_vectors

    def forward(
        self,
        history_item_indices: torch.Tensor,
        item_indices: torch.Tensor,
        history_event_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.score_pairs(history_item_indices, item_indices, history_event_weights=history_event_weights)

    def score_pairs(
        self,
        history_item_indices: torch.Tensor,
        item_indices: torch.Tensor,
        history_event_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_vectors = self.encode_queries(history_item_indices, history_event_weights=history_event_weights)
        item_vectors = self.encode_items(item_indices)
        return torch.sum(query_vectors * item_vectors, dim=-1)

    def pairwise_logits(
        self,
        history_item_indices: torch.Tensor,
        item_indices: torch.Tensor,
        history_event_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_vectors = self.encode_queries(history_item_indices, history_event_weights=history_event_weights)
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


def category_tree_ids_from_csv(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "categoryid" not in df.columns:
        raise ValueError("category_tree csv must contain categoryid")
    return set(df["categoryid"].dropna().astype(str).tolist())


def build_item_category_map(
    item_properties_paths: Sequence[Path],
    item_ids: Optional[set[str]] = None,
    category_ids: Optional[set[str]] = None,
    max_timestamp_ms: Optional[int] = None,
    chunksize: int = 500_000,
) -> Dict[str, str]:
    latest_by_item: Dict[str, Tuple[int, str]] = {}
    item_id_filter = set(item_ids) if item_ids is not None else None
    category_filter = set(category_ids) if category_ids is not None else None

    for path in item_properties_paths:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            required = {"timestamp", "itemid", "property", "value"}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"item properties csv missing columns: {sorted(missing)}")

            category_chunk = chunk[chunk["property"].astype(str) == "categoryid"][["timestamp", "itemid", "value"]].copy()
            if category_chunk.empty:
                continue

            category_chunk["itemid"] = category_chunk["itemid"].astype(str)
            category_chunk["value"] = category_chunk["value"].astype(str)
            category_chunk["timestamp"] = category_chunk["timestamp"].astype(int)

            if item_id_filter is not None:
                category_chunk = category_chunk[category_chunk["itemid"].isin(item_id_filter)]
            if max_timestamp_ms is not None:
                category_chunk = category_chunk[category_chunk["timestamp"] <= int(max_timestamp_ms)]
            if category_filter is not None:
                category_chunk = category_chunk[category_chunk["value"].isin(category_filter)]

            if category_chunk.empty:
                continue

            category_chunk = category_chunk.sort_values(["itemid", "timestamp"], kind="stable")
            latest = category_chunk.groupby("itemid", sort=False).tail(1)
            for row in latest.itertuples(index=False):
                previous = latest_by_item.get(row.itemid)
                if previous is None or int(row.timestamp) >= previous[0]:
                    latest_by_item[row.itemid] = (int(row.timestamp), str(row.value))

    return {item_id: category_id for item_id, (_, category_id) in latest_by_item.items()}


def build_category_vocab(item_category_map: Dict[str, str]) -> Dict[str, int]:
    categories = sorted(set(item_category_map.values()))
    return {category: index + 1 for index, category in enumerate(categories)}


def build_item_category_index_tensor(
    vocabs: Vocabularies,
    item_category_map: Dict[str, str],
    category_to_idx: Dict[str, int],
) -> torch.Tensor:
    item_category_indices = torch.zeros(vocabs.num_items, dtype=torch.long)
    for item_id, item_index in vocabs.item_to_idx.items():
        category_id = item_category_map.get(str(item_id))
        if category_id is None:
            continue
        item_category_indices[item_index] = int(category_to_idx.get(str(category_id), 0))
    return item_category_indices


def count_items_with_categories(vocabs: Vocabularies, item_category_map: Dict[str, str]) -> int:
    return sum(1 for item_id in vocabs.item_to_idx if str(item_id) in item_category_map)


def timestamp_to_epoch_ms(value) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.value // 1_000_000)


def count_users_with_non_empty_history(train_df: pd.DataFrame) -> int:
    if train_df.empty:
        return 0
    counts = train_df["visitorid"].value_counts()
    return int((counts >= 2).sum())


def build_train_history_maps(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    max_history_length: Optional[int] = None,
    use_event_weights: bool = False,
    event_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    _validate_max_history_length(max_history_length)

    if train_df.empty:
        return {}, {}

    ordered_df = train_df.copy()
    if "timestamp" in ordered_df.columns:
        ordered_df = ordered_df.sort_values(["timestamp", "visitorid", "itemid"], kind="stable")

    history_map: Dict[str, List[int]] = {}
    history_weight_map: Dict[str, List[float]] = {}
    for user_id, frame in ordered_df.groupby("visitorid", sort=False):
        item_sequence = []
        weight_sequence = []
        for item_value, event_value in zip(frame["itemid"].tolist(), frame.get("event", pd.Series([None] * len(frame))).tolist()):
            item_index = vocabs.item_to_idx.get(_standardize_id(item_value), 0)
            if item_index == 0:
                continue
            item_sequence.append(item_index)
            weight_sequence.append(_event_weight_for_value(event_value, event_weights) if use_event_weights else 1.0)
        if max_history_length is not None:
            item_sequence = item_sequence[-int(max_history_length):]
            weight_sequence = weight_sequence[-int(max_history_length):]
        history_map[str(user_id)] = item_sequence
        history_weight_map[str(user_id)] = weight_sequence
    return history_map, history_weight_map


def build_train_history_map(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    max_history_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    history_map, _ = build_train_history_maps(train_df, vocabs, max_history_length=max_history_length)
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


def _history_weights_to_padded_tensor(
    history_weights: List[List[float]],
    max_history_length: Optional[int],
) -> torch.Tensor:
    _validate_max_history_length(max_history_length)

    if not history_weights:
        width = int(max_history_length or 0)
        return torch.zeros((0, width), dtype=torch.float32)

    effective_width = max((len(items) for items in history_weights), default=0)
    if max_history_length is not None:
        effective_width = min(effective_width, int(max_history_length))

    if effective_width == 0:
        effective_width = int(max_history_length or 1)

    tensor = torch.zeros((len(history_weights), effective_width), dtype=torch.float32)
    for row_index, weights in enumerate(history_weights):
        trimmed = weights[-effective_width:]
        if trimmed:
            tensor[row_index, -len(trimmed):] = torch.tensor(trimmed, dtype=torch.float32)
    return tensor


def build_eval_history_tensor(
    user_ids: Iterable,
    history_map: Dict[str, List[int]],
    max_history_length: Optional[int],
) -> torch.Tensor:
    history_items = [history_map.get(_standardize_id(user_id), []) for user_id in user_ids]
    return _history_to_padded_tensor(history_items, max_history_length=max_history_length)


def build_eval_history_tensors(
    user_ids: Iterable,
    history_map: Dict[str, List[int]],
    history_weight_map: Dict[str, List[float]],
    max_history_length: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    history_items = [history_map.get(_standardize_id(user_id), []) for user_id in user_ids]
    history_weights = [history_weight_map.get(_standardize_id(user_id), []) for user_id in user_ids]
    return (
        _history_to_padded_tensor(history_items, max_history_length=max_history_length),
        _history_weights_to_padded_tensor(history_weights, max_history_length=max_history_length),
    )


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
    def __init__(self, histories: torch.Tensor, history_weights: torch.Tensor, items: torch.Tensor):
        self.histories = histories
        self.history_weights = history_weights
        self.items = items

    def __len__(self) -> int:
        return int(self.items.shape[0])

    def __getitem__(self, index: int):
        return self.histories[index], self.history_weights[index], self.items[index]


def build_history_training_loader(
    train_df: pd.DataFrame,
    vocabs: Vocabularies,
    batch_size: int,
    max_history_length: Optional[int] = 20,
    use_event_weights: bool = False,
    event_weights: Optional[Dict[str, float]] = None,
) -> DataLoader:
    _validate_max_history_length(max_history_length)

    if train_df.empty:
        raise ValueError("train_df must not be empty")

    ordered_df = train_df.copy()
    ordered_df["_row_order"] = range(len(ordered_df))
    if "timestamp" in ordered_df.columns:
        ordered_df = ordered_df.sort_values(["timestamp", "_row_order"], kind="stable")

    running_histories: Dict[str, List[int]] = {}
    running_history_weights: Dict[str, List[float]] = {}
    example_histories: List[List[int]] = []
    example_history_weights: List[List[float]] = []
    target_items: List[int] = []

    for row in ordered_df.itertuples(index=False):
        user_id = _standardize_id(row.visitorid)
        item_index = vocabs.item_to_idx.get(_standardize_id(row.itemid), 0)
        prior_history = list(running_histories.get(user_id, []))
        prior_history_weights = list(running_history_weights.get(user_id, []))
        example_histories.append(prior_history)
        example_history_weights.append(prior_history_weights)
        target_items.append(item_index)

        if item_index != 0:
            updated_history = prior_history + [item_index]
            current_weight = _event_weight_for_value(getattr(row, "event", None), event_weights) if use_event_weights else 1.0
            updated_history_weights = prior_history_weights + [current_weight]
            if max_history_length is not None:
                updated_history = updated_history[-int(max_history_length):]
                updated_history_weights = updated_history_weights[-int(max_history_length):]
            running_histories[user_id] = updated_history
            running_history_weights[user_id] = updated_history_weights
        else:
            running_histories[user_id] = prior_history
            running_history_weights[user_id] = prior_history_weights

    history_tensor = _history_to_padded_tensor(example_histories, max_history_length=max_history_length)
    history_weight_tensor = _history_weights_to_padded_tensor(example_history_weights, max_history_length=max_history_length)
    item_tensor = torch.tensor(target_items, dtype=torch.long)
    dataset = HistoryInteractionDataset(history_tensor, history_weight_tensor, item_tensor)
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
        for batch_histories, batch_history_weights, batch_items in loader:
            batch_histories = batch_histories.to(device)
            batch_history_weights = batch_history_weights.to(device)
            batch_items = batch_items.to(device)

            logits = model.pairwise_logits(batch_histories, batch_items, history_event_weights=batch_history_weights)
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
        num_items = int(model.item_embedding.num_embeddings)
        item_indices = torch.arange(num_items, dtype=torch.long, device=device)
        return model.encode_items(item_indices).detach().clone()


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
    history_event_weights: Optional[torch.Tensor],
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
        weight_tensor = None
        if history_event_weights is not None:
            weight_tensor = history_event_weights.to(device)
            if weight_tensor.ndim == 1:
                weight_tensor = weight_tensor.unsqueeze(0)
        query_vector = model.encode_queries(history_tensor, history_event_weights=weight_tensor).squeeze(0)
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
    if hasattr(model, "category_to_idx") and getattr(model, "category_embedding", None) is not None:
        payload["category_features"] = {
            "category_to_idx": dict(getattr(model, "category_to_idx", {})),
            "item_category_indices": getattr(model, "item_category_indices").detach().cpu().tolist(),
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
        category_features = payload.get("category_features")
        if config.use_category_embeddings:
            if not category_features:
                raise ValueError("checkpoint missing category_features for category-aware history model")
            category_to_idx = {str(key): int(value) for key, value in category_features["category_to_idx"].items()}
            item_category_indices = torch.tensor(category_features["item_category_indices"], dtype=torch.long)
            num_categories = max(category_to_idx.values(), default=0) + 1
            model = HistoryQueryTwoTowerRetrievalModel(
                vocabs.num_items,
                embedding_dim=config.embedding_dim,
                num_categories=num_categories,
                item_category_indices=item_category_indices,
                category_to_idx=category_to_idx,
            )
        else:
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
