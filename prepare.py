from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import STATION_TYPES, TARGET, make_rentals

SEED = 2026
NUMERIC = ["temperature_c", "rain_mm", "wind_kph"]
HOURS = list(range(24))


class RentalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


@dataclass
class Prepared:
    prep: ColumnTransformer
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def datasets(self):
        return (
            RentalDataset(self.X_train, self.y_train),
            RentalDataset(self.X_val, self.y_val),
            RentalDataset(self.X_test, self.y_test),
        )


def split_table(frame: pd.DataFrame, seed: int = SEED):
    X, y = frame.drop(columns=[TARGET]), frame[TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(categories=[STATION_TYPES, HOURS], sparse_output=False),
                ["station_type", "hour"],
            ),
            ("pass", "passthrough", ["is_weekend"]),
        ]
    )


def prepare(n: int = 6000, seed: int = SEED) -> Prepared:
    frame = make_rentals(n=n, seed=seed)
    X_train, X_val, X_test, y_train, y_val, y_test = split_table(frame, seed)
    prep = build_preprocessor()
    return Prepared(
        prep=prep,
        X_train=prep.fit_transform(X_train).astype("float32"),
        X_val=prep.transform(X_val).astype("float32"),
        X_test=prep.transform(X_test).astype("float32"),
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        train_index=X_train.index,
        val_index=X_val.index,
        test_index=X_test.index,
    )


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = False, seed: int = SEED
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
