import pytest
import torch

from prepare import make_loader, prepare


@pytest.fixture(scope="session")
def data():
    return prepare()


@pytest.fixture(scope="session")
def datasets(data):
    return data.datasets()


@pytest.fixture
def small_loaders(datasets):
    train_ds, val_ds, _ = datasets
    small_train = torch.utils.data.Subset(train_ds, range(256))
    small_val = torch.utils.data.Subset(val_ds, range(128))
    return (
        make_loader(small_train, 64, shuffle=True),
        make_loader(small_val, 64),
    )


@pytest.fixture
def one_batch(datasets):
    train_ds, _, _ = datasets
    return next(iter(make_loader(train_ds, 32)))
