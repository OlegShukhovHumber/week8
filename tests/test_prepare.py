import numpy as np
import torch

from prepare import NUMERIC, make_loader


def test_dataset_length_and_dtypes(data, datasets):
    train_ds, _, _ = datasets
    x, y = train_ds[0]
    assert len(train_ds) == len(data.X_train)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (data.n_features,)
    assert y.shape == ()


def test_split_sizes_add_up_and_do_not_overlap(data):
    train, val, test = (
        set(data.train_index),
        set(data.val_index),
        set(data.test_index),
    )
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == 6000


def test_scaler_is_fitted_on_training_rows_only(data):
    k = len(NUMERIC)
    assert np.allclose(data.X_train[:, :k].mean(axis=0), 0, atol=1e-5)
    assert not np.allclose(data.X_val[:, :k].mean(axis=0), 0, atol=1e-5)


def test_train_loader_shuffles_but_is_repeatable(datasets):
    train_ds, val_ds, _ = datasets
    first = next(iter(make_loader(train_ds, 64, shuffle=True, seed=7)))
    again = next(iter(make_loader(train_ds, 64, shuffle=True, seed=7)))
    other = next(iter(make_loader(train_ds, 64, shuffle=True, seed=8)))
    ordered = next(iter(make_loader(train_ds, 64)))
    assert torch.equal(first[1], again[1])
    assert not torch.equal(first[1], other[1])
    assert not torch.equal(first[1], ordered[1])
    val_a = next(iter(make_loader(val_ds, 64)))
    val_b = next(iter(make_loader(val_ds, 64)))
    assert torch.equal(val_a[1], val_b[1])
