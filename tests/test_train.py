import copy

import pytest
import torch
from torch import nn, optim

from model import RentalModel
from train import fit, run_epoch


def test_validation_epoch_leaves_weights_unchanged(small_loaders, data):
    _, val_loader = small_loaders
    model = RentalModel(data.n_features, hidden=16)
    before = copy.deepcopy(model.state_dict())
    run_epoch(model, val_loader, nn.MSELoss())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_training_epoch_reduces_the_loss(small_loaders, data):
    train_loader, _ = small_loaders
    torch.manual_seed(0)
    model = RentalModel(data.n_features, hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    first, _ = run_epoch(model, train_loader, loss_fn, optimizer)
    for _ in range(30):
        last, _ = run_epoch(model, train_loader, loss_fn, optimizer)
    assert last < 0.7 * first


def test_gradients_are_cleared_between_steps(one_batch):
    x, y = one_batch
    torch.manual_seed(0)
    model = RentalModel(x.shape[1], hidden=8)
    reference = copy.deepcopy(model)
    lr = 1e-3
    loss_fn = nn.MSELoss()

    run_epoch(model, [(x, y), (x, y)], loss_fn, optim.SGD(model.parameters(), lr=lr))

    for _ in range(2):
        reference.zero_grad()
        loss_fn(reference(x), y).backward()
        with torch.no_grad():
            for p in reference.parameters():
                p -= lr * p.grad

    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_fit_logs_every_epoch_and_restores_best_weights(small_loaders, data):
    train_loader, val_loader = small_loaders
    torch.manual_seed(0)
    model = RentalModel(data.n_features, hidden=16)
    loss_fn = nn.MSELoss()
    history, best = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optim.Adam(model.parameters(), lr=1e-2),
        epochs=6,
    )
    assert list(history.columns) == ["epoch", "train_loss", "val_loss", "val_mae"]
    assert len(history) == 6
    assert best == int(history.val_loss.idxmin()) + 1
    restored, _ = run_epoch(model, val_loader, loss_fn)
    assert restored == pytest.approx(history.val_loss.min(), rel=1e-5)
