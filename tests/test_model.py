import copy

import pytest
import torch
from torch import nn, optim

from model import RentalModel


@pytest.mark.parametrize("n_hidden", [2, 3])
@pytest.mark.parametrize("hidden", [4, 16, 64])
def test_output_has_one_value_per_row(one_batch, hidden, n_hidden):
    x, y = one_batch
    out = RentalModel(x.shape[1], hidden, n_hidden)(x)
    assert out.shape == y.shape == (len(x),)


def test_one_optimizer_step_changes_every_layer(one_batch):
    x, y = one_batch
    torch.manual_seed(0)
    model = RentalModel(x.shape[1], hidden=16)
    before = copy.deepcopy(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss = nn.MSELoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        assert not torch.equal(value, before[name]), name


def test_network_can_memorize_one_small_batch(one_batch):
    x, y = one_batch
    torch.manual_seed(0)
    model = RentalModel(x.shape[1], hidden=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    start = loss_fn(model(x), y).item()
    for _ in range(300):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05 * start
