import torch
from torch import nn


class RentalModel(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, n_hidden: int = 2):
        super().__init__()
        layers, n = [], in_features
        for _ in range(n_hidden):
            layers += [nn.Linear(n, hidden), nn.ReLU()]
            n = hidden
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
