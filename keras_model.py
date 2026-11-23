"""The Lab 07 network written again in Keras, and the weight copy from PyTorch."""

import keras
import numpy as np
import torch
from torch import nn


def build_keras_model(
    in_features: int, hidden: int = 64, n_hidden: int = 2
) -> keras.Sequential:
    """Same architecture as model.RentalModel: n_hidden Dense+ReLU, one linear out."""
    layers = [keras.Input(shape=(in_features,))]
    for i in range(n_hidden):
        layers.append(keras.layers.Dense(hidden, activation="relu", name=f"hidden_{i}"))
    layers.append(keras.layers.Dense(1, name="output"))
    return keras.Sequential(layers, name="rental_model_keras")


def count_torch_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def copy_torch_to_keras(torch_net: nn.Module, keras_net: keras.Model) -> None:
    """Copy every nn.Linear into the matching Dense layer.

    PyTorch stores a weight as (outputs, inputs), Keras a kernel as
    (inputs, outputs), so each weight matrix is transposed.
    """
    linears = [m for m in torch_net.modules() if isinstance(m, nn.Linear)]
    denses = [
        layer for layer in keras_net.layers if isinstance(layer, keras.layers.Dense)
    ]
    if len(linears) != len(denses):
        raise ValueError(f"{len(linears)} Linear layers but {len(denses)} Dense layers")
    for linear, dense in zip(linears, denses, strict=True):
        dense.set_weights(
            [
                linear.weight.detach().cpu().numpy().T,
                linear.bias.detach().cpu().numpy(),
            ]
        )


def torch_predict(torch_net: nn.Module, x: np.ndarray) -> np.ndarray:
    torch_net.eval()
    with torch.no_grad():
        return torch_net(torch.as_tensor(x, dtype=torch.float32)).numpy().ravel()


def keras_predict(keras_net: keras.Model, x: np.ndarray) -> np.ndarray:
    # predict returns (N, 1); flatten so it lines up with targets of shape (N,)
    return keras_net.predict(x, verbose=0).ravel()


def max_prediction_gap(
    torch_net: nn.Module, keras_net: keras.Model, x: np.ndarray
) -> float:
    return float(
        np.abs(torch_predict(torch_net, x) - keras_predict(keras_net, x)).max()
    )
