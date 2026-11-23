"""Check both versions are the same function, then train both on several seeds."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch import nn, optim

from keras_model import (
    build_keras_model,
    copy_torch_to_keras,
    keras_predict,
    max_prediction_gap,
)
from model import RentalModel
from prepare import SEED, Prepared, make_loader, prepare
from train import EPOCHS, MAIN, fit, run_epoch
from train_keras import CHECKPOINTS, fit_keras, keras_arrays

REPORTS = Path("reports")
SEEDS = (2026, 7, 42)
TOLERANCE = 1e-5


def check_equivalence(data: Prepared, seed: int = SEED) -> float:
    torch.manual_seed(seed)
    torch_net = RentalModel(data.n_features, MAIN["hidden"], MAIN["n_hidden"])
    keras_net = build_keras_model(data.n_features, MAIN["hidden"], MAIN["n_hidden"])
    copy_torch_to_keras(torch_net, keras_net)
    return max_prediction_gap(torch_net, keras_net, data.X_val)


def run_torch(data: Prepared, seed: int, epochs: int = EPOCHS, **_) -> dict:
    torch.manual_seed(seed)
    model = RentalModel(data.n_features, MAIN["hidden"], MAIN["n_hidden"])
    train_ds, val_ds, test_ds = data.datasets()
    train_loader = make_loader(train_ds, MAIN["batch_size"], shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, 256)
    start = time.perf_counter()
    history, best = fit(
        model,
        train_loader,
        val_loader,
        nn.MSELoss(),
        optim.Adam(model.parameters(), lr=MAIN["lr"]),
        epochs,
    )
    seconds = time.perf_counter() - start
    _, test_mae = run_epoch(model, make_loader(test_ds, 256), nn.MSELoss())
    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best,
        "val_mae": float(history.val_mae.iloc[best - 1]),
        "test_mae": test_mae,
        "seconds": round(seconds, 1),
    }


def run_keras(
    data: Prepared, seed: int, epochs: int = EPOCHS, checkpoint_dir=CHECKPOINTS
) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = keras_arrays(data)
    run = fit_keras(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        seed=seed,
        checkpoint_path=Path(checkpoint_dir) / f"rental_keras_seed{seed}.keras",
    )
    test_mae = float(np.abs(keras_predict(run.model, x_test) - y_test).mean())
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": run.best_epoch,
        "val_mae": float(run.history.val_mae.iloc[run.best_epoch - 1]),
        "test_mae": test_mae,
        "seconds": round(run.seconds, 1),
    }


def compare(
    data: Prepared, seeds=SEEDS, epochs: int = EPOCHS, checkpoint_dir=CHECKPOINTS
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for runner in (run_torch, run_keras):
            row = runner(data, seed, epochs, checkpoint_dir=checkpoint_dir)
            rows.append(row)
            print(
                f"{row['framework']:8s} seed {seed:5d}  best epoch "
                f"{row['best_epoch']:3d}  val MAE {row['val_mae']:.3f}  "
                f"test MAE {row['test_mae']:.3f}  {row['seconds']:.1f}s"
            )
    return pd.DataFrame(rows)


def main():
    tf.config.experimental.enable_op_determinism()
    REPORTS.mkdir(exist_ok=True)
    data = prepare()

    gap = check_equivalence(data)
    print(f"Same weights, largest prediction gap on validation rows: {gap:.2e}")
    if gap > TOLERANCE:
        raise SystemExit(f"Gap {gap:.2e} is above the tolerance {TOLERANCE:.0e}")

    table = compare(data)
    table.to_csv(REPORTS / "comparison.csv", index=False)
    summary = table.groupby("framework")[
        ["best_epoch", "val_mae", "test_mae", "seconds"]
    ].agg(["mean", "std"])
    print(summary.round(3).to_string())
    print("Saved reports/comparison.csv")


if __name__ == "__main__":
    main()
