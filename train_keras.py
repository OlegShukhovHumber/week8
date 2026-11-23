"""Train the Keras version with compile and fit, using the same rows and settings."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras_model import build_keras_model, keras_predict
from prepare import SEED, Prepared, prepare
from train import EPOCHS, MAIN, plot_curves

REPORTS = Path("reports")
CHECKPOINTS = Path("checkpoints")


@dataclass
class KerasRun:
    model: keras.Model
    history: pd.DataFrame
    best_epoch: int
    seconds: float


def keras_arrays(data: Prepared):
    """The exact arrays PyTorch's datasets are built from, as float32."""
    return tuple(
        np.asarray(a, dtype="float32")
        for a in (
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            data.X_test,
            data.y_test,
        )
    )


def fit_keras(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    hidden=MAIN["hidden"],
    n_hidden=MAIN["n_hidden"],
    lr=MAIN["lr"],
    batch_size=MAIN["batch_size"],
    epochs=EPOCHS,
    seed=SEED,
    checkpoint_path=None,
    verbose=0,
) -> KerasRun:
    keras.utils.set_random_seed(seed)
    model = build_keras_model(x_train.shape[1], hidden, n_hidden)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        # patience = epochs: never stops early, so both frameworks train equally long
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs, restore_best_weights=True
        ),
    ]
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True
            )
        )
    start = time.perf_counter()
    fitted = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    seconds = time.perf_counter() - start
    h = fitted.history
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(h["loss"]) + 1),
            "train_loss": h["loss"],
            "val_loss": h["val_loss"],
            "val_mae": h["val_mae"],
        }
    )
    best_epoch = int(history.val_loss.idxmin()) + 1
    return KerasRun(model, history, best_epoch, seconds)


def main():
    plt.switch_backend("Agg")
    tf.config.experimental.enable_op_determinism()
    REPORTS.mkdir(exist_ok=True)
    data = prepare()
    x_train, y_train, x_val, y_val, x_test, y_test = keras_arrays(data)

    run = fit_keras(
        x_train,
        y_train,
        x_val,
        y_val,
        checkpoint_path=CHECKPOINTS / "rental_keras.keras",
        verbose=2,
    )
    run.history.to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(
        run.history,
        run.best_epoch,
        "main run (Keras)",
        path=REPORTS / "learning_curves_keras.png",
    )

    # scored on the test rows once, with the best weights already restored
    best = run.history.iloc[run.best_epoch - 1]
    test_pred = keras_predict(run.model, x_test)
    metrics = {
        "framework": "keras",
        "seed": SEED,
        "best_epoch": run.best_epoch,
        "val_mae": float(best.val_mae),
        "val_mse": float(best.val_loss),
        "test_mae": float(np.abs(test_pred - y_test).mean()),
        "test_mse": float(((test_pred - y_test) ** 2).mean()),
        "seconds": round(run.seconds, 1),
    }
    (REPORTS / "metrics_keras.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(
        "Saved checkpoints/rental_keras.keras, reports/history_keras.csv, "
        "reports/learning_curves_keras.png, reports/metrics_keras.json"
    )


if __name__ == "__main__":
    main()
