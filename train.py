import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn, optim

from bikes_data import TARGET, make_rentals
from model import RentalModel
from prepare import SEED, make_loader, prepare

REPORTS = Path("reports")
EPOCHS = 150
MAIN = {"hidden": 64, "n_hidden": 2, "lr": 3e-3, "batch_size": 64}
OTHER_RUNS = [
    ("lr too small", {"lr": 1e-5}),
    ("lr too large", {"lr": 1.0}),
    ("huge net", {"hidden": 512, "n_hidden": 3}),
    ("big batch", {"batch_size": 1024}),
]


def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_abs, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs += (pred.detach() - y).abs().sum().item()
            n += len(y)
    return total_loss / n, total_abs / n


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, verbose=0):
    rows = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and epoch % verbose == 0:
            print(
                f"{epoch:4d}  train={train_loss:8.3f}  "
                f"val={val_loss:8.3f}  mae={val_mae:.3f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return pd.DataFrame(rows), best_epoch


def run_experiment(data, name, config, epochs=EPOCHS, verbose=0):
    cfg = {**MAIN, **config}
    torch.manual_seed(SEED)
    model = RentalModel(data.n_features, cfg["hidden"], cfg["n_hidden"])
    train_ds, val_ds, _ = data.datasets()
    train_loader = make_loader(train_ds, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_ds, 256)
    start = time.perf_counter()
    history, best = fit(
        model,
        train_loader,
        val_loader,
        nn.MSELoss(),
        optim.Adam(model.parameters(), lr=cfg["lr"]),
        epochs,
        verbose=verbose,
    )
    row = {
        "run": name,
        "hidden": cfg["hidden"],
        "layers": cfg["n_hidden"],
        "lr": cfg["lr"],
        "batch": cfg["batch_size"],
        "epochs": epochs,
        "best_epoch": best,
        "best_val_mae": history.val_mae.iloc[best - 1],
        "final_train_loss": history.train_loss.iloc[-1],
        "final_val_loss": history.val_loss.iloc[-1],
        "seconds": round(time.perf_counter() - start, 1),
    }
    return model, history, row


def baselines(data):
    mean = float(data.y_train.mean())
    linear = LinearRegression().fit(data.X_train, data.y_train)
    out = {}
    for split, X, y in [
        ("val", data.X_val, data.y_val),
        ("test", data.X_test, data.y_test),
    ]:
        out[f"mean_{split}"] = {
            "mae": mean_absolute_error(y, np.full(len(y), mean)),
            "mse": mean_squared_error(y, np.full(len(y), mean)),
        }
        out[f"linear_{split}"] = {
            "mae": mean_absolute_error(y, linear.predict(X)),
            "mse": mean_squared_error(y, linear.predict(X)),
        }
    return out


def plot_curves(history, best_epoch, title="", ax=None, path=None):
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(history.epoch, history.train_loss, label="train")
    ax.plot(history.epoch, history.val_loss, label="val")
    ax.axvline(best_epoch, ls="--", color="gray", label=f"best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE loss", title=title)
    ax.grid()
    ax.legend()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.close()


def plot_rentals_vs_hour(path):
    frame = make_rentals(n=6000, seed=SEED)
    _, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    frame.groupby("hour")[TARGET].mean().plot(marker="o", ax=axes[0])
    axes[0].set(xlabel="hour", ylabel="mean rentals", title="Rentals vs hour")
    axes[0].grid()
    frame.groupby(["hour", "station_type"])[TARGET].mean().unstack().plot(ax=axes[1])
    axes[1].set(xlabel="hour", ylabel="mean rentals", title="By station type")
    axes[1].grid()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    plt.switch_backend("Agg")
    REPORTS.mkdir(exist_ok=True)
    data = prepare()
    plot_rentals_vs_hour(REPORTS / "rentals_vs_hour.png")
    base = baselines(data)

    results = []
    for name, config in [("main", {})] + OTHER_RUNS:
        model, history, row = run_experiment(
            data, name, config, verbose=10 if name == "main" else 0
        )
        results.append((name, model, history, row))
        print(
            f"{name:14s} best epoch {row['best_epoch']:3d}  "
            f"val MAE {row['best_val_mae']:.3f}"
        )

    _, main_model, main_history, main_row = results[0]
    main_history.to_csv(REPORTS / "history.csv", index=False)
    plot_curves(
        main_history,
        main_row["best_epoch"],
        "main run",
        path=REPORTS / "learning_curves.png",
    )
    experiments = pd.DataFrame([row for *_, row in results])
    experiments.to_csv(REPORTS / "experiments.csv", index=False)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6))
    for ax, (name, _, history, row) in zip(axes, results, strict=True):
        plot_curves(history, row["best_epoch"], name, ax=ax)
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(REPORTS / "experiment_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    _, _, test_ds = data.datasets()
    test_loss, test_mae = run_epoch(main_model, make_loader(test_ds, 256), nn.MSELoss())
    metrics = {
        "baselines": base,
        "network": {
            "val_mae": main_row["best_val_mae"],
            "val_mse": main_history.val_loss.iloc[main_row["best_epoch"] - 1],
            "test_mae": test_mae,
            "test_mse": test_loss,
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
