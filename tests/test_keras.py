import keras
import numpy as np
import pytest
import torch

from compare import compare
from keras_model import (
    build_keras_model,
    copy_torch_to_keras,
    count_torch_params,
    keras_predict,
    max_prediction_gap,
    torch_predict,
)
from model import RentalModel
from train_keras import fit_keras, keras_arrays

TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def arrays(data):
    return keras_arrays(data)


@pytest.fixture(scope="module")
def small(arrays):
    x_train, y_train, x_val, y_val, *_ = arrays
    return x_train[:256], y_train[:256], x_val[:128], y_val[:128]


def test_keras_returns_one_value_per_row(data):
    net = build_keras_model(data.n_features)
    out = net.predict(data.X_val, verbose=0)
    assert out.shape == (len(data.X_val), 1)
    assert keras_predict(net, data.X_val).shape == (len(data.X_val),)


@pytest.mark.parametrize(("hidden", "n_hidden"), [(4, 1), (16, 2), (64, 2), (32, 3)])
def test_parameter_count_matches_pytorch(data, hidden, n_hidden):
    torch_net = RentalModel(data.n_features, hidden, n_hidden)
    keras_net = build_keras_model(data.n_features, hidden, n_hidden)
    assert keras_net.count_params() == count_torch_params(torch_net)


def test_same_weights_give_same_predictions(data):
    # random, non-symmetric weights and non-zero biases: a copy that forgot to
    # transpose, or skipped the biases, would not reproduce these predictions
    torch.manual_seed(0)
    torch_net = RentalModel(data.n_features, 16, 2)
    keras_net = build_keras_model(data.n_features, 16, 2)
    copy_torch_to_keras(torch_net, keras_net)
    assert max_prediction_gap(torch_net, keras_net, data.X_val) < TOLERANCE


def test_copy_without_transpose_is_caught():
    # square layers, so an untransposed copy would load without a shape error
    torch.manual_seed(0)
    torch_net = RentalModel(8, 8, 2)
    keras_net = build_keras_model(8, 8, 2)
    linears = [m for m in torch_net.modules() if isinstance(m, torch.nn.Linear)]
    for linear, dense in zip(linears[:-1], keras_net.layers[:-1], strict=True):
        dense.set_weights(
            [linear.weight.detach().numpy(), linear.bias.detach().numpy()]
        )
    keras_net.layers[-1].set_weights(
        [linears[-1].weight.detach().numpy().T, linears[-1].bias.detach().numpy()]
    )
    x = np.random.default_rng(0).normal(size=(64, 8)).astype("float32")
    assert max_prediction_gap(torch_net, keras_net, x) > 1e-3
    copy_torch_to_keras(torch_net, keras_net)
    assert max_prediction_gap(torch_net, keras_net, x) < TOLERANCE


def test_both_frameworks_get_the_same_rows(data, datasets, arrays):
    x_train, y_train, x_val, y_val, x_test, y_test = arrays
    for ds, x, y in zip(
        datasets,
        [x_train, x_val, x_test],
        [y_train, y_val, y_test],
        strict=True,
    ):
        np.testing.assert_array_equal(ds.features.numpy(), x)
        np.testing.assert_array_equal(ds.targets.numpy(), y)


def test_keras_memorizes_one_batch(arrays):
    x, y = arrays[0][:32], arrays[1][:32]
    keras.utils.set_random_seed(0)
    net = build_keras_model(x.shape[1])
    net.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    start = net.evaluate(x, y, verbose=0)
    net.fit(x, y, batch_size=32, epochs=400, verbose=0)
    end = net.evaluate(x, y, verbose=0)
    assert end < 0.5
    assert end < start / 100


def test_saved_and_reloaded_model_predicts_the_same(data, tmp_path):
    keras.utils.set_random_seed(0)
    net = build_keras_model(data.n_features)
    path = tmp_path / "model.keras"
    net.save(path)
    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        keras_predict(reloaded, data.X_val), keras_predict(net, data.X_val)
    )


def test_best_weights_are_restored(small):
    x_train, y_train, x_val, y_val = small
    # a large learning rate on few rows makes the validation loss bounce,
    # so the best epoch is not the last one
    run = fit_keras(x_train, y_train, x_val, y_val, lr=0.05, epochs=40, seed=1)
    assert run.best_epoch < 40
    restored = run.model.evaluate(x_val, y_val, batch_size=64, verbose=0)[0]
    assert restored == pytest.approx(run.history.val_loss.min(), rel=1e-4)
    assert restored != pytest.approx(run.history.val_loss.iloc[-1], rel=1e-4)


def test_same_seed_repeats_a_short_run(small):
    first = fit_keras(*small, epochs=3, seed=5)
    second = fit_keras(*small, epochs=3, seed=5)
    np.testing.assert_allclose(first.history.val_loss, second.history.val_loss)
    x_val = small[2]
    np.testing.assert_allclose(
        keras_predict(first.model, x_val), keras_predict(second.model, x_val)
    )


def test_comparison_table_has_required_columns(data, tmp_path):
    table = compare(data, seeds=(1,), epochs=2, checkpoint_dir=tmp_path)
    assert set(table.framework) == {"pytorch", "keras"}
    assert list(table.columns) == [
        "framework",
        "seed",
        "best_epoch",
        "val_mae",
        "test_mae",
        "seconds",
    ]
    assert table.notna().all().all()


def test_torch_predict_flattens(data):
    net = RentalModel(data.n_features)
    assert torch_predict(net, data.X_val).shape == (len(data.X_val),)
