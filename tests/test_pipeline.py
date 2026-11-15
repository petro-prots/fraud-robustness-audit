import keras
import numpy as np
import pytest

from fraudaudit.data import period
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.model.artifacts import load_artifacts
from fraudaudit.model.pipeline import FraudModel, build_datasets, build_mlp, train_model


def test_architecture_matches_listing():
    mlp = build_mlp(len(FEATURE_COLUMNS) + 1)
    assert mlp.input_shape == (None, 35)
    layers = mlp.layers
    kinds = [type(l).__name__ for l in layers]
    assert kinds == ["Dense", "BatchNormalization", "Dense", "BatchNormalization",
                     "Dense", "BatchNormalization", "Dense"]
    dense = [l for l in layers if isinstance(l, keras.layers.Dense)]
    assert [l.units for l in dense] == [55, 26, 12, 1]
    assert [l.activation.__name__ for l in dense] == ["relu", "sigmoid", "sigmoid", "sigmoid"]
    assert sum(isinstance(l, keras.layers.BatchNormalization) for l in layers) == 3
    assert isinstance(mlp.optimizer, keras.optimizers.Adamax)
    assert float(mlp.optimizer.learning_rate.numpy()) == pytest.approx(0.01)
    assert isinstance(mlp.loss, keras.losses.BinaryCrossentropy)


def test_balancing_only_fit(sparkov_like):
    df = sparkov_like
    ds = build_datasets(df, seed=0)
    per = period(df)
    fit_rows = set(df.loc[per == "fit", "row"])
    # training set: only fit rows, fraud upsampled to the number of legit fit rows
    assert set(ds.train_rows) <= fit_rows
    assert int((ds.y_train == 0).sum()) == int((ds.y_train == 1).sum())
    assert int((ds.y_train == 0).sum()) == int(((per == "fit") & (df["is_fraud"] == 0)).sum())
    # val / test: unbalanced, every row exactly once
    for name in ("val", "test"):
        rows = ds.rows[name]
        assert len(rows) == len(set(rows))
        assert set(rows) == set(df.loc[per == name, "row"])
        assert ds.F[name].shape == (len(rows), len(FEATURE_COLUMNS))
        assert np.array_equal(ds.y[name], df.set_index("row").loc[rows, "is_fraud"].to_numpy())


def test_train_smoke(sparkov_like, tmp_path):
    model, report = train_model(sparkov_like, seed=0, epochs=1)
    assert isinstance(model, FraudModel)
    for name in ("val", "test"):
        assert {"recall", "fpr", "precision", "n", "n_fraud"} <= set(report[name])
    assert 0.0 <= model.tau <= 1.0
    model.save(tmp_path, meta={"seed": 0})
    loaded = FraudModel.from_artifacts(load_artifacts(tmp_path))
    ds = build_datasets(sparkov_like, seed=0)
    F = ds.F["test"]
    p = loaded.predict_proba(F)
    assert p.shape == (len(F),)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p, model.predict_proba(F), atol=1e-6)
    one = loaded.predict_proba(F[:1])
    assert one.shape == (1,) and one[0] == pytest.approx(p[0], abs=1e-6)
    assert loaded.tau == pytest.approx(model.tau)
