"""Tests du forecaster quantile sur données synthétiques."""

import numpy as np
import pandas as pd

from zonefleet.features import FEATURES
from zonefleet.model import QuantileForecaster, seasonal_naive


def toy_features(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    df["zone"] = pd.Categorical(rng.integers(1, 4, size=n))
    df["lag_168"] = rng.uniform(0, 4, size=n)
    df["y"] = df["lag_168"] + rng.normal(scale=0.3, size=n)
    df["ts"] = pd.date_range("2019-01-01", periods=n, freq="h")
    return df


def test_quantiles_are_ordered_and_non_negative():
    df = toy_features()
    m = QuantileForecaster(params={"n_estimators": 30, "verbose": -1}).fit(df)
    p = m.predict(df)
    assert list(p.columns) == ["q10", "q50", "q90"]
    assert (p["q10"] <= p["q50"]).all() and (p["q50"] <= p["q90"]).all()
    assert (p >= 0).all().all()


def test_lag_168_dominates_importance():
    m = QuantileForecaster(params={"n_estimators": 30, "verbose": -1}).fit(toy_features())
    assert m.feature_importance().index[0] == "lag_168"


def test_save_and_load_roundtrip(tmp_path):
    df = toy_features(500)
    m = QuantileForecaster(params={"n_estimators": 10, "verbose": -1}).fit(df)
    m.save(tmp_path / "m.pkl")
    m2 = QuantileForecaster.load(tmp_path / "m.pkl")
    np.testing.assert_allclose(m.predict(df), m2.predict(df))
    assert m2.trained_on == m.trained_on


def test_seasonal_naive_inverts_log():
    df = pd.DataFrame({"lag_168": np.log1p([0.0, 9.0])})
    np.testing.assert_allclose(seasonal_naive(df), [0.0, 9.0])
