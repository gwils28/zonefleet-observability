"""Tests anti-fuite de la matrice de features."""

import numpy as np
import pandas as pd
import pytest

import zonefleet.features as F


def toy_panel(n_hours=800, zones=(1, 2)):
    ts = pd.date_range("2019-01-01", periods=n_hours, freq="h")
    rows = [
        pd.DataFrame({"zone": z, "ts": ts, "pickups": np.arange(n_hours) + 1000 * z}) for z in zones
    ]
    return pd.concat(rows, ignore_index=True)


def test_lags_are_shifted_by_at_least_horizon():
    df = toy_panel()
    df["y"] = df["pickups"].astype(float)
    out = F.add_lag_features(df).set_index(["zone", "ts"])
    t = pd.Timestamp("2019-01-20 05:00")
    for k in F.LAGS:
        assert out.loc[(1, t), f"lag_{k}"] == out.loc[(1, t - pd.Timedelta(hours=k)), "y"]


def test_rolling_mean_ends_at_t_minus_24():
    df = toy_panel()
    df["y"] = df["pickups"].astype(float)
    out = F.add_lag_features(df).set_index(["zone", "ts"])
    t = pd.Timestamp("2019-01-20 05:00")
    window = [out.loc[(1, t - pd.Timedelta(hours=24 + i)), "y"] for i in range(24)]
    assert out.loc[(1, t), "roll_mean_24"] == pytest.approx(np.mean(window))


def test_no_lag_crosses_zones():
    df = toy_panel()
    df["y"] = df["pickups"].astype(float)
    out = F.add_lag_features(df)
    first_zone2 = out[out["zone"] == 2].iloc[: min(F.LAGS)]
    assert first_zone2["lag_24"].isna().all()


def test_no_lag_crosses_the_2020_2024_gap():
    a = toy_panel(400, zones=(1,))
    b = a.assign(ts=a["ts"] + pd.DateOffset(years=5))
    df = pd.concat([a, b], ignore_index=True)
    df["y"] = df["pickups"].astype(float)
    out = F.add_lag_features(df)
    start_b = out[out["ts"].dt.year == 2024].sort_values("ts").iloc[:24]
    assert start_b["lag_24"].isna().all()


def test_short_lag_is_rejected(monkeypatch):
    monkeypatch.setattr(F, "LAGS", (1, 24))
    df = toy_panel(100)
    df["y"] = 1.0
    with pytest.raises(ValueError, match="fuite"):
        F.add_lag_features(df)


def test_feature_list_excludes_seasonal_extrapolation():
    assert "doy_sin" not in F.FEATURES and "doy_cos" not in F.FEATURES
    assert all(c in F.FEATURES for c in F.LAG_COLS)
