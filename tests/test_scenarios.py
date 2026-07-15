"""Tests des injections : chaque dérive est pure par construction."""

import numpy as np
import pandas as pd
import pytest

from zonefleet import scenarios as S


def analysis_frame(n_days=10, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.repeat(pd.date_range("2025-01-01", periods=n_days, freq="D"), 144)
    ts = pd.to_datetime(ts) + pd.to_timedelta(np.tile(np.arange(144) % 24, n_days), unit="h")
    df = pd.DataFrame(
        {"ts": ts, "lag_168": rng.normal(size=len(ts)), "cbd": rng.integers(0, 2, len(ts))}
    )
    df["pickups"] = np.exp(2 + df["lag_168"])
    df["q50"] = df["pickups"] * 1.1
    return df


ONSET = pd.Timestamp("2025-01-06")


def test_covariate_shift_keeps_rows_real_and_chunk_size():
    a = analysis_frame()
    out = S.covariate_shift(a, ONSET, kappa=2.0, seed=1)
    assert len(out) == len(a)
    assert out.groupby(out["ts"].dt.floor("D")).size().eq(144).all()
    # chaque ligne rééchantillonnée existe telle quelle dans l'original : P(y|X) intact
    merged = out.merge(a, on=["ts", "lag_168", "pickups", "q50", "cbd"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_covariate_shift_moves_distribution_in_requested_direction():
    a = analysis_frame()
    low = S.covariate_shift(a, ONSET, kappa=1.5, direction=-1, seed=1)
    high = S.covariate_shift(a, ONSET, kappa=1.5, direction=+1, seed=1)
    after = lambda d: d[d["ts"] >= ONSET]["lag_168"].mean()  # noqa: E731
    assert after(low) < after(a) - 0.5 < after(a) + 0.5 < after(high)
    pd.testing.assert_frame_equal(
        low[low["ts"] < ONSET].reset_index(drop=True), a[a["ts"] < ONSET].reset_index(drop=True)
    )


def test_concept_drift_freezes_inputs_and_predictions():
    a = analysis_frame()
    out = S.concept_drift(a, ONSET, factor=0.5)
    pd.testing.assert_series_equal(out["lag_168"], a["lag_168"])
    pd.testing.assert_series_equal(out["q50"], a["q50"])
    post = out["ts"] >= ONSET
    np.testing.assert_allclose(out.loc[post, "pickups"], 0.5 * a.loc[post, "pickups"])
    np.testing.assert_allclose(out.loc[~post, "pickups"], a.loc[~post, "pickups"])


def test_concept_drift_on_subset_only():
    a = analysis_frame()
    out = S.concept_drift(a, ONSET, factor=0.5, subset="cbd")
    untouched = (a["cbd"] == 0) | (a["ts"] < ONSET)
    np.testing.assert_allclose(out.loc[untouched, "pickups"], a.loc[untouched, "pickups"])


def test_subset_masks():
    a = analysis_frame(1)
    assert S.subset_mask(a, "night").sum() == 6 * 6
    with pytest.raises(ValueError):
        S.subset_mask(a, "weekend")


def test_inject_none_is_identity():
    a = analysis_frame()
    sc = S.Scenario("none", 0.0, ONSET)
    assert S.inject(sc, a) is a
    assert S.Scenario("concept", 0.7, ONSET).label == "concept_0.7"


def test_level_shock_propagates_through_lags_with_their_delay():
    from zonefleet.features import build_features
    from zonefleet.model import QuantileForecaster

    ts = pd.date_range("2025-01-01", "2025-03-01", freq="h", inclusive="left")
    panel = pd.concat(
        [pd.DataFrame({"zone": z, "ts": ts, "pickups": 100.0 + 10 * (ts.hour % 6)}) for z in (1, 2)]
    )
    weather = pd.DataFrame({"ts": ts, "temperature_2m": 5.0, "precipitation": 0.0})
    zones = pd.DataFrame({"zone": [1, 2], "cbd": [True, False]})
    model = QuantileForecaster(params={"n_estimators": 5, "verbose": -1})
    model.fit(build_features(panel, weather, zones))
    onset = pd.Timestamp("2025-02-10")
    out = S.level_shock(
        panel,
        weather,
        model,
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-03-01"),
        onset,
        factor=0.5,
        zones=zones,
    ).set_index(["zone", "ts"])
    base = np.log1p(100.0)
    t1 = onset + pd.Timedelta(hours=30)  # lag_24 a vu le choc, lag_168 non
    assert out.loc[(1, t1.floor("6h")), "lag_24"] == pytest.approx(np.log1p(50.0))
    assert out.loc[(1, t1.floor("6h")), "lag_168"] == pytest.approx(base)
    assert out.loc[(1, t1.floor("6h")), "pickups"] == pytest.approx(50.0)
