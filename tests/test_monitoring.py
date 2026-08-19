"""Tests des briques de monitoring sur des données synthétiques contrôlées."""

import numpy as np
import pandas as pd
import pytest

from zonefleet import monitoring as M


def synthetic(n_days=40, shift=None, seed=0, start="2021-01-01"):
    """Panel jouet au format des prédictions : features, q50, pickups."""
    rng = np.random.default_rng(seed)
    n = n_days * 48
    ts = pd.date_range(start, periods=n, freq="30min")
    df = pd.DataFrame({"ts": ts, "zone": pd.Categorical(rng.integers(1, 3, n))})
    for c in M.DRIFT_FEATURES:
        df[c] = rng.normal(size=n)
    if shift is not None:
        df = shift(df)
    level = np.exp(2 + 0.5 * df["lag_24"])
    noise = rng.normal(scale=0.2 + 0.4 * np.abs(df["lag_24"]), size=n)
    df["q50"] = level
    df["pickups"] = np.clip(level * np.exp(noise), 0, None)
    return df


def test_psi_is_zero_on_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert M.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert M.psi(a, a + 0.1) < M.psi(a, a + 1.0)
    assert M.psi(a, a + 1.0) > 0.25


def test_flatten_multiindex():
    cols = pd.MultiIndex.from_tuples([("chunk", "key"), ("mae", "value")])
    assert list(M.flatten(pd.DataFrame([[1, 2]], columns=cols)).columns) == ["key", "mae__value"]


def test_realized_performance_matches_manual():
    df = synthetic(3)
    out = M.realized_performance(df, "count").query("metric == 'mae_count'")
    day0 = df[df["ts"].dt.date == df["ts"].dt.date.iloc[0]]
    assert out["realized"].iloc[0] == pytest.approx((day0.pickups - day0.q50).abs().mean())


def test_prepare_rejects_unknown_scale():
    with pytest.raises(ValueError):
        M.prepare(synthetic(1), "percent")


def test_dle_never_sees_analysis_labels():
    ref = synthetic(30)
    an = synthetic(10, seed=1, start="2021-03-01").assign(pickups=np.nan)
    est = M.estimate_performance(ref, an, "log", metrics=("mae",))
    assert est["estimated"].notna().all() and len(est) == 10


def test_dle_tracks_heteroscedastic_covariate_shift():
    ref = synthetic(30)
    widen = lambda d: d.assign(lag_24=d["lag_24"] * 2.5)  # noqa: E731
    an = synthetic(10, seed=1, shift=widen, start="2021-03-01")
    est = M.estimate_performance(ref, an, "log", metrics=("mae",))
    real = M.realized_performance(an, "log").query("metric == 'mae_log'")
    base = M.realized_performance(ref, "log").query("metric == 'mae_log'")["realized"].mean()
    assert est["estimated"].mean() > 1.3 * base
    assert est["estimated"].mean() == pytest.approx(real["realized"].mean(), rel=0.3)


def test_univariate_drift_flags_shifted_column_only():
    ref = synthetic(30)
    an = synthetic(10, seed=1, shift=lambda d: d.assign(lag_168=d["lag_168"] + 1.5))
    u = M.univariate_drift(ref, an, columns=["lag_168", "lag_336"], methods=("kolmogorov_smirnov",))
    rate = u.groupby("feature")["alert"].mean()
    assert rate["lag_168"] == 1.0 and rate["lag_336"] < 0.3


def test_pca_reconstruction_sees_broken_correlation_but_not_shift_along_it():
    """Limite documentée : un déplacement le long d'une direction principale reste invisible."""

    def correlated(d, rho_break=False, offset=0.0):
        z = d["lag_24"].to_numpy()
        for i, c in enumerate(M.LAG_COLS + M.ROLL_COLS):
            rng = np.random.default_rng(100 + i)
            base = rng.normal(size=len(d)) if rho_break else z
            d[c] = base + rng.normal(scale=0.1, size=len(d)) + offset
        return d

    cols = M.LAG_COLS + M.ROLL_COLS
    ref = correlated(synthetic(30))
    along = correlated(synthetic(10, seed=1), offset=0.5)
    broken = correlated(synthetic(10, seed=1), rho_break=True)
    mv_along = M.multivariate_drift(ref, along, columns=cols)
    mv_broken = M.multivariate_drift(ref, broken, columns=cols)
    assert mv_broken["alert"].mean() == 1.0
    assert mv_along["alert"].mean() == 0.0


def test_seasonal_baselines_on_a_shift_that_repeats_every_year():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2019-01-01", "2021-01-01", freq="h", inclusive="left")
    seasonal = 10 * np.sin(2 * np.pi * ts.dayofyear / 365)
    df = pd.DataFrame({"ts": ts, "temperature_2m": seasonal + rng.normal(size=len(ts))})
    hist, an = df[df["ts"].dt.year == 2019], df[df["ts"].dt.year == 2020]
    autumn = hist[hist["ts"].dt.month >= 9]
    winter = an[an["ts"].dt.month <= 2]
    z = M.seasonal_zscore(hist, winter, ["temperature_2m"])
    fixed = M.daily_psi(autumn, winter, ["temperature_2m"])
    assert z["z"].abs().median() < 3 and fixed["psi"].median() > 0.25
    # Limite documentée : le PSI journalier contre une fenêtre de 28 jours reste élevé,
    # car une journée a une distribution bien plus étroite que la fenêtre.
    s = M.seasonal_psi(hist, winter, ["temperature_2m"])
    assert s["psi"].median() > 0.25


def test_seasonal_psi_skips_days_without_history():
    ts = pd.date_range("2020-01-01", periods=48, freq="h")
    df = pd.DataFrame({"ts": ts, "x": 1.0})
    assert M.seasonal_psi(df, df, ["x"]).empty


def test_psi_by_day_matches_reference_implementation():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=5000)
    days = np.repeat(pd.date_range("2025-01-01", periods=20, freq="D"), 300)
    vals = rng.normal(loc=np.repeat(np.linspace(0, 1, 20), 300), size=len(days))
    fast = M.psi_by_day(ref, vals, days)
    slow = pd.Series(vals).groupby(days).apply(lambda s: M.psi(ref, s.to_numpy()))
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), rtol=1e-12, atol=1e-12)


def test_psi_by_day_with_ties_on_bin_edges():
    """Les lags sont des log1p d'entiers : beaucoup de valeurs tombent sur les bornes."""
    rng = np.random.default_rng(4)
    ref = np.log1p(rng.poisson(5, 4000))
    days = np.repeat(pd.date_range("2025-01-01", periods=10, freq="D"), 200)
    vals = np.log1p(rng.poisson(np.repeat(np.linspace(4, 8, 10), 200)))
    fast = M.psi_by_day(ref, vals, days)
    slow = pd.Series(vals).groupby(days).apply(lambda s: M.psi(ref, s.to_numpy()))
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), rtol=1e-12, atol=1e-12)
