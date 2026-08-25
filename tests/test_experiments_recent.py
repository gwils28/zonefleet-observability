"""Pilotes récents (E0 bis, E7, E8, E9) et générateur de tables, sur données jouets."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import toy_panel


def test_e0_conformal_periods_split_reference(toy_world):
    from zonefleet.experiments import e0_conformal as E

    parts = E.periods("covid")
    assert list(parts) == [
        "calibration",
        "test (fin de référence)",
        "analysis, 3 premiers mois",
        "analysis, 3 mois suivants",
    ]
    assert parts["calibration"]["ts"].max() < parts["test (fin de référence)"]["ts"].min()


def test_e7_summarize_counts_alerts_before_and_after():
    from zonefleet.experiments.e7_seasonal import summarize

    dates = pd.date_range("2020-03-01", periods=20, freq="D")
    sig = pd.DataFrame(
        {
            "date": dates,
            "signal": "z-score saisonnier",
            "feature": "lag_24",
            "alert": [False] * 12 + [True] * 8,
        }
    )
    s = summarize(sig, pd.Timestamp("2020-03-10")).iloc[0]
    # alertes brutes à partir du 13 mars, persistance de 2 jours : alerte le 14, soit 4 jours
    assert s["share_alert_before"] == 0 and s["delay_days"] == 4


def test_e8_analysis_from_bugged_panel(toy_world, monkeypatch):
    from zonefleet import quality as Q
    from zonefleet.experiments import e8_quality as E

    zones = pd.DataFrame({"zone": [1, 2], "cbd": [True, False]})
    monkeypatch.setattr(E, "zone_table", lambda: zones)
    monkeypatch.setattr(E, "load_weather", toy_world.load_weather)
    monkeypatch.setitem(E.SPLITS, "congestion", E.SPLITS["covid"])
    monkeypatch.setattr(E, "frozen_model", lambda split: toy_world.frozen_model("covid"))
    p = toy_panel()
    bugged = Q.inject_bug(p, "duplication", pd.Timestamp("2020-03-02"))
    an = E.analysis_from_panel(bugged)
    assert {"q50", "snaive", "pickups"} <= set(an.columns)
    day = an[an["ts"].dt.floor("D") == pd.Timestamp("2020-03-03")]
    ref = an[an["ts"].dt.floor("D") == pd.Timestamp("2020-02-25")]
    assert day["pickups"].mean() > 1.5 * ref["pickups"].mean()


def test_e9_forecast_error_columns(monkeypatch):
    from zonefleet.experiments import e9_weather as E

    ts = pd.date_range("2024-02-01", periods=100, freq="h")
    w = pd.DataFrame({"ts": ts, "temperature_2m": np.linspace(0, 5, 100), "precipitation": 0.0})
    f = w.assign(temperature_2m=w["temperature_2m"] - 1, precipitation=0.0)
    monkeypatch.setattr(E, "load_weather", lambda: w)
    monkeypatch.setattr(E, "load_weather_forecast", lambda: f)
    e = E.forecast_error().iloc[0]
    assert e["mae_temperature"] == pytest.approx(1.0) and e["biais_temperature"] == pytest.approx(
        -1.0
    )


def test_make_tables_french_number_format():
    from scripts.make_tables import fr

    assert fr(1234.5, 1) == "1 234,5"
    assert fr(float("nan")) == "-"
    assert fr("texte") == "texte"


def test_e10_group_signals_and_joint_condition():
    from zonefleet.experiments import e10_zones as E

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2025-01-01", periods=30, freq="D"), 20)
    rows = pd.DataFrame(
        {
            "date": dates,
            "est": 0.3,
            "recon": 1.0,
            "real": 0.3,
            "lag_24": rng.normal(size=600),
            "lag_168": rng.normal(size=600),
            "cbd": np.tile([0, 1], 300),
        }
    )
    daily = E.group_daily(rows, rows)
    assert set(daily) == {"ville", "cbd", "hors_cbd"}
    thr = E.thresholds(daily["ville"])
    d = daily["cbd"].copy()
    d.loc[d.index[-5:], "est"] = 0.9
    d.loc[d.index[-5:], "psi_lag_24"] = 0.5
    j = E.joint(d, {**thr, "est": 0.5})
    assert j.iloc[-1] and not j.iloc[:-5].any()


def test_e4_seasonal_z_is_large_for_a_level_jump():
    from zonefleet.experiments.e4_seasonal import seasonal_z

    base_idx = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    base = pd.DataFrame(
        {
            "lag_24": np.random.default_rng(1).normal(5, 0.1, len(base_idx)),
            "lag_168": 5.0 + np.random.default_rng(2).normal(0, 0.1, len(base_idx)),
        },
        index=base_idx,
    )
    days = pd.date_range("2025-02-01", periods=10, freq="D")
    daily = pd.DataFrame({"mean_lag_24": 5.0, "mean_lag_168": 3.0}, index=days)
    z = seasonal_z(daily, base)
    assert (z["z_lag_168"] > 3).all() and (z["z_lag_24"] < 3).all()


def test_e1_chunks_run_on_toy_world(toy_world):
    from zonefleet.experiments.e1_chunks import run_chunk, summarize

    p = toy_world.predictions("covid")
    ref, an = p[p.period == "reference"], p[p.period == "analysis"]
    d = run_chunk(ref, an, "semaine", "W")
    assert set(d["chunk"]) == {"semaine"} and (d["ci_rel_width"] > 0).all()
    assert summarize(d)["n_chunks"] == len(d)


def test_e3_seeds_stability_and_separation():
    from zonefleet.experiments.e3_seeds import families_separated, stability

    per_seed = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "scenario": ["cov", "con", "cov", "con"],
            "family": ["covariate nuisible", "concept pur"] * 2,
            "severity": [1.0, 0.7] * 2,
            "tracking_steady": [0.9, 0.0, 0.1, 0.2],
        }
    )
    st = stability(per_seed).set_index("scenario")
    assert st.loc["cov", "min"] == 0.1 and st.loc["cov", "max"] == 0.9
    sep = families_separated(per_seed)
    assert sep[1] and not sep[2]
