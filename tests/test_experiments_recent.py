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
