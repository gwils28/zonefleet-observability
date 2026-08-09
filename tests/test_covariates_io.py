"""Météo Open-Meteo : requête simulée, aucun appel réseau."""

import pandas as pd

from zonefleet import covariates as C


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_weather_parses_hourly_payload(monkeypatch):
    payload = {
        "hourly": {
            "time": ["2019-01-01T00:00", "2019-01-01T01:00"],
            "temperature_2m": [1.5, 1.0],
            "precipitation": [0.0, 0.2],
        }
    }
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return FakeResponse(payload)

    monkeypatch.setattr(C.requests, "get", fake_get)
    df = C.fetch_weather("2019-01-01", "2019-01-01")
    assert list(df.columns) == ["ts_utc", "temperature_2m", "precipitation"]
    assert df["ts_utc"].dt.tz is not None
    assert seen["timezone"] == "GMT" and "temperature_2m" in seen["hourly"]


def test_load_weather_roundtrip(tmp_path):
    p = tmp_path / "w.csv"
    pd.DataFrame(
        {"ts": ["2019-01-01 00:00"], "temperature_2m": [1.0], "precipitation": [0.0]}
    ).to_csv(p, index=False)
    w = C.load_weather(p)
    assert pd.api.types.is_datetime64_any_dtype(w["ts"])


def test_fetch_weather_forecast_renames_previous_day_columns(monkeypatch):
    payload = {
        "hourly": {
            "time": ["2024-02-01T00:00"],
            "temperature_2m_previous_day1": [2.0],
            "precipitation_previous_day1": [0.5],
        }
    }
    monkeypatch.setattr(C.requests, "get", lambda url, params, timeout: FakeResponse(payload))
    df = C.fetch_weather_forecast("2024-02-01", "2024-02-01")
    assert list(df.columns) == ["ts_utc", "temperature_2m", "precipitation"]
    assert df["temperature_2m"].iloc[0] == 2.0
