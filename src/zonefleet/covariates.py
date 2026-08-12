"""Covariables exogènes : météo horaire (Open-Meteo) et calendrier US.

Hypothèse assumée (SPEC, section 2) : la météo réalisée tient lieu de prévision météo à
24 h. C'est optimiste ; l'erreur d'une vraie prévision météo à J+1 dégraderait un peu
le modèle, sans changer la logique du monitoring.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import requests

from zonefleet.config import DATA_PROCESSED, TZ

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MANHATTAN = (40.758, -73.985)  # Times Square, centre de gravité de la demande
WEATHER_VARS = ("temperature_2m", "precipitation")


def fetch_weather(start: str, end: str, lat_lon: tuple[float, float] = MANHATTAN) -> pd.DataFrame:
    """Météo horaire ERA5 en UTC entre deux dates incluses (AAAA-MM-JJ)."""
    params: dict[str, str | float] = {
        "latitude": lat_lon[0],
        "longitude": lat_lon[1],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "GMT",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"ts_utc": pd.to_datetime(h["time"], utc=True)})
    for v in WEATHER_VARS:
        df[v] = h[v]
    return df


def to_local_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Passe de l'UTC à l'heure locale naïve, sur la même convention que le panel TLC.

    Automne : deux heures UTC tombent sur la même heure locale, on en fait la moyenne.
    Printemps : l'heure locale manquante est interpolée.
    """
    local = df["ts_utc"].dt.tz_convert(TZ).dt.tz_localize(None)
    out = df.drop(columns="ts_utc").assign(ts=local).groupby("ts").mean()
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    return out.reindex(full).interpolate(limit=2).rename_axis("ts").reset_index()


def load_weather(path: Path | None = None) -> pd.DataFrame:
    """Météo locale mise en cache (`data/processed/weather.csv`)."""
    return pd.read_csv(path or DATA_PROCESSED / "weather.csv", parse_dates=["ts"])


def us_holidays(years: list[int]) -> set[pd.Timestamp]:
    """Jours fériés fédéraux US (et observés) pour les années données."""
    return {pd.Timestamp(d) for d in holidays.US(years=years)}


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    """Features calendaires d'une série d'horodatages horaires locaux."""
    ts = pd.to_datetime(ts)
    days = ts.dt.normalize()
    hol = us_holidays(sorted(ts.dt.year.unique().tolist()))
    is_hol = days.isin(hol)
    return pd.DataFrame(
        {
            "hour": ts.dt.hour.astype(np.int16),
            "dow": ts.dt.dayofweek.astype(np.int16),
            "is_weekend": (ts.dt.dayofweek >= 5).astype(np.int8),
            "is_holiday": is_hol.astype(np.int8),
            "doy_sin": np.sin(2 * np.pi * ts.dt.dayofyear / 365.25),
            "doy_cos": np.cos(2 * np.pi * ts.dt.dayofyear / 365.25),
        },
        index=ts.index,
    )


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_weather_forecast(
    start: str, end: str, lat_lon: tuple[float, float] = MANHATTAN
) -> pd.DataFrame:
    """Prévision émise la veille (variables `*_previous_day1`), en UTC, colonnes renommées
    comme la météo réalisée pour être interchangeables dans les features."""
    params: dict[str, str | float] = {
        "latitude": lat_lon[0],
        "longitude": lat_lon[1],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(f"{v}_previous_day1" for v in WEATHER_VARS),
        "timezone": "GMT",
    }
    r = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"ts_utc": pd.to_datetime(h["time"], utc=True)})
    for v in WEATHER_VARS:
        df[v] = h[f"{v}_previous_day1"]
    return df


def load_weather_forecast(path: Path | None = None) -> pd.DataFrame:
    """Prévision de la veille mise en cache (`data/processed/weather_forecast.csv`)."""
    return pd.read_csv(path or DATA_PROCESSED / "weather_forecast.csv", parse_dates=["ts"])
