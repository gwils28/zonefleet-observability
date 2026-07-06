"""Matrice de features du forecaster, sans fuite temporelle.

Règle des trois horloges (SPEC, section 3) : la prévision de la cible t est émise à
τ = t - 24 h. Toute feature doit donc être connue à τ : aucun lag de la demande
inférieur à 24 h, et les moyennes glissantes se terminent à t - 24 h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from zonefleet.config import CONFIGS, HORIZON_H
from zonefleet.covariates import calendar_features

LAGS = (24, 48, 72, 168, 336)
ROLL_WINDOWS = (24, 168)
# doy_sin/doy_cos (saison) sont calculés mais exclus : entraîné de janvier à août, le modèle
# les extrapolerait sur la référence (septembre-décembre). Voir docs/notes/implementation.md.
CALENDAR = ["hour", "dow", "is_weekend", "is_holiday"]
WEATHER = ["temperature_2m", "precipitation"]
LAG_COLS = [f"lag_{k}" for k in LAGS]
ROLL_COLS = [f"roll_mean_{w}" for w in ROLL_WINDOWS]
FEATURES = ["zone", "cbd", *LAG_COLS, *ROLL_COLS, *CALENDAR, *WEATHER]
CATEGORICAL = ["zone"]
TARGET = "y"  # log1p(pickups)


def zone_table() -> pd.DataFrame:
    """Métadonnées des 60 zones (`configs/zones.yaml`)."""
    zones = yaml.safe_load((CONFIGS / "zones.yaml").read_text())["zones"]
    return pd.DataFrame(zones)


def contiguous_block(ts: pd.Series) -> pd.Series:
    """Identifiant de bloc temporel contigu (le panel saute de 2020 à 2024)."""
    return (ts.dt.year >= 2022).astype(int)


def add_lag_features(panel: pd.DataFrame, value: str = TARGET) -> pd.DataFrame:
    """Ajoute lags et moyennes glissantes de `value` par zone et par bloc contigu.

    Le panel doit être complet (une ligne par zone et par heure, sans trou).
    """
    df = panel.sort_values(["zone", "ts"]).copy()
    g = df.groupby(["zone", contiguous_block(df["ts"])], sort=False)[value]
    for k in LAGS:
        if k < HORIZON_H:
            raise ValueError(f"lag {k} h < horizon {HORIZON_H} h : fuite temporelle")
        df[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(HORIZON_H)
    keys = [df["zone"], contiguous_block(df["ts"])]
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby(keys).transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )
    return df


def build_features(
    panel: pd.DataFrame, weather: pd.DataFrame, zones: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Assemble la matrice complète : cible, lags, calendrier, météo, zone.

    Les lignes dont un lag est indisponible (début de bloc) sont retirées.
    """
    zones = zone_table() if zones is None else zones
    df = panel[["zone", "ts", "pickups"]].copy()
    df[TARGET] = np.log1p(df["pickups"].astype(float))
    df = add_lag_features(df)
    df = df.merge(zones[["zone", "cbd"]], on="zone", how="left")
    df["cbd"] = df["cbd"].astype(np.int8)
    df = df.merge(weather[["ts", *WEATHER]], on="ts", how="left")
    df = pd.concat([df, calendar_features(df["ts"])], axis=1)
    df = df.dropna(subset=[*LAG_COLS, *ROLL_COLS, *WEATHER]).reset_index(drop=True)
    df["zone"] = df["zone"].astype("category")
    return df


def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Lignes dont la cible t est dans [start, end)."""
    return df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
