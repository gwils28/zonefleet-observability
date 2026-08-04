"""Étapes partagées par les expériences : features, modèle figé, prédictions en cache.

Le cache (`results/cache/`) n'est pas versionné : il se régénère à l'identique (graines
fixées). Les tables de résultats, elles, sont versionnées dans `results/tables/`.
"""

from __future__ import annotations

import json

import pandas as pd

from zonefleet.config import ALL_SPLITS as SPLITS
from zonefleet.config import RESULTS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.features import build_features, slice_period
from zonefleet.metrics import seasonal_scale
from zonefleet.model import QuantileForecaster, seasonal_naive

CACHE = RESULTS / "cache"
TABLES = RESULTS / "tables"


def features(refresh: bool = False) -> pd.DataFrame:
    """Matrice de features complète (deux blocs), mise en cache."""
    path = CACHE / "features.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        df["zone"] = df["zone"].astype("category")
        return df
    df = build_features(load_panel(), load_weather())
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def frozen_model(split: str, refresh: bool = False) -> QuantileForecaster:
    """Modèle figé d'une expérience : entraîné une fois, puis seulement relu."""
    path = CACHE / "models" / f"{split}.pkl"
    if path.exists() and not refresh:
        return QuantileForecaster.load(path)
    df = features()
    model = QuantileForecaster().fit(slice_period(df, *SPLITS[split].period("train")))
    model.save(path)
    return model


def mase_scale(split: str) -> float:
    """Échelle MASE figée : MAE du seasonal naive sur l'entraînement de l'expérience."""
    path = CACHE / "scales.json"
    scales = json.loads(path.read_text()) if path.exists() else {}
    if split not in scales:
        tr = slice_period(features(), *SPLITS[split].period("train")).sort_values(["zone", "ts"])
        scales[split] = seasonal_scale(tr["pickups"], tr["zone"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scales, indent=2))
    return scales[split]


def predictions(split: str, refresh: bool = False) -> pd.DataFrame:
    """Prédictions du modèle figé sur la référence et l'analysis, avec la vérité terrain.

    Colonnes ajoutées : period ('reference' ou 'analysis'), q10, q50, q90, snaive.
    """
    path = CACHE / f"predictions_{split}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        df["zone"] = df["zone"].astype("category")
        return df
    model = frozen_model(split)
    df = features()
    parts = []
    for period in ("reference", "analysis"):
        sub = slice_period(df, *SPLITS[split].period(period))
        sub = pd.concat([sub, model.predict(sub)], axis=1)
        sub["snaive"] = seasonal_naive(sub)
        sub["period"] = period
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path, index=False)
    return out
