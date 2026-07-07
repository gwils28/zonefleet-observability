"""Briques de monitoring : dérive des entrées et des prédictions, performance estimée
sans labels (NannyML DLE) et performance réalisée.

Toutes les fonctions renvoient des tables « longues » à une ligne par chunk (jour), avec
une colonne `date`, pour que les politiques d'alerte (alerting.py) les combinent sans
connaître NannyML.

Deux échelles de performance sont surveillées :
- `count` : erreur en nombre de courses (lecture métier, mais dépend du niveau) ;
- `log` : erreur sur log1p (≈ erreur relative), stable quand le volume change d'échelle.
"""

from __future__ import annotations

import logging

import nannyml as nml
import numpy as np
import pandas as pd

from zonefleet.features import CALENDAR, LAG_COLS, ROLL_COLS, WEATHER

logging.getLogger("nannyml").setLevel(logging.WARNING)

DRIFT_FEATURES = [*LAG_COLS, *ROLL_COLS, *CALENDAR, *WEATHER]
DLE_FEATURES = ["zone", *DRIFT_FEATURES]
TS = "ts"
CHUNK = "D"


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatit les colonnes multi-index NannyML en `niveau1__niveau2`."""
    out = df.copy()
    out.columns = ["__".join(str(c) for c in col if c != "chunk") or "chunk" for col in df.columns]
    return out


def prepare(df: pd.DataFrame, scale: str = "count") -> pd.DataFrame:
    """Colonnes cible/prédiction selon l'échelle (`y_true`, `y_pred`)."""
    out = df.copy()
    if scale == "count":
        out["y_true"], out["y_pred"] = out["pickups"].astype(float), out["q50"]
    elif scale == "log":
        out["y_true"], out["y_pred"] = np.log1p(out["pickups"]), np.log1p(out["q50"])
    else:
        raise ValueError(f"échelle inconnue : {scale}")
    return out


def _dates(flat: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(flat["start_date"]).dt.normalize()


def estimate_performance(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    scale: str = "count",
    metrics: tuple[str, ...] = ("mae", "rmse"),
    chunk_period: str = CHUNK,
    seed: int = 0,
) -> pd.DataFrame:
    """Performance estimée par DLE sur l'analysis, **sans lire** `y_true` de l'analysis.

    DLE entraîne un « nanny model » sur la référence pour prédire la perte de chaque
    observation à partir des features et de la prédiction, puis agrège par chunk.
    """
    ref, an = prepare(reference, scale), prepare(analysis, scale)
    est = nml.DLE(
        feature_column_names=DLE_FEATURES,
        y_pred="y_pred",
        y_true="y_true",
        timestamp_column_name=TS,
        metrics=list(metrics),
        chunk_period=chunk_period,
        tune_hyperparameters=False,
        hyperparameters={"random_state": seed, "verbosity": -1},
    )
    est.fit(ref[[TS, *DLE_FEATURES, "y_pred", "y_true"]])
    blind = an[[TS, *DLE_FEATURES, "y_pred"]]  # la vérité terrain n'est pas transmise
    flat = flatten(est.estimate(blind).filter(period="analysis").to_df())
    rows = []
    for m in metrics:
        rows.append(
            pd.DataFrame(
                {
                    "date": _dates(flat),
                    "metric": f"{m}_{scale}",
                    "estimated": flat[f"{m}__value"],
                    "lower": flat[f"{m}__lower_confidence_boundary"],
                    "upper": flat[f"{m}__upper_confidence_boundary"],
                    "threshold": flat[f"{m}__upper_threshold"],
                    "alert": flat[f"{m}__alert"].astype(bool),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def realized_performance(df: pd.DataFrame, scale: str = "count", freq: str = CHUNK) -> pd.DataFrame:
    """MAE et RMSE réalisées par chunk, une fois les labels révélés."""
    d = prepare(df, scale)
    err = d["y_true"] - d["y_pred"]
    key = d[TS].dt.floor(freq)
    g = pd.DataFrame({"abs": err.abs(), "sq": err**2}).groupby(key)
    out = pd.DataFrame({"mae": g["abs"].mean(), "rmse": np.sqrt(g["sq"].mean())})
    out = (
        out.rename_axis("date").reset_index().melt("date", var_name="metric", value_name="realized")
    )
    out["metric"] = out["metric"] + f"_{scale}"
    return out
