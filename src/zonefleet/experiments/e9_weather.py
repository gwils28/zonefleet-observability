"""E9 : la vraie prévision météo de la veille à la place de la météo réalisée.

Usage : uv run python -m zonefleet.experiments.e9_weather
Sorties : results/tables/e9_*.csv

Découpage E2 (référence sept.-déc. 2024, analysis janv.-juin 2025). Trois variantes :
(a) entraîné et servi sur la météo réalisée (hypothèse du projet) ;
(b) entraîné sur le réalisé, servi sur la prévision (le modèle actuel en production) ;
(c) entraîné et servi sur la prévision (entraînement à partir du 20 janvier 2024 : l'archive
    des prévisions commence le 19 janvier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather, load_weather_forecast
from zonefleet.data import load_panel
from zonefleet.features import build_features, slice_period, zone_table
from zonefleet.model import QuantileForecaster, seasonal_naive
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.rowscore import RowScorer

SPLIT = "congestion"
TRAIN_START_FORECAST = pd.Timestamp("2024-01-20")


def forecast_error() -> pd.DataFrame:
    w = load_weather().set_index("ts")
    f = load_weather_forecast().set_index("ts")
    j = w.join(f, rsuffix="_prev", how="inner").dropna()
    j = j[j.index >= TRAIN_START_FORECAST]
    rain, rain_f = j["precipitation"] > 0.1, j["precipitation_prev"] > 0.1
    return pd.DataFrame(
        [
            {
                "heures": len(j),
                "mae_temperature": (j["temperature_2m"] - j["temperature_2m_prev"]).abs().mean(),
                "biais_temperature": (j["temperature_2m_prev"] - j["temperature_2m"]).mean(),
                "corr_temperature": j["temperature_2m"].corr(j["temperature_2m_prev"]),
                "pluie_detectee": (rain & rain_f).sum() / max(rain.sum(), 1),
                "fausse_pluie": (~rain & rain_f).sum() / max((~rain).sum(), 1),
            }
        ]
    )


def with_forecast_features() -> pd.DataFrame:
    panel = load_panel()
    panel = panel[panel["ts"] >= pd.Timestamp("2023-12-01")]
    return build_features(panel, load_weather_forecast(), zone_table())


def predict(model: QuantileForecaster, feats: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for period in ("reference", "analysis"):
        sub = slice_period(feats, *SPLITS[SPLIT].period(period))
        sub = pd.concat([sub, model.predict(sub)], axis=1)
        parts.append(sub.assign(snaive=seasonal_naive(sub), period=period))
    return pd.concat(parts, ignore_index=True)


def metrics(pred: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    scorer = RowScorer().fit(pred[pred.period == "reference"])
    for period, d in pred.groupby("period"):
        rows.append(
            {
                "variant": variant,
                "period": period,
                "mae": (d["pickups"] - d["q50"]).abs().mean(),
                "mae_log": np.abs(np.log1p(d["pickups"]) - np.log1p(d["q50"])).mean(),
                "dle_est_mean": scorer.daily(d.reset_index(drop=True))["est"].mean(),
            }
        )
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    forecast_error().round(4).to_csv(TABLES / "e9_forecast_error.csv", index=False)
    feats_f = with_forecast_features()
    a = predictions(SPLIT)
    b = predict(frozen_model(SPLIT), feats_f)
    train = slice_period(feats_f, TRAIN_START_FORECAST, SPLITS[SPLIT].period("train")[1])
    c = predict(QuantileForecaster().fit(train), feats_f)
    rows = (
        metrics(a, "a : réalisé / réalisé")
        + metrics(b, "b : réalisé / prévision")
        + metrics(c, "c : prévision / prévision")
    )
    out = pd.DataFrame(rows)
    base_mae = out[out["variant"].str.startswith("a")].set_index("period")["mae"]
    out["delta_mae_pct"] = 100 * (out["mae"] / out["period"].map(base_mae) - 1)
    out.round(4).to_csv(TABLES / "e9_variants.csv", index=False)
    print(pd.read_csv(TABLES / "e9_forecast_error.csv").round(3).to_string())
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
