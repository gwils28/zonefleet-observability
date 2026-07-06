"""Baselines : seasonal naive (dans model.py) et MSTL glissant (statsforecast).

MSTL est réajusté chaque jour sur les 8 dernières semaines, avec les données arrêtées à
la fin du jour J-2 ; il prévoit les 24 heures du jour J (horizons 25 à 48 h). C'est un
peu plus difficile que le forecaster (horizon 24 h exact) : la comparaison est
conservatrice en faveur de LightGBM, ce que la documentation signale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL

WINDOW_H = 8 * 168


def mstl_rolling(panel: pd.DataFrame, zones: list[int], start: str, end: str) -> pd.DataFrame:
    """Prévisions MSTL journalières pour des zones, cibles dans [start, end)."""
    days = pd.date_range(start, end, freq="D", inclusive="left")
    sub = panel[panel["zone"].isin(zones)].copy()
    sub["y"] = np.log1p(sub["pickups"].astype(float))
    out = []
    for day in days:
        cutoff = day - pd.Timedelta(days=1)  # dernière heure connue : J-2 23:00
        hist = sub[(sub["ts"] < cutoff) & (sub["ts"] >= cutoff - pd.Timedelta(hours=WINDOW_H))]
        df = hist.rename(columns={"zone": "unique_id", "ts": "ds"})[["unique_id", "ds", "y"]]
        sf = StatsForecast(models=[MSTL(season_length=[24, 168])], freq="h", n_jobs=1)
        fc = sf.forecast(df=df, h=48)
        fc = fc[fc["ds"] >= day]
        out.append(fc.rename(columns={"unique_id": "zone", "ds": "ts", "MSTL": "mstl_log"}))
    res = pd.concat(out, ignore_index=True)
    res["mstl"] = np.clip(np.expm1(res["mstl_log"]), 0, None)
    return res[["zone", "ts", "mstl"]]
