"""Calibration conforme des quantiles (CQR, Romano, Patterson et Candès, 2019).

Pour un intervalle [q_lo, q_hi] de niveau nominal 1 - α, le score de conformité d'une
observation est E = max(q_lo - y, y - q_hi) : positif si y sort de l'intervalle. Sur un
échantillon de calibration de taille n, la correction est le quantile
⌈(n + 1)(1 - α)⌉ / n des scores ; l'intervalle calibré est [q_lo - Q, q_hi + Q].
La garantie de couverture suppose l'échangeabilité calibration / test : elle n'a
aucune raison de tenir à travers un choc.

Le calcul se fait en échelle log1p (où le modèle est appris), puis on revient en courses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Scores de conformité CQR."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return np.maximum(lo - y, y - hi)


def correction(s: np.ndarray, alpha: float = 0.2) -> float:
    """Quantile conforme des scores (avec la correction de taille finie)."""
    s = np.sort(np.asarray(s, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def fit(cal: pd.DataFrame, alpha: float = 0.2, by: str | None = None) -> dict:
    """Corrections globales ou par groupe (Mondrian), en échelle log1p."""
    y = np.log1p(cal["pickups"])
    s = scores(y, np.log1p(cal["q10"]), np.log1p(cal["q90"]))
    if by is None:
        return {"__all__": correction(s, alpha)}
    return {k: correction(s[cal[by] == k], alpha) for k in sorted(cal[by].unique())}


def apply(df: pd.DataFrame, corr: dict, by: str | None = None) -> pd.DataFrame:
    """Intervalle calibré (colonnes lo_cal, hi_cal, en courses)."""
    q = df["hour"].map(corr) if by else pd.Series(corr["__all__"], index=df.index)
    lo = np.expm1(np.log1p(df["q10"]) - q).clip(lower=0)
    hi = np.expm1(np.log1p(df["q90"]) + q)
    return df.assign(lo_cal=lo, hi_cal=hi)


def coverage_width(df: pd.DataFrame, lo: str, hi: str) -> tuple[float, float]:
    inside = (df["pickups"] >= df[lo]) & (df["pickups"] <= df[hi])
    return float(inside.mean()), float((df[hi] - df[lo]).mean())
