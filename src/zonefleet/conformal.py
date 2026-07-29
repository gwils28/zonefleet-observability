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


def rolling(
    df: pd.DataFrame, window_days: int = 28, delay_days: int = 1, alpha: float = 0.2
) -> pd.DataFrame:
    """Intervalle calibré jour par jour sur les `window_days` derniers jours de labels connus.

    Pour le jour cible D, la correction utilise les lignes dont la cible est dans
    [D - delay - window, D - delay) : aucun label non publié n'est lu. Les jours sans
    assez d'historique gardent l'intervalle brut (correction nulle).
    """
    d = df.copy()
    d["day"] = d["ts"].dt.floor("D")
    s = scores(np.log1p(d["pickups"]), np.log1p(d["q10"]), np.log1p(d["q90"]))
    by_day = pd.Series(s, index=d.index).groupby(d["day"])
    day_scores = {k: v.to_numpy() for k, v in by_day}
    days = sorted(day_scores)
    corr = {}
    for day in days:
        lo, hi = (
            day - pd.Timedelta(days=delay_days + window_days),
            day - pd.Timedelta(days=delay_days),
        )
        hist = [day_scores[k] for k in days if lo <= k < hi]
        corr[day] = (
            correction(np.concatenate(hist), alpha) if len(hist) >= window_days // 2 else 0.0
        )
    q = d["day"].map(corr)
    d["lo_cal"] = np.expm1(np.log1p(d["q10"]) - q).clip(lower=0)
    d["hi_cal"] = np.expm1(np.log1p(d["q90"]) + q)
    d["correction"] = q
    return d.drop(columns="day")
