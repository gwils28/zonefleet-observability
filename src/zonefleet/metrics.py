"""Métriques de prévision réalisée (échelle des courses).

MASE : l'échelle est l'erreur absolue moyenne du seasonal naive hebdomadaire (lag 168 h)
calculée sur la période d'entraînement ; elle est fixée une fois pour toutes, sinon la
métrique bouge avec le régime et ne se compare plus d'une semaine à l'autre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def mae(y: ArrayLike, p: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y: ArrayLike, p: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y: ArrayLike, p: ArrayLike) -> float:
    """sMAPE en %, terme nul quand prévision et réalisé valent 0 (demande intermittente)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(y - p), den, out=np.zeros_like(den), where=den > 0)
    return float(100 * ratio.mean())


def mase(y: ArrayLike, p: ArrayLike, scale: float) -> float:
    return mae(y, p) / scale


def pinball(y: ArrayLike, q: ArrayLike, alpha: float) -> float:
    """Perte quantile moyenne au niveau alpha."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Part des réalisations dans [lo, hi]."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def seasonal_scale(pickups: pd.Series, zone: pd.Series, season: int = 168) -> float:
    """MAE du seasonal naive par zone (panel trié par zone puis heure), toutes zones confondues."""
    diff = pickups.groupby(zone, observed=True).diff(season).abs()
    return float(diff.mean())


def summarize(
    y: ArrayLike,
    q10: ArrayLike,
    q50: ArrayLike,
    q90: ArrayLike,
    scale: float,
) -> dict[str, float]:
    """Toutes les métriques du SPEC pour une prévision quantile."""
    return {
        "mae": mae(y, q50),
        "rmse": rmse(y, q50),
        "smape": smape(y, q50),
        "mase": mase(y, q50, scale),
        "pinball_mean": float(
            np.mean([pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)])
        ),
        "coverage_80": coverage(y, q10, q90),
    }
