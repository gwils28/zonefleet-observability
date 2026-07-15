"""Injection de dérives contrôlées dans une période d'analysis (E3, E4).

Trois familles, chacune « pure » par construction :

- `covariate_shift` : sous-échantillonnage pondéré de lignes **réelles**, jour par jour,
  avec des poids qui ne dépendent que de X. Chaque ligne garde son vrai couple (X, y) :
  P(y | X) est intact, seule P(X) change.
- `concept_drift` : la cible est multipliée sur un sous-ensemble de lignes, **sans toucher
  aux features ni aux prédictions** (les lags restent calculés sur l'ancienne demande) :
  P(X) est intacte, seule P(y | X) change.
- `level_shock` : la demande du panel est multipliée à partir d'une date, les features
  sont recalculées (les lags absorbent le choc avec leur retard) et le modèle figé est
  ré-appliqué. Cas réaliste, mêlé : concept drift au début, covariate shift ensuite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zonefleet.features import build_features, slice_period
from zonefleet.model import QuantileForecaster, seasonal_naive

ROWS_PER_DAY = 1440


@dataclass(frozen=True)
class Scenario:
    """Description d'une dérive injectée."""

    kind: str  # "none", "covariate", "concept", "level"
    severity: float
    onset: pd.Timestamp
    column: str = "lag_168"
    direction: int = -1  # covariate : -1 vers les faibles valeurs, +1 vers les fortes
    subset: str = "all"  # concept/level : "all", "cbd", "hors_cbd", "night", "evening"
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or f"{self.kind}_{self.severity:g}"


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    """Lignes concernées par une dérive locale."""
    if subset == "all":
        return pd.Series(True, index=df.index)
    if subset == "cbd":
        return df["cbd"] == 1
    if subset == "hors_cbd":
        return df["cbd"] == 0
    if subset == "night":
        return df["ts"].dt.hour.isin([0, 1, 2, 3, 4, 5])
    if subset == "evening":
        return df["ts"].dt.hour.isin([17, 18, 19, 20, 21, 22])
    raise ValueError(f"sous-ensemble inconnu : {subset}")


def covariate_shift(
    analysis: pd.DataFrame,
    onset: pd.Timestamp,
    kappa: float,
    column: str = "lag_168",
    direction: int = -1,
    ref_stats: tuple[float, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Rééchantillonne chaque jour après `onset` avec des poids exp(direction * kappa * z).

    z est la colonne standardisée (moyenne et écart-type de la référence si fournis).
    La taille de chaque chunk journalier est conservée (tirage avec remise).
    """
    rng = np.random.default_rng(seed)
    mu, sd = ref_stats or (analysis[column].mean(), analysis[column].std())
    before = analysis[analysis["ts"] < onset]
    parts = [before]
    after = analysis[analysis["ts"] >= onset]
    for _, day in after.groupby(after["ts"].dt.floor("D")):
        z = (day[column].to_numpy() - mu) / sd
        w = np.exp(direction * kappa * z)
        idx = rng.choice(len(day), size=len(day), replace=True, p=w / w.sum())
        parts.append(day.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def concept_drift(
    analysis: pd.DataFrame, onset: pd.Timestamp, factor: float, subset: str = "all"
) -> pd.DataFrame:
    """Multiplie la demande réalisée après `onset`, X et prédictions gelés."""
    out = analysis.copy()
    mask = (out["ts"] >= onset) & subset_mask(out, subset)
    out.loc[mask, "pickups"] = out.loc[mask, "pickups"] * factor
    return out


def level_shock(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    model: QuantileForecaster,
    start: pd.Timestamp,
    end: pd.Timestamp,
    onset: pd.Timestamp,
    factor: float,
    subset: str = "all",
    zones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Choc de niveau propagé : demande x factor à partir d'onset, features recalculées.

    Renvoie l'analysis [start, end) au format des prédictions (features, q10, q50, q90).
    Le panel est limité à 30 jours avant `start` pour que les lags existent.
    """
    p = panel[(panel["ts"] >= start - pd.Timedelta(days=30)) & (panel["ts"] < end)].copy()
    if zones is not None:
        p = p.merge(zones[["zone", "cbd"]], on="zone", how="left")
        p["cbd"] = p["cbd"].astype(int)
    else:
        p["cbd"] = 0
    mask = (p["ts"] >= onset) & subset_mask(p, subset)
    p.loc[mask, "pickups"] = p.loc[mask, "pickups"] * factor
    feats = build_features(p.drop(columns="cbd"), weather, zones)
    an = slice_period(feats, start, end)
    an = pd.concat([an, model.predict(an)], axis=1)
    an["snaive"] = seasonal_naive(an)
    an["period"] = "analysis"
    return an


def inject(
    scenario: Scenario,
    analysis: pd.DataFrame,
    ref_stats: dict[str, tuple[float, float]] | None = None,
    level_inputs: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Applique un scénario à une analysis au format `pipeline.predictions`."""
    if scenario.kind == "none":
        return analysis
    if scenario.kind == "covariate":
        stats = (ref_stats or {}).get(scenario.column)
        return covariate_shift(
            analysis,
            scenario.onset,
            scenario.severity,
            scenario.column,
            scenario.direction,
            stats,
            seed,
        )
    if scenario.kind == "concept":
        return concept_drift(analysis, scenario.onset, scenario.severity, scenario.subset)
    if scenario.kind == "level":
        if level_inputs is None:
            raise ValueError("level_shock exige panel, météo, modèle, bornes")
        return level_shock(
            onset=scenario.onset, factor=scenario.severity, subset=scenario.subset, **level_inputs
        )
    raise ValueError(f"type de scénario inconnu : {scenario.kind}")
