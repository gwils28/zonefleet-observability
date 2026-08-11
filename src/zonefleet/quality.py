"""Contrôles de qualité journaliers du panel, avant tout monitoring de modèle.

Sculley et al. (section 7) : les producteurs de données amont doivent être surveillés,
et leurs alertes propagées au plan de contrôle du système ML. Un fichier tronqué ou un
horodatage décalé produit exactement les symptômes d'une dérive ; il faut l'attraper
avant d'accuser le modèle.

Trois contrôles par jour, tous comparés aux 4 mêmes jours de semaine précédents
(robustes à la saisonnalité hebdomadaire) :
- `volume_ratio` : volume total du jour / médiane des 4 mêmes jours précédents ;
- `zero_zones` : nombre de zones sans aucune course sur la journée ;
- `profile_corr` : corrélation du profil horaire ville (24 valeurs) avec la moyenne des
  4 mêmes jours précédents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zonefleet.covariates import us_holidays

VOLUME_BAND = (0.7, 1.4)
PROFILE_MIN = 0.97
WEEKS = 4


def known_calendar_days(index: pd.Index) -> pd.Series:
    """Jours fériés US, la veille et le lendemain, et la semaine du 24 décembre au 1er janvier.

    Ces jours ont un volume et un profil atypiques connus d'avance : un contrôle de
    qualité qui les signale produit des fausses alertes certaines.
    """
    days = pd.DatetimeIndex(index)
    hol = us_holidays(sorted(set(days.year) | {y - 1 for y in days.year}))
    near = {h + pd.Timedelta(days=k) for h in hol for k in (-1, 0, 1)}
    year_end = (days.month == 12) & (days.day >= 24) | (days.month == 1) & (days.day == 1)
    return pd.Series(days.isin(list(near)) | year_end, index=days)


def calibrate(checks: pd.DataFrame, q: float = 0.005) -> dict[str, float]:
    """Seuils empiriques (quantiles q et 1 - q) sur des jours ordinaires de référence."""
    ordinary = checks[~known_calendar_days(checks.index)]
    return {
        "volume_low": float(ordinary["volume_ratio"].quantile(q)),
        "volume_high": float(ordinary["volume_ratio"].quantile(1 - q)),
        "profile_min": float(ordinary["profile_corr"].quantile(q)),
    }


def daily_checks(panel: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Une ligne par jour : indicateurs et drapeaux de qualité (jours calendaires connus exclus)."""
    p = panel.assign(day=panel["ts"].dt.floor("D"), hour=panel["ts"].dt.hour)
    total = p.groupby("day")["pickups"].sum()
    zero = p.groupby(["day", "zone"])["pickups"].sum().eq(0).groupby(level="day").sum()
    prof = p.groupby(["day", "hour"])["pickups"].sum().unstack("hour")
    rows = []
    for day in total.index:
        past = [day - pd.Timedelta(weeks=w) for w in range(1, WEEKS + 1)]
        past = [d for d in past if d in total.index]
        if len(past) < WEEKS:
            continue
        ratio = total[day] / np.median(total[past])
        ref_prof = prof.loc[past].mean()
        corr = float(np.corrcoef(prof.loc[day], ref_prof)[0, 1])
        rows.append(
            {"date": day, "volume_ratio": ratio, "zero_zones": int(zero[day]), "profile_corr": corr}
        )
    out = pd.DataFrame(rows).set_index("date")
    t = thresholds or {
        "volume_low": VOLUME_BAND[0],
        "volume_high": VOLUME_BAND[1],
        "profile_min": PROFILE_MIN,
    }
    known = known_calendar_days(out.index).to_numpy()
    low, high = out["volume_ratio"] < t["volume_low"], out["volume_ratio"] > t["volume_high"]
    out["known_calendar_day"] = known
    out["flag_volume"] = (low | high) & ~known
    out["flag_zero"] = out["zero_zones"] > 0
    out["flag_profile"] = (out["profile_corr"] < t["profile_min"]) & ~known
    out["flag_any"] = out[["flag_volume", "flag_zero", "flag_profile"]].any(axis=1)
    return out


def inject_bug(
    panel: pd.DataFrame,
    kind: str,
    start: pd.Timestamp,
    days: int = 7,
    zones: tuple[int, ...] = (161, 162, 230),
    seed: int = 0,
) -> pd.DataFrame:
    """Injecte un bug de données dans le panel sur [start, start + days)."""
    p = panel.copy()
    m = (p["ts"] >= start) & (p["ts"] < start + pd.Timedelta(days=days))
    if kind == "troncature":
        rng = np.random.default_rng(seed)
        p.loc[m, "pickups"] = rng.binomial(p.loc[m, "pickups"].astype(int), 0.6).astype(float)
    elif kind == "duplication":
        p.loc[m, "pickups"] = p.loc[m, "pickups"] * 2
    elif kind == "decalage_1h":
        p = p.sort_values(["zone", "ts"])
        shifted = p.groupby("zone")["pickups"].shift(1)
        p.loc[m, "pickups"] = shifted[m].fillna(p.loc[m, "pickups"])
    elif kind == "zone_muette":
        p.loc[m & p["zone"].isin(zones), "pickups"] = 0.0
    else:
        raise ValueError(f"bug inconnu : {kind}")
    return p
