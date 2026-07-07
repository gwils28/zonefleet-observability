"""Politiques d'alerte et horloge de décision.

Signaux journaliers (un booléen par jour cible) :
- `drift` : dérive des entrées, PCA en alerte OU PSI maximal > 0,25 sur les features
  clés (la PCA seule rate les déplacements le long des directions principales) ;
- `perf_est` : MAE estimée par DLE (échelle log) au-dessus du seuil NannyML ;
- `perf_real` : MAE réalisée (échelle log) au-dessus du même seuil, connue avec retard.

Politiques (SPEC, E4) :
- P1 dérive seule, P2 DLE seul, P3 dérive ET DLE, P4 labels retardés.

Une alerte n'est levée qu'après `persistence` jours consécutifs en dépassement, pour ne
pas réveiller l'astreinte sur un jour atypique isolé.

Horloge de décision : les signaux avancés d'un jour cible D sont calculables à
l'émission (D - 1 jour, H = 24 h) ; le signal réalisé n'existe qu'à D + L.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.config import HORIZON_H, LABEL_DELAY_DAYS

PSI_ALERT = 0.25
POLICIES = ("P1_drift", "P2_dle", "P3_drift_and_dle", "P4_realized")
LEAD = pd.Timedelta(hours=HORIZON_H)
LAG = pd.Timedelta(days=LABEL_DELAY_DAYS)


def daily_signals(signals: dict[str, pd.DataFrame], metric: str = "mae_log") -> pd.DataFrame:
    """Assemble les signaux booléens journaliers à partir de la sortie de `monitor`."""
    perf = signals["performance"].query("metric == @metric").set_index("date")
    mv = signals["multivariate"].set_index("date")["alert"]
    psi_max = signals["psi"].groupby("date")["psi"].max()
    out = pd.DataFrame(index=perf.index)
    out["pca_alert"] = mv.reindex(out.index).fillna(False).astype(bool)
    out["psi_max"] = psi_max.reindex(out.index)
    out["drift"] = out["pca_alert"] | (out["psi_max"] > PSI_ALERT)
    out["perf_est"] = perf["estimated"] > perf["threshold"]
    out["perf_real"] = perf["realized"] > perf["threshold"]
    out["estimated"], out["realized"], out["threshold"] = (
        perf["estimated"],
        perf["realized"],
        perf["threshold"],
    )
    return out


def persistent(flag: pd.Series, k: int) -> pd.Series:
    """Vrai le jour où `flag` est vrai depuis au moins k jours consécutifs."""
    run = flag.astype(int).groupby((~flag.astype(bool)).cumsum()).cumsum()
    return run >= k


def policy_flags(sig: pd.DataFrame, persistence: int = 2) -> pd.DataFrame:
    """Alertes (par jour cible) de chaque politique, après persistance."""
    raw = {
        "P1_drift": sig["drift"],
        "P2_dle": sig["perf_est"],
        "P3_drift_and_dle": sig["drift"] & sig["perf_est"],
        "P4_realized": sig["perf_real"],
    }
    return pd.DataFrame({k: persistent(v, persistence) for k, v in raw.items()})


def decision_time(target_day: pd.Timestamp, policy: str) -> pd.Timestamp:
    """Date à laquelle l'alerte d'un jour cible peut effectivement être levée."""
    return target_day + LAG if policy == "P4_realized" else target_day - LEAD


def first_alert(flags: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Première alerte de chaque politique à partir d'un jour cible, et son délai.

    Le délai est mesuré sur l'horloge de décision, en jours depuis `after` (début du choc).
    """
    rows = []
    for pol in flags.columns:
        hits = flags.index[(flags.index >= after) & flags[pol]]
        if len(hits) == 0:
            rows.append(
                {"policy": pol, "target_day": pd.NaT, "decision": pd.NaT, "delay_days": None}
            )
            continue
        dec = decision_time(hits[0], pol)
        delay = (dec - after) / pd.Timedelta(days=1)
        rows.append({"policy": pol, "target_day": hits[0], "decision": dec, "delay_days": delay})
    return pd.DataFrame(rows)


def false_alert_days(flags: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    """Nombre de jours en alerte avant le début du choc (fausses alertes)."""
    return flags[flags.index < before].sum()
