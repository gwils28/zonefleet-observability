"""Décisions opérationnelles, en fonctions pures (testées sans Dagster).

- `alert_severity` : du jour de signaux à un niveau d'alerte (page, ticket, rien).
- `retrain_decision` : réentraîner ou non, et pourquoi (politique gardée de E5).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zonefleet.alerting import persistent

PAGE, TICKET, NONE = "page", "ticket", "aucune"


def alert_severity(sig: pd.DataFrame, persistence: int = 2, ticket_days: int = 7) -> str:
    """Sévérité du dernier jour de `sig` (signaux journaliers de `alerting.daily_signals`).

    - page : dérive nettoyée ET perte estimée au-dessus du seuil, `persistence` jours de suite,
      ou perte réalisée au-dessus du seuil (labels arrivés) ;
    - ticket : dérive nettoyée seule pendant `ticket_days` jours (revue hebdomadaire) ;
    - aucune sinon.
    """
    drift = sig["drift_clean"] if "drift_clean" in sig else sig["drift"]
    joint = persistent(drift & sig["perf_est"], persistence)
    realized = persistent(sig["perf_real"].fillna(False).astype(bool), persistence)
    if joint.iloc[-1] or realized.iloc[-1]:
        return PAGE
    if persistent(drift, ticket_days).iloc[-1]:
        return TICKET
    return NONE


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reason: str


def retrain_decision(
    drift_confirmed: bool,
    post_alert_label_days: int,
    min_label_days: int = 28,
    feedback_guard_ok: bool = True,
) -> RetrainDecision:
    """Politique gardée : dérive confirmée ET assez de labels ET pas de boucle de rétroaction.

    `feedback_guard_ok` est faux si la fenêtre de labels candidate a été entièrement
    pilotée par le modèle lui-même (positionnement de flotte 100 % automatique, sans zones
    témoins) : on refuse d'apprendre sur une demande censurée par sa propre offre.
    """
    if not drift_confirmed:
        return RetrainDecision(False, "pas de dérive confirmée : réentraîner serait du bruit")
    if post_alert_label_days < min_label_days:
        return RetrainDecision(
            False, f"{post_alert_label_days} j de labels post-alerte < {min_label_days} j requis"
        )
    if not feedback_guard_ok:
        return RetrainDecision(
            False, "fenêtre pilotée par le modèle : risque de boucle de rétroaction"
        )
    return RetrainDecision(True, "entraîner un challenger, promotion si gain >= 5 % en validation")
