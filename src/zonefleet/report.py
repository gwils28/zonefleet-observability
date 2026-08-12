"""Rapport quotidien de monitoring, en Markdown.

Fonction pure : elle reçoit les signaux journaliers (`alerting.daily_signals`), les
pertes estimées par zone et les PSI par feature, et produit le texte que l'astreinte
lit le matin. Aucun accès aux données ici : le pilote (CLI, asset Dagster) les fournit.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.alerting import persistent
from zonefleet.config import LABEL_DELAY_DAYS
from zonefleet.ops.decisions import PAGE, TICKET, alert_severity, retrain_decision

ICON = {PAGE: "PAGE", TICKET: "TICKET", "aucune": "RAS", "donnees": "INCIDENT DONNÉES"}


def _fmt(x: float, d: int = 3) -> str:
    return "-" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def build(
    day: pd.Timestamp,
    signals: pd.DataFrame,
    zone_loss: pd.DataFrame | None = None,
    psi: pd.DataFrame | None = None,
    model_version: str = "?",
    label_delay_days: int = LABEL_DELAY_DAYS,
    quality: pd.Series | None = None,
) -> str:
    """Rapport du jour `day` (signaux jusqu'à ce jour inclus).

    `zone_loss` : colonnes zone, name, est (perte estimée du jour) ;
    `psi` : colonnes date, feature, psi.
    """
    sig = signals[signals.index <= day].copy()
    if sig.empty:
        raise ValueError(f"aucun signal jusqu'au {day.date()}")
    # Horloge des labels : un rejeu contient des labels du futur ; on ne garde que ceux
    # publiés à la date du rapport (jour cible + délai <= jour du rapport).
    unpublished = pd.DatetimeIndex(sig.index) + pd.Timedelta(days=label_delay_days) > day
    sig.loc[unpublished, "realized"] = float("nan")
    sig.loc[unpublished, "perf_real"] = False
    sev = alert_severity(sig)
    data_incident = quality is not None and bool(quality.get("flag_any", False))
    if data_incident:
        sev = "donnees"  # un bug amont prime : les signaux du modèle ne sont pas fiables (E8)
    last = sig.iloc[-1]
    persisted_joint = persistent(sig["drift_clean"] & sig["perf_est"], 2)
    joint = persisted_joint | persistent(sig["perf_real"].astype(bool), 2)
    confirmed = bool(joint.any())
    first = sig.index[joint]
    label_days = int(sig.loc[first[0] :, "realized"].notna().sum()) if len(first) else 0
    rec = retrain_decision(confirmed, label_days)
    lines = [
        f"# Monitoring ZoneFleet : {day.date()}",
        "",
        f"**Sévérité : {ICON.get(sev, sev)}** · modèle `zonefleet-demand` v{model_version}",
        "",
        "## Signaux du jour",
        "",
        "| Signal | Valeur | Seuil | État |",
        "|---|---|---|---|",
        f"| perte estimée (MAE log, DLE) | {_fmt(last['estimated'])} | {_fmt(last['threshold'])} | "
        f"{'au-dessus' if last['perf_est'] else 'normal'} |",
        f"| dérive nettoyée (PCA ou PSI lags) | {'oui' if last['drift_clean'] else 'non'} | - | "
        f"{'dérive' if last['drift_clean'] else 'normal'} |",
        f"| perte réalisée (labels) | {_fmt(last['realized'])} | {_fmt(last['threshold'])} | "
        f"{'disponible' if pd.notna(last['realized']) else 'en attente'} |",
        "",
    ]
    if quality is not None:
        lines += [
            "## Qualité des données (avant le modèle)",
            "",
            f"- volume / médiane des 4 mêmes jours : {_fmt(quality['volume_ratio'], 2)}",
            f"- corrélation du profil horaire : {_fmt(quality['profile_corr'], 3)}",
            f"- zones sans aucune course : {int(quality['zero_zones'])}",
            "",
        ]
        if data_incident:
            lines += [
                "> Incident données : traiter l'amont avant toute lecture des signaux du"
                " modèle ; exclure ces jours de tout réentraînement.",
                "",
            ]
    if zone_loss is not None and len(zone_loss):
        top = zone_loss.sort_values("est", ascending=False).head(5)
        lines += ["## Zones en tête de perte estimée", "", "| Zone | Perte estimée |", "|---|---|"]
        lines += [
            f"| {r['zone']} {r.get('name', '')} | {_fmt(r['est'])} |" for _, r in top.iterrows()
        ]
        lines += [""]
    if psi is not None and len(psi):
        today = psi[psi["date"] == day].sort_values("psi", ascending=False)
        drifting = today[today["psi"] > 0.25]
        lines += ["## Features en dérive (PSI > 0,25)", ""]
        lines += [f"- `{r['feature']}` : {_fmt(r['psi'], 2)}" for _, r in drifting.iterrows()] or [
            "- aucune"
        ]
        lines += [""]
    lines += [
        "## Recommandation",
        "",
        f"- Réentraînement : **{'oui' if rec.retrain else 'non'}** ({rec.reason}).",
    ]
    if sev == PAGE:
        lines += [
            "- Suivre le runbook : contrat de données, carte de perte estimée, événement, repli."
        ]
    lines += [
        "",
        "> Angle mort : un changement de la relation entre entrées et demande (concept drift pur)"
        " n'apparaît dans aucun de ces signaux avant l'arrivée des labels.",
        "",
    ]
    return "\n".join(lines)
