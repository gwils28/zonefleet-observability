"""Rapport quotidien de monitoring, en Markdown.

Fonction pure : elle reçoit les signaux journaliers (`alerting.daily_signals`), les
pertes estimées par zone et les PSI par feature, et produit le texte que l'astreinte
lit le matin. Aucun accès aux données ici : le pilote (CLI, asset Dagster) les fournit.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.ops.decisions import PAGE, TICKET, alert_severity, retrain_decision

ICON = {PAGE: "PAGE", TICKET: "TICKET", "aucune": "RAS"}


def _fmt(x: float, d: int = 3) -> str:
    return "-" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


def build(
    day: pd.Timestamp,
    signals: pd.DataFrame,
    zone_loss: pd.DataFrame | None = None,
    psi: pd.DataFrame | None = None,
    model_version: str = "?",
) -> str:
    """Rapport du jour `day` (signaux jusqu'à ce jour inclus).

    `zone_loss` : colonnes zone, name, est (perte estimée du jour) ;
    `psi` : colonnes date, feature, psi.
    """
    sig = signals[signals.index <= day]
    if sig.empty:
        raise ValueError(f"aucun signal jusqu'au {day.date()}")
    sev = alert_severity(sig)
    last = sig.iloc[-1]
    confirmed = bool(
        ((sig["drift_clean"] & sig["perf_est"]) | sig["perf_real"].fillna(False)).any()
    )
    rec = retrain_decision(confirmed, int(sig["realized"].notna().sum()) if confirmed else 0)
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
