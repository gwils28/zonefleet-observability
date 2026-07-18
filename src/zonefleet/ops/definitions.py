"""Orchestration Dagster du monitoring ZoneFleet.

Lancer l'interface : `uv run dagster dev -m zonefleet.ops.definitions`
Matérialiser en ligne de commande :
`uv run dagster asset materialize -m zonefleet.ops.definitions --select '*'`

Graphe d'assets (un asset = une donnée versionnable, pas une tâche) :

    tlc_schema_report -> panel -> feature_matrix -> predictions -> monitoring_signals
                                      weather -/        champion_model -/        |
                                                                    alert_decision, retrain_decision

Les contrôles qualité sont des asset checks : data contract (aucune violation), panel
(aucun doublon, plafonds), fraîcheur des prédictions.
"""

# Pas de `from __future__ import annotations` : Dagster résout les annotations de
# configuration à l'exécution.
import dagster as dg
import pandas as pd

from zonefleet.alerting import daily_signals
from zonefleet.config import SPLITS
from zonefleet.ops.decisions import alert_severity, retrain_decision


class MonitoringConfig(dg.Config):
    """Paramètres d'une exécution : expérience (référence et analysis du SPEC)."""

    split: str = "congestion"


@dg.asset(group_name="donnees", description="Journal du data contract sur les fichiers TLC bruts.")
def tlc_schema_report() -> pd.DataFrame:
    from zonefleet.contract import schema_timeline
    from zonefleet.data import tlc_files

    files = tlc_files()
    reps = schema_timeline(files) if files else []
    return pd.DataFrame(
        [
            {
                "file": r.file,
                "ok": r.ok,
                "violations": "; ".join(r.violations),
                "unknown_breaks": "; ".join(r.unknown_breaks),
            }
            for r in reps
        ]
    )


@dg.asset_check(asset=tlc_schema_report, description="Aucun fichier ne viole le contrat.")
def contract_respected(tlc_schema_report: pd.DataFrame) -> dg.AssetCheckResult:
    bad = int((~tlc_schema_report["ok"]).sum()) if len(tlc_schema_report) else 0
    unknown = int((tlc_schema_report.get("unknown_breaks", pd.Series(dtype=str)) != "").sum())
    return dg.AssetCheckResult(
        passed=bad == 0,
        metadata={
            "fichiers": len(tlc_schema_report),
            "violations": bad,
            "ruptures_inconnues": unknown,
        },
    )


@dg.asset(group_name="donnees", deps=[tlc_schema_report], description="Panel zone x heure.")
def panel() -> pd.DataFrame:
    from zonefleet.data import load_panel

    return load_panel()


@dg.asset_check(asset=panel, description="Doublons, volumes négatifs, plafond du contrat.")
def panel_is_valid(panel: pd.DataFrame) -> dg.AssetCheckResult:
    from zonefleet.data import check_panel

    issues = check_panel(panel)
    return dg.AssetCheckResult(
        passed=not issues, metadata={"problemes": "; ".join(issues) or "aucun"}
    )


@dg.asset(group_name="donnees", description="Météo horaire locale (cache Open-Meteo).")
def weather() -> pd.DataFrame:
    from zonefleet.covariates import load_weather

    return load_weather()


@dg.asset(group_name="modele", description="Matrice de features (lags >= 24 h).")
def feature_matrix(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    from zonefleet.features import build_features

    return build_features(panel, weather)


@dg.asset(group_name="modele", description="Modèle champion (alias MLflow, sinon modèle figé).")
def champion_model(config: MonitoringConfig):
    from zonefleet.pipeline import frozen_model

    try:
        from zonefleet.ops.registry import load

        return load("champion")
    except Exception:  # registry vide : on retombe sur le modèle figé du SPEC
        return frozen_model(config.split)


@dg.asset(group_name="modele", description="Prévisions du champion sur référence et analysis.")
def predictions(
    config: MonitoringConfig, feature_matrix: pd.DataFrame, champion_model
) -> pd.DataFrame:
    from zonefleet.features import slice_period
    from zonefleet.model import seasonal_naive

    parts = []
    for period in ("reference", "analysis"):
        sub = slice_period(feature_matrix, *SPLITS[config.split].period(period))
        sub = pd.concat([sub, champion_model.predict(sub)], axis=1)
        parts.append(sub.assign(snaive=seasonal_naive(sub), period=period))
    return pd.concat(parts, ignore_index=True)


@dg.asset(group_name="monitoring", description="Signaux journaliers : dérive, DLE, réalisé.")
def monitoring_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    from zonefleet.monitoring import monitor

    ref = predictions[predictions["period"] == "reference"]
    an = predictions[predictions["period"] == "analysis"]
    return daily_signals(monitor(ref, an))


@dg.asset(group_name="monitoring", description="Sévérité d'alerte du dernier jour.")
def alert_decision(context: dg.AssetExecutionContext, monitoring_signals: pd.DataFrame) -> str:
    sev = alert_severity(monitoring_signals)
    context.add_output_metadata({"severite": sev, "jour": str(monitoring_signals.index[-1].date())})
    return sev


@dg.asset(
    group_name="monitoring", description="Recommandation de réentraînement (politique gardée)."
)
def retrain_recommendation(
    context: dg.AssetExecutionContext, monitoring_signals: pd.DataFrame
) -> dict:
    s = monitoring_signals
    confirmed = bool((s["drift_clean"] & s["perf_est"]).any() or s["perf_real"].any())
    first = s.index[(s["drift_clean"] & s["perf_est"]) | s["perf_real"].fillna(False)]
    label_days = int(s.loc[first[0] :, "realized"].notna().sum()) if len(first) else 0
    d = retrain_decision(confirmed, label_days)
    context.add_output_metadata({"reentrainer": d.retrain, "raison": d.reason})
    return {"retrain": d.retrain, "reason": d.reason}


monitoring_job = dg.define_asset_job("monitoring_quotidien", selection="*")

defs = dg.Definitions(
    assets=[
        tlc_schema_report,
        panel,
        weather,
        feature_matrix,
        champion_model,
        predictions,
        monitoring_signals,
        alert_decision,
        retrain_recommendation,
    ],
    asset_checks=[contract_respected, panel_is_valid],
    jobs=[monitoring_job],
    schedules=[
        dg.ScheduleDefinition(
            job=monitoring_job, cron_schedule="0 6 * * *", execution_timezone="America/New_York"
        )
    ],
)
