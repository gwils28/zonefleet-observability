"""E3 : où la performance estimée décroche (covariate shift vs concept drift).

Usage : uv run python -m zonefleet.experiments.e3_frontier
Sorties : results/tables/e3_*.csv

Base : le modèle 2024 et sa référence (sept.-déc. 2024) ; analysis janv.-juin 2025, que
E2 a montrée calme. Les dérives sont injectées à partir du lundi 3 mars 2025.
Mesure centrale, le **taux de suivi** : (hausse de l'erreur estimée) / (hausse de
l'erreur réalisée) entre avant et après l'injection. 1 = DLE suit, 0 = DLE aveugle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zonefleet.alerting import daily_signals, policy_flags
from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.features import zone_table
from zonefleet.monitoring import monitor
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.scenarios import Scenario, inject

SPLIT = "congestion"
ONSET = pd.Timestamp("2025-03-03")
PRE = (pd.Timestamp("2025-01-06"), ONSET)
TRANSITION_DAYS = 14

SCENARIOS = [
    Scenario("none", 0.0, ONSET, name="aucune dérive"),
    *[
        Scenario("covariate", k, ONSET, direction=-1, name=f"covariate nuisible κ={k:g}")
        for k in (0.5, 1.0, 2.0)
    ],
    *[
        Scenario("covariate", k, ONSET, direction=+1, name=f"covariate bénin κ={k:g}")
        for k in (0.5, 1.0, 2.0)
    ],
    *[Scenario("concept", m, ONSET, name=f"concept pur x{m:g}") for m in (0.85, 0.7, 0.5)],
    *[Scenario("level", m, ONSET, name=f"choc propagé x{m:g}") for m in (0.85, 0.7, 0.5)],
]


def family(sc: Scenario) -> str:
    if sc.kind == "covariate":
        return "covariate nuisible" if sc.direction < 0 else "covariate bénin"
    return {"none": "aucune", "concept": "concept pur", "level": "choc propagé"}[sc.kind]


WINDOWS = {
    "post": (0, 10_000),
    "transition": (0, TRANSITION_DAYS),
    "steady": (TRANSITION_DAYS, 10_000),
}


def summarize(daily: pd.DataFrame, flags_share: pd.DataFrame) -> pd.DataFrame:
    """Effet injecté = scénario moins contrefactuel sans dérive, sur les mêmes jours.

    Comparer avant/après l'injection mélangerait l'effet injecté et l'évolution naturelle
    de la période (hiver puis printemps) ; le contrefactuel « aucune dérive » l'annule.
    """
    base = daily[daily["scenario"] == "aucune dérive"]
    rows = []
    for (sc, fam, sev), d in daily.groupby(["scenario", "family", "severity"], sort=False):
        row = {"scenario": sc, "family": fam, "severity": sev}
        for w, (a, b) in WINDOWS.items():
            lo, hi = ONSET + pd.Timedelta(days=a), ONSET + pd.Timedelta(days=b)
            m = (d.index >= lo) & (d.index < hi)
            mb = (base.index >= lo) & (base.index < hi)
            de = d.loc[m, "estimated"].mean() - base.loc[mb, "estimated"].mean()
            dr = d.loc[m, "realized"].mean() - base.loc[mb, "realized"].mean()
            row[f"est_{w}"] = d.loc[m, "estimated"].mean()
            row[f"real_{w}"] = d.loc[m, "realized"].mean()
            row[f"delta_est_{w}"] = de
            row[f"delta_real_{w}"] = dr
            row[f"tracking_{w}"] = de / dr if abs(dr) > 0.01 else np.nan
        post = d[d.index >= ONSET]
        row["corr_post"] = post["estimated"].corr(post["realized"])
        row["share_drift"] = post["drift"].mean()
        row["share_drift_clean"] = post["drift_clean"].mean()
        row["share_pca"] = post["pca_alert"].mean()
        rows.append(row)
    return pd.DataFrame(rows).merge(flags_share, on="scenario")


def main() -> None:
    pred = predictions(SPLIT)
    ref, an = pred[pred.period == "reference"], pred[pred.period == "analysis"]
    ref_stats = {"lag_168": (ref["lag_168"].mean(), ref["lag_168"].std())}
    start, end = SPLITS[SPLIT].period("analysis")
    level_inputs = {
        "panel": load_panel(),
        "weather": load_weather(),
        "model": frozen_model(SPLIT),
        "start": start,
        "end": end,
        "zones": zone_table(),
    }
    shares, daily = [], []
    cols = [
        "estimated",
        "realized",
        "threshold",
        "drift",
        "drift_clean",
        "pca_alert",
        "psi_clean_max",
    ]
    for sc in SCENARIOS:
        an2 = inject(sc, an, ref_stats=ref_stats, level_inputs=level_inputs, seed=7)
        sig = daily_signals(monitor(ref, an2))
        flags = policy_flags(sig)
        post = flags[flags.index >= ONSET]
        shares.append({"scenario": sc.label, **{f"share_{c}": post[c].mean() for c in post}})
        daily.append(sig[cols].assign(scenario=sc.label, family=family(sc), severity=sc.severity))
    daily = pd.concat(daily)
    summary = summarize(daily, pd.DataFrame(shares))
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(TABLES / "e3_summary.csv", index=False)
    daily.round(5).to_csv(TABLES / "e3_daily.csv", index_label="date")
    show = [
        "scenario",
        "delta_est_transition",
        "delta_real_transition",
        "tracking_transition",
        "delta_est_steady",
        "delta_real_steady",
        "tracking_steady",
        "share_drift_clean",
    ]
    print(summary[show].round(3).to_string())


if __name__ == "__main__":
    main()
