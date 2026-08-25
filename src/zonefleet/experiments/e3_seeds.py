"""E3 bis (extension) : stabilité du taux de suivi de E3 selon les graines.

Usage : uv run python -m zonefleet.experiments.e3_seeds
Sorties : results/tables/e3_seeds*.csv

Mêmes 13 scénarios, même période que E3. Deux sources d'aléa varient ensemble, graine
s de 1 à 5 : le rééchantillonnage du covariate shift et le nanny model de DLE. Le
contrefactuel « aucune dérive » est recalculé pour chaque graine (même nanny model).
Critère déclaré avant exécution (issue) : la conclusion de E3 tient si, pour chaque
graine, le taux de suivi en régime établi de toute dérive covariable nuisible dépasse
celui de tout concept drift pur.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.alerting import daily_signals
from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.experiments.e3_frontier import SCENARIOS, SPLIT, family, summarize
from zonefleet.features import zone_table
from zonefleet.monitoring import monitor
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.scenarios import inject

SEEDS = (1, 2, 3, 4, 5)
COLS = ["estimated", "realized", "threshold", "drift", "drift_clean", "pca_alert"]


def run_seed(ref, an, ref_stats, level_cache, seed: int) -> pd.DataFrame:
    """Signaux journaliers des 13 scénarios pour une graine."""
    parts = []
    for sc in SCENARIOS:
        if sc.kind == "level":
            an2 = level_cache[sc.label]
        else:
            an2 = inject(sc, an, ref_stats=ref_stats, seed=seed)
        sig = daily_signals(monitor(ref, an2, seed=seed))
        parts.append(sig[COLS].assign(scenario=sc.label, family=family(sc), severity=sc.severity))
    return pd.concat(parts)


def stability(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Min, médiane, max du taux de suivi établi par scénario, sur les graines."""
    g = per_seed.groupby(["scenario", "family", "severity"], sort=False)["tracking_steady"]
    return g.agg(["min", "median", "max"]).reset_index()


def families_separated(per_seed: pd.DataFrame) -> pd.Series:
    """Par graine : le pire covariate nuisible bat-il le meilleur concept pur ?"""
    out = {}
    for seed, d in per_seed.groupby("seed"):
        cov = d.loc[d["family"] == "covariate nuisible", "tracking_steady"].min()
        con = d.loc[d["family"] == "concept pur", "tracking_steady"].max()
        out[seed] = bool(cov > con)
    return pd.Series(out, name="separated")


def main() -> None:
    pred = predictions(SPLIT)
    ref, an = pred[pred.period == "reference"], pred[pred.period == "analysis"]
    ref_stats = {"lag_168": (ref["lag_168"].mean(), ref["lag_168"].std())}
    start, end = SPLITS[SPLIT].period("analysis")
    inputs = {
        "panel": load_panel(),
        "weather": load_weather(),
        "model": frozen_model(SPLIT),
        "start": start,
        "end": end,
        "zones": zone_table(),
    }
    # le choc propagé est déterministe : construit une fois, scoré avec chaque nanny model
    level_cache = {
        sc.label: inject(sc, an, level_inputs=inputs) for sc in SCENARIOS if sc.kind == "level"
    }
    rows = []
    for seed in SEEDS:
        daily = run_seed(ref, an, ref_stats, level_cache, seed)
        s = summarize(daily, pd.DataFrame({"scenario": daily["scenario"].unique()}))
        rows.append(s.assign(seed=seed))
        print(f"graine {seed} faite")
    per_seed = pd.concat(rows, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    keep = ["seed", "scenario", "family", "severity", "delta_est_steady", "delta_real_steady"]
    keep += ["tracking_transition", "tracking_steady", "corr_post"]
    per_seed[keep].round(4).to_csv(TABLES / "e3_seeds.csv", index=False)
    stab = stability(per_seed)
    stab.round(4).to_csv(TABLES / "e3_seeds_summary.csv", index=False)
    sep = families_separated(per_seed)
    sep.to_csv(TABLES / "e3_seeds_separation.csv", index_label="seed")
    print(stab.round(3).to_string())
    print(sep.to_string())


if __name__ == "__main__":
    main()
