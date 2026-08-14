"""E4 bis : sensibilité des conclusions de E4 aux choix du protocole.

Usage : uv run python -m zonefleet.experiments.e4_sensitivity
Entrée : results/cache/e4_daily.parquet (signaux journaliers des 240 scénarios, écrits
par E4) ; aucune injection n'est relancée.
Sortie : results/tables/e4_sensitivity.csv

Un seul paramètre varie à la fois autour du protocole de référence (persistance 2 jours,
nuisance +10 %, fenêtre 28 jours, k = 1, signal lissé 7 jours) :
- persistance : 1, 2, 3 jours ;
- seuil de nuisance : +5 %, +10 %, +20 % ;
- fenêtre de détection et de nuisance : 14, 28, 42 jours.
"""

from __future__ import annotations

import warnings

import pandas as pd

from zonefleet.experiments.e4_backtest import (
    HARM,
    PERSISTENCE,
    WINDOW_DAYS,
    evaluate,
    harm_of,
    policy_flags,
    scenario_frame,
    smooth,
    summarize,
    thresholds,
)
from zonefleet.pipeline import CACHE, TABLES, predictions
from zonefleet.rowscore import RowScorer

K, SMOOTH_W = 1.0, 7
VARIANTS = (
    [("persistance", p, {"persistence": p}) for p in (1, 2, 3)]
    + [("nuisance", h, {"harm_threshold": h}) for h in (0.05, 0.10, 0.20)]
    + [("fenêtre", w, {"window": w}) for w in (14, 28, 42)]
)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    daily = pd.read_parquet(CACHE / "e4_daily.parquet")
    scen = pd.read_csv(TABLES / "e4_scenarios.csv", parse_dates=["onset"])
    pred = predictions("congestion")
    ref_daily = RowScorer().fit(pred[pred.period == "reference"]).reference_daily
    thr = thresholds(smooth(ref_daily, SMOOTH_W), K)
    base = scenario_frame(daily, scen.loc[scen["kind"] == "none", "id"].iloc[0])
    rows = []
    for param, value, kw in VARIANTS:
        persistence = int(kw.get("persistence", PERSISTENCE))
        window = int(kw.get("window", WINDOW_DAYS))
        harm_threshold = float(kw.get("harm_threshold", HARM))
        records: list[dict] = []
        for sc in scen.itertuples():
            raw = scenario_frame(daily, sc.id)
            harm = harm_of(sc, raw, base, window)
            d = smooth(raw, SMOOTH_W)
            flags = policy_flags(d, thr, persistence)
            records += evaluate(sc, d, harm, flags, window, harm_threshold)
        res = pd.DataFrame(records).assign(smooth=SMOOTH_W, k=K)
        summ = summarize(res, n_boot=500)
        rows.append(summ.assign(parameter=param, value=value))
        print(f"{param:12s} {value:<5} fait")
    out = pd.concat(rows, ignore_index=True)
    out.round(4).to_csv(TABLES / "e4_sensitivity.csv", index=False)
    show = out[
        out["policy"].isin(["P1b_drift_clean", "P2_dle", "P3b_clean_and_dle", "P4_realized"])
    ]
    print(
        show[
            [
                "parameter",
                "value",
                "policy",
                "recall",
                "false_alerts_per_30d",
                "median_delay",
                "n_harmful",
            ]
        ]
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
