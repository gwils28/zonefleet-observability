"""E4 ter (extension) : fausses alertes sur des périodes calmes réelles, sans injection.

Usage : uv run python -m zonefleet.experiments.e4_calm
Sortie : results/tables/e4_calm.csv

Trois analysis sans choc connu : sept.-déc. 2019, sept.-déc. 2024 (découpages ajoutés,
`config.CALM_SPLITS`) et janv.-juin 2025 (E4). Toute alerte y est fausse ; on vérifie que
l'erreur réalisée reste sous son seuil de référence.
"""

from __future__ import annotations

import warnings

import pandas as pd

from zonefleet.experiments.e4_backtest import policy_flags, rising, smooth, thresholds
from zonefleet.pipeline import TABLES, predictions
from zonefleet.rowscore import RowScorer

PERIODS = {
    "calme_2019": "sept.-déc. 2019",
    "calme_2024": "sept.-déc. 2024",
    "congestion": "janv.-juin 2025",
}
K = 1.0


def evaluate_period(split: str) -> list[dict]:
    p = predictions(split)
    ref, an = p[p.period == "reference"], p[p.period == "analysis"].reset_index(drop=True)
    scorer = RowScorer().fit(ref)
    daily = scorer.aggregate(scorer.score_rows(an))
    rows = []
    for w in (1, 7):
        thr = thresholds(smooth(scorer.reference_daily, w), K)
        d = smooth(daily, w)
        flags = policy_flags(d, thr)
        real_ok = (d["real"].dropna() <= thr["real"]).mean()
        truth = flags["P4_realized"]  # dégradation réelle (labels), persistante 2 jours
        for pol in flags.columns:
            f = flags[pol]
            hit = (f & truth).sum()
            rows.append(
                {
                    "period": PERIODS[split],
                    "smooth": w,
                    "policy": pol,
                    "days": len(f),
                    "alert_events_per_30d": 30 * rising(f).sum() / len(f),
                    "share_days_in_alert": f.mean(),
                    "share_days_realized_below_threshold": real_ok,
                    "share_alert_days_on_real_degradation": hit / f.sum()
                    if f.sum()
                    else float("nan"),
                    "share_degradation_days_covered": hit / truth.sum()
                    if truth.sum()
                    else float("nan"),
                }
            )
    return rows


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rows = []
    for split in PERIODS:
        rows += evaluate_period(split)
        print(split, "fait")
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(TABLES / "e4_calm.csv", index=False)
    print(out[out["smooth"] == 7].round(3).to_string())


if __name__ == "__main__":
    main()
