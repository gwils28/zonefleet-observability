"""E8 : bugs de données amont, contrôles de qualité contre moniteur de modèle.

Usage : uv run python -m zonefleet.experiments.e8_quality
Sorties : results/tables/e8_*.csv

Quatre bugs injectés dans le panel de l'analysis 2025 à partir du lundi 3 mars, pendant
7 jours. Pour chacun : (a) les contrôles de qualité, calibrés sur 2024, le voient-ils et
quand ? (b) que disent les politiques du moniteur de modèle, une fois les features et les
prévisions recalculées sur le panel bogué ?
"""

from __future__ import annotations

import warnings

import pandas as pd

from zonefleet import quality as Q
from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.experiments.e4_backtest import policy_flags, smooth, thresholds
from zonefleet.features import build_features, slice_period, zone_table
from zonefleet.model import seasonal_naive
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.rowscore import RowScorer

ONSET = pd.Timestamp("2025-03-03")
DAYS = 7
BUGS = ("aucun", "troncature", "duplication", "decalage_1h", "zone_muette")
POLICIES = ("P1b_drift_clean", "P2_dle", "P3b_clean_and_dle")


def analysis_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    start, end = SPLITS["congestion"].period("analysis")
    p = panel[(panel["ts"] >= start - pd.Timedelta(days=30)) & (panel["ts"] < end)]
    feats = build_features(p, load_weather(), zone_table())
    an = slice_period(feats, start, end)
    an = pd.concat([an, frozen_model("congestion").predict(an)], axis=1)
    an["snaive"] = seasonal_naive(an)
    return an.reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    panel = load_panel()
    p24 = panel[panel["ts"].dt.year == 2024]
    thr_q = Q.calibrate(Q.daily_checks(p24))
    pred = predictions("congestion")
    scorer = RowScorer().fit(pred[pred.period == "reference"])
    thr_m = thresholds(smooth(scorer.reference_daily, 1), 1.0)
    window = (ONSET, ONSET + pd.Timedelta(days=DAYS + 14))
    rows, daily_q = [], []
    for bug in BUGS:
        bugged = panel if bug == "aucun" else Q.inject_bug(panel, bug, ONSET, DAYS)
        q = Q.daily_checks(bugged[bugged["ts"].dt.year >= 2024], thr_q)
        q = q[q.index >= "2025-01-01"]
        daily_q.append(q.assign(bug=bug))
        qwin = q[(q.index >= window[0]) & (q.index < window[1])]
        first_q = qwin.index[qwin["flag_any"]].min() if qwin["flag_any"].any() else pd.NaT
        an = analysis_from_panel(bugged)
        d = scorer.aggregate(scorer.score_rows(an))
        flags = policy_flags(d, thr_m)
        row = {
            "bug": bug,
            "quality_first_flag": first_q,
            "quality_flag_days": int(qwin["flag_any"].sum()),
            "quality_flags": ", ".join(
                c[5:] for c in ("flag_volume", "flag_zero", "flag_profile") if qwin[c].any()
            ),
        }
        for pol in POLICIES:
            f = flags[pol]
            fw = f[(f.index >= window[0]) & (f.index < window[1])]
            row[f"{pol}_days"] = int(fw.sum())
            row[f"{pol}_first"] = fw.index[fw.to_numpy()].min() if fw.any() else pd.NaT
        dw = d[(d.index >= ONSET) & (d.index < ONSET + pd.Timedelta(days=DAYS))]
        row["est_mean_bug_days"] = dw["est"].mean()
        row["real_mean_bug_days"] = dw["real"].mean()
        rows.append(row)
        print(bug, "fait")
    TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TABLES / "e8_summary.csv", index=False)
    pd.concat(daily_q).round(4).to_csv(TABLES / "e8_quality_daily.csv", index_label="date")
    pd.DataFrame([thr_q]).round(4).to_csv(TABLES / "e8_thresholds.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
