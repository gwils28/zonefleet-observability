"""E4 quater : la dérive saisonnière (z-score de E7) dans la condition jointe.

Usage : uv run python -m zonefleet.experiments.e4_seasonal
Entrées : cache de E4 (signaux journaliers des 240 scénarios, avec moyennes de lags),
features 2024 pour la base saisonnière.
Sorties : results/tables/e4_seasonal*.csv

Politiques ajoutées, comparées à P1b et P3b :
- P1s : PCA en alerte OU |z saisonnier| > 3 sur la moyenne journalière de `lag_24` ou
  `lag_168` (base : 28 jours centrés sur la même date, 364 jours plus tôt) ;
- P3s : P1s ET perte estimée (DLE) au-dessus du seuil.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from zonefleet.alerting import LAG, LEAD, persistent
from zonefleet.experiments.e4_backtest import (
    HARM,
    PRE_START,
    WINDOW_DAYS,
    harm_of,
    policy_flags,
    rising,
    smooth,
    summarize,
    thresholds,
)
from zonefleet.features import slice_period
from zonefleet.pipeline import CACHE, TABLES, features, predictions
from zonefleet.rowscore import RowScorer

Z_ALERT = 3.0
COLS = ("lag_24", "lag_168")


def seasonal_base() -> pd.DataFrame:
    """Moyennes journalières de 2024 (base saisonnière des analysis 2025)."""
    f = slice_period(features(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-08-01"))
    return f.groupby(f["ts"].dt.floor("D"))[list(COLS)].mean()


def seasonal_z(daily: pd.DataFrame, base: pd.DataFrame, half: int = 14) -> pd.DataFrame:
    """|z| saisonnier de chaque moyenne journalière de lag."""
    out = pd.DataFrame(index=daily.index)
    for c in COLS:
        z = []
        for day, v in daily[f"mean_{c}"].items():
            center = day - pd.Timedelta(days=364)
            b = base.loc[
                (base.index >= center - pd.Timedelta(days=half))
                & (base.index < center + pd.Timedelta(days=half)),
                c,
            ]
            z.append(abs(v - b.mean()) / b.std() if len(b) >= half and b.std() > 0 else np.nan)
        out[f"z_{c}"] = z
    return out


def flags_with_seasonal(d: pd.DataFrame, z: pd.DataFrame, thr: dict) -> pd.DataFrame:
    flags = policy_flags(d, thr)
    pca = d["recon"] > thr["recon"]
    seas = pca | (z.max(axis=1) > Z_ALERT)
    est = d["est"] > thr["est"]
    flags["P1s_drift_seasonal"] = persistent(seas, 2)
    flags["P3s_seasonal_and_dle"] = persistent(seas & est, 2)
    return flags


def evaluate(sc, d, harm, flags) -> list[dict]:
    onset = sc.onset
    win = (d.index >= onset) & (d.index < onset + pd.Timedelta(days=WINDOW_DAYS))
    null = (d.index >= PRE_START) & ((d.index < onset) | (sc.kind == "none"))
    rows = []
    for pol in flags.columns:
        f = flags[pol]
        ev = rising(f)
        hits = f.index[win & ev.to_numpy()]
        dec = (hits[0] + LAG if pol == "P4_realized" else hits[0] - LEAD) if len(hits) else None
        rows.append(
            {
                "id": sc.id,
                "kind": sc.kind,
                "severity": sc.severity,
                "subset": sc.subset,
                "onset": onset,
                "harm": harm,
                "harmful": harm > HARM,
                "policy": pol,
                "detected": len(hits) > 0,
                "delay_days": (dec - onset) / pd.Timedelta(days=1) if dec is not None else np.nan,
                "false_alert_events": int(ev[null].sum()),
                "false_alert_days": int(f[null].sum()),
                "null_days": int(null.sum()),
            }
        )
    return rows


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    daily = pd.read_parquet(CACHE / "e4_daily.parquet")
    scen = pd.read_csv(TABLES / "e4_scenarios.csv", parse_dates=["onset"])
    pred = predictions("congestion")
    ref_daily = RowScorer().fit(pred[pred.period == "reference"]).reference_daily
    base_raw = daily.xs(scen.loc[scen["kind"] == "none", "id"].iloc[0])
    base = seasonal_base()
    res = []
    for w in (1, 7):
        thr = thresholds(smooth(ref_daily, w), 1.0)
        for sc in scen.itertuples():
            raw = daily.xs(sc.id)
            z = seasonal_z(raw, base)
            d = smooth(raw, w)
            res += [
                dict(r, smooth=w, k=1.0)
                for r in evaluate(sc, d, harm_of(sc, raw, base_raw), flags_with_seasonal(d, z, thr))
            ]
    res = pd.DataFrame(res)
    res.to_csv(TABLES / "e4_seasonal_results.csv", index=False)
    summ = summarize(res, n_boot=1000)
    summ.round(4).to_csv(TABLES / "e4_seasonal.csv", index=False)
    by_kind = (
        res[res["harmful"]]
        .groupby(["smooth", "policy", "kind"])["detected"]
        .mean()
        .unstack()
        .reset_index()
    )
    by_kind.round(4).to_csv(TABLES / "e4_seasonal_by_kind.csv", index=False)
    cols = [
        "smooth",
        "policy",
        "recall",
        "median_delay",
        "false_alerts_per_30d",
        "share_null_days_in_alert",
        "benign_alert_rate",
    ]
    print(summ[cols].round(3).to_string())
    print(by_kind.round(2).to_string())


if __name__ == "__main__":
    main()
