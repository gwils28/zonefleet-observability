"""E10 : alerter par groupe de zones (CBD, hors CBD) plutôt qu'à la ville.

Usage : uv run python -m zonefleet.experiments.e10_zones
Sorties : results/tables/e10_*.csv

Mêmes 240 scénarios que E4 (tirage, nuisance et injections identiques). Les scores par
ligne sont agrégés par groupe ; chaque groupe a ses seuils de référence ; la politique
« par groupe » alerte si l'un des groupes satisfait la condition jointe nettoyée.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from zonefleet.alerting import LEAD, PSI_ALERT, persistent
from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.experiments.e4_backtest import PRE_START, WINDOW_DAYS, resample_rows, rising
from zonefleet.features import zone_table
from zonefleet.monitoring import psi
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.rowscore import RowScorer
from zonefleet.scenarios import level_shock, subset_mask

GROUPS = {"cbd": 1, "hors_cbd": 0}


class Scen(NamedTuple):
    """Scénario de E4, champs typés."""

    id: int
    kind: str
    severity: float
    subset: str
    onset: pd.Timestamp


def load_scenarios() -> list[Scen]:
    df = pd.read_csv(TABLES / "e4_scenarios.csv", parse_dates=["onset"])
    return [
        Scen(
            int(r["id"]),
            str(r["kind"]),
            float(r["severity"]),
            str(r["subset"]),
            pd.Timestamp(r["onset"]),
        )
        for _, r in df.iterrows()
    ]


def rows_with_group(scorer: RowScorer, df: pd.DataFrame) -> pd.DataFrame:
    r = scorer.score_rows(df)
    r["cbd"] = df["cbd"].to_numpy()
    return r


def group_daily(rows: pd.DataFrame, ref_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"ville": rows}
    out.update({g: rows[rows["cbd"] == v] for g, v in GROUPS.items()})
    res = {}
    for g, r in out.items():
        ref = ref_rows if g == "ville" else ref_rows[ref_rows["cbd"] == GROUPS[g]]
        gb = r.groupby("date")
        d = gb[["est", "recon", "real"]].mean()
        for c in ("lag_24", "lag_168"):
            ref_vals = ref[c].to_numpy()

            def day_psi(s: pd.Series, rv: np.ndarray = ref_vals) -> float:
                return psi(rv, s.to_numpy())

            d[f"psi_{c}"] = gb[c].apply(day_psi)
        res[g] = d
    return res


def thresholds(ref_daily: pd.DataFrame) -> dict[str, float]:
    """Perte estimée : référence lissée 7 jours, k = 1 ; PCA : référence brute, 3 écarts-types."""
    est = ref_daily["est"].rolling(7, min_periods=7).mean().dropna()
    return {
        "est": est.mean() + est.std(),
        "recon": ref_daily["recon"].mean() + 3 * ref_daily["recon"].std(),
    }


def joint(d: pd.DataFrame, thr: dict) -> pd.Series:
    drift = (d["recon"] > thr["recon"]) | (d[["psi_lag_24", "psi_lag_168"]].max(axis=1) > PSI_ALERT)
    return persistent(drift & (d["est"] > thr["est"]), 2)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pred = predictions("congestion")
    ref, an = (
        pred[pred.period == "reference"],
        pred[pred.period == "analysis"].reset_index(drop=True),
    )
    scorer = RowScorer().fit(ref)
    ref_rows = rows_with_group(scorer, ref)
    ref_daily = group_daily(ref_rows, ref_rows)
    thr = {g: thresholds(d) for g, d in ref_daily.items()}
    base_rows = rows_with_group(scorer, an)
    stats = (ref["lag_168"].mean(), ref["lag_168"].std())
    start, end = SPLITS["congestion"].period("analysis")
    panel, weather, model, zones = (
        load_panel(),
        load_weather(),
        frozen_model("congestion"),
        zone_table(),
    )
    scen = load_scenarios()
    harm = (
        pd.read_csv(TABLES / "e4_results.csv")
        .query("smooth == 1 and k == 1.0")
        .drop_duplicates("id")
        .set_index("id")["harmful"]
    )
    out = []
    for sc in scen:
        if sc.kind == "none":
            rows = base_rows
        elif sc.kind.startswith("covariate"):
            direction = -1 if sc.kind == "covariate_harmful" else 1
            rows = resample_rows(base_rows, sc.onset, sc.severity, direction, stats, seed=sc.id)
        elif sc.kind == "concept":
            mask = ((an["ts"] >= sc.onset) & subset_mask(an, sc.subset)).to_numpy()
            y = an["pickups"].to_numpy() * np.where(mask, sc.severity, 1.0)
            rows = base_rows.copy()
            rows["real"] = np.abs(np.log1p(y) - np.log1p(an["q50"].to_numpy()))
        else:
            df = level_shock(
                panel, weather, model, start, end, sc.onset, sc.severity, sc.subset, zones
            )
            rows = rows_with_group(scorer, df.reset_index(drop=True))
        daily = {
            g: d.assign(est=d["est"].rolling(7, min_periods=7).mean())
            for g, d in group_daily(rows, ref_rows).items()
        }
        flags = {g: joint(daily[g], thr[g]) for g in daily}
        flags["par_groupe"] = flags["cbd"] | flags["hors_cbd"]
        idx = daily["ville"].index
        win = (idx >= sc.onset) & (idx < sc.onset + pd.Timedelta(days=WINDOW_DAYS))
        null = (idx >= PRE_START) & ((idx < sc.onset) | bool(sc.kind == "none"))
        for pol in ("ville", "par_groupe"):
            f = flags[pol].reindex(idx).fillna(False)
            ev = rising(f)
            hits = idx[win & ev.to_numpy()]
            out.append(
                {
                    "id": sc.id,
                    "kind": sc.kind,
                    "subset": sc.subset,
                    "harmful": bool(harm[sc.id]),
                    "policy": pol,
                    "detected": len(hits) > 0,
                    "delay_days": ((hits[0] - LEAD) - sc.onset) / pd.Timedelta(days=1)
                    if len(hits)
                    else np.nan,
                    "false_alert_events": int(ev[null].sum()),
                    "null_days": int(null.sum()),
                }
            )
    res = pd.DataFrame(out)
    res.to_csv(TABLES / "e10_results.csv", index=False)
    h = res[res["harmful"]]
    summary = pd.DataFrame(
        [
            {
                "policy": pol,
                "recall": d.loc[d["harmful"], "detected"].mean(),
                "median_delay": d.loc[d["harmful"] & d["detected"], "delay_days"].median(),
                "false_alerts_per_30d": 30 * d["false_alert_events"].sum() / d["null_days"].sum(),
            }
            for pol, d in res.groupby("policy")
        ]
    )
    by_subset = h.groupby(["policy", "subset"])["detected"].mean().unstack().reset_index()
    n_subset = h.drop_duplicates("id")["subset"].value_counts().rename("n").reset_index()
    summary.round(4).to_csv(TABLES / "e10_summary.csv", index=False)
    by_subset.round(4).to_csv(TABLES / "e10_by_subset.csv", index=False)
    n_subset.to_csv(TABLES / "e10_subset_counts.csv", index=False)
    print(summary.round(3).to_string())
    print(by_subset.round(2).to_string())
    print(n_subset.to_string())


if __name__ == "__main__":
    main()
