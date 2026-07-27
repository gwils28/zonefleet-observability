"""E0 : qualité du modèle surveillé sur la référence (et l'analysis, pour mémoire).

Usage : uv run python -m zonefleet.experiments.e0_model
Sorties : results/tables/e0_*.csv
"""

from __future__ import annotations

import pandas as pd

from zonefleet.baselines import mstl_rolling
from zonefleet.config import SPLITS
from zonefleet.data import load_panel
from zonefleet.metrics import mae, smape, summarize
from zonefleet.pipeline import TABLES, frozen_model, mase_scale, predictions

MSTL_ZONES = [7, 48, 74, 79, 132, 138, 161, 237]


def metrics_table() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        scale = mase_scale(split)
        pred = predictions(split)
        for period, d in pred.groupby("period"):
            y = d["pickups"].to_numpy()
            lgbm = summarize(y, d["q10"], d["q50"], d["q90"], scale)
            rows.append({"split": split, "period": period, "model": "lightgbm", **lgbm})
            sn = d["snaive"].to_numpy()
            rows.append(
                {
                    "split": split,
                    "period": period,
                    "model": "seasonal_naive",
                    "mae": mae(y, sn),
                    "smape": smape(y, sn),
                    "mase": mae(y, sn) / scale,
                }
            )
    return pd.DataFrame(rows)


def by_zone(split: str) -> pd.DataFrame:
    """Erreur, gain sur le seasonal naive et couverture, par zone (référence)."""
    ref = predictions(split).query("period == 'reference'").copy()
    ref["abs_lgbm"] = (ref["pickups"] - ref["q50"]).abs()
    ref["abs_snaive"] = (ref["pickups"] - ref["snaive"]).abs()
    ref["inside"] = (ref["pickups"] >= ref["q10"]) & (ref["pickups"] <= ref["q90"])
    g = ref.groupby("zone", observed=True)
    out = pd.DataFrame(
        {
            "mean_pickups": g["pickups"].mean(),
            "mae_lightgbm": g["abs_lgbm"].mean(),
            "mae_snaive": g["abs_snaive"].mean(),
            "coverage_80": g["inside"].mean(),
            "cbd": g["cbd"].first(),
        }
    )
    out["skill_vs_snaive"] = 1 - out["mae_lightgbm"] / out["mae_snaive"]
    return out.reset_index().assign(split=split)


def coverage_by_hour(split: str) -> pd.DataFrame:
    ref = predictions(split).query("period == 'reference'")
    inside = (ref.pickups >= ref.q10) & (ref.pickups <= ref.q90)
    return (
        inside.groupby(ref["hour"]).mean().rename("coverage_80").reset_index().assign(split=split)
    )


def mstl_comparison() -> pd.DataFrame:
    panel = load_panel()
    rows = []
    for split, sp in SPLITS.items():
        start, end = sp.reference
        fc = mstl_rolling(panel, MSTL_ZONES, start, str(pd.Timestamp(end) + pd.Timedelta(days=1)))
        ref = predictions(split).query("period == 'reference'")
        ref = ref[ref["zone"].isin(MSTL_ZONES)].astype({"zone": int})
        m = ref.merge(fc, on=["zone", "ts"], how="inner")
        for name, col in [("lightgbm", "q50"), ("seasonal_naive", "snaive"), ("mstl", "mstl")]:
            rows.append({"split": split, "model": name, "mae": mae(m.pickups, m[col]), "n": len(m)})
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics_table().round(4).to_csv(TABLES / "e0_metrics.csv", index=False)
    pd.concat([by_zone(s) for s in SPLITS]).round(4).to_csv(TABLES / "e0_by_zone.csv", index=False)
    cov = pd.concat([coverage_by_hour(s) for s in SPLITS])
    cov.round(4).to_csv(TABLES / "e0_coverage_by_hour.csv", index=False)
    imp = pd.concat([frozen_model(s).feature_importance().rename(s) for s in SPLITS], axis=1)
    (imp / imp.sum()).round(4).rename_axis("feature").to_csv(TABLES / "e0_feature_importance.csv")
    mstl_comparison().round(4).to_csv(TABLES / "e0_mstl.csv", index=False)
    print(pd.read_csv(TABLES / "e0_metrics.csv").to_string())
    print(pd.read_csv(TABLES / "e0_mstl.csv").to_string())


if __name__ == "__main__":
    main()
