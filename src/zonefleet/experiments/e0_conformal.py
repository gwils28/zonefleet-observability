"""E0 bis (extension hors SPEC) : calibration conforme des quantiles.

Usage : uv run python -m zonefleet.experiments.e0_conformal
Sortie : results/tables/e0_conformal*.csv

Calibration sur septembre-octobre de la référence, évaluation sur novembre-décembre
(même régime), puis sur les analysis (régime qui change).
"""

from __future__ import annotations

import pandas as pd

from zonefleet import conformal as C
from zonefleet.pipeline import TABLES, predictions

ALPHA = 0.2


def periods(split: str) -> dict[str, pd.DataFrame]:
    p = predictions(split)
    ref = p[p["period"] == "reference"]
    cut = ref["ts"].min() + pd.DateOffset(months=2)
    an = p[p["period"] == "analysis"]
    mid = an["ts"].min() + pd.DateOffset(months=3)
    return {
        "calibration": ref[ref["ts"] < cut],
        "test (fin de référence)": ref[ref["ts"] >= cut],
        "analysis, 3 premiers mois": an[an["ts"] < mid],
        "analysis, 3 mois suivants": an[an["ts"] >= mid],
    }


def main() -> None:
    rows, hourly = [], []
    for split in ("covid", "congestion"):
        parts = periods(split)
        cal = parts["calibration"]
        corrs = {
            "globale": (C.fit(cal, ALPHA), None),
            "par heure": (C.fit(cal, ALPHA, by="hour"), "hour"),
        }
        for name, df in parts.items():
            if name == "calibration":
                continue
            cov, width = C.coverage_width(df, "q10", "q90")
            rows.append(
                {
                    "split": split,
                    "period": name,
                    "method": "aucune",
                    "coverage": cov,
                    "width": width,
                }
            )
            for m, (corr, by) in corrs.items():
                out = C.apply(df, corr, by)
                cov, width = C.coverage_width(out, "lo_cal", "hi_cal")
                rows.append(
                    {"split": split, "period": name, "method": m, "coverage": cov, "width": width}
                )
                if name == "test (fin de référence)":
                    inside = (out["pickups"] >= out["lo_cal"]) & (out["pickups"] <= out["hi_cal"])
                    h = inside.groupby(out["hour"]).mean().rename("coverage").reset_index()
                    hourly.append(h.assign(split=split, method=m))
            raw = (df["pickups"] >= df["q10"]) & (df["pickups"] <= df["q90"])
            if name == "test (fin de référence)":
                h = raw.groupby(df["hour"]).mean().rename("coverage").reset_index()
                hourly.append(h.assign(split=split, method="aucune"))
        pd.DataFrame(
            [
                {"split": split, "hour": k, "correction_log": v}
                for k, v in corrs["par heure"][0].items()
            ]
            + [{"split": split, "hour": "toutes", "correction_log": corrs["globale"][0]["__all__"]}]
        ).to_csv(TABLES / f"e0_conformal_corrections_{split}.csv", index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TABLES / "e0_conformal.csv", index=False)
    pd.concat(hourly).round(4).to_csv(TABLES / "e0_conformal_by_hour.csv", index=False)
    print(out.round(3).to_string())
    rolling_main()


if __name__ == "__main__":
    main()


def rolling_study(window_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration glissante (labels à J + 1 et J + 60) sur référence + analysis."""
    daily, rows = [], []
    for split in ("covid", "congestion"):
        p = predictions(split).sort_values("ts")
        for delay in (1, 60):
            out = C.rolling(p, window_days=window_days, delay_days=delay, alpha=ALPHA)
            out["inside"] = (out["pickups"] >= out["lo_cal"]) & (out["pickups"] <= out["hi_cal"])
            out["inside_raw"] = (out["pickups"] >= out["q10"]) & (out["pickups"] <= out["q90"])
            d = out.groupby(out["ts"].dt.floor("D")).agg(
                coverage=("inside", "mean"),
                coverage_raw=("inside_raw", "mean"),
                correction=("correction", "first"),
            )
            daily.append(d.rename_axis("date").reset_index().assign(split=split, delay=delay))
            for name, df in periods(split).items():
                if name == "calibration":
                    continue
                sub = out[out["ts"].isin(df["ts"]) & out["zone"].isin(df["zone"].unique())]
                rows.append(
                    {
                        "split": split,
                        "delay": delay,
                        "period": name,
                        "coverage": sub["inside"].mean(),
                        "width": (sub["hi_cal"] - sub["lo_cal"]).mean(),
                    }
                )
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def rolling_main() -> None:
    summ, daily = rolling_study()
    summ.round(4).to_csv(TABLES / "e0_conformal_rolling.csv", index=False)
    daily.round(4).to_csv(TABLES / "e0_conformal_rolling_daily.csv", index=False)
    print(summ.round(3).to_string())
