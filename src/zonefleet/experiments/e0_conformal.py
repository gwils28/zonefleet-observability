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


if __name__ == "__main__":
    main()
