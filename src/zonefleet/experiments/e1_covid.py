"""E1 : choc COVID 2020, performance estimée (DLE) vs réalisée.

Usage : uv run python -m zonefleet.experiments.e1_covid
Sorties : results/tables/e1_*.csv
"""

from __future__ import annotations

import pandas as pd

from zonefleet.alerting import daily_signals, false_alert_days, first_alert, policy_flags
from zonefleet.config import COVID_EMERGENCY
from zonefleet.monitoring import monitor
from zonefleet.pipeline import TABLES, predictions

SPLIT = "covid"
PHASES = {
    "avant (janv.-12 mars)": ("2020-01-01", "2020-03-13"),
    "choc (13 mars-12 avril)": ("2020-03-13", "2020-04-13"),
    "plancher (13 avril-30 juin)": ("2020-04-13", "2020-07-01"),
}


def phase_table(perf: pd.DataFrame) -> pd.DataFrame:
    """Erreur estimée vs réalisée par phase du choc, et ratio de sous-estimation."""
    rows = []
    for metric, d in perf.groupby("metric"):
        for name, (a, b) in PHASES.items():
            sub = d[(d["date"] >= a) & (d["date"] < b)]
            est, real = sub["estimated"].mean(), sub["realized"].mean()
            corr = sub["estimated"].corr(sub["realized"])
            rows.append(
                {
                    "metric": metric,
                    "phase": name,
                    "estimated": est,
                    "realized": real,
                    "ratio_est_real": est / real,
                    "corr_daily": corr,
                }
            )
    return pd.DataFrame(rows)


def run(split: str, shock: pd.Timestamp, prefix: str) -> dict[str, pd.DataFrame]:
    pred = predictions(split)
    ref, an = pred[pred.period == "reference"], pred[pred.period == "analysis"]
    out = monitor(ref, an)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.round(5).to_csv(TABLES / f"{prefix}_{name}.csv", index=False)
    sig = daily_signals(out)
    flags = policy_flags(sig)
    sig.join(flags).round(5).to_csv(TABLES / f"{prefix}_signals.csv", index_label="date")
    det = first_alert(flags, after=shock)
    det["false_alert_days_before"] = det["policy"].map(false_alert_days(flags, shock))
    det.to_csv(TABLES / f"{prefix}_detection.csv", index=False)
    return {"signals": sig, "detection": det, **out}


def main() -> None:
    res = run(SPLIT, COVID_EMERGENCY, "e1")
    phases = phase_table(res["performance"])
    phases.round(4).to_csv(TABLES / "e1_phases.csv", index=False)
    print(phases.round(3).to_string())
    print(res["detection"].to_string())


if __name__ == "__main__":
    main()
