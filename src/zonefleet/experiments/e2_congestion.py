"""E2 : tarification de congestion du 5 janvier 2025, zones CBD vs hors CBD.

Usage : uv run python -m zonefleet.experiments.e2_congestion
Sorties : results/tables/e2_*.csv

Trois questions :
1. Le choc est-il visible dans la demande, et où ? (évolution annuelle par groupe)
2. Dégrade-t-il le modèle figé ? (erreur réalisée par groupe, avant/après)
3. Le moniteur le voit-il, globalement et par groupe ? (DLE, dérive, politiques)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zonefleet.alerting import daily_signals, false_alert_days, first_alert, policy_flags
from zonefleet.config import CONGESTION_START
from zonefleet.data import load_panel
from zonefleet.experiments.e1_covid import run
from zonefleet.features import zone_table
from zonefleet.monitoring import monitor
from zonefleet.pipeline import TABLES, predictions

SPLIT = "congestion"
GROUPS = {"cbd": True, "hors_cbd": False}


def demand_change() -> pd.DataFrame:
    """Variation de la demande janv.-juin 2025 vs janv.-juin 2024, par groupe et par heure.

    Une différence de différences simple : l'évolution CBD moins l'évolution hors CBD
    neutralise la tendance commune (reprise post-COVID, météo, calendrier).
    """
    p = load_panel().merge(zone_table()[["zone", "cbd"]], on="zone")
    p = p[p["ts"].dt.month <= 6]
    p["year"], p["hour"] = p["ts"].dt.year, p["ts"].dt.hour
    p = p[p["year"].isin([2024, 2025])]
    g = p.groupby(["cbd", "year", "hour"])["pickups"].mean().unstack("year")
    g["change_pct"] = 100 * (g[2025] / g[2024] - 1)
    out = g["change_pct"].unstack("cbd").rename(columns={True: "cbd", False: "hors_cbd"})
    out["did_pts"] = out["cbd"] - out["hors_cbd"]
    return out.reset_index()


def demand_change_daily() -> pd.DataFrame:
    """Rapport journalier CBD / hors CBD (moyenne 7 jours) sur 2024-2025."""
    p = load_panel().merge(zone_table()[["zone", "cbd"]], on="zone")
    p = p[p["ts"].dt.year >= 2024]
    d = p.groupby([p["ts"].dt.floor("D"), "cbd"])["pickups"].sum().unstack("cbd")
    d.columns = ["hors_cbd", "cbd"]
    d["ratio_cbd"] = d["cbd"] / (d["cbd"] + d["hors_cbd"])
    return d.rename_axis("date").reset_index()


def realized_by_group() -> pd.DataFrame:
    """MAE réalisée (log et count) par groupe, référence vs avant/après le 5 janvier."""
    pred = predictions(SPLIT)
    rows = []
    phases = {
        "référence (sept.-déc. 2024)": pred["period"] == "reference",
        "1er-4 janv. 2025": (pred["period"] == "analysis") & (pred["ts"] < CONGESTION_START),
        "après le 5 janv. 2025": (pred["period"] == "analysis") & (pred["ts"] >= CONGESTION_START),
    }
    for gname, flag in GROUPS.items():
        for pname, mask in phases.items():
            d = pred[mask & (pred["cbd"] == int(flag))]
            rows.append(
                {
                    "group": gname,
                    "phase": pname,
                    "mae_count": (d["pickups"] - d["q50"]).abs().mean(),
                    "mae_log": (np.log1p(d["pickups"]) - np.log1p(d["q50"])).abs().mean(),
                    "bias_log": (np.log1p(d["pickups"]) - np.log1p(d["q50"])).mean(),
                    "n": len(d),
                }
            )
    return pd.DataFrame(rows)


def monitor_by_group() -> pd.DataFrame:
    """Monitoring complet séparé par groupe de zones ; détection après le 5 janvier."""
    pred = predictions(SPLIT)
    dets = []
    for gname, flag in GROUPS.items():
        sub = pred[pred["cbd"] == int(flag)]
        ref, an = sub[sub.period == "reference"], sub[sub.period == "analysis"]
        out = monitor(ref, an)
        out["performance"].round(5).to_csv(TABLES / f"e2_performance_{gname}.csv", index=False)
        sig = daily_signals(out)
        flags = policy_flags(sig)
        sig.join(flags).round(5).to_csv(TABLES / f"e2_signals_{gname}.csv", index_label="date")
        det = first_alert(flags, after=CONGESTION_START)
        det["false_alert_days_before"] = det["policy"].map(
            false_alert_days(flags, CONGESTION_START)
        )
        det["alert_days_after"] = det["policy"].map(flags[flags.index >= CONGESTION_START].sum())
        dets.append(det.assign(group=gname))
    return pd.concat(dets, ignore_index=True)


def main() -> None:
    res = run(SPLIT, CONGESTION_START, "e2")
    dc = demand_change()
    dc.round(3).to_csv(TABLES / "e2_demand_change_by_hour.csv", index=False)
    demand_change_daily().round(4).to_csv(TABLES / "e2_demand_daily.csv", index=False)
    rg = realized_by_group()
    rg.round(4).to_csv(TABLES / "e2_realized_by_group.csv", index=False)
    mg = monitor_by_group()
    mg.to_csv(TABLES / "e2_detection_by_group.csv", index=False)
    print(res["detection"].to_string())
    print(dc.round(1).to_string())
    print(rg.round(3).to_string())
    print(mg.to_string())


if __name__ == "__main__":
    main()
