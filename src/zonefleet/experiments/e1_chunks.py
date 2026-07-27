"""E1 bis : taille de chunk, jour ou semaine, pour DLE et la dérive (choc COVID).

Usage : uv run python -m zonefleet.experiments.e1_chunks
Sorties : results/tables/e1_chunks_*.csv

Un chunk plus grand réduit le bruit d'échantillonnage (intervalle de confiance de DLE,
fluctuations du PSI) mais retarde la décision : une semaine n'est complète que le
dimanche. On compare, sur la même analysis 2020 :
- la qualité de suivi (corrélation estimé / réalisé avant et après le choc) ;
- la largeur relative de l'intervalle de confiance de DLE ;
- la première décision possible après le 13 mars (horloge de décision : veille du
  dernier jour cible du chunk), alerte = estimation au-dessus du seuil NannyML.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.alerting import LEAD
from zonefleet.config import COVID_EMERGENCY
from zonefleet.monitoring import estimate_performance, realized_performance
from zonefleet.pipeline import TABLES, predictions

CHUNKS = {"jour": "D", "semaine": "W"}


def chunk_end(start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Dernier jour cible couvert par un chunk commençant à `start`."""
    return start if freq == "D" else start + pd.Timedelta(days=6)


def run_chunk(ref: pd.DataFrame, an: pd.DataFrame, label: str, freq: str) -> pd.DataFrame:
    est = estimate_performance(ref, an, "log", metrics=("mae",), chunk_period=freq)
    real = realized_performance(an, "log", freq="D")
    real = real[real["metric"] == "mae_log"].set_index("date")["realized"]
    if freq == "W":
        # même découpage que NannyML : semaines calendaires, rattachées à leur premier jour
        rows = []
        for d in est["date"]:
            rows.append(real[(real.index >= d) & (real.index < d + pd.Timedelta(days=7))].mean())
        est["realized"] = rows
    else:
        est["realized"] = est["date"].map(real)
    est["chunk"] = label
    est["ci_rel_width"] = (est["upper"] - est["lower"]) / est["estimated"]
    est["decision"] = [chunk_end(d, freq) - LEAD for d in est["date"]]
    return est


def summarize(d: pd.DataFrame) -> dict:
    pre = d[d["date"] < COVID_EMERGENCY]
    post = d[d["date"] >= COVID_EMERGENCY - pd.Timedelta(days=6)]
    alerts = d[(d["estimated"] > d["threshold"]) & (d["decision"] >= COVID_EMERGENCY - LEAD)]
    first = alerts["decision"].min()
    pre_alerts = d[(d["estimated"] > d["threshold"]) & (d["date"] < COVID_EMERGENCY)]
    return {
        "chunk": d["chunk"].iloc[0],
        "n_chunks": len(d),
        "corr_pre": pre["estimated"].corr(pre["realized"]),
        "corr_post": post["estimated"].corr(post["realized"]),
        "ci_rel_width_median": d["ci_rel_width"].median(),
        "first_decision": first,
        "delay_days": (first - COVID_EMERGENCY) / pd.Timedelta(days=1) if pd.notna(first) else None,
        "false_alert_chunks_before": len(pre_alerts),
    }


def main() -> None:
    p = predictions("covid")
    ref, an = p[p.period == "reference"], p[p.period == "analysis"]
    parts = [run_chunk(ref, an, label, freq) for label, freq in CHUNKS.items()]
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).round(5).to_csv(TABLES / "e1_chunks_series.csv", index=False)
    summ = pd.DataFrame([summarize(d) for d in parts])
    summ.round(4).to_csv(TABLES / "e1_chunks_summary.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
