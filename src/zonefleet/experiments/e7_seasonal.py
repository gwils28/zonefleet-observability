"""E7 : dérive contre une base saisonnière plutôt qu'une référence fixe.

Usage : uv run python -m zonefleet.experiments.e7_seasonal
Sorties : results/tables/e7_*.csv

Trois signaux par jour et par feature (`temperature_2m`, `lag_24`, `lag_168`) :
- PSI contre la référence fixe (sept.-déc. de l'année précédente), le signal de E1/E2 ;
- PSI contre la base saisonnière (28 jours centrés sur la même date un an plus tôt),
  le signal déclaré dans l'issue #53 ;
- z-score saisonnier de la moyenne journalière, ajouté après coup (voir les notes).
Alerte : PSI > 0,25 ou |z| > 3, persistance de 2 jours.
"""

from __future__ import annotations

import pandas as pd

from zonefleet.alerting import persistent
from zonefleet.config import CONGESTION_START, COVID_EMERGENCY, SPLITS
from zonefleet.features import slice_period
from zonefleet.monitoring import daily_psi, seasonal_psi, seasonal_zscore
from zonefleet.pipeline import TABLES, features

COLUMNS = ["temperature_2m", "lag_24", "lag_168"]
CASES = {"covid": COVID_EMERGENCY, "congestion": CONGESTION_START}


def signals(split: str) -> pd.DataFrame:
    df = features()
    sp = SPLITS[split]
    an = slice_period(df, *sp.period("analysis"))
    ref = slice_period(df, *sp.period("reference"))
    a0, a1 = sp.period("analysis")
    hist = slice_period(df, a0 - pd.Timedelta(days=400), a1 - pd.Timedelta(days=300))
    fixed = (
        daily_psi(ref, an, COLUMNS)
        .rename(columns={"psi": "value"})
        .assign(signal="PSI référence fixe")
    )
    seas = (
        seasonal_psi(hist, an, COLUMNS)
        .rename(columns={"psi": "value"})
        .assign(signal="PSI base saisonnière")
    )
    z = (
        seasonal_zscore(hist, an, COLUMNS)
        .rename(columns={"z": "value"})
        .assign(signal="z-score saisonnier")
    )
    out = pd.concat([fixed, seas, z], ignore_index=True)
    out["alert"] = (out["signal"].str.startswith("PSI") & (out["value"] > 0.25)) | (
        (out["signal"] == "z-score saisonnier") & (out["value"].abs() > 3)
    )
    return out.assign(split=split)


def summarize(sig: pd.DataFrame, shock: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for (signal, feature), d in sig.groupby(["signal", "feature"]):
        f = persistent(d.set_index("date")["alert"].sort_index(), 2)
        before = f[f.index < shock]
        after = f[f.index >= shock]
        first = after.index[after.to_numpy()].min() if after.any() else pd.NaT
        rows.append(
            {
                "signal": signal,
                "feature": feature,
                "share_alert_before": before.mean(),
                "first_alert_after": first,
                "delay_days": (first - shock) / pd.Timedelta(days=1) if pd.notna(first) else None,
                "share_alert_after": after.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    allsig, summ = [], []
    for split, shock in CASES.items():
        s = signals(split)
        allsig.append(s)
        summ.append(summarize(s, shock).assign(split=split))
    pd.concat(allsig).round(4).to_csv(TABLES / "e7_signals.csv", index=False)
    out = pd.concat(summ, ignore_index=True)
    out.round(4).to_csv(TABLES / "e7_summary.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
