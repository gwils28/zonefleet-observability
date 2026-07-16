"""E4 : backtest du moniteur lui-même (rappel x délai x fausses alertes).

Usage : uv run python -m zonefleet.experiments.e4_backtest [--n 240]
Sorties : results/tables/e4_*.csv

Un moniteur non testé est un faux sentiment de sécurité. On tire au hasard des scénarios
de dérive (type, intensité, sous-ensemble, date de début) injectés dans l'analysis calme
de 2025, dont 20 % de scénarios nuls, et on mesure pour chaque politique d'alerte :
- le **rappel** sur les scénarios nuisibles (MAE log réalisée +10 % sur 28 jours, contre
  le contrefactuel sans dérive), détection comptée si un jour cible des 28 jours suivant
  l'injection est en alerte ;
- le **délai** de la première alerte, sur l'horloge de décision ;
- les **fausses alertes** : jours en alerte avant l'injection et dans les scénarios nuls,
  rapportés à 30 jours ;
- les **alertes inutiles** : scénarios non nuisibles (dérive bénigne) qui déclenchent.

Les seuils de DLE et de l'erreur réalisée valent moyenne + k écarts-types des jours de
référence ; k est balayé pour tracer le compromis rappel / fausses alertes.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from zonefleet.alerting import LAG, LEAD, PSI_ALERT, persistent
from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.features import zone_table
from zonefleet.pipeline import CACHE, TABLES, frozen_model, predictions
from zonefleet.rowscore import RowScorer
from zonefleet.scenarios import level_shock, subset_mask

SPLIT = "congestion"
WINDOW_DAYS = 28
HARM = 0.10
PRE_START = pd.Timestamp("2025-01-06")
ONSET_RANGE = (pd.Timestamp("2025-02-03"), pd.Timestamp("2025-05-15"))
KINDS = ("none", "covariate_harmful", "covariate_benign", "concept", "level")
KIND_P = (0.2, 0.2, 0.2, 0.2, 0.2)
SUBSETS = ("all", "cbd", "hors_cbd", "evening", "night")
K_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SMOOTH = (1, 7)  # signal journalier brut, ou moyenne glissante sur 7 jours
PERSISTENCE = 2


def draw_scenarios(n: int, seed: int = 2026) -> pd.DataFrame:
    """Tire n scénarios au hasard (graine fixée)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(*ONSET_RANGE, freq="D")
    rows = []
    for i in range(n):
        kind = rng.choice(KINDS, p=KIND_P)
        sev = {
            "none": 0.0,
            "covariate_harmful": rng.uniform(0.25, 2.0),
            "covariate_benign": rng.uniform(0.25, 2.0),
            "concept": rng.uniform(0.5, 0.95),
            "level": rng.uniform(0.5, 0.95),
        }[kind]
        subset = rng.choice(SUBSETS) if kind in ("concept", "level") else "all"
        rows.append(
            {
                "id": i,
                "kind": kind,
                "severity": round(float(sev), 3),
                "subset": subset,
                "onset": days[rng.integers(len(days))],
            }
        )
    return pd.DataFrame(rows)


def resample_rows(rows: pd.DataFrame, onset, kappa, direction, stats, seed) -> pd.DataFrame:
    """Covariate shift sur la table des scores par ligne (même logique que scenarios.py)."""
    rng = np.random.default_rng(seed)
    mu, sd = stats
    parts = [rows[rows["date"] < onset]]
    after = rows[rows["date"] >= onset]
    for _, day in after.groupby("date"):
        w = np.exp(direction * kappa * (day["lag_168"].to_numpy() - mu) / sd)
        idx = rng.choice(len(day), size=len(day), replace=True, p=w / w.sum())
        parts.append(day.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def scenario_daily(sc, scorer, base_rows, analysis, level_inputs, stats) -> pd.DataFrame:
    """Signaux journaliers d'un scénario."""
    if sc.kind == "none":
        return scorer.aggregate(base_rows)
    if sc.kind.startswith("covariate"):
        direction = -1 if sc.kind == "covariate_harmful" else +1
        return scorer.aggregate(
            resample_rows(base_rows, sc.onset, sc.severity, direction, stats, seed=sc.id)
        )
    if sc.kind == "concept":
        mask = ((analysis["ts"] >= sc.onset) & subset_mask(analysis, sc.subset)).to_numpy()
        y = analysis["pickups"].to_numpy() * np.where(mask, sc.severity, 1.0)
        rows = base_rows.copy()
        rows["real"] = np.abs(np.log1p(y) - np.log1p(analysis["q50"].to_numpy()))
        return scorer.aggregate(rows)
    an = level_shock(onset=sc.onset, factor=sc.severity, subset=sc.subset, **level_inputs)
    return scorer.daily(an)


def smooth(daily: pd.DataFrame, w: int) -> pd.DataFrame:
    """Moyenne glissante (passée) de l'erreur estimée et réalisée ; dérive inchangée."""
    if w == 1:
        return daily
    out = daily.copy()
    for c in ("est", "real"):
        out[c] = daily[c].rolling(w, min_periods=w).mean()
    return out


def thresholds(ref_daily: pd.DataFrame, k: float) -> dict[str, float]:
    m, s = ref_daily.mean(), ref_daily.std()
    return {
        "est": m["est"] + k * s["est"],
        "real": m["real"] + k * s["real"],
        "recon": m["recon"] + 3 * s["recon"],  # seuil NannyML par défaut, non balayé
    }


def policy_flags(daily: pd.DataFrame, thr: dict[str, float]) -> pd.DataFrame:
    psi_all = daily[[c for c in daily if c.startswith("psi_")]].max(axis=1)
    psi_clean = daily[["psi_lag_24", "psi_lag_168"]].max(axis=1)
    pca = daily["recon"] > thr["recon"]
    drift, clean = pca | (psi_all > PSI_ALERT), pca | (psi_clean > PSI_ALERT)
    est, real = daily["est"] > thr["est"], daily["real"] > thr["real"]
    raw = {
        "P1_drift": drift,
        "P1b_drift_clean": clean,
        "P2_dle": est,
        "P3_drift_and_dle": drift & est,
        "P3b_clean_and_dle": clean & est,
        "P4_realized": real,
    }
    return pd.DataFrame({k: persistent(v, PERSISTENCE) for k, v in raw.items()})


def rising(flag: pd.Series) -> pd.Series:
    """Jours où une alerte s'allume (front montant) : une alarme déjà allumée ne détecte rien."""
    f = flag.astype(bool)
    return f & ~f.shift(1, fill_value=False)


def harm_of(sc, raw: pd.DataFrame, base_raw: pd.DataFrame) -> float:
    """Hausse relative de la MAE log réalisée sur 28 jours, contre le contrefactuel."""
    win = (raw.index >= sc.onset) & (raw.index < sc.onset + pd.Timedelta(days=WINDOW_DAYS))
    return raw.loc[win, "real"].mean() / base_raw.loc[win, "real"].mean() - 1


def evaluate(sc, daily, harm, flags) -> list[dict]:
    """Une ligne par politique : détection, délai, fausses alertes."""
    onset = sc.onset
    win = (daily.index >= onset) & (daily.index < onset + pd.Timedelta(days=WINDOW_DAYS))
    null = (daily.index >= PRE_START) & ((daily.index < onset) | (sc.kind == "none"))
    out = []
    for pol in flags.columns:
        f = flags[pol]
        events = rising(f)
        hits = f.index[win & events.to_numpy()]
        if len(hits):
            dec = hits[0] + LAG if pol == "P4_realized" else hits[0] - LEAD
            delay = (dec - onset) / pd.Timedelta(days=1)
        else:
            delay = np.nan
        out.append(
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
                "delay_days": delay,
                "in_alert_at_onset": bool(f[daily.index < onset].iloc[-1])
                if (daily.index < onset).any()
                else False,
                "false_alert_events": int(events[null].sum()),
                "false_alert_days": int(f[null].sum()),
                "null_days": int(null.sum()),
            }
        )
    return out


def summarize(res: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Rappel, délai médian, fausses alertes (événements / 30 j et part de jours),
    alertes sur dérive non nuisible ; intervalles bootstrap 90 % sur les scénarios."""
    rng = np.random.default_rng(seed)
    rows = []

    def stats(sub):
        harmful = sub[sub["harmful"]]
        benign = sub[~sub["harmful"] & (sub["kind"] != "none")]
        delays = harmful.loc[harmful["detected"], "delay_days"]
        return (
            harmful["detected"].mean(),
            delays.median() if len(delays) else np.nan,
            30 * sub["false_alert_events"].sum() / sub["null_days"].sum(),
            sub["false_alert_days"].sum() / sub["null_days"].sum(),
            benign["detected"].mean(),
        )

    for (w, k, pol), d in res.groupby(["smooth", "k", "policy"]):
        ids = d["id"].unique()
        by_id = {i: g for i, g in d.groupby("id")}
        point = stats(d)
        boots = np.array(
            [
                stats(pd.concat([by_id[i] for i in rng.choice(ids, size=len(ids))]))
                for _ in range(n_boot)
            ],
            dtype=float,
        )
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(boots, 5, axis=0)
            hi = np.nanpercentile(boots, 95, axis=0)
        rows.append(
            {
                "smooth": w,
                "k": k,
                "policy": pol,
                "recall": point[0],
                "recall_lo": lo[0],
                "recall_hi": hi[0],
                "median_delay": point[1],
                "delay_lo": lo[1],
                "delay_hi": hi[1],
                "false_alerts_per_30d": point[2],
                "fa_lo": lo[2],
                "fa_hi": hi[2],
                "share_null_days_in_alert": point[3],
                "benign_alert_rate": point[4],
                "n_harmful": int(d.loc[d["harmful"], "id"].nunique()),
                "n_scenarios": int(len(ids)),
            }
        )
    return pd.DataFrame(rows)


def main(n: int = 240) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pred = predictions(SPLIT)
    ref, an = pred[pred.period == "reference"], pred[pred.period == "analysis"]
    an = an.reset_index(drop=True)
    scorer = RowScorer().fit(ref)
    base_rows = scorer.score_rows(an)
    base = scorer.aggregate(base_rows)
    stats = (ref["lag_168"].mean(), ref["lag_168"].std())
    start, end = SPLITS[SPLIT].period("analysis")
    level_inputs = {
        "panel": load_panel(),
        "weather": load_weather(),
        "model": frozen_model(SPLIT),
        "start": start,
        "end": end,
        "zones": zone_table(),
    }
    scen = draw_scenarios(n)
    TABLES.mkdir(parents=True, exist_ok=True)
    scen.to_csv(TABLES / "e4_scenarios.csv", index=False)
    dailies = {}
    for sc in scen.itertuples():
        dailies[sc.id] = scenario_daily(sc, scorer, base_rows, an, level_inputs, stats)
    pd.concat(dailies, names=["id", "date"]).to_parquet(CACHE / "e4_daily.parquet")
    results, thr_rows = [], []
    harms = {sc.id: harm_of(sc, dailies[sc.id], base) for sc in scen.itertuples()}
    for w in SMOOTH:
        ref_w = smooth(scorer.reference_daily, w)
        for k in K_GRID:
            thr = thresholds(ref_w, k)
            thr_rows.append({**thr, "k": k, "smooth": w})
            for sc in scen.itertuples():
                d = smooth(dailies[sc.id], w)
                for r in evaluate(sc, d, harms[sc.id], policy_flags(d, thr)):
                    results.append({**r, "k": k, "smooth": w})
    res = pd.DataFrame(results)
    res.to_csv(TABLES / "e4_results.csv", index=False)
    summ = summarize(res)
    summ.round(4).to_csv(TABLES / "e4_summary.csv", index=False)
    pd.DataFrame(thr_rows).round(5).to_csv(TABLES / "e4_thresholds.csv", index=False)
    print(scen["kind"].value_counts().to_string())
    print(res.drop_duplicates("id")["harmful"].groupby(res.drop_duplicates("id")["kind"]).mean())
    cols = [
        "smooth",
        "k",
        "policy",
        "recall",
        "median_delay",
        "false_alerts_per_30d",
        "share_null_days_in_alert",
        "benign_alert_rate",
    ]
    print(summ[summ["k"].isin([1.0, 2.0])][cols].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    main(ap.parse_args().n)
