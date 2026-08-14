"""E5 : politique de réentraînement après un choc (COVID 2020).

Usage : uv run python -m zonefleet.experiments.e5_retraining
Sorties : results/tables/e5_*.csv

Le modèle champion est celui de E1 (entraîné janv.-août 2019). On rejoue l'année 2020
jour par jour sous cinq politiques, pour deux délais de labels :
- L = 60 jours (publication TLC, le cas du projet) ;
- L = 1 jour (journaux de courses propres à l'opérateur : le cas ZoneFleet réaliste).

Politiques :
- `jamais` : le modèle de 2019 reste en place ;
- `calendrier` : réentraînement le 1er de chaque mois sur les 8 derniers mois de labels
  disponibles (pratique courante « tous les mois ») ;
- `réflexe` : réentraînement dès la première alerte P3 de E1 (décision le 22 mars), sur
  les 8 derniers mois de labels disponibles ;
- `réflexe court` : à la même date, réentraînement sur les seuls labels postérieurs au
  début du choc (« coller au nouveau régime » avec très peu de données) ;
- `gardé` : dérive confirmée (alerte P3) ET au moins 28 jours de labels postérieurs à
  l'alerte ; un challenger est alors entraîné chaque semaine et ne remplace le champion
  que s'il fait mieux de 5 % sur les 14 derniers jours de labels disponibles.

Garde-fou commun : aucun modèle n'est entraîné sur des labels non encore disponibles à la
date de décision (horloge des labels, `available_until`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from zonefleet.features import slice_period
from zonefleet.model import QuantileForecaster
from zonefleet.pipeline import TABLES, features, frozen_model

YEAR = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))
SHOCK = pd.Timestamp("2020-03-13")
FIRST_ALERT = pd.Timestamp("2020-03-22")  # décision P3 de E1 (results/tables/e1_detection.csv)
TRAIN_MONTHS = 8
MIN_POST_LABEL_DAYS = 28
VALIDATION_DAYS = 14
PROMOTION_MARGIN = 0.05
DELAYS = (1, 60)
FAST_PARAMS = {"n_estimators": 300}  # réentraînements : un peu plus léger que le modèle initial


def available_until(decision: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Dernier instant (exclu) dont le label est connu à la date de décision."""
    return decision - pd.Timedelta(days=delay_days)


def train_window(
    df: pd.DataFrame,
    decision: pd.Timestamp,
    delay: int,
    months: int = TRAIN_MONTHS,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Lignes d'entraînement disponibles à la décision : [fin - months, fin)."""
    end = available_until(decision, delay)
    start = start if start is not None else end - pd.DateOffset(months=months)
    return slice_period(df, start, end)


def fit(df: pd.DataFrame) -> QuantileForecaster:
    from zonefleet.model import PARAMS

    return QuantileForecaster(params={**PARAMS, **FAST_PARAMS}).fit(df)


def mae_log(df: pd.DataFrame, model: QuantileForecaster) -> float:
    q50 = model.predict(df)["q50"]
    return float(np.mean(np.abs(np.log1p(df["pickups"]) - np.log1p(q50))))


@dataclass
class Run:
    """Historique d'une politique : modèles successifs et leurs dates de mise en service."""

    name: str
    delay: int
    switches: list[tuple[pd.Timestamp, QuantileForecaster, str]] = field(default_factory=list)


def simulate(name: str, delay: int, df: pd.DataFrame, champion: QuantileForecaster) -> Run:
    run = Run(name, delay, [(YEAR[0], champion, "modèle 2019")])
    if name == "jamais":
        return run
    if name == "calendrier":
        for month in pd.date_range("2020-02-01", YEAR[1] - pd.Timedelta(days=1), freq="MS"):
            run.switches.append((month, fit(train_window(df, month, delay)), "mensuel"))
        return run
    if name == "réflexe":
        run.switches.append((FIRST_ALERT, fit(train_window(df, FIRST_ALERT, delay)), "alerte"))
        return run
    if name == "réflexe court":
        tr = train_window(df, FIRST_ALERT, delay, start=SHOCK)
        if len(tr) == 0:  # L = 60 : aucun label post-choc disponible, rien à apprendre
            return run
        run.switches.append(
            (FIRST_ALERT, fit(tr), f"{tr['ts'].dt.normalize().nunique()} j de labels")
        )
        return run
    if name == "gardé":
        current = champion
        earliest = FIRST_ALERT + pd.Timedelta(days=MIN_POST_LABEL_DAYS + delay)
        for week in pd.date_range(earliest, YEAR[1], freq="7D", inclusive="left"):
            end = available_until(week, delay)
            val = slice_period(df, end - pd.Timedelta(days=VALIDATION_DAYS), end)
            tr = train_window(df, week - pd.Timedelta(days=VALIDATION_DAYS), delay)
            challenger = fit(tr)
            champ_err, chall_err = mae_log(val, current), mae_log(val, challenger)
            if chall_err < (1 - PROMOTION_MARGIN) * champ_err:
                current = fit(train_window(df, week, delay))  # réentraîné jusqu'au bout
                run.switches.append((week, current, f"promu ({chall_err:.3f} < {champ_err:.3f})"))
        return run
    raise ValueError(name)


def realized(run: Run, df: pd.DataFrame) -> pd.DataFrame:
    """Erreur réalisée jour par jour du modèle en service selon la politique."""
    year = slice_period(df, *YEAR)
    parts = []
    bounds = [s[0] for s in run.switches[1:]] + [YEAR[1]]
    for (start, model, _), end in zip(run.switches, bounds, strict=True):
        sub = year[(year["ts"] >= start) & (year["ts"] < end)]
        if len(sub):
            q = model.predict(sub)
            parts.append(sub[["ts", "zone", "pickups"]].assign(q50=q["q50"].to_numpy()))
    d = pd.concat(parts)
    d["err_log"] = np.abs(np.log1p(d["pickups"]) - np.log1p(d["q50"]))
    d["err"] = np.abs(d["pickups"] - d["q50"])
    daily = d.groupby(d["ts"].dt.floor("D")).agg(mae_log=("err_log", "mean"), mae=("err", "mean"))
    return daily.rename_axis("date").reset_index().assign(policy=run.name, delay=run.delay)


PERIODS = {
    "choc (13 mars-12 avril)": (SHOCK, pd.Timestamp("2020-04-13")),
    "printemps (13 avril-30 juin)": (pd.Timestamp("2020-04-13"), pd.Timestamp("2020-07-01")),
    "second semestre": (pd.Timestamp("2020-07-01"), pd.Timestamp("2021-01-01")),
    "13 mars-31 déc.": (SHOCK, pd.Timestamp("2021-01-01")),
    "2021, 1er semestre": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-07-01")),
    "2021, 2nd semestre": (pd.Timestamp("2021-07-01"), pd.Timestamp("2022-01-01")),
    "2021": (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")),
}


def summarize(daily: pd.DataFrame, switches: pd.DataFrame) -> pd.DataFrame:
    """MAE log moyenne par période (bornes fixes) et nombre de remplacements par politique.

    Les périodes non couvertes par la simulation sont omises.
    """
    end = daily["date"].max()
    periods = {k: v for k, v in PERIODS.items() if v[0] <= end}
    rows = []
    for (pol, delay), d in daily.groupby(["policy", "delay"], sort=False):
        row = {"policy": pol, "delay": delay}
        for pname, (a, b) in periods.items():
            m = (d["date"] >= a) & (d["date"] < b)
            row[pname] = d.loc[m, "mae_log"].mean()
        sw = (
            switches[(switches["policy"] == pol) & (switches["delay"] == delay)]
            if len(switches)
            else switches
        )
        row["retrains"] = len(sw)
        rows.append(row)
    return pd.DataFrame(rows)


def main(end: str = "2021-01-01", suffix: str = "") -> None:
    """Rejoue l'année 2020 (défaut) ou jusqu'à `end` ; `suffix` distingue les sorties."""
    global YEAR
    YEAR = (YEAR[0], pd.Timestamp(end))
    df = features()
    champion = frozen_model("covid")
    policies = ["jamais", "calendrier", "réflexe", "réflexe court", "gardé"]
    parts: list[pd.DataFrame] = []
    switches: list[dict] = []
    for delay in DELAYS:
        for name in policies:
            run = simulate(name, delay, df, champion)
            parts.append(realized(run, df))
            switches += [
                {"policy": name, "delay": delay, "date": d, "note": note}
                for d, _, note in run.switches[1:]
            ]
            print(f"L={delay:>2} {name:14s} {len(run.switches) - 1} remplacement(s)")
    daily = pd.concat(parts, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    daily.round(5).to_csv(TABLES / f"e5{suffix}_daily.csv", index=False)
    pd.DataFrame(switches).to_csv(TABLES / f"e5{suffix}_switches.csv", index=False)
    summary = summarize(daily, pd.DataFrame(switches))
    summary.round(4).to_csv(TABLES / f"e5{suffix}_summary.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jusqua", default="2021-01-01", help="fin exclue de la simulation")
    ap.add_argument("--suffixe", default="")
    a = ap.parse_args()
    main(a.jusqua, a.suffixe)
