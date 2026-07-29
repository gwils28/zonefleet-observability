"""Briques de monitoring : dérive des entrées et des prédictions, performance estimée
sans labels (NannyML DLE) et performance réalisée.

Toutes les fonctions renvoient des tables « longues » à une ligne par chunk (jour), avec
une colonne `date`, pour que les politiques d'alerte (alerting.py) les combinent sans
connaître NannyML.

Deux échelles de performance sont surveillées :
- `count` : erreur en nombre de courses (lecture métier, mais dépend du niveau) ;
- `log` : erreur sur log1p (≈ erreur relative), stable quand le volume change d'échelle.
"""

from __future__ import annotations

import logging
import os

# NannyML envoie par défaut des statistiques d'usage à un service tiers : désactivé.
os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")

import nannyml as nml  # noqa: E402
import numpy as np
import pandas as pd

from zonefleet.features import CALENDAR, LAG_COLS, ROLL_COLS, WEATHER

logging.getLogger("nannyml").setLevel(logging.WARNING)

DRIFT_FEATURES = [*LAG_COLS, *ROLL_COLS, *CALENDAR, *WEATHER]
DLE_FEATURES = ["zone", *DRIFT_FEATURES]
TS = "ts"
CHUNK = "D"


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatit les colonnes multi-index NannyML en `niveau1__niveau2`."""
    out = df.copy()
    out.columns = ["__".join(str(c) for c in col if c != "chunk") or "chunk" for col in df.columns]
    return out


def prepare(df: pd.DataFrame, scale: str = "count") -> pd.DataFrame:
    """Colonnes cible/prédiction selon l'échelle (`y_true`, `y_pred`)."""
    out = df.copy()
    if scale == "count":
        out["y_true"], out["y_pred"] = out["pickups"].astype(float), out["q50"]
    elif scale == "log":
        out["y_true"], out["y_pred"] = np.log1p(out["pickups"]), np.log1p(out["q50"])
    else:
        raise ValueError(f"échelle inconnue : {scale}")
    return out


def _dates(flat: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(flat["start_date"]).dt.normalize()


def estimate_performance(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    scale: str = "count",
    metrics: tuple[str, ...] = ("mae", "rmse"),
    chunk_period: str = CHUNK,
    seed: int = 0,
) -> pd.DataFrame:
    """Performance estimée par DLE sur l'analysis, **sans lire** `y_true` de l'analysis.

    DLE entraîne un « nanny model » sur la référence pour prédire la perte de chaque
    observation à partir des features et de la prédiction, puis agrège par chunk.
    """
    ref, an = prepare(reference, scale), prepare(analysis, scale)
    est = nml.DLE(
        feature_column_names=DLE_FEATURES,
        y_pred="y_pred",
        y_true="y_true",
        timestamp_column_name=TS,
        metrics=list(metrics),
        chunk_period=chunk_period,
        tune_hyperparameters=False,
        hyperparameters={"random_state": seed, "verbosity": -1},
    )
    est.fit(ref[[TS, *DLE_FEATURES, "y_pred", "y_true"]])
    blind = an[[TS, *DLE_FEATURES, "y_pred"]]  # la vérité terrain n'est pas transmise
    flat = flatten(est.estimate(blind).filter(period="analysis").to_df())
    rows = []
    for m in metrics:
        rows.append(
            pd.DataFrame(
                {
                    "date": _dates(flat),
                    "metric": f"{m}_{scale}",
                    "estimated": flat[f"{m}__value"],
                    "lower": flat[f"{m}__lower_confidence_boundary"],
                    "upper": flat[f"{m}__upper_confidence_boundary"],
                    "threshold": flat[f"{m}__upper_threshold"],
                    "alert": flat[f"{m}__alert"].astype(bool),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def realized_performance(df: pd.DataFrame, scale: str = "count", freq: str = CHUNK) -> pd.DataFrame:
    """MAE et RMSE réalisées par chunk, une fois les labels révélés."""
    d = prepare(df, scale)
    err = d["y_true"] - d["y_pred"]
    key = d[TS].dt.floor(freq)
    g = pd.DataFrame({"abs": err.abs(), "sq": err**2}).groupby(key)
    out = pd.DataFrame({"mae": g["abs"].mean(), "rmse": np.sqrt(g["sq"].mean())})
    out = (
        out.rename_axis("date").reset_index().melt("date", var_name="metric", value_name="realized")
    )
    out["metric"] = out["metric"] + f"_{scale}"
    return out


def univariate_drift(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    columns: list[str] | None = None,
    methods: tuple[str, ...] = ("kolmogorov_smirnov", "wasserstein"),
    chunk_period: str = CHUNK,
) -> pd.DataFrame:
    """Dérive univariée (KS, Wasserstein) par colonne et par chunk."""
    columns = columns or DRIFT_FEATURES
    calc = nml.UnivariateDriftCalculator(
        column_names=columns,
        treat_as_numerical=columns,
        timestamp_column_name=TS,
        continuous_methods=list(methods),
        chunk_period=chunk_period,
    )
    calc.fit(reference[[TS, *columns]])
    res = calc.calculate(analysis[[TS, *columns]]).filter(period="analysis").to_df()
    dates = pd.to_datetime(res[("chunk", "chunk", "start_date")]).dt.normalize().to_numpy()
    rows = []
    for col in columns:
        for m in methods:
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "feature": col,
                        "method": m,
                        "value": res[(col, m, "value")].to_numpy(),
                        "threshold": res[(col, m, "upper_threshold")].to_numpy(),
                        "alert": res[(col, m, "alert")].astype(bool).to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def multivariate_drift(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    columns: list[str] | None = None,
    chunk_period: str = CHUNK,
    n_components: float = 0.65,
) -> pd.DataFrame:
    """Erreur de reconstruction PCA par chunk (NannyML DataReconstructionDriftCalculator).

    La PCA, ajustée sur la référence standardisée, garde `n_components` de la variance ;
    une hausse de l'erreur de reconstruction signale que la structure jointe des entrées
    a changé, même si chaque marginale bouge peu.
    """
    columns = columns or DRIFT_FEATURES
    calc = nml.DataReconstructionDriftCalculator(
        column_names=columns,
        timestamp_column_name=TS,
        chunk_period=chunk_period,
        n_components=n_components,
    )
    calc.fit(reference[[TS, *columns]])
    flat = flatten(calc.calculate(analysis[[TS, *columns]]).filter(period="analysis").to_df())
    return pd.DataFrame(
        {
            "date": _dates(flat),
            "value": flat["reconstruction_error__value"],
            "threshold": flat["reconstruction_error__upper_threshold"],
            "alert": flat["reconstruction_error__alert"].astype(bool),
        }
    )


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index, bacs définis par les quantiles de la référence.

    Repères usuels : < 0,1 stable ; 0,1 à 0,25 dérive modérée ; > 0,25 dérive forte.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / max(len(current), 1)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def daily_psi(
    reference: pd.DataFrame, analysis: pd.DataFrame, columns: list[str], bins: int = 10
) -> pd.DataFrame:
    """PSI journalier de chaque colonne contre la référence entière."""
    rows = []
    for day, d in analysis.groupby(analysis[TS].dt.floor(CHUNK)):
        for col in columns:
            rows.append({"date": day, "feature": col, "psi": psi(reference[col], d[col], bins)})
    return pd.DataFrame(rows)


def prediction_drift(reference: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    """Dérive de la distribution des prévisions (médiane, échelle log) : KS et Wasserstein."""
    ref = reference.assign(pred_log=np.log1p(reference["q50"]))
    an = analysis.assign(pred_log=np.log1p(analysis["q50"]))
    return univariate_drift(ref, an, columns=["pred_log"])


def monitor(
    reference: pd.DataFrame, analysis: pd.DataFrame, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Exécute toutes les briques et renvoie un tableau par signal."""
    est = pd.concat(
        [estimate_performance(reference, analysis, s, seed=seed) for s in ("count", "log")],
        ignore_index=True,
    )
    real = pd.concat([realized_performance(analysis, s) for s in ("count", "log")])
    perf = est.merge(real, on=["date", "metric"], how="left")
    return {
        "performance": perf,
        "univariate": univariate_drift(reference, analysis),
        "multivariate": multivariate_drift(reference, analysis),
        "prediction": prediction_drift(reference, analysis),
        "psi": daily_psi(
            reference, analysis, ["lag_24", "lag_168", "roll_mean_168", "temperature_2m"]
        ),
    }


def seasonal_psi(
    history: pd.DataFrame,
    analysis: pd.DataFrame,
    columns: list[str],
    half_window_days: int = 14,
    lag_days: int = 364,
    bins: int = 10,
) -> pd.DataFrame:
    """PSI journalier contre une base saisonnière : les 2 x `half_window_days` jours centrés
    sur la même date `lag_days` plus tôt (364 = même jour de semaine un an avant).

    `history` doit contenir l'année précédente ; une date sans base complète est ignorée.
    """
    hist_day = history[TS].dt.floor(CHUNK)
    rows = []
    for day, d in analysis.groupby(analysis[TS].dt.floor(CHUNK)):
        center = day - pd.Timedelta(days=lag_days)
        lo, hi = (
            center - pd.Timedelta(days=half_window_days),
            center + pd.Timedelta(days=half_window_days),
        )
        base = history[(hist_day >= lo) & (hist_day < hi)]
        if base.empty:
            continue
        for col in columns:
            rows.append({"date": day, "feature": col, "psi": psi(base[col], d[col], bins)})
    return pd.DataFrame(rows)


def seasonal_zscore(
    history: pd.DataFrame,
    analysis: pd.DataFrame,
    columns: list[str],
    half_window_days: int = 14,
    lag_days: int = 364,
) -> pd.DataFrame:
    """Écart standardisé de la moyenne du jour à la base saisonnière des moyennes journalières.

    Corrige le défaut du PSI journalier contre une fenêtre longue (distributions de
    largeurs incomparables) : on compare une moyenne journalière à des moyennes
    journalières. |z| > 3 signale un jour hors de la variabilité de la même saison.
    """
    hist_daily = history.groupby(history[TS].dt.floor(CHUNK))[columns].mean()
    an_daily = analysis.groupby(analysis[TS].dt.floor(CHUNK))[columns].mean()
    rows = []
    for day, vals in an_daily.iterrows():
        center = day - pd.Timedelta(days=lag_days)
        base = hist_daily[
            (hist_daily.index >= center - pd.Timedelta(days=half_window_days))
            & (hist_daily.index < center + pd.Timedelta(days=half_window_days))
        ]
        if len(base) < half_window_days:
            continue
        for col in columns:
            sd = base[col].std()
            z = (vals[col] - base[col].mean()) / sd if sd > 0 else 0.0
            rows.append({"date": day, "feature": col, "z": float(z)})
    return pd.DataFrame(rows)
