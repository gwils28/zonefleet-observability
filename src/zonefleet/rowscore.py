"""Scores par ligne réutilisant les modèles ajustés par NannyML (E4 à grande échelle).

NannyML agrège par chunk : la MAE estimée d'un chunk est la moyenne des pertes prédites
par le nanny model, bornées à 0 ; l'erreur de reconstruction d'un chunk est la moyenne
des distances par ligne. Ces scores par ligne ne dépendent que de la ligne. Pour un
covariate shift (rééchantillonnage) ou un concept drift (cible modifiée, X gelé), on
peut donc les calculer **une fois** puis agréger chaque scénario en millisecondes.

Le prétraitement reproduit exactement celui de NannyML 0.13 (imputation et encodage
ordinal des catégorielles, +1) ; `tests/test_rowscore.py` vérifie l'égalité avec les
sorties officielles chunk par chunk.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")

import nannyml as nml  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zonefleet.monitoring import DLE_FEATURES, DRIFT_FEATURES, TS, prepare, psi_by_day  # noqa: E402

logging.getLogger("nannyml").setLevel(logging.WARNING)

PSI_COLUMNS = ("lag_24", "lag_168", "roll_mean_168", "temperature_2m")


class RowScorer:
    """Ajuste DLE (MAE, échelle log) et la PCA sur une référence, puis score des lignes."""

    def __init__(self, seed: int = 0, n_components: float = 0.65):
        self.seed = seed
        self.n_components = n_components

    def fit(self, reference: pd.DataFrame) -> RowScorer:
        ref = prepare(reference, "log")
        self.dle = nml.DLE(
            feature_column_names=DLE_FEATURES,
            y_pred="y_pred",
            y_true="y_true",
            timestamp_column_name=TS,
            metrics=["mae"],
            chunk_period="D",
            tune_hyperparameters=False,
            hyperparameters={"random_state": self.seed, "verbosity": -1},
        ).fit(ref[[TS, *DLE_FEATURES, "y_pred", "y_true"]])
        self.pca = nml.DataReconstructionDriftCalculator(
            column_names=DRIFT_FEATURES,
            timestamp_column_name=TS,
            chunk_period="D",
            n_components=self.n_components,
        ).fit(reference[[TS, *DRIFT_FEATURES]])
        self.ref_psi = {c: reference[c].to_numpy() for c in PSI_COLUMNS}
        self.reference_daily = self.daily(reference)
        return self

    # ------------------------------------------------------------------ par ligne
    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df[[*DLE_FEATURES, "y_pred"]].copy()
        col = "zone"
        values = data[col].astype("object").to_numpy().reshape(-1, 1)
        values = self.dle._categorical_imputer.transform(values)
        data[col] = self.dle._categorical_encoders[col].transform(values).ravel() + 1
        return data

    def row_loss(self, df: pd.DataFrame) -> np.ndarray:
        """Perte estimée par ligne (MAE sur log1p), sans la vérité terrain."""
        d = prepare(df, "log")
        metric = self.dle.metrics[0]
        X = self._encode(d)[metric.feature_column_names + [metric.y_pred]]
        return np.maximum(0, metric._dee_model.predict(X))

    def row_recon(self, df: pd.DataFrame) -> np.ndarray:
        """Erreur de reconstruction PCA par ligne."""
        c = self.pca
        data = df[c.column_names].copy()
        if c.continuous_column_names:
            data[c.continuous_column_names] = c._imputer_continuous.transform(
                data[c.continuous_column_names]
            )
        z = c._scaler.transform(c._encoder.transform(data[c.column_names]).to_numpy())
        rec = c._pca.inverse_transform(c._pca.transform(z))
        return np.linalg.norm(z - rec, axis=1)

    @staticmethod
    def row_realized(df: pd.DataFrame) -> np.ndarray:
        """Erreur réalisée par ligne (MAE sur log1p) ; exige la vérité terrain."""
        return np.abs(np.log1p(df["pickups"].to_numpy()) - np.log1p(df["q50"].to_numpy()))

    def score_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tous les scores par ligne, plus la date du chunk."""
        return pd.DataFrame(
            {
                "date": df[TS].dt.floor("D").to_numpy(),
                "est": self.row_loss(df),
                "recon": self.row_recon(df),
                "real": self.row_realized(df),
                **{c: df[c].to_numpy() for c in PSI_COLUMNS},
            }
        )

    # ------------------------------------------------------------------ par jour
    def aggregate(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Signaux journaliers à partir de scores par ligne (éventuellement rééchantillonnés)."""
        g = rows.groupby("date")
        out = g[["est", "recon", "real"]].mean()
        for c in ("lag_24", "lag_168"):
            out[f"mean_{c}"] = g[c].mean()  # pour la dérive saisonnière (E4 quater)
        days = rows["date"].to_numpy()
        for c in PSI_COLUMNS:
            out[f"psi_{c}"] = (
                psi_by_day(self.ref_psi[c], rows[c], days).reindex(out.index).to_numpy()
            )
        return out

    def daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(self.score_rows(df))
