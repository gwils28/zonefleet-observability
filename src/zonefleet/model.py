"""Le modèle surveillé : LightGBM global à trois quantiles, puis figé.

Une fois entraîné, l'objet est sérialisé et n'est plus modifié : c'est lui que le
monitoring observe. Les hyperparamètres sont fixés a priori (pas de recherche), parce
que la question du projet n'est pas la précision mais l'observabilité.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from zonefleet.config import QUANTILES
from zonefleet.features import FEATURES, TARGET

PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass
class QuantileForecaster:
    """Trois LightGBM quantiles sur log1p(demande) ; prévisions renvoyées en courses."""

    quantiles: tuple[float, ...] = QUANTILES
    params: dict = field(default_factory=lambda: dict(PARAMS))
    seed: int = 0
    models: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)
    trained_on: tuple[str, str] | None = None

    def fit(self, df: pd.DataFrame) -> QuantileForecaster:
        X, y = df[FEATURES], df[TARGET]
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q, random_state=self.seed, **self.params
            )
            m.fit(X, y, categorical_feature=["zone"])
            self.models[q] = m
        self.trained_on = (str(df["ts"].min()), str(df["ts"].max()))
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Quantiles en échelle log1p, triés par ligne (pas de croisement de quantiles)."""
        preds = np.column_stack([self.models[q].predict(df[FEATURES]) for q in self.quantiles])
        return np.sort(preds, axis=1)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quantiles en nombre de courses (q10, q50, q90), bornés à 0."""
        p = np.clip(np.expm1(self.predict_log(df)), 0, None)
        cols = [f"q{int(round(q * 100))}" for q in self.quantiles]
        return pd.DataFrame(p, columns=cols, index=df.index)

    def feature_importance(self, q: float = 0.5) -> pd.Series:
        """Gain par feature pour le modèle du quantile q."""
        m = self.models[q]
        gain = m.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=FEATURES).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> QuantileForecaster:
        return pickle.loads(path.read_bytes())


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Baseline : même heure la semaine précédente (lag 168 h), en courses."""
    return np.expm1(df["lag_168"].to_numpy())
