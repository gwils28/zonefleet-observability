"""E4 bis (sensibilité) et E4 ter (périodes calmes), sur signaux synthétiques et monde jouet."""

import numpy as np
import pandas as pd
import pytest

DAYS = pd.date_range("2025-01-01", "2025-06-30", freq="D")
ONSET = pd.Timestamp("2025-03-03")


def _signals(seed: int, shift: float = 0.0, real_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    after = (DAYS >= ONSET).astype(float)
    n = len(DAYS)
    return pd.DataFrame(
        {
            "est": 0.30 + 0.01 * rng.normal(size=n) + shift * after,
            "recon": 2.0 + 0.05 * rng.normal(size=n),
            "real": 0.30 + 0.01 * rng.normal(size=n) + real_shift * after,
            "mean_lag_24": 3.0,
            "mean_lag_168": 3.0,
            "psi_lag_24": 0.05 + 0.5 * shift / 0.1 * after,
            "psi_lag_168": 0.05,
            "psi_roll_mean_168": 0.05,
            "psi_temperature_2m": 0.05,
        },
        index=pd.Index(DAYS, name="date"),
    )


@pytest.fixture
def synthetic_e4(tmp_path, monkeypatch):
    from zonefleet.experiments import e4_sensitivity as E

    scen = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "kind": ["none", "level", "covariate_benign"],
            "severity": [0.0, 0.9, 0.5],
            "subset": ["all", "all", "all"],
            "onset": [ONSET] * 3,
        }
    )
    frames = {
        0: _signals(0),
        1: _signals(1, shift=0.1, real_shift=0.05),  # +17 % d'erreur réalisée
        2: _signals(2),
    }
    daily = pd.concat(frames, names=["id"])
    (tmp_path / "cache").mkdir()
    (tmp_path / "tables").mkdir()
    daily.to_parquet(tmp_path / "cache" / "e4_daily.parquet")
    scen.to_csv(tmp_path / "tables" / "e4_scenarios.csv", index=False)

    class FakeScorer:
        def fit(self, ref):
            self.reference_daily = _signals(9).iloc[:60]
            return self

    monkeypatch.setattr(E, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(E, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(E, "RowScorer", FakeScorer)
    monkeypatch.setattr(E, "predictions", lambda split: pd.DataFrame({"period": ["reference"]}))
    return E, tmp_path / "tables"


def test_e4_sensitivity_writes_every_variant(synthetic_e4):
    E, tables = synthetic_e4
    E.main()
    out = pd.read_csv(tables / "e4_sensitivity.csv")
    assert set(zip(out["parameter"], out["value"], strict=True)) == {
        (p, v) for p, v, _ in E.VARIANTS
    }


def test_e4_sensitivity_harm_threshold_changes_what_counts(synthetic_e4):
    E, tables = synthetic_e4
    E.main()
    out = pd.read_csv(tables / "e4_sensitivity.csv")
    joint = out[(out["policy"] == "P3b_clean_and_dle") & (out["parameter"] == "nuisance")]
    n = joint.set_index("value")["n_harmful"]
    # +17 % d'erreur : nuisible aux seuils de 5 et 10 %, pas à 20 %
    assert n[0.05] == 1 and n[0.10] == 1 and n[0.20] == 0
    assert joint.set_index("value").loc[0.10, "recall"] == 1.0


def test_e4_calm_evaluate_period_on_toy_world(toy_world, monkeypatch):
    from zonefleet.experiments import e4_calm as E

    monkeypatch.setattr(E, "predictions", toy_world.predictions)
    monkeypatch.setitem(E.PERIODS, "covid", "jouet")
    rows = pd.DataFrame(E.evaluate_period("covid"))
    assert set(rows["smooth"]) == {1, 7} and set(rows["period"]) == {"jouet"}
    assert rows["share_days_in_alert"].between(0, 1).all()
    assert (rows["days"] > 0).all()
