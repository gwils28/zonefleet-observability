"""Monde jouet partagé : panel, météo et zones factices, cache temporaire.

Remplace les accès aux vraies données dans `pipeline` pour tester les pilotes
d'expérience sans toucher `data/` ni `results/cache/`.
"""

import numpy as np
import pandas as pd
import pytest


def toy_panel(start="2019-01-01", end="2020-07-01", zones=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h", inclusive="left")
    daily = 1 + 0.6 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
    weekly = 1 + 0.2 * (ts.dayofweek < 5)
    parts = []
    for z in zones:
        lam = 20 * z * daily * weekly
        parts.append(pd.DataFrame({"zone": z, "ts": ts, "pickups": rng.poisson(lam).astype(float)}))
    return pd.concat(parts, ignore_index=True)


def toy_weather(start="2018-12-01", end="2021-01-01"):
    ts = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "ts": ts,
            "temperature_2m": 10 + 8 * np.sin(2 * np.pi * ts.dayofyear / 365),
            "precipitation": 0.0,
        }
    )


@pytest.fixture
def toy_world(tmp_path, monkeypatch):
    """Pipeline branché sur des données jouets et un cache temporaire."""
    import zonefleet.features as F
    import zonefleet.model as M
    import zonefleet.pipeline as P

    zones = pd.DataFrame(
        {"zone": [1, 2], "cbd": [True, False], "name": ["A", "B"], "borough": ["X", "Y"]}
    )
    monkeypatch.setattr(P, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(P, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(P, "load_panel", lambda: toy_panel())
    monkeypatch.setattr(P, "load_weather", lambda: toy_weather())
    monkeypatch.setattr(F, "zone_table", lambda: zones)
    monkeypatch.setattr(M, "PARAMS", {**M.PARAMS, "n_estimators": 20})
    return P
