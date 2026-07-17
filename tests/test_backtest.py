"""Tests des briques du backtest du moniteur (E4)."""

import numpy as np
import pandas as pd

from zonefleet.experiments import e4_backtest as E


def test_draw_is_reproducible_and_covers_kinds():
    a, b = E.draw_scenarios(200, seed=1), E.draw_scenarios(200, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["kind"]) == set(E.KINDS)
    assert a["onset"].between(*E.ONSET_RANGE).all()
    assert (a.loc[a["kind"] == "none", "severity"] == 0).all()


def test_rising_edge_ignores_alarm_already_on():
    f = pd.Series([True, True, False, True, True, False, True])
    assert E.rising(f).tolist() == [True, False, False, True, False, False, True]


def test_smooth_only_touches_performance_columns():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    d = pd.DataFrame({"est": np.arange(10.0), "real": np.arange(10.0), "recon": 1.0}, index=idx)
    s = E.smooth(d, 7)
    assert s["est"].isna().sum() == 6 and s["est"].iloc[-1] == np.mean(range(3, 10))
    assert (s["recon"] == 1.0).all()
    assert E.smooth(d, 1) is d


def test_harm_is_relative_to_counterfactual():
    idx = pd.date_range("2025-03-01", periods=40, freq="D")
    base = pd.DataFrame({"real": 0.2}, index=idx)
    hurt = pd.DataFrame({"real": np.where(idx >= "2025-03-10", 0.3, 0.2)}, index=idx)
    sc = E.draw_scenarios(1).iloc[0].copy()
    sc["onset"] = pd.Timestamp("2025-03-10")
    assert abs(E.harm_of(sc, hurt, base) - 0.5) < 1e-9


def test_policy_flags_and_condition():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    d = pd.DataFrame(
        {
            "est": [0.1, 0.5, 0.5, 0.5, 0.1, 0.1],
            "real": 0.1,
            "recon": 0.0,
            "psi_lag_24": [0.0, 0.0, 0.5, 0.5, 0.5, 0.0],
            "psi_lag_168": 0.0,
            "psi_roll_mean_168": 0.0,
            "psi_temperature_2m": 0.0,
        },
        index=idx,
    )
    flags = E.policy_flags(d, {"est": 0.3, "real": 0.3, "recon": 1.0})
    assert flags["P2_dle"].tolist() == [False, False, True, True, False, False]
    assert flags["P3b_clean_and_dle"].tolist() == [False, False, False, True, False, False]
