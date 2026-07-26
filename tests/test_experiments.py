"""Pilotes d'expérience sur le monde jouet (exécution de bout en bout, sans vraies données)."""

import pandas as pd

from tests.conftest import toy_panel


def test_e1_run_writes_all_tables(toy_world, tmp_path, monkeypatch):
    from zonefleet.experiments import e1_covid as E1

    monkeypatch.setattr(E1, "TABLES", tmp_path / "t")
    res = E1.run("covid", pd.Timestamp("2020-03-13"), "x1")
    for name in (
        "performance",
        "univariate",
        "multivariate",
        "prediction",
        "psi",
        "signals",
        "detection",
    ):
        assert (tmp_path / "t" / f"x1_{name}.csv").exists()
    assert set(res["detection"]["policy"]) >= {
        "P1_drift",
        "P2_dle",
        "P3_drift_and_dle",
        "P4_realized",
    }


def test_e1_phase_table_ratios():
    from zonefleet.experiments.e1_covid import PHASES, phase_table

    dates = pd.date_range("2020-01-01", "2020-06-30", freq="D")
    perf = pd.DataFrame({"date": dates, "metric": "mae_log", "estimated": 0.2, "realized": 0.4})
    t = phase_table(perf)
    assert len(t) == len(PHASES)
    assert (t["ratio_est_real"] == 0.5).all()


def test_e2_demand_change_did(monkeypatch):
    from zonefleet.experiments import e2_congestion as E2

    p = pd.concat(
        [toy_panel("2024-01-01", "2024-07-01"), toy_panel("2025-01-01", "2025-07-01", seed=1)]
    )
    p.loc[(p["zone"] == 1) & (p["ts"].dt.year == 2025), "pickups"] *= 0.5
    monkeypatch.setattr(E2, "load_panel", lambda: p)
    monkeypatch.setattr(
        E2, "zone_table", lambda: pd.DataFrame({"zone": [1, 2], "cbd": [True, False]})
    )
    dc = E2.demand_change()
    assert len(dc) == 24
    assert (dc["cbd"] < dc["hors_cbd"]).all()  # la zone CBD jouet a perdu la moitié de sa demande
    daily = E2.demand_change_daily()
    assert daily["ratio_cbd"].between(0, 1).all()


def test_e3_summary_is_relative_to_counterfactual():
    from zonefleet.experiments.e3_frontier import ONSET, summarize

    idx = pd.date_range(ONSET - pd.Timedelta(days=10), ONSET + pd.Timedelta(days=30), freq="D")
    base = pd.DataFrame(
        {
            "estimated": 0.3,
            "realized": 0.3,
            "drift": False,
            "drift_clean": False,
            "pca_alert": False,
        },
        index=idx,
    )
    shifted = base.copy()
    post = idx >= ONSET
    shifted.loc[post, "estimated"] = 0.35
    shifted.loc[post, "realized"] = 0.40
    daily = pd.concat(
        [
            base.assign(scenario="aucune dérive", family="aucune", severity=0.0),
            shifted.assign(scenario="s", family="covariate nuisible", severity=1.0),
        ]
    )
    shares = pd.DataFrame({"scenario": ["aucune dérive", "s"], "share_P2_dle": [0.0, 0.5]})
    out = summarize(daily, shares).set_index("scenario")
    assert abs(out.loc["s", "tracking_steady"] - 0.5) < 1e-9
    assert abs(out.loc["s", "delta_real_transition"] - 0.10) < 1e-9
