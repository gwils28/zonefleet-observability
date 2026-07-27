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


def test_e5_policies_on_toy_world(toy_world, monkeypatch):
    from zonefleet.experiments import e5_retraining as E5

    df = toy_world.features()
    champion = toy_world.frozen_model("covid")
    monkeypatch.setattr(E5, "YEAR", (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01")))
    never = E5.simulate("jamais", 1, df, champion)
    assert len(never.switches) == 1
    reflex = E5.simulate("réflexe", 60, df, champion)
    assert reflex.switches[1][0] == E5.FIRST_ALERT
    short60 = E5.simulate("réflexe court", 60, df, champion)
    assert len(short60.switches) == 1  # aucun label post-choc à L = 60 : pas de remplacement
    daily = E5.realized(reflex, df)
    assert set(daily.columns) == {"date", "mae_log", "mae", "policy", "delay"}
    assert daily["date"].min() == pd.Timestamp("2020-01-01")


def test_e6_zone_delta_and_auc():
    import numpy as np

    from zonefleet.experiments import e6_localisation as E6

    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    rows = pd.DataFrame(
        {
            "date": np.tile(dates, 2),
            "zone": np.repeat([7, 13], 20),
            "est": np.r_[np.full(20, 0.2), np.r_[np.full(10, 0.2), np.full(10, 0.5)]],
            "real": 0.3,
            "lag_168": np.random.default_rng(0).normal(size=40),
        }
    )
    z = E6.zone_delta(rows, rows, ("2025-01-11", "2025-01-21"), ("2025-01-01", "2025-01-11"))
    assert z.loc[13, "delta_est"] > z.loc[7, "delta_est"]
    assert E6.auc(pd.Series([True, False, True]), pd.Series([0.9, 0.1, 0.8])) == 1.0


def test_e0_tables_on_toy_world(toy_world, monkeypatch):
    from zonefleet.experiments import e0_model as E0

    monkeypatch.setattr(E0, "SPLITS", {"covid": E0.SPLITS["covid"]})
    m = E0.metrics_table()
    assert set(m["model"]) == {"lightgbm", "seasonal_naive"}
    assert (m.loc[m["model"] == "lightgbm", "coverage_80"].between(0, 1)).all()
    z = E0.by_zone("covid")
    assert set(z["zone"]) == {1, 2} and "skill_vs_snaive" in z
    c = E0.coverage_by_hour("covid")
    assert len(c) == 24


def test_e2_realized_by_group(toy_world, monkeypatch):
    from zonefleet.experiments import e2_congestion as E2

    monkeypatch.setattr(E2, "SPLIT", "covid")
    monkeypatch.setattr(E2, "CONGESTION_START", pd.Timestamp("2020-01-05"))
    r = E2.realized_by_group()
    assert set(r["group"]) == {"cbd", "hors_cbd"} and len(r) == 6
    assert (r["mae_log"] >= 0).all()


def test_e4_scenario_daily_covariate_and_concept(toy_world):
    from zonefleet.experiments import e4_backtest as E4
    from zonefleet.rowscore import RowScorer

    pred = toy_world.predictions("covid")
    ref, an = (
        pred[pred.period == "reference"],
        pred[pred.period == "analysis"].reset_index(drop=True),
    )
    scorer = RowScorer().fit(ref)
    base_rows = scorer.score_rows(an)
    stats = (ref["lag_168"].mean(), ref["lag_168"].std())
    onset = pd.Timestamp("2020-03-02")
    Sc = __import__("collections").namedtuple("Sc", "id kind severity subset onset")
    cov = E4.scenario_daily(
        Sc(1, "covariate_harmful", 2.0, "all", onset), scorer, base_rows, an, None, stats
    )
    con = E4.scenario_daily(Sc(2, "concept", 0.5, "all", onset), scorer, base_rows, an, None, stats)
    none = E4.scenario_daily(Sc(3, "none", 0.0, "all", onset), scorer, base_rows, an, None, stats)
    post = none.index >= onset
    assert con.loc[post, "real"].mean() > none.loc[post, "real"].mean()
    assert abs(con.loc[post, "est"].mean() - none.loc[post, "est"].mean()) < 1e-12
    assert cov.loc[post, "est"].mean() != none.loc[post, "est"].mean()
