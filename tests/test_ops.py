"""Tests des décisions opérationnelles, du registry MLflow et du graphe Dagster."""

import numpy as np
import pandas as pd
import pytest

from zonefleet.ops.decisions import NONE, PAGE, TICKET, alert_severity, retrain_decision


def signals(drift, est, real=None):
    n = len(drift)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "drift": drift,
            "drift_clean": drift,
            "perf_est": est,
            "perf_real": real if real is not None else [False] * n,
        },
        index=idx,
    ).astype(bool)


def test_page_requires_joint_condition_for_two_days():
    assert alert_severity(signals([1, 1, 1], [0, 1, 1])) == PAGE
    assert alert_severity(signals([1, 1, 1], [0, 0, 1])) == NONE


def test_drift_alone_becomes_a_ticket_after_a_week():
    assert alert_severity(signals([1] * 7, [0] * 7)) == TICKET
    assert alert_severity(signals([1] * 6, [0] * 6)) == NONE


def test_realized_degradation_pages_even_without_drift():
    assert alert_severity(signals([0, 0], [0, 0], [1, 1])) == PAGE


@pytest.mark.parametrize(
    "confirmed,days,guard,expected",
    [
        (False, 90, True, False),
        (True, 10, True, False),
        (True, 30, False, False),
        (True, 30, True, True),
    ],
)
def test_retrain_decision(confirmed, days, guard, expected):
    d = retrain_decision(confirmed, days, feedback_guard_ok=guard)
    assert d.retrain is expected and d.reason


def test_registry_roundtrip_and_promotion(tmp_path):
    pytest.importorskip("mlflow")
    from zonefleet.features import FEATURES
    from zonefleet.model import QuantileForecaster
    from zonefleet.ops import registry

    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=400) for c in FEATURES})
    df["zone"] = pd.Categorical(rng.integers(1, 3, 400))
    df["y"] = rng.normal(size=400)
    df["ts"] = pd.date_range("2019-01-01", periods=400, freq="h")
    m = QuantileForecaster(params={"n_estimators": 5, "verbose": -1}).fit(df)
    v1 = registry.register(m, {"mae_log_ref": 0.25}, directory=tmp_path)
    registry.set_alias("champion", v1, directory=tmp_path)
    v2 = registry.register(m, {"mae_log_ref": 0.24}, directory=tmp_path)
    assert not registry.promote_if_better(0.25, 0.245, v2, directory=tmp_path)  # gain < 5 %
    assert registry.promote_if_better(0.25, 0.20, v2, directory=tmp_path)
    loaded = registry.load("champion", directory=tmp_path)
    np.testing.assert_allclose(loaded.predict(df), m.predict(df))


def test_dagster_graph_loads():
    pytest.importorskip("dagster")
    from zonefleet.ops.definitions import defs

    keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    assert {
        "panel",
        "monitoring_signals",
        "alert_decision",
        "retrain_recommendation",
        "daily_report",
    } <= keys


def test_monitoring_assets_on_toy_world(toy_world, monkeypatch):
    dg = pytest.importorskip("dagster")
    import zonefleet.features as F
    from tests.conftest import toy_panel, toy_weather
    from zonefleet.ops import definitions as D

    monkeypatch.setattr(
        F, "zone_table", lambda: pd.DataFrame({"zone": [1, 2], "cbd": [True, False]})
    )
    cfg = D.MonitoringConfig(split="covid")
    feats = D.feature_matrix(panel=toy_panel(), weather=toy_weather())
    model = toy_world.frozen_model("covid")
    pred = D.predictions(config=cfg, feature_matrix=feats, champion_model=model)
    assert set(pred["period"]) == {"reference", "analysis"}
    sig = D.monitoring_signals(predictions=pred)
    assert {"drift", "drift_clean", "perf_est", "perf_real"} <= set(sig.columns)
    sev = D.alert_decision(dg.build_asset_context(), monitoring_signals=sig)
    assert sev in {"page", "ticket", "aucune"}
    rec = D.retrain_recommendation(dg.build_asset_context(), monitoring_signals=sig)
    assert set(rec) == {"retrain", "reason"}
    import zonefleet.config as Z

    monkeypatch.setattr(Z, "RESULTS", toy_world.CACHE.parent)
    text = D.daily_report(dg.build_asset_context(), config=cfg, monitoring_signals=sig)
    assert text.startswith("# Monitoring ZoneFleet")
    assert list((toy_world.CACHE.parent / "reports").glob("covid_*.md"))


def test_estimated_loss_alone_opens_a_ticket():
    assert alert_severity(signals([0, 0, 0], [0, 1, 1])) == TICKET


def test_data_quality_check_on_toy_panel():
    pytest.importorskip("dagster")
    from tests.conftest import toy_panel
    from zonefleet.ops.definitions import data_quality

    p = toy_panel("2019-01-01", "2020-03-01", zones=(161, 162))
    assert data_quality(p).passed
    p.loc[p["ts"] >= "2020-02-29", "pickups"] *= 3
    assert not data_quality(p).passed
