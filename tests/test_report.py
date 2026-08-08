"""Tests du rapport quotidien."""

import pandas as pd
import pytest

from zonefleet.report import build


def signals(est, drift, real=None, n=5):
    idx = pd.date_range("2025-03-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "estimated": est,
            "threshold": 0.4,
            "realized": real if real is not None else [float("nan")] * n,
            "perf_est": [e > 0.4 for e in est],
            "perf_real": [False] * n,
            "drift": drift,
            "drift_clean": drift,
        },
        index=idx,
    )


def test_quiet_day_report():
    md = build(pd.Timestamp("2025-03-05"), signals([0.3] * 5, [False] * 5), model_version="2")
    assert "Sévérité : RAS" in md and "v2" in md
    assert "Réentraînement : **non**" in md and "Angle mort" in md


def test_page_report_lists_zones_and_drifting_features():
    sig = signals([0.3, 0.3, 0.3, 0.5, 0.6], [False, False, True, True, True])
    zones = pd.DataFrame(
        {
            "zone": [161, 7, 79],
            "name": ["Midtown", "Astoria", "East Village"],
            "est": [0.6, 0.2, 0.5],
        }
    )
    psi = pd.DataFrame(
        {"date": pd.Timestamp("2025-03-05"), "feature": ["lag_24", "lag_168"], "psi": [0.6, 0.1]}
    )
    md = build(pd.Timestamp("2025-03-05"), sig, zones, psi)
    assert "Sévérité : PAGE" in md and "runbook" in md
    assert md.index("161 Midtown") < md.index("79 East Village") < md.index("7 Astoria")
    assert (
        "`lag_24` : 0,60" in md
        and "lag_168" not in md.split("Features en dérive")[1].split("##")[0]
    )


def test_report_before_first_signal_fails():
    with pytest.raises(ValueError):
        build(pd.Timestamp("2024-01-01"), signals([0.3] * 5, [False] * 5))


def test_unpublished_labels_are_hidden_on_the_decision_clock():
    sig = signals(
        [0.3, 0.3, 0.5, 0.6, 0.6], [False, True, True, True, True], real=[0.3, 0.3, 1.2, 1.4, 1.5]
    )
    md = build(pd.Timestamp("2025-03-05"), sig, label_delay_days=60)
    assert "en attente" in md and "1,500" not in md
    assert "Réentraînement : **non**" in md  # aucun label post-alerte publié
    md1 = build(pd.Timestamp("2025-03-05"), sig, label_delay_days=0)
    assert "disponible" in md1


def test_data_incident_overrides_model_severity():
    sig = signals([0.3] * 5, [False] * 5)
    q = pd.Series({"volume_ratio": 0.55, "profile_corr": 0.99, "zero_zones": 0, "flag_any": True})
    md = build(pd.Timestamp("2025-03-05"), sig, quality=q)
    assert "Sévérité : INCIDENT DONNÉES" in md and "exclure ces jours" in md
    ok = q.copy()
    ok["flag_any"] = False
    assert "Sévérité : RAS" in build(pd.Timestamp("2025-03-05"), sig, quality=ok)
