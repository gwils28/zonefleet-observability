"""Tests des politiques d'alerte et de l'horloge de décision."""

import pandas as pd

from zonefleet import alerting as A


def days(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def test_persistence_requires_consecutive_days():
    f = pd.Series([True, False, True, True, True, False], index=days(6))
    assert A.persistent(f, 2).tolist() == [False, False, False, True, True, False]
    assert A.persistent(f, 1).tolist() == f.tolist()


def signals(drift, est, real):
    idx = days(len(drift))
    return pd.DataFrame({"drift": drift, "perf_est": est, "perf_real": real}, index=idx)


def test_and_policy_is_intersection():
    s = signals([1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0]).astype(bool)
    flags = A.policy_flags(s, persistence=1)
    assert flags["P3_drift_and_dle"].tolist() == [False, True, True, False]
    assert flags["P1_drift"].sum() == 3 and flags["P2_dle"].sum() == 3


def test_decision_clock():
    d = pd.Timestamp("2020-03-20")
    assert A.decision_time(d, "P2_dle") == pd.Timestamp("2020-03-19")
    assert A.decision_time(d, "P4_realized") == pd.Timestamp("2020-05-19")


def test_first_alert_delay_on_decision_clock():
    s = signals([0] * 5 + [1] * 5, [0] * 6 + [1] * 4, [0] * 7 + [1] * 3).astype(bool)
    flags = A.policy_flags(s, persistence=1)
    res = A.first_alert(flags, after=pd.Timestamp("2020-01-05")).set_index("policy")
    assert res.loc["P1_drift", "delay_days"] == 4 - 4  # cible 01-06, décision 01-05
    assert res.loc["P2_dle", "delay_days"] == 1
    assert res.loc["P4_realized", "delay_days"] == 3 + 60


def test_missing_alert_has_no_delay():
    s = signals([0, 0], [0, 0], [0, 0]).astype(bool)
    res = A.first_alert(A.policy_flags(s, 1), after=pd.Timestamp("2020-01-01"))
    assert res["delay_days"].isna().all()


def test_false_alert_days_before_shock():
    s = signals([1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]).astype(bool)
    fa = A.false_alert_days(A.policy_flags(s, 1), before=pd.Timestamp("2020-01-04"))
    assert fa["P1_drift"] == 2 and fa["P2_dle"] == 0
