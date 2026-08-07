"""Tests des contrôles de qualité du panel."""

import pandas as pd

from tests.conftest import toy_panel
from zonefleet import quality as Q

START = pd.Timestamp("2019-03-04")


def test_clean_panel_raises_no_flag():
    c = Q.daily_checks(toy_panel("2019-01-01", "2019-04-01", zones=(161, 162)))
    assert not c["flag_any"].any()


def test_each_bug_is_flagged_on_its_first_day():
    p = toy_panel("2019-01-01", "2019-04-01", zones=(161, 162, 7))
    expected = {
        "troncature": "flag_volume",
        "duplication": "flag_volume",
        "decalage_1h": "flag_profile",
        "zone_muette": "flag_zero",
    }
    for kind, flag in expected.items():
        c = Q.daily_checks(Q.inject_bug(p, kind, START, zones=(161,)))
        assert c.loc[START, flag], kind


def test_unknown_bug():
    import pytest

    with pytest.raises(ValueError):
        Q.inject_bug(toy_panel("2019-01-01", "2019-01-10"), "inversion", START)


def test_known_calendar_days_and_calibration():
    days = pd.date_range("2024-12-20", "2025-01-03", freq="D")
    k = Q.known_calendar_days(days)
    assert (
        k[pd.Timestamp("2024-12-25")]
        and k[pd.Timestamp("2025-01-01")]
        and not k[pd.Timestamp("2024-12-20")]
    )
    c = Q.daily_checks(toy_panel("2019-01-01", "2019-06-01", zones=(161, 162)))
    t = Q.calibrate(c)
    assert t["volume_low"] < 1 < t["volume_high"] and 0 < t["profile_min"] < 1
