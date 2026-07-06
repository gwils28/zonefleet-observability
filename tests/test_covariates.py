"""Tests des covariables : conversion d'heure locale et calendrier."""

import pandas as pd

from zonefleet.covariates import calendar_features, to_local_naive, us_holidays


def utc_frame(start, periods):
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({"ts_utc": ts, "temperature_2m": range(periods), "precipitation": 0.0})


def test_fall_back_averages_duplicate_local_hour():
    # 2019-11-03 05:00 et 06:00 UTC tombent tous deux sur 01:00 locale
    df = utc_frame("2019-11-03 04:00", 4)
    out = to_local_naive(df).set_index("ts")
    assert out.loc["2019-11-03 01:00", "temperature_2m"] == 1.5
    assert out.index.is_unique


def test_spring_forward_fills_missing_local_hour():
    df = utc_frame("2019-03-10 05:00", 4)  # 00:00, 01:00, 03:00, 04:00 locales
    out = to_local_naive(df).set_index("ts")
    assert pd.Timestamp("2019-03-10 02:00") in out.index
    assert out.loc["2019-03-10 02:00", "temperature_2m"] == 1.5


def test_holidays_include_thanksgiving_and_july_4th():
    hol = us_holidays([2019])
    assert pd.Timestamp("2019-11-28") in hol
    assert pd.Timestamp("2019-07-04") in hol


def test_calendar_features():
    ts = pd.Series(pd.to_datetime(["2019-12-25 08:00", "2019-12-28 23:00"]))
    cal = calendar_features(ts)
    assert cal["is_holiday"].tolist() == [1, 0]
    assert cal["is_weekend"].tolist() == [0, 1]
    assert cal["hour"].tolist() == [8, 23]
    assert cal["doy_sin"].between(-1, 1).all()
