"""Tests de l'horloge des labels dans la simulation de réentraînement (E5)."""

import numpy as np
import pandas as pd

from zonefleet.experiments import e5_retraining as E


def frame():
    ts = pd.date_range("2019-06-01", "2020-12-31", freq="D")
    return pd.DataFrame({"ts": ts, "pickups": np.arange(len(ts), dtype=float)})


def test_available_until_respects_label_delay():
    d = pd.Timestamp("2020-03-22")
    assert E.available_until(d, 60) == pd.Timestamp("2020-01-22")
    assert E.available_until(d, 1) == pd.Timestamp("2020-03-21")


def test_train_window_never_reads_unpublished_labels():
    df = frame()
    for delay in (1, 60):
        w = E.train_window(df, pd.Timestamp("2020-03-22"), delay)
        assert w["ts"].max() < E.available_until(pd.Timestamp("2020-03-22"), delay)
        assert w["ts"].min() >= pd.Timestamp("2020-03-22") - pd.Timedelta(
            days=delay
        ) - pd.DateOffset(months=8)


def test_short_reflex_window_is_empty_with_slow_labels():
    df = frame()
    w = E.train_window(df, E.FIRST_ALERT, 60, start=E.SHOCK)
    assert w.empty  # à L = 60 jours, aucun label postérieur au choc n'existe le 22 mars
    assert not E.train_window(df, E.FIRST_ALERT, 1, start=E.SHOCK).empty


def test_summarize_uses_fixed_periods_and_counts_switches():
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "mae_log": np.where(dates.year == 2021, 0.2, 0.5),
            "policy": "p",
            "delay": 1,
        }
    )
    sw = pd.DataFrame({"policy": ["p", "p", "q"], "delay": [1, 1, 1], "date": dates[:3]})
    s = E.summarize(daily, sw).iloc[0]
    assert s["second semestre"] == 0.5 and s["2021"] == 0.2 and s["retrains"] == 2
    short = E.summarize(daily[daily["date"] < "2021-01-01"], sw)
    assert "2021" not in short.columns
