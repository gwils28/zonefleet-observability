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
