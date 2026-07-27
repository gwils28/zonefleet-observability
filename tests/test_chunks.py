"""Tests de la variante E1 bis (fin de chunk et horloge de décision)."""

import pandas as pd

from zonefleet.experiments.e1_chunks import chunk_end, summarize


def test_chunk_end():
    monday = pd.Timestamp("2020-03-16")
    assert chunk_end(monday, "D") == monday
    assert chunk_end(monday, "W") == pd.Timestamp("2020-03-22")


def test_summarize_counts_first_decision_and_false_alerts():
    dates = pd.date_range("2020-03-01", periods=30, freq="D")
    d = pd.DataFrame(
        {
            "date": dates,
            "estimated": [0.5 if (x.day in (3, 20)) else 0.2 for x in dates],
            "realized": 0.3,
            "threshold": 0.4,
            "ci_rel_width": 0.1,
            "chunk": "jour",
        }
    )
    d["decision"] = d["date"] - pd.Timedelta(days=1)
    s = summarize(d)
    assert s["false_alert_chunks_before"] == 1
    assert s["first_decision"] == pd.Timestamp("2020-03-19")
