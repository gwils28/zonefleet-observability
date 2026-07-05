"""Tests de l'agrégation en panel (DST, zéros explicites, contrôles)."""

import pandas as pd

from zonefleet.data import build_panel, check_panel, dst_hours, top_zones


def long_frame(ts, zone=1, service="yellow", n=10.0):
    return pd.DataFrame({"zone": zone, "ts": pd.to_datetime(ts), "service": service, "pickups": n})


def test_dst_hours_2019():
    idx = pd.date_range("2019-01-01", "2020-01-01", freq="h", inclusive="left")
    spring, fall = dst_hours(idx)
    assert list(spring) == [pd.Timestamp("2019-03-10 02:00")]
    assert list(fall) == [pd.Timestamp("2019-11-03 01:00")]


def test_panel_fills_missing_hours_with_zero():
    lf = long_frame(["2019-01-01 00:00", "2019-01-01 03:00"])
    p = build_panel(lf, [1, 2], "2019-01-01", "2019-01-02")
    assert len(p) == 2 * 24
    assert p.loc[p["zone"] == 2, "pickups"].sum() == 0
    assert p.loc[(p["zone"] == 1) & (p["ts"] == "2019-01-01 01:00"), "pickups"].item() == 0


def test_panel_sums_services():
    lf = pd.concat(
        [long_frame(["2019-01-01 05:00"]), long_frame(["2019-01-01 05:00"], service="green", n=4.0)]
    )
    p = build_panel(lf, [1], "2019-01-01", "2019-01-02")
    row = p[p["ts"] == "2019-01-01 05:00"].iloc[0]
    assert (row["yellow"], row["green"], row["pickups"]) == (10.0, 4.0, 14.0)


def test_spring_forward_hour_is_interpolated():
    ts = ["2019-03-10 01:00", "2019-03-10 03:00"]
    lf = pd.concat([long_frame(ts[:1], n=20.0), long_frame(ts[1:], n=40.0)])
    p = build_panel(lf, [1], "2019-03-10", "2019-03-11").set_index("ts")
    assert p.loc["2019-03-10 02:00", "pickups"] == 30.0
    assert p.loc["2019-03-10 02:00", "dst_repaired"]


def test_fall_back_hour_is_halved():
    lf = long_frame(["2019-11-03 01:00"], n=50.0)
    p = build_panel(lf, [1], "2019-11-03", "2019-11-04").set_index("ts")
    assert p.loc["2019-11-03 01:00", "pickups"] == 25.0


def test_check_panel_flags_duplicates_and_excess():
    p = pd.DataFrame(
        {"zone": [1, 1], "ts": pd.to_datetime(["2019-01-01"] * 2), "pickups": [1.0, 5000.0]}
    )
    issues = check_panel(p, max_per_hour=3000)
    assert "doublons zone x heure" in issues
    assert any("3000" in i for i in issues)


def test_top_zones_ranks_by_volume():
    lf = pd.concat([long_frame(["2019-05-01"], zone=z, n=float(z)) for z in range(1, 6)])
    assert top_zones(lf, year=2019, n=2) == [4, 5]
