"""Tests de la calibration conforme."""

import numpy as np
import pandas as pd

from zonefleet import conformal as C


def test_scores_sign():
    s = C.scores(
        np.array([5.0, 0.0, 12.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0])
    )
    assert list(s) == [-4.0, 1.0, 2.0]


def test_correction_finite_sample_quantile():
    s = np.arange(1, 10, dtype=float)  # n = 9, alpha = 0,2 : k = ceil(10 * 0,8) = 8
    assert C.correction(s, 0.2) == 8.0


def test_calibration_restores_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.lognormal(3, 0.5, n)
    df = pd.DataFrame(
        {"pickups": y, "q10": y * 0.9, "q90": y * 1.1, "hour": rng.integers(0, 24, n)}
    )
    df["pickups"] = df["pickups"] * rng.lognormal(0, 0.3, n)  # intervalles trop étroits
    cal, test = df.iloc[: n // 2], df.iloc[n // 2 :]
    before, _ = C.coverage_width(test, "q10", "q90")
    out = C.apply(test, C.fit(cal, 0.2))
    after, _ = C.coverage_width(out, "lo_cal", "hi_cal")
    assert before < 0.5 and abs(after - 0.8) < 0.02
    by_hour = C.apply(test, C.fit(cal, 0.2, by="hour"), by="hour")
    assert abs(C.coverage_width(by_hour, "lo_cal", "hi_cal")[0] - 0.8) < 0.03


def test_rolling_uses_only_published_labels():
    rng = np.random.default_rng(1)
    ts = pd.date_range("2020-01-01", periods=24 * 60, freq="h")
    y = rng.lognormal(3, 0.3, len(ts))
    df = pd.DataFrame({"ts": ts, "pickups": y, "q10": y * 0.95, "q90": y * 1.05})
    late = ts >= "2020-02-15"
    df.loc[late, "pickups"] *= 3  # rupture : seuls les labels publiés peuvent la révéler
    out = C.rolling(df, window_days=14, delay_days=7)
    c = out.groupby(out["ts"].dt.date)["correction"].first()
    d = pd.Timestamp
    assert c[d("2020-01-03").date()] == 0.0  # pas encore d'historique
    # le 20 février, les labels connus s'arrêtent au 13 : la rupture du 15 est invisible
    assert c[d("2020-02-20").date()] < 0.5 < c[d("2020-02-28").date()]
