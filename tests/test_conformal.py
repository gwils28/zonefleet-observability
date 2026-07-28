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
