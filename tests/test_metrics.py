"""Tests des métriques réalisées."""

import numpy as np
import pandas as pd
import pytest

from zonefleet import metrics as M


def test_basic_errors():
    y, p = np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0])
    assert M.mae(y, p) == pytest.approx(1.0)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))


def test_smape_handles_double_zero():
    assert M.smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert M.smape(np.array([0.0]), np.array([1.0])) == pytest.approx(200.0)


def test_pinball_asymmetry():
    y = np.array([10.0])
    assert M.pinball(y, np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert M.pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.2)


def test_coverage():
    y = np.array([1.0, 5.0, 9.0])
    assert M.coverage(y, np.array([0, 0, 0]), np.array([6, 6, 6])) == pytest.approx(2 / 3)


def test_seasonal_scale_is_per_zone():
    zone = pd.Series([1] * 4 + [2] * 4)
    pickups = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 100.0])
    assert M.seasonal_scale(pickups, zone, season=2) == pytest.approx(1.0)


def test_summarize_keys():
    y = np.arange(10.0)
    out = M.summarize(y, y - 1, y, y + 1, scale=1.0)
    assert out["mae"] == 0 and out["coverage_80"] == 1.0
    assert set(out) == {"mae", "rmse", "smape", "mase", "pinball_mean", "coverage_80"}
