"""Le scoreur par ligne reproduit exactement les agrégats NannyML."""

import numpy as np
import pytest

from tests.test_monitoring import synthetic
from zonefleet import monitoring as M
from zonefleet.rowscore import RowScorer


@pytest.fixture(scope="module")
def fitted():
    ref = synthetic(30)
    an = synthetic(
        8, seed=3, start="2021-03-01", shift=lambda d: d.assign(lag_24=d["lag_24"] * 1.8)
    )
    return ref, an, RowScorer().fit(ref)


def test_daily_estimate_matches_nannyml(fitted):
    ref, an, rs = fitted
    official = M.estimate_performance(ref, an, "log", metrics=("mae",))
    ours = rs.daily(an)
    np.testing.assert_allclose(ours["est"].to_numpy(), official["estimated"].to_numpy(), rtol=1e-6)


def test_daily_reconstruction_matches_nannyml(fitted):
    ref, an, rs = fitted
    official = M.multivariate_drift(ref, an)
    np.testing.assert_allclose(
        rs.daily(an)["recon"].to_numpy(), official["value"].to_numpy(), rtol=1e-6
    )


def test_aggregate_of_resampled_rows_is_cheap_and_consistent(fitted):
    _, an, rs = fitted
    rows = rs.score_rows(an)
    resampled = rows.sample(frac=1.0, replace=True, random_state=0)
    out = rs.aggregate(resampled)
    assert {"est", "recon", "real", "psi_lag_24", "psi_lag_168"} <= set(out.columns)
    assert (out["est"] >= 0).all()
