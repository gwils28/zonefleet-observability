"""Tests du pipeline partagé sur le monde jouet."""

import json


def test_features_are_cached(toy_world):
    P = toy_world
    df = P.features()
    assert (P.CACHE / "features.parquet").exists()
    again = P.features()
    assert len(again) == len(df) and str(again["zone"].dtype) == "category"


def test_frozen_model_is_trained_once_then_reloaded(toy_world):
    P = toy_world
    m1 = P.frozen_model("covid")
    path = P.CACHE / "models" / "covid.pkl"
    mtime = path.stat().st_mtime
    m2 = P.frozen_model("covid")
    assert path.stat().st_mtime == mtime
    assert m1.trained_on == m2.trained_on and m1.trained_on[1] < "2019-09-01"


def test_predictions_cover_reference_and_analysis(toy_world):
    P = toy_world
    pred = P.predictions("covid")
    assert set(pred["period"]) == {"reference", "analysis"}
    assert {"q10", "q50", "q90", "snaive"} <= set(pred.columns)
    ref = pred[pred["period"] == "reference"]
    assert ref["ts"].min() >= __import__("pandas").Timestamp("2019-09-01")


def test_mase_scale_is_computed_once_and_stored(toy_world):
    P = toy_world
    s = P.mase_scale("covid")
    stored = json.loads((P.CACHE / "scales.json").read_text())
    assert stored["covid"] == s > 0
