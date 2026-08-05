"""Cohérence des libellés partagés."""

from zonefleet import labels as L
from zonefleet.alerting import POLICIES


def test_every_policy_has_long_and_short_labels():
    assert set(L.POLICY_LONG) == set(POLICIES) == set(L.POLICY_SHORT)


def test_colors_are_distinct_within_each_role():
    for mapping in (L.POLICY_COLOR, L.FAMILY_COLOR, L.RETRAIN_COLOR):
        assert len(set(mapping.values())) == len(mapping)
