"""Tests du style graphique."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from zonefleet import viz  # noqa: E402


def test_setup_and_save(tmp_path):
    viz.setup()
    assert matplotlib.rcParams["axes.spines.top"] is False
    fig, ax = plt.subplots()
    ax.plot([1, 10, 100], [1, 2, 3])
    ax.set_xscale("log")
    viz.plain_log_axis(ax)
    viz.label_end(ax, 100, 3, "fin", viz.BLUE)
    path = viz.save(fig, "essai", folder=tmp_path)
    assert path.exists() and path.stat().st_size > 1000


def test_palette_roles_are_distinct():
    assert len(set(viz.SERIES)) == len(viz.SERIES)
    assert viz.ALERT not in viz.SERIES and viz.BLUE != viz.ORANGE
