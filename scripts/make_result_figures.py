"""Figures des pages Modèle et Expériences, lues dans results/ (et le cache pour les traces).

Usage : uv run python scripts/make_result_figures.py [groupe ...]
Groupes : e0, e1, e2, e3, e4, e5 (tous par défaut).
Sortie : site/figures/<groupe>_*.png
"""

from __future__ import annotations

import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zonefleet import viz
from zonefleet.config import RESULTS

T = RESULTS / "tables"
GROUPS: dict[str, list] = {}


def group(name: str):
    def deco(fn):
        GROUPS.setdefault(name, []).append(fn)
        return fn

    return deco


# --------------------------------------------------------------------------- E0
@group("e0")
def e0_importance() -> None:
    imp = pd.read_csv(T / "e0_feature_importance.csv").set_index("feature")
    imp = imp.sort_values("covid")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(imp))
    ax.barh(y + 0.2, imp["covid"], height=0.38, color=viz.BLUE, label="modèle 2019 (E1)")
    ax.barh(y - 0.2, imp["congestion"], height=0.38, color=viz.ORANGE, label="modèle 2024 (E2)")
    ax.set_yticks(y, imp.index)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("part du gain total (quantile 0,5)")
    ax.set_title("Le lag hebdomadaire porte l'essentiel du modèle")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, "e0_importance")


@group("e0")
def e0_skill_by_zone() -> None:
    z = pd.read_csv(T / "e0_by_zone.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, split, title in [
        (axes[0], "covid", "Référence sept.-déc. 2019"),
        (axes[1], "congestion", "Référence sept.-déc. 2024"),
    ]:
        s = z[z["split"] == split]
        for flag, c, lab in [(True, viz.BLUE, "CBD"), (False, viz.AQUA, "hors CBD")]:
            t = s[s["cbd"] == flag]
            ax.scatter(
                t["mean_pickups"],
                100 * t["skill_vs_snaive"],
                s=26,
                color=c,
                label=lab,
                edgecolor=viz.SURFACE,
                lw=1,
            )
        ax.axhline(0, color=viz.MUTED, lw=1)
        ax.set_xscale("log")
        viz.plain_log_axis(ax)
        ax.set_xlabel("courses moyennes par heure (log)")
        ax.set_title(title)
    axes[0].set_ylabel("gain de MAE vs seasonal naive (%)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, "e0_skill_zones")


@group("e0")
def e0_coverage_by_hour() -> None:
    c = pd.read_csv(T / "e0_coverage_by_hour.csv")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for split, col, lab in [
        ("covid", viz.BLUE, "référence 2019"),
        ("congestion", viz.ORANGE, "référence 2024"),
    ]:
        s = c[c["split"] == split]
        ax.plot(s["hour"], 100 * s["coverage_80"], color=col, marker="o", ms=4, label=lab)
    ax.axhline(80, color=viz.MUTED, ls="--", lw=1)
    ax.annotate(
        "cible 80 %",
        (23, 80),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=viz.INK2,
    )
    ax.set_xticks(range(0, 24, 3), [f"{h}h" for h in range(0, 24, 3)])
    ax.set_ylabel("couverture [q10, q90] (%)")
    ax.set_title("Intervalle 80 % : sous-couverture partout, maximale à la pointe du matin")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, "e0_couverture_heure")


@group("e0")
def e0_traces() -> None:
    from zonefleet.pipeline import predictions

    p = predictions("covid")
    p = p[(p["period"] == "reference") & (p["ts"] >= "2019-10-07") & (p["ts"] < "2019-10-14")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
    for ax, zone, name in [
        (axes[0], 161, "Midtown Center (CBD)"),
        (axes[1], 7, "Astoria (Queens)"),
    ]:
        s = p[p["zone"].astype(int) == zone].sort_values("ts")
        ax.fill_between(
            s["ts"],
            s["q10"],
            s["q90"],
            color=viz.BLUE,
            alpha=0.18,
            lw=0,
            label="intervalle [q10, q90]",
        )
        ax.plot(
            s["ts"], s["q50"], color=viz.BLUE, lw=1.6, label="prévision médiane (émise 24 h avant)"
        )
        ax.plot(s["ts"], s["pickups"], color=viz.ORANGE, lw=1.3, label="courses réalisées")
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("courses / heure")
    axes[0].legend(loc="upper left", ncols=3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    fig.suptitle("Une semaine de référence (octobre 2019)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, "e0_traces")


def main(argv: list[str]) -> None:
    viz.setup()
    names = argv or list(GROUPS)
    for name in names:
        for fn in GROUPS[name]:
            fn()


if __name__ == "__main__":
    main(sys.argv[1:])
