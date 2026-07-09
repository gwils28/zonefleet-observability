"""Style graphique commun (palette catégorielle validée, marques fines, grille discrète).

Rôles fixes, jamais permutés d'une figure à l'autre :
- bleu : estimé / sans labels ; orange : réalisé / vérité terrain ;
- aqua, jaune, magenta, vert, violet : séries supplémentaires, dans cet ordre ;
- rouge réservé aux alertes (statut), gris aux références et zones neutres.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from zonefleet.config import FIGURES

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e3e2de", "#fcfcfb"
ALERT = "#c62828"
NEUTRAL = "#b5b4ae"


def setup() -> None:
    """Applique le style à matplotlib."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def save(fig: plt.Figure, name: str, folder: Path | None = None) -> Path:
    """Enregistre une figure PNG dans `site/figures/` et la ferme."""
    folder = folder or FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def label_end(ax, x, y, text: str, color: str, dy: float = 0.0) -> None:
    """Étiquette directe en bout de courbe (texte en encre, pastille de couleur)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9.5,
        color=INK2,
    )
    ax.plot([x], [y], "o", ms=4, color=color)
