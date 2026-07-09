"""Figures pédagogiques des pages Concepts (données synthétiques, graines fixées).

Usage : uv run python scripts/make_concept_figures.py
Sortie : site/figures/concept_*.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from zonefleet import viz
from zonefleet.monitoring import psi

RNG = np.random.default_rng(42)


def three_clocks() -> None:
    """Frise des trois horloges : émission, cible, label."""
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(-8, 70)
    ax.set_ylim(-1.6, 1.4)
    ax.axis("off")
    ax.axhline(0, color=viz.GRID, lw=1.5)
    ax.axvspan(1, 61, ymin=0.47, ymax=0.60, color=viz.BLUE, alpha=0.10)
    ax.text(
        31,
        0.12,
        "fenêtre aveugle : signaux avancés seulement (dérive, prédictions, DLE)",
        ha="center",
        color=viz.BLUE,
        fontsize=9,
    )
    ax.plot([0], [0], "o", ms=9, color=viz.BLUE)
    ax.annotate(
        "τ : émission\n(veille, positionnement)",
        (0, 0),
        xytext=(-4, -34),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
    )
    ax.plot([1], [0], "o", ms=9, color=viz.INK2)
    ax.annotate(
        "t = τ + 24 h : cible",
        (1, 0),
        xytext=(4, 26),
        textcoords="offset points",
        ha="left",
        fontsize=9.5,
    )
    ax.plot([61], [0], "o", ms=9, color=viz.ORANGE)
    ax.annotate(
        "t + 60 j : label publié",
        (61, 0),
        xytext=(0, 26),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
    )
    ax.annotate(
        "", xy=(61, -0.9), xytext=(1, -0.9), arrowprops=dict(arrowstyle="<->", color=viz.MUTED)
    )
    ax.text(
        31,
        -1.3,
        "H + L ≈ 61 jours sans métrique réalisée",
        ha="center",
        color=viz.INK2,
        fontsize=9.5,
    )
    viz.save(fig, "concept_trois_horloges")


def drift_types() -> None:
    """Covariate shift vs concept drift sur une relation x -> y hétéroscédastique."""
    f = lambda x: 2 + 1.2 * x  # noqa: E731

    def sample(n, mu, slope=1.2):
        x = RNG.normal(mu, 1, n)
        return x, 2 + slope * x + RNG.normal(0, 0.25 + 0.35 * np.abs(x), n)

    xs = np.linspace(-4, 6, 50)
    cases = [
        ("Référence", sample(400, 0)),
        ("Covariate shift : P(X) bouge", sample(400, 2.5)),
        ("Concept drift : P(y|X) bouge", sample(400, 0, slope=0.4)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (title, (x, y)) in zip(axes, cases, strict=True):
        ax.scatter(x, y, s=8, color=viz.ORANGE, alpha=0.35, lw=0, label="observations")
        ax.plot(xs, f(xs), color=viz.BLUE, label="modèle appris")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("feature x")
    axes[0].set_ylabel("cible y")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Même modèle, deux dérives : seule la première laisse une trace dans X",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    viz.save(fig, "concept_types_derive")


def dle_mechanism() -> None:
    """DLE : la carte des erreurs apprise sur la référence prédit l'erreur sur l'analysis."""
    noise = lambda x: 0.2 + 0.5 * np.abs(x)  # noqa: E731
    xr = RNG.normal(0, 1, 3000)
    err_r = np.abs(RNG.normal(0, noise(xr)))
    bins = np.linspace(-4, 5, 37)
    centers = 0.5 * (bins[1:] + bins[:-1])
    idx = np.digitize(xr, bins) - 1
    nanny = np.array(
        [err_r[idx == i].mean() if (idx == i).sum() > 10 else np.nan for i in range(len(centers))]
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    ax = axes[0]
    ax.scatter(xr, err_r, s=6, color=viz.ORANGE, alpha=0.25, lw=0, label="erreur |y - f(x)|")
    ax.plot(centers, nanny, color=viz.BLUE, label="nanny model")
    ax.set_title("1. Référence : apprendre où\nle modèle se trompe")
    ax.set_xlabel("feature x")
    ax.set_ylabel("erreur absolue")
    ax.legend(loc="upper left")
    ax = axes[1]
    samples = {}
    for mu, c, lab in [(0, viz.NEUTRAL, "référence"), (1.5, viz.AQUA, "analysis")]:
        x = RNG.normal(mu, 1, 3000)
        samples[lab] = x
        ax.hist(x, bins=bins, color=c, alpha=0.6, density=True, label=lab)
    ax.set_title("2. Analysis : X se déplace vers\nles zones à forte erreur")
    ax.set_xlabel("feature x")
    ax.set_ylabel("densité")
    ax.legend(loc="upper left")
    ax = axes[2]
    lookup = lambda x: np.interp(x, centers[~np.isnan(nanny)], nanny[~np.isnan(nanny)])  # noqa: E731
    est = [lookup(samples[k]).mean() for k in ("référence", "analysis")]
    real = [np.abs(RNG.normal(0, noise(samples[k]))).mean() for k in ("référence", "analysis")]
    xs = np.arange(2)
    ax.bar(xs - 0.2, est, width=0.38, color=viz.BLUE, label="MAE estimée (sans labels)")
    ax.bar(xs + 0.2, real, width=0.38, color=viz.ORANGE, label="MAE réalisée")
    ax.set_xticks(xs, ["référence", "analysis"])
    ax.set_title("3. L'estimation suit la hausse\nréelle de l'erreur")
    ax.legend(loc="upper left")
    fig.tight_layout()
    viz.save(fig, "concept_dle_mecanisme")


def univariate_stats() -> None:
    """KS, Wasserstein et PSI sur deux distributions."""
    a = RNG.normal(0, 1, 4000)
    b = RNG.normal(0.6, 1.3, 4000)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    grid = np.linspace(-5, 6, 400)
    ax = axes[0]
    ax.hist(a, bins=60, density=True, color=viz.NEUTRAL, alpha=0.7, label="référence")
    ax.hist(b, bins=60, density=True, color=viz.AQUA, alpha=0.55, label="chunk courant")
    ax.set_title("Deux distributions")
    ax.legend(loc="upper left")
    ax = axes[1]
    ca = np.searchsorted(np.sort(a), grid) / len(a)
    cb = np.searchsorted(np.sort(b), grid) / len(b)
    k = np.argmax(np.abs(ca - cb))
    ax.plot(grid, ca, color=viz.INK2, label="FdR référence")
    ax.plot(grid, cb, color=viz.AQUA, label="FdR courant")
    ax.fill_between(grid, ca, cb, color=viz.YELLOW, alpha=0.25, label="aire = Wasserstein")
    ax.vlines(grid[k], cb[k], ca[k], color=viz.ALERT, lw=2, label="écart max = KS")
    ks = stats.ks_2samp(a, b).statistic
    ws = stats.wasserstein_distance(a, b)
    ax.set_title(f"KS = {ks:.2f} · Wasserstein = {ws:.2f}")
    ax.legend(loc="upper left", fontsize=8.5)
    ax = axes[2]
    edges = np.quantile(a, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    ra = np.histogram(a, edges)[0] / len(a)
    rb = np.histogram(b, edges)[0] / len(b)
    xs = np.arange(10)
    ax.bar(xs - 0.2, ra, width=0.38, color=viz.NEUTRAL, label="référence (10 % par bac)")
    ax.bar(xs + 0.2, rb, width=0.38, color=viz.AQUA, label="courant")
    ax.set_xticks(xs, [f"D{i + 1}" for i in xs])
    ax.set_title(f"PSI = {psi(a, b):.2f} (bacs = déciles de la référence)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    viz.save(fig, "concept_derive_univariee")


def pca_limit() -> None:
    """Reconstruction PCA : corrélation cassée vue, déplacement le long de l'axe invisible."""
    z = RNG.normal(0, 1, 600)
    ref = np.column_stack([z, z + RNG.normal(0, 0.15, 600)])
    zb = RNG.normal(0, 1, 300)
    along = np.column_stack([zb + 2.5, zb + 2.5 + RNG.normal(0, 0.15, 300)])
    broken = np.column_stack([RNG.normal(0, 1, 300), RNG.normal(0, 1, 300)])
    u = np.array([1, 1]) / np.sqrt(2)

    def rec_err(p):
        proj = (p @ u)[:, None] * u
        return np.linalg.norm(p - proj, axis=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, (pts, c, title) in zip(
        axes,
        [
            (along, viz.AQUA, "Déplacement le long de l'axe principal"),
            (broken, viz.MAGENTA, "Corrélation cassée"),
        ],
        strict=True,
    ):
        ax.scatter(*ref.T, s=7, color=viz.NEUTRAL, alpha=0.6, lw=0, label="référence")
        ax.scatter(*pts.T, s=7, color=c, alpha=0.6, lw=0, label="analysis")
        t = np.linspace(-4, 6, 2)
        ax.plot(t, t, color=viz.BLUE, lw=1.5, label="1re composante (65 % var.)")
        ax.set_title(
            f"{title}\nerreur de reconstruction : {rec_err(pts):.2f} (réf. {rec_err(ref):.2f})",
            fontsize=10.5,
        )
        ax.set_xlabel("lag_24 (standardisé)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("lag_168 (standardisé)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    viz.save(fig, "concept_pca_limite")


def metric_scales() -> None:
    """Échelle count vs log quand le volume s'effondre."""
    days = np.arange(120)
    level = np.where(days < 60, 300, 300 * np.exp(-(days - 60) / 6).clip(0.08, 1))
    level = np.maximum(level, 25)
    pred = np.concatenate([level[:60], np.full(60, 300.0)])
    pred = np.where(
        days >= 60, np.maximum(level * 1.0, 0) + (300 - level) * np.exp(-(days - 60) / 14), pred
    )
    real = level * np.exp(RNG.normal(0, 0.15, 120))
    mae_count = np.abs(real - pred)
    mae_log = np.abs(np.log1p(real) - np.log1p(pred))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(days, real, color=viz.ORANGE, label="demande réalisée")
    axes[0].plot(days, pred, color=viz.BLUE, label="prévision (lags en retard)")
    axes[0].set_title("Un choc de niveau\n(demande en courses)")
    axes[0].legend(loc="lower left")
    axes[1].plot(days, mae_count, color=viz.INK2)
    axes[1].set_title("Erreur en courses :\nle choc passe, puis « s'améliore »")
    axes[2].plot(days, mae_log, color=viz.INK2)
    axes[2].set_title("Erreur relative (log) :\nle choc reste visible")
    for ax in axes:
        ax.axvline(60, color=viz.MUTED, lw=1, ls="--")
        ax.set_xlabel("jour")
    fig.tight_layout()
    viz.save(fig, "concept_metriques_echelles")


def main() -> None:
    viz.setup()
    three_clocks()
    drift_types()
    dle_mechanism()
    univariate_stats()
    pca_limit()
    metric_scales()


if __name__ == "__main__":
    main()
