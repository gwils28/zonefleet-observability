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


# --------------------------------------------------------------------------- E1/E2
POLICY_FR = {
    "P1_drift": "P1 dérive seule",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE seul",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b nettoyée ET DLE",
    "P4_realized": "P4 labels (D + 60 j)",
}


def perf_panel(ax, perf: pd.DataFrame, metric: str, title: str, shock=None, phases=None) -> None:
    """Erreur estimée (bande de confiance) vs réalisée, journalière, avec seuil."""
    d = perf[perf["metric"] == metric].sort_values("date")
    if phases:
        for (a, b), shade in phases:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=shade, alpha=0.08, lw=0)
    ax.fill_between(d["date"], d["lower"], d["upper"], color=viz.BLUE, alpha=0.15, lw=0)
    ax.plot(d["date"], d["estimated"], color=viz.BLUE, label="estimée par DLE (sans labels)")
    ax.plot(d["date"], d["realized"], color=viz.ORANGE, label="réalisée (labels, D + 60 j)")
    ax.plot(
        d["date"], d["threshold"], color=viz.MUTED, ls="--", lw=1, label="seuil (référence + 3 σ)"
    )
    if shock is not None:
        ax.axvline(shock, color=viz.INK2, lw=1)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def policy_timeline(ax, sig: pd.DataFrame, shock) -> None:
    """Jours d'alerte par politique (horloge cible), début du choc en trait vertical."""
    for i, col in enumerate(POLICY_FR):
        days = sig.index[sig[col].astype(bool)]
        ax.scatter(days, [i] * len(days), marker="|", s=120, color=viz.ALERT, lw=1.6)
    ax.set_yticks(range(len(POLICY_FR)), list(POLICY_FR.values()))
    ax.set_ylim(-0.6, len(POLICY_FR) - 0.4)
    ax.invert_yaxis()
    ax.axvline(shock, color=viz.INK2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


E1_PHASES = [(("2020-03-13", "2020-04-13"), viz.ALERT)]


@group("e1")
def e1_estimated_vs_realized() -> None:
    from zonefleet.config import COVID_EMERGENCY

    perf = pd.read_csv(T / "e1_performance.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True)
    perf_panel(
        axes[0],
        perf,
        "mae_log",
        "Erreur relative (MAE sur log1p) : DLE voit le choc mais le sous-estime d'un facteur 2,7",
        COVID_EMERGENCY,
        E1_PHASES,
    )
    perf_panel(
        axes[1],
        perf,
        "mae_count",
        "Erreur en courses (MAE) : la mesure absolue « s'améliore » au plancher",
        COVID_EMERGENCY,
        E1_PHASES,
    )
    axes[0].legend(loc="upper left")
    axes[0].annotate(
        "13 mars : état d'urgence",
        (COVID_EMERGENCY, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color=viz.INK2,
        va="top",
    )
    fig.tight_layout()
    viz.save(fig, "e1_estime_vs_realise")


@group("e1")
def e1_scatter() -> None:
    perf = pd.read_csv(T / "e1_performance.csv", parse_dates=["date"])
    d = perf[perf["metric"] == "mae_log"]
    phases = [
        ("avant", d["date"] < "2020-03-13", viz.NEUTRAL),
        (
            "choc (13 mars-12 avril)",
            (d["date"] >= "2020-03-13") & (d["date"] < "2020-04-13"),
            viz.MAGENTA,
        ),
        ("plancher (13 avril-30 juin)", d["date"] >= "2020-04-13", viz.VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for lab, m, c in phases:
        ax.scatter(
            d.loc[m, "realized"],
            d.loc[m, "estimated"],
            s=22,
            color=c,
            label=lab,
            edgecolor=viz.SURFACE,
            lw=0.8,
        )
    lim = [0, max(d["realized"].max(), d["estimated"].max()) * 1.05]
    ax.plot(lim, lim, color=viz.MUTED, lw=1, ls="--")
    ax.annotate(
        "estimé = réalisé", (lim[1] * 0.62, lim[1] * 0.66), rotation=38, fontsize=9, color=viz.INK2
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MAE log réalisée")
    ax.set_ylabel("MAE log estimée (DLE)")
    ax.set_title("Chaque point est un jour de 2020")
    ax.legend(loc="upper left")
    ax.grid(axis="x")
    fig.tight_layout()
    viz.save(fig, "e1_scatter")


@group("e1")
def e1_drift() -> None:
    from zonefleet.config import COVID_EMERGENCY

    u = pd.read_csv(T / "e1_univariate.csv", parse_dates=["date"])
    u = u[u["method"] == "kolmogorov_smirnov"]
    w = u.pivot_table(index="feature", columns=pd.Grouper(key="date", freq="W"), values="value")
    order = [
        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",
        "lag_336",
        "roll_mean_24",
        "roll_mean_168",
        "temperature_2m",
        "precipitation",
        "hour",
        "dow",
        "is_weekend",
        "is_holiday",
    ]
    w = w.reindex([o for o in order if o in w.index])
    mv = pd.read_csv(T / "e1_multivariate.csv", parse_dates=["date"])
    psi = pd.read_csv(T / "e1_psi.csv", parse_dates=["date"]).groupby("date")["psi"].max()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.5, 1]})
    im = axes[0].imshow(w, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_yticks(range(len(w)), w.index)
    ticks = list(range(0, w.shape[1], 4))
    axes[0].set_xticks(ticks, [w.columns[i].strftime("%d %b") for i in ticks])
    axes[0].grid(False)
    axes[0].set_title("Statistique KS hebdomadaire par feature (0 = identique à la référence)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    ax = axes[1]
    ax.plot(
        mv["date"], mv["value"] / mv["threshold"], color=viz.VIOLET, label="PCA : erreur / seuil"
    )
    ax.plot(psi.index, psi / 0.25, color=viz.AQUA, label="PSI max / 0,25")
    ax.axhline(1, color=viz.ALERT, lw=1, ls="--")
    ax.axvline(COVID_EMERGENCY, color=viz.INK2, lw=1)
    ax.set_yscale("log")
    viz.plain_log_axis(ax, "y")
    ax.set_ylabel("rapport au seuil (log)")
    ax.set_title(
        "Au-dessus de 1 : alerte. PCA sous son seuil ; PSI au-dessus dès janvier (température)"
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    viz.save(fig, "e1_derive")


@group("e1")
def e1_timeline() -> None:
    from zonefleet.config import COVID_EMERGENCY

    sig = pd.read_csv(T / "e1_signals.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    policy_timeline(ax, sig, COVID_EMERGENCY)
    ax.set_title("Jours en alerte par politique (après persistance de 2 jours), E1")
    fig.tight_layout()
    viz.save(fig, "e1_alertes")


@group("e2")
def e2_demand() -> None:
    from zonefleet.config import CONGESTION_START

    h = pd.read_csv(T / "e2_demand_change_by_hour.csv")
    d = pd.read_csv(T / "e2_demand_daily.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.bar(h["hour"] - 0.2, h["cbd"], width=0.38, color=viz.BLUE, label="zones CBD")
    ax.bar(h["hour"] + 0.2, h["hors_cbd"], width=0.38, color=viz.AQUA, label="zones hors CBD")
    ax.set_xticks(range(0, 24, 3), [f"{x}h" for x in range(0, 24, 3)])
    ax.set_ylabel("évolution janv.-juin 2025 / 2024 (%)")
    ax.set_title("Toute la ville progresse ; la nuit, le CBD progresse moins")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(d["date"], 100 * d["ratio_cbd"].rolling(7).mean(), color=viz.BLUE)
    ax.axvline(CONGESTION_START, color=viz.INK2, lw=1)
    ax.set_ylabel("part du CBD dans les 60 zones (%)")
    ax.set_title("Part du CBD (moyenne 7 jours)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout()
    viz.save(fig, "e2_demande")


@group("e2")
def e2_estimated_vs_realized() -> None:
    from zonefleet.config import CONGESTION_START

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, g, title in [(axes[0], "cbd", "Zones CBD"), (axes[1], "hors_cbd", "Zones hors CBD")]:
        perf = pd.read_csv(T / f"e2_performance_{g}.csv", parse_dates=["date"])
        perf_panel(ax, perf, "mae_log", f"{title} : erreur relative (MAE log)", CONGESTION_START)
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    viz.save(fig, "e2_estime_vs_realise")


@group("e2")
def e2_false_alarms() -> None:
    rows = []
    for e, lab in [("e1", "E1 (janv.-juin 2020)"), ("e2", "E2 (janv.-juin 2025)")]:
        p = pd.read_csv(T / f"{e}_psi.csv")
        r = p.assign(a=p["psi"] > 0.25).groupby("feature")["a"].mean()
        rows.append(r.rename(lab))
    t = pd.concat(rows, axis=1).loc[["temperature_2m", "roll_mean_168", "lag_168", "lag_24"]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(t))
    ax.barh(y + 0.2, 100 * t.iloc[:, 0], height=0.38, color=viz.VIOLET, label=t.columns[0])
    ax.barh(y - 0.2, 100 * t.iloc[:, 1], height=0.38, color=viz.YELLOW, label=t.columns[1])
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% des jours avec PSI > 0,25")
    ax.set_title("Anatomie des fausses alertes de la dérive seule")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, "e2_fausses_alertes")


@group("e2")
def e2_timeline() -> None:
    from zonefleet.config import CONGESTION_START

    sig = pd.read_csv(T / "e2_signals.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    policy_timeline(ax, sig, CONGESTION_START)
    ax.set_title("Jours en alerte par politique, E2 : seule la dérive seule alerte, tous les jours")
    fig.tight_layout()
    viz.save(fig, "e2_alertes")


# --------------------------------------------------------------------------- E3
FAMILY_COLOR = {
    "covariate nuisible": viz.BLUE,
    "covariate bénin": viz.AQUA,
    "concept pur": viz.MAGENTA,
    "choc propagé": viz.YELLOW,
}


@group("e3")
def e3_frontier() -> None:
    s = pd.read_csv(T / "e3_summary.csv")
    s = s[s["family"] != "aucune"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, w, title in [
        (axes[0], "transition", "14 premiers jours après l'injection"),
        (axes[1], "steady", "Régime établi (jour 15 et après)"),
    ]:
        lim = (-0.15, 0.42)
        ax.axhline(0, color=viz.GRID, lw=1)
        ax.axvline(0, color=viz.GRID, lw=1)
        ax.plot(lim, lim, color=viz.MUTED, ls="--", lw=1)
        for fam, c in FAMILY_COLOR.items():
            t = s[s["family"] == fam]
            ax.scatter(
                t[f"delta_real_{w}"],
                t[f"delta_est_{w}"],
                s=30 + 120 * t["severity"].rank(pct=True),
                color=c,
                label=fam,
                edgecolor=viz.SURFACE,
                lw=1,
            )
        ax.set_xlim(lim)
        ax.set_ylim(-0.15, 0.25)
        ax.set_xlabel("hausse de l'erreur réalisée (MAE log, vs sans dérive)")
        ax.set_title(title)
        ax.grid(axis="x")
    axes[0].set_ylabel("hausse de l'erreur estimée par DLE")
    axes[0].annotate("estimé = réalisé", (0.17, 0.19), rotation=33, fontsize=9, color=viz.INK2)
    axes[1].annotate("concept pur : DLE ne bouge pas", (0.20, 0.012), fontsize=9, color=viz.INK2)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    viz.save(fig, "e3_frontiere")


@group("e3")
def e3_series() -> None:
    from zonefleet.experiments.e3_frontier import ONSET

    d = pd.read_csv(T / "e3_daily.csv", parse_dates=["date"]).set_index("date")
    base = d[d["scenario"] == "aucune dérive"]
    picks = [
        ("covariate nuisible κ=2", "Covariate shift nuisible (κ = 2)"),
        ("covariate bénin κ=2", "Covariate shift bénin (κ = 2)"),
        ("concept pur x0.5", "Concept drift pur (demande x0,5, X gelé)"),
        ("choc propagé x0.5", "Choc de niveau propagé (x0,5)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, (sc, title) in zip(axes.flat, picks, strict=True):
        t = d[d["scenario"] == sc]
        ax.plot(base.index, base["realized"], color=viz.NEUTRAL, lw=1, label="réalisée sans dérive")
        ax.plot(t.index, t["estimated"], color=viz.BLUE, lw=1.6, label="estimée (DLE)")
        ax.plot(t.index, t["realized"], color=viz.ORANGE, lw=1.3, label="réalisée")
        ax.axvline(ONSET, color=viz.INK2, lw=1)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0, 0].legend(loc="upper right", fontsize=8.5)
    axes[0, 0].set_ylabel("MAE log journalière")
    axes[1, 0].set_ylabel("MAE log journalière")
    fig.tight_layout()
    viz.save(fig, "e3_series")


@group("e3")
def e3_transition() -> None:
    from zonefleet.experiments.e3_frontier import ONSET

    d = pd.read_csv(T / "e3_daily.csv", parse_dates=["date"]).set_index("date")
    t = d[d["scenario"] == "choc propagé x0.5"]
    t = t[(t.index >= ONSET - pd.Timedelta(days=14)) & (t.index < ONSET + pd.Timedelta(days=35))]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for k, lab in [(1, "lag 24 h"), (7, "lag 168 h"), (14, "lag 336 h")]:
        x = ONSET + pd.Timedelta(days=k)
        ax.axvline(x, color=viz.GRID, lw=1)
        ax.annotate(
            f"{lab} absorbe le choc",
            (x, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=viz.INK2,
            va="top",
        )
    ax.plot(t.index, t["estimated"], color=viz.BLUE, label="estimée (DLE)")
    ax.plot(t.index, t["realized"], color=viz.ORANGE, label="réalisée")
    ax.axvline(ONSET, color=viz.INK2, lw=1.2)
    ax.set_ylabel("MAE log journalière")
    ax.set_title(
        "Choc x0,5 le 3 mars : DLE ne voit la transition qu'à mesure que les lags la voient"
    )
    ax.legend(loc="center right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    viz.save(fig, "e3_transition")


# --------------------------------------------------------------------------- E4
POLICY_SHORT = {
    "P1_drift": "P1 dérive",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b dérive nettoyée ET DLE",
    "P4_realized": "P4 labels (D + 60 j)",
}
POLICY_COLOR = {
    "P1b_drift_clean": viz.AQUA,
    "P2_dle": viz.BLUE,
    "P3b_clean_and_dle": viz.VIOLET,
    "P4_realized": viz.ORANGE,
}


@group("e4")
def e4_tradeoff() -> None:
    s = pd.read_csv(T / "e4_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, w, title in [
        (axes[0], 1, "Signal journalier"),
        (axes[1], 7, "Signal lissé sur 7 jours"),
    ]:
        for pol, c in POLICY_COLOR.items():
            d = s[(s["smooth"] == w) & (s["policy"] == pol)].sort_values("k")
            ax.plot(
                d["false_alerts_per_30d"],
                100 * d["recall"],
                color=c,
                marker="o",
                ms=5,
                label=POLICY_SHORT[pol],
            )
            if pol in ("P2_dle", "P3b_clean_and_dle"):
                for _, r in d.iloc[[0, -1]].iterrows():
                    ax.annotate(
                        f"k={r['k']:g}",
                        (r["false_alerts_per_30d"], 100 * r["recall"]),
                        xytext=(5, 3),
                        textcoords="offset points",
                        fontsize=8,
                        color=viz.INK2,
                    )
        ax.set_xlabel("fausses alertes par 30 jours (allumages)")
        ax.set_title(title)
        ax.grid(axis="x")
    axes[0].set_ylabel("rappel sur les scénarios nuisibles (%)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        "Compromis rappel / fausses alertes, seuils k de 0,5 à 3 écarts-types",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    viz.save(fig, "e4_compromis")


@group("e4")
def e4_recall_by_kind() -> None:
    r = pd.read_csv(T / "e4_results.csv")
    r = r[(r["smooth"] == 1) & (r["k"] == 1.0) & r["harmful"]]
    t = r.groupby(["policy", "kind"])["detected"].mean().unstack()
    pols = ["P1b_drift_clean", "P2_dle", "P3b_clean_and_dle", "P4_realized"]
    kinds = [
        ("covariate_harmful", "covariate nuisible", viz.BLUE),
        ("level", "choc propagé", viz.YELLOW),
        ("concept", "concept pur", viz.MAGENTA),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(pols))
    for j, (kind, lab, c) in enumerate(kinds):
        n = r.loc[r["kind"] == kind, "id"].nunique()
        ax.bar(
            x + (j - 1) * 0.26,
            100 * t.loc[pols, kind],
            width=0.25,
            color=c,
            label=f"{lab} (n = {n})",
        )
    ax.set_xticks(x, [POLICY_SHORT[p] for p in pols])
    ax.set_ylabel("rappel (%)")
    ax.set_title("Rappel par type de dérive nuisible (k = 1, signal journalier)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, fontsize=9)
    fig.tight_layout()
    viz.save(fig, "e4_rappel_par_type")


@group("e4")
def e4_harm() -> None:
    r = pd.read_csv(T / "e4_results.csv")
    r = r[(r["smooth"] == 1) & (r["k"] == 1.0)].drop_duplicates("id")
    kinds = [
        ("none", "aucune", viz.NEUTRAL),
        ("covariate_benign", "covariate bénin", viz.AQUA),
        ("covariate_harmful", "covariate nuisible", viz.BLUE),
        ("level", "choc propagé", viz.YELLOW),
        ("concept", "concept pur", viz.MAGENTA),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rng = np.random.default_rng(0)
    for i, (kind, _lab, c) in enumerate(kinds):
        h = 100 * r.loc[r["kind"] == kind, "harm"]
        ax.scatter(
            h, i + rng.uniform(-0.22, 0.22, len(h)), s=16, color=c, edgecolor=viz.SURFACE, lw=0.6
        )
    ax.axvline(10, color=viz.ALERT, ls="--", lw=1)
    ax.annotate(
        "seuil de nuisance : +10 %",
        (10, 4.45),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color=viz.INK2,
    )
    ax.set_yticks(range(len(kinds)), [k[1] for k in kinds])
    ax.set_xlabel("hausse de la MAE log réalisée sur 28 jours, vs sans dérive (%)")
    ax.set_title("240 scénarios : qui est réellement nuisible ?")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    viz.save(fig, "e4_nuisance")


@group("e4")
def e4_delays() -> None:
    r = pd.read_csv(T / "e4_results.csv")
    r = r[(r["k"] == 1.0) & r["harmful"] & r["detected"]]
    pols = ["P1b_drift_clean", "P2_dle", "P3b_clean_and_dle", "P4_realized"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    rng = np.random.default_rng(1)
    for i, pol in enumerate(pols):
        for w, dy, mk in [(1, -0.15, "o"), (7, 0.15, "s")]:
            d = r[(r["policy"] == pol) & (r["smooth"] == w)]["delay_days"]
            ax.scatter(
                d,
                i + dy + rng.uniform(-0.06, 0.06, len(d)),
                s=14,
                marker=mk,
                color=POLICY_COLOR[pol],
                alpha=0.7 if w == 1 else 0.45,
                lw=0,
            )
    ax.set_yticks(range(len(pols)), [POLICY_SHORT[p] for p in pols])
    ax.set_xlabel("délai de détection sur l'horloge de décision (jours après l'injection)")
    ax.set_title("Délais : quelques jours pour les signaux avancés, deux mois pour les labels")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.annotate(
        "ronds : signal journalier · carrés : lissé 7 j",
        (0.99, 0.03),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color=viz.INK2,
    )
    fig.tight_layout()
    viz.save(fig, "e4_delais")


# --------------------------------------------------------------------------- E5
POLICY_E5 = {
    "jamais": viz.NEUTRAL,
    "calendrier": viz.BLUE,
    "réflexe": viz.YELLOW,
    "réflexe court": viz.MAGENTA,
    "gardé": viz.VIOLET,
}


@group("e5")
def e5_trajectories() -> None:
    d = pd.read_csv(T / "e5_daily.csv", parse_dates=["date"])
    sw = pd.read_csv(T / "e5_switches.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, delay, title in [
        (axes[0], 1, "Labels à J + 1 (journaux de courses de l'opérateur)"),
        (axes[1], 60, "Labels à J + 60 (publication TLC)"),
    ]:
        for pol, c in POLICY_E5.items():
            t = d[(d["delay"] == delay) & (d["policy"] == pol)].set_index("date")["mae_log"]
            ax.plot(
                t.index,
                t.rolling(7, min_periods=1).mean(),
                color=c,
                lw=1.8 if pol != "jamais" else 2.4,
                label=pol,
            )
            for x in sw[(sw["delay"] == delay) & (sw["policy"] == pol)]["date"]:
                ax.plot([x], [0.05], marker="|", ms=10, color=c, mew=2)
        ax.axvline(pd.Timestamp("2020-03-13"), color=viz.INK2, lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("MAE log (moy. 7 j)")
        ax.set_ylim(0, 1.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].legend(loc="upper right", ncols=5, fontsize=9)
    axes[0].annotate(
        "traits en bas : remplacements de modèle",
        (0.01, 0.12),
        xycoords="axes fraction",
        fontsize=8.5,
        color=viz.INK2,
    )
    fig.suptitle(
        "Erreur réalisée du modèle en service en 2020, selon la politique de réentraînement",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    viz.save(fig, "e5_trajectoires")


@group("e5")
def e5_bars() -> None:
    s = pd.read_csv(T / "e5_summary.csv")
    periods = ["choc (13 mars-12 avril)", "printemps (13 avril-30 juin)", "second semestre"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, delay in [(axes[0], 1), (axes[1], 60)]:
        t = s[s["delay"] == delay].set_index("policy")
        x = np.arange(len(periods))
        for j, (pol, c) in enumerate(POLICY_E5.items()):
            ax.bar(x + (j - 2) * 0.16, t.loc[pol, periods], width=0.15, color=c, label=pol)
        ax.set_xticks(x, ["choc", "printemps", "2nd semestre"])
        ax.set_title(f"L = {delay} jour{'s' if delay > 1 else ''}")
    axes[0].set_ylabel("MAE log moyenne")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    viz.save(fig, "e5_periodes")


@group("e4")
def e4_sensitivity() -> None:
    s = pd.read_csv(T / "e4_sensitivity.csv")
    params = [
        ("persistance", "persistance (jours)"),
        ("nuisance", "seuil de nuisance"),
        ("fenêtre", "fenêtre (jours)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharey="row")
    for j, (p, lab) in enumerate(params):
        for pol, c in POLICY_COLOR.items():
            d = s[(s["parameter"] == p) & (s["policy"] == pol)].sort_values("value")
            x = d["value"] * (100 if p == "nuisance" else 1)
            axes[0, j].plot(
                x, 100 * d["recall"], color=c, marker="o", ms=5, label=POLICY_SHORT[pol]
            )
            axes[1, j].plot(x, d["false_alerts_per_30d"], color=c, marker="o", ms=5)
        axes[1, j].set_xlabel(lab + (" (%)" if p == "nuisance" else ""))
        axes[0, j].set_xticks(x)
        axes[1, j].set_xticks(x)
    axes[0, 0].set_ylabel("rappel (%)")
    axes[1, 0].set_ylabel("fausses alertes / 30 j")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Sensibilité de E4 (k = 1, lissé 7 j) : un paramètre varie, les autres au protocole",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    viz.save(fig, "e4_sensibilite")


def main(argv: list[str]) -> None:
    viz.setup()
    names = argv or list(GROUPS)
    for name in names:
        for fn in GROUPS[name]:
            fn()


if __name__ == "__main__":
    main(sys.argv[1:])
