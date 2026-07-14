"""Figures des pages Données et Modèle (données réelles).

Usage : uv run python scripts/make_data_figures.py
Entrées : data/processed/*, results/tables/*, configs/zones.yaml,
          data/raw/ref/taxi_zones.zip (carte, optionnelle : `make data`)
Sortie : site/figures/data_*.png
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zonefleet import viz
from zonefleet.config import CONGESTION_START, COVID_EMERGENCY, DATA_RAW, RESULTS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.features import zone_table

T = RESULTS / "tables"


def city_volumes() -> None:
    """Courses journalières (jaune, vert) sur les deux blocs, chocs annotés."""
    d = pd.read_csv(T / "city_daily_pickups.csv", parse_dates=["day"]).set_index("day")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (a, b), shock, lab in [
        (axes[0], ("2019-01-01", "2020-12-31"), COVID_EMERGENCY, "état d'urgence\n13 mars 2020"),
        (
            axes[1],
            ("2024-01-01", "2025-12-31"),
            CONGESTION_START,
            "tarification de\ncongestion, 5 janv. 2025",
        ),
    ]:
        s = d.loc[a:b]
        ax.plot(
            s.index, s["yellow"].rolling(7).mean() / 1e3, color=viz.YELLOW, label="taxis jaunes"
        )
        ax.plot(s.index, s["green"].rolling(7).mean() / 1e3, color=viz.GREEN, label="taxis verts")
        ax.axvline(shock, color=viz.MUTED, ls="--", lw=1)
        ax.annotate(
            lab,
            (shock, 0.92),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=viz.INK2,
            va="top",
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[0].set_ylabel("milliers de courses / jour (moy. 7 j)")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.55, 0.55))
    axes[0].set_title("2019-2020 : le choc COVID")
    axes[1].set_title("2024-2025 : la tarification de congestion")
    fig.tight_layout()
    viz.save(fig, "data_volumes_ville")


def schema_timeline() -> None:
    """Présence et type des colonnes mois par mois (taxis jaunes)."""
    s = pd.read_csv(T / "schema_columns.csv", keep_default_na=False)  # "null" est un type
    s = s[s["service"] == "yellow"]
    s["col"] = s["column"].str.lower()
    keep = [
        "pulocationid",
        "tpep_pickup_datetime",
        "passenger_count",
        "ratecodeid",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
    ]
    s = s[s["col"].isin(keep)]
    months = sorted(s["month"].unique())
    codes = {"null": 1, "float": 2, "integer": 3, "timestamp": 4}
    mat = np.zeros((len(keep), len(months)))
    for _, r in s.iterrows():
        mat[keep.index(r["col"]), months.index(r["month"])] = codes.get(r["type"], 2)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([viz.SURFACE, viz.NEUTRAL, viz.AQUA, viz.BLUE, viz.VIOLET])
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_yticks(range(len(keep)), keep)
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(ticks, [months[i][:4] for i in ticks])
    ax.grid(False)
    ax.set_title("Schéma des fichiers jaunes, mois par mois (48 mois)")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=c, edgecolor=viz.MUTED, label=lab)
        for c, lab in [
            (viz.NEUTRAL, "type nul"),
            (viz.AQUA, "flottant"),
            (viz.BLUE, "entier"),
            (viz.VIOLET, "horodatage"),
            (viz.SURFACE, "absente"),
        ]
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    viz.save(fig, "data_schema_timeline")


def cbd_bimodal() -> None:
    """Distribution de la part de courses taxées par zone (2025)."""
    c = pd.read_csv(T / "cbd_share_all_zones.csv")
    c = c[c["trips"] > 1000]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(c["cbd_share"], bins=40, color=viz.BLUE, alpha=0.85)
    ax.axvline(0.8, color=viz.ALERT, lw=1.5, ls="--")
    ax.annotate(
        "seuil 0,8",
        (0.8, 0.9),
        xycoords=("data", "axes fraction"),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color=viz.INK2,
    )
    ax.set_xlabel("part 2025 des courses avec cbd_congestion_fee > 0 (zone de prise en charge)")
    ax.set_ylabel("nombre de zones")
    ax.set_title("La zone de tarification se lit dans les données : distribution bimodale")
    fig.tight_layout()
    viz.save(fig, "data_cbd_bimodal")


def zone_map() -> None:
    """Carte des 60 zones surveillées, CBD vs hors CBD."""
    shp = DATA_RAW / "ref" / "taxi_zones.zip"
    if not shp.exists():
        print("shapefile absent : carte non régénérée (make data)")
        return
    import geopandas as gpd

    g = gpd.read_file(f"zip://{shp}!taxi_zones/taxi_zones.shp")
    z = zone_table().rename(columns={"zone": "LocationID"})
    g = g.merge(z[["LocationID", "cbd"]], on="LocationID", how="left")
    g["cat"] = np.select([g["cbd"].eq(True), g["cbd"].eq(False)], ["cbd", "hors"], "autre")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, sub in zip(axes, [g, g[g["borough"] == "Manhattan"]], strict=True):
        colors = sub["cat"].map({"cbd": viz.BLUE, "hors": viz.AQUA, "autre": "#ecebe7"})
        sub.plot(ax=ax, color=colors, edgecolor=viz.SURFACE, linewidth=0.5)
        ax.set_axis_off()
    axes[0].set_title("Les 60 zones surveillées (New York)")
    axes[1].set_title("Zoom Manhattan : la zone de congestion")
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(color=viz.BLUE, label="surveillée, CBD (32)"),
            Patch(color=viz.AQUA, label="surveillée, hors CBD (28)"),
            Patch(color="#ecebe7", label="non surveillée"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    viz.save(fig, "data_carte_zones")


def hourly_profiles() -> None:
    """Profil heure x jour de la demande moyenne (2019), CBD vs hors CBD."""
    p = load_panel()
    p = p[p["ts"].dt.year == 2019].merge(zone_table()[["zone", "cbd"]], on="zone")
    p["hour"], p["dow"] = p["ts"].dt.hour, p["ts"].dt.dayofweek
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    days = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
    for ax, flag, title in [(axes[0], True, "Zones CBD"), (axes[1], False, "Zones hors CBD")]:
        m = p[p["cbd"] == flag].pivot_table(
            index="dow", columns="hour", values="pickups", aggfunc="mean"
        )
        im = ax.imshow(m, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_yticks(range(7), days)
        ax.set_xticks(range(0, 24, 3), [f"{h}h" for h in range(0, 24, 3)])
        ax.grid(False)
        ax.set_title(f"{title} (2019)")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "Courses moyennes par zone, selon l'heure et le jour", x=0.01, ha="left", fontweight="bold"
    )
    fig.tight_layout()
    viz.save(fig, "data_profils_horaires")


def intermittency() -> None:
    """Niveau moyen et part d'heures à zéro par zone (2019)."""
    p = load_panel()
    p = p[p["ts"].dt.year == 2019]
    g = p.groupby("zone")["pickups"].agg(mean="mean", zero=lambda s: (s == 0).mean())
    g = g.join(zone_table().set_index("zone")["cbd"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for flag, c, lab in [(True, viz.BLUE, "CBD"), (False, viz.AQUA, "hors CBD")]:
        s = g[g["cbd"] == flag]
        ax.scatter(
            s["mean"], 100 * s["zero"], s=28, color=c, label=lab, edgecolor=viz.SURFACE, lw=1
        )
    ax.set_xscale("log")
    viz.plain_log_axis(ax)
    ax.set_xlabel("courses moyennes par heure (échelle log)")
    ax.set_ylabel("% d'heures sans course")
    ax.set_title("60 zones : peu d'heures à zéro, mais deux ordres de grandeur de niveau")
    ax.legend(loc="upper right")
    fig.tight_layout()
    viz.save(fig, "data_intermittence")


def weather() -> None:
    """Température et précipitations journalières."""
    w = load_weather().set_index("ts")
    daily = w.resample("D").agg({"temperature_2m": "mean", "precipitation": "sum"}).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.2), sharex=True)
    axes[0].plot(daily.index, daily["temperature_2m"], color=viz.ORANGE, lw=1)
    axes[0].set_ylabel("°C (moy. jour)")
    axes[1].bar(daily.index, daily["precipitation"], color=viz.BLUE, width=1)
    axes[1].set_ylabel("mm / jour")
    axes[0].set_title("Météo Manhattan (Open-Meteo, ERA5) : la saison dérive par nature")
    fig.tight_layout()
    viz.save(fig, "data_meteo")


def summary() -> None:
    """Chiffres clés des pages Données (results/tables/data_summary.csv)."""
    d = pd.read_csv(T / "city_daily_pickups.csv", parse_dates=["day"]).set_index("day").fillna(0)
    tot = d["yellow"] + d["green"]
    p = load_panel()
    rows = {
        "jours_2019_moyenne_courses": tot.loc["2019"].mean(),
        "fev_2020_moyenne_courses": tot.loc["2020-02"].mean(),
        "avril_2020_moyenne_courses": tot.loc["2020-04"].mean(),
        "ratio_avril_fevrier_2020": tot.loc["2020-04"].mean() / tot.loc["2020-02"].mean(),
        "dec_2024_moyenne_courses": tot.loc["2024-12"].mean(),
        "fev_2025_moyenne_courses": tot.loc["2025-02"].mean(),
        "part_vert_2019": d.loc["2019", "green"].sum() / tot.loc["2019"].sum(),
        "part_vert_2025": d.loc["2025", "green"].sum() / tot.loc["2025"].sum(),
        "panel_lignes": len(p),
        "panel_zones": p["zone"].nunique(),
        "panel_heures_reparees_dst": int(p["dst_repaired"].sum() / p["zone"].nunique()),
        "part_60_zones_courses_2019": p.loc[p["ts"].dt.year == 2019, "pickups"].sum()
        / tot.loc["2019"].sum(),
    }
    pd.Series(rows).rename("valeur").round(4).to_csv(
        T / "data_summary.csv", index_label="indicateur"
    )


def main() -> None:
    viz.setup()
    summary()
    city_volumes()
    schema_timeline()
    cbd_bimodal()
    zone_map()
    hourly_profiles()
    intermittency()
    weather()


if __name__ == "__main__":
    main()
