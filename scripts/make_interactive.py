"""Graphiques interactifs (plotly) des pages E1, E3 et E4, lus dans results/.

Usage : uv run python scripts/make_interactive.py
Sortie : site/_widgets/*.md (blocs HTML bruts inclus par {{< include >}}).
La bibliothèque plotly.js est chargée une fois par page depuis le CDN officiel.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from zonefleet import viz
from zonefleet.config import RESULTS, ROOT

T = RESULTS / "tables"
OUT = ROOT / "site" / "_widgets"
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=viz.INK2)


def layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=viz.INK)),
        paper_bgcolor=viz.SURFACE,
        plot_bgcolor=viz.SURFACE,
        font=FONT,
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0),
        separators=", ",
    )
    fig.update_xaxes(title=xtitle, showgrid=False, linecolor=viz.GRID)
    fig.update_yaxes(title=ytitle, gridcolor=viz.GRID, zeroline=False)
    return fig


def write(fig: go.Figure, name: str, first: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        div_id=f"zf-{name.replace('_', '-')}",
        full_html=False,
        include_plotlyjs="cdn" if first else False,
        config={"displaylogo": False, "responsive": True},
    )
    (OUT / f"{name}.md").write_text("```{=html}\n" + html + "\n```\n")


def e1() -> None:
    perf = pd.read_csv(T / "e1_performance.csv", parse_dates=["date"])
    d = perf[perf["metric"] == "mae_log"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"], y=d["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["lower"],
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(42,120,214,0.15)",
            name="IC de DLE",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["estimated"],
            name="estimée (sans labels)",
            line=dict(color=viz.BLUE, width=2),
            hovertemplate="%{y:.3f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["realized"],
            name="réalisée (D + 60 j)",
            line=dict(color=viz.ORANGE, width=2),
            hovertemplate="%{y:.3f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["threshold"],
            name="seuil",
            line=dict(color=viz.MUTED, width=1, dash="dash"),
            hovertemplate="%{y:.3f}",
        )
    )
    fig.add_vline(
        x=pd.Timestamp("2020-03-13").timestamp() * 1000, line=dict(color=viz.INK2, width=1)
    )
    write(
        layout(fig, "E1 : erreur relative estimée et réalisée, jour par jour", "", "MAE log"),
        "e1_estime_vs_realise",
    )


def e3() -> None:
    s = pd.read_csv(T / "e3_summary.csv")
    s = s[s["family"] != "aucune"]
    colors = {
        "covariate nuisible": viz.BLUE,
        "covariate bénin": viz.AQUA,
        "concept pur": viz.MAGENTA,
        "choc propagé": viz.YELLOW,
    }
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-0.15, 0.42],
            y=[-0.15, 0.42],
            mode="lines",
            name="estimé = réalisé",
            line=dict(color=viz.MUTED, dash="dash", width=1),
            hoverinfo="skip",
        )
    )
    for fam, c in colors.items():
        t = s[s["family"] == fam]
        fig.add_trace(
            go.Scatter(
                x=t["delta_real_steady"],
                y=t["delta_est_steady"],
                mode="markers",
                name=fam,
                marker=dict(color=c, size=12, line=dict(color=viz.SURFACE, width=1.5)),
                text=t["scenario"],
                customdata=t["tracking_steady"],
                hovertemplate=(
                    "<b>%{text}</b><br>réalisé %{x:+.3f}<br>estimé %{y:+.3f}"
                    "<br>suivi %{customdata:.2f}<extra></extra>"
                ),
            )
        )
    fig = layout(
        fig,
        "E3 : hausse estimée contre hausse réalisée (régime établi)",
        "hausse réalisée (MAE log, vs sans dérive)",
        "hausse estimée par DLE",
    )
    fig.update_layout(hovermode="closest")
    write(fig, "e3_frontiere")


def e4() -> None:
    s = pd.read_csv(T / "e4_summary.csv")
    names = {
        "P1b_drift_clean": ("P1b dérive nettoyée", viz.AQUA),
        "P2_dle": ("P2 DLE", viz.BLUE),
        "P3b_clean_and_dle": ("P3b nettoyée ET DLE", viz.VIOLET),
        "P4_realized": ("P4 labels", viz.ORANGE),
    }
    fig = go.Figure()
    for pol, (lab, c) in names.items():
        d = s[(s["policy"] == pol) & (s["smooth"] == 7)].sort_values("k")
        fig.add_trace(
            go.Scatter(
                x=d["false_alerts_per_30d"],
                y=100 * d["recall"],
                mode="lines+markers",
                name=lab,
                line=dict(color=c, width=2),
                marker=dict(size=9),
                customdata=d[["k", "median_delay"]],
                hovertemplate=(
                    f"<b>{lab}</b><br>k = %{{customdata[0]}}<br>rappel %{{y:.0f}} %"
                    "<br>fausses alertes %{x:.2f} / 30 j"
                    "<br>délai médian %{customdata[1]:.0f} j<extra></extra>"
                ),
            )
        )
    fig = layout(
        fig,
        "E4 : compromis rappel / fausses alertes (signal lissé 7 jours)",
        "fausses alertes par 30 jours",
        "rappel (%)",
    )
    fig.update_layout(hovermode="closest")
    write(fig, "e4_compromis")


def main() -> None:
    e1()
    e3()
    e4()
    print(sorted(p.name for p in OUT.glob("*.md")))


if __name__ == "__main__":
    main()
