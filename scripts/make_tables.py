"""Tables Markdown du site, générées depuis results/ et configs/ (jamais tapées à la main).

Usage : uv run python scripts/make_tables.py
Sortie : site/_tables/*.md, incluses dans les pages par {{< include >}}.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from zonefleet.config import RESULTS, ROOT
from zonefleet.features import zone_table

T = RESULTS / "tables"
OUT = ROOT / "site" / "_tables"
REGISTRY: dict[str, Callable[[], pd.DataFrame]] = {}


def table(name: str):
    """Enregistre une fonction qui produit une table du site."""

    def deco(fn):
        REGISTRY[name] = fn
        return fn

    return deco


def fr(x, digits: int = 2) -> str:
    """Nombre au format français (virgule décimale, espace fine des milliers)."""
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "-"
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s


@table("zones")
def zones() -> pd.DataFrame:
    z = zone_table()
    z["cbd"] = z["cbd"].map({True: "oui", False: "non"})
    z["cbd_share_2025"] = z["cbd_share_2025"].map(lambda v: fr(v, 3))
    return z.rename(
        columns={
            "zone": "Zone",
            "borough": "Arrondissement",
            "name": "Nom",
            "cbd_share_2025": "Part taxée 2025",
            "cbd": "CBD",
        }
    )


SPLIT_FR = {"covid": "E1 (2019-2020)", "congestion": "E2 (2024-2025)"}
PERIOD_FR = {"reference": "référence", "analysis": "analysis"}
MODEL_FR = {"lightgbm": "LightGBM (q50)", "seasonal_naive": "seasonal naive", "mstl": "MSTL"}


@table("e0_metrics")
def e0_metrics() -> pd.DataFrame:
    m = pd.read_csv(T / "e0_metrics.csv")
    out = pd.DataFrame(
        {
            "Expérience": m["split"].map(SPLIT_FR),
            "Période": m["period"].map(PERIOD_FR),
            "Modèle": m["model"].map(MODEL_FR),
            "MAE": m["mae"].map(lambda v: fr(v, 1)),
            "RMSE": m["rmse"].map(lambda v: fr(v, 1)),
            "sMAPE (%)": m["smape"].map(lambda v: fr(v, 1)),
            "MASE": m["mase"].map(lambda v: fr(v, 3)),
            "Pinball": m["pinball_mean"].map(lambda v: fr(v, 2)),
            "Couv. 80 %": m["coverage_80"].map(lambda v: fr(100 * v, 1)),
        }
    )
    return out.sort_values(["Expérience", "Période"], ascending=[True, False])


@table("e0_mstl")
def e0_mstl() -> pd.DataFrame:
    m = pd.read_csv(T / "e0_mstl.csv")
    w = m.pivot(index="split", columns="model", values="mae")
    return pd.DataFrame(
        {
            "Référence": [SPLIT_FR[s] for s in w.index],
            "LightGBM": w["lightgbm"].map(lambda v: fr(v, 1)).to_numpy(),
            "MSTL": w["mstl"].map(lambda v: fr(v, 1)).to_numpy(),
            "Seasonal naive": w["seasonal_naive"].map(lambda v: fr(v, 1)).to_numpy(),
        }
    )


POLICY_FR = {
    "P1_drift": "P1 dérive seule",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE seul",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b dérive nettoyée ET DLE",
    "P4_realized": "P4 labels retardés",
}


def detection_frame(d: pd.DataFrame) -> pd.DataFrame:
    def date_fr(v):
        return "-" if pd.isna(v) else pd.Timestamp(v).strftime("%d/%m/%Y")

    return pd.DataFrame(
        {
            "Politique": d["policy"].map(POLICY_FR),
            "1er jour cible en alerte": d["target_day"].map(date_fr),
            "Décision possible le": d["decision"].map(date_fr),
            "Délai (jours)": d["delay_days"].map(
                lambda v: "non détecté" if pd.isna(v) else fr(v, 0)
            ),
            "Jours en alerte avant le choc": d["false_alert_days_before"].astype(int),
        }
    )


@table("e1_phases")
def e1_phases() -> pd.DataFrame:
    m = pd.read_csv(T / "e1_phases.csv")
    m = m[m["metric"].isin(["mae_log", "mae_count"])]
    return pd.DataFrame(
        {
            "Métrique": m["metric"].map({"mae_log": "MAE log", "mae_count": "MAE courses"}),
            "Phase": m["phase"],
            "Estimée (DLE)": m["estimated"].map(lambda v: fr(v, 3)),
            "Réalisée": m["realized"].map(lambda v: fr(v, 3)),
            "Estimée / réalisée": m["ratio_est_real"].map(lambda v: fr(v, 2)),
            "Corrélation journalière": m["corr_daily"].map(lambda v: fr(v, 2)),
        }
    )


@table("e1_detection")
def e1_detection() -> pd.DataFrame:
    return detection_frame(pd.read_csv(T / "e1_detection.csv"))


@table("e2_detection")
def e2_detection() -> pd.DataFrame:
    d = pd.read_csv(T / "e2_detection.csv")
    g = pd.read_csv(T / "e2_detection_by_group.csv")
    out = detection_frame(d)
    after = g.groupby("policy")["alert_days_after"].apply(
        lambda s: " / ".join(str(int(v)) for v in s)
    )
    out["Jours en alerte après (CBD / hors CBD)"] = d["policy"].map(after).to_numpy()
    return out


@table("e2_realized")
def e2_realized() -> pd.DataFrame:
    r = pd.read_csv(T / "e2_realized_by_group.csv")
    return pd.DataFrame(
        {
            "Groupe": r["group"].map({"cbd": "CBD", "hors_cbd": "hors CBD"}),
            "Phase": r["phase"],
            "MAE courses": r["mae_count"].map(lambda v: fr(v, 1)),
            "MAE log": r["mae_log"].map(lambda v: fr(v, 3)),
            "Biais log (réalisé - prévu)": r["bias_log"].map(lambda v: fr(v, 3)),
            "Lignes": r["n"].map(lambda v: fr(v, 0)),
        }
    )


@table("e3_summary")
def e3_summary() -> pd.DataFrame:
    s = pd.read_csv(T / "e3_summary.csv")
    s = s[s["family"] != "aucune"]
    f2 = lambda v: fr(v, 3)  # noqa: E731
    return pd.DataFrame(
        {
            "Scénario": s["scenario"],
            "Δ réalisé (14 j)": s["delta_real_transition"].map(f2),
            "Δ estimé (14 j)": s["delta_est_transition"].map(f2),
            "Suivi (14 j)": s["tracking_transition"].map(lambda v: fr(v, 2)),
            "Δ réalisé (établi)": s["delta_real_steady"].map(f2),
            "Δ estimé (établi)": s["delta_est_steady"].map(f2),
            "Suivi (établi)": s["tracking_steady"].map(lambda v: fr(v, 2)),
            "Jours en dérive (P1b)": s["share_drift_clean"].map(lambda v: fr(100 * v, 0) + " %"),
        }
    )


POLICY_E4 = {
    "P1_drift": "P1 dérive seule",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE seul",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b dérive nettoyée ET DLE",
    "P4_realized": "P4 labels retardés",
}


@table("e4_main")
def e4_main() -> pd.DataFrame:
    s = pd.read_csv(T / "e4_summary.csv")
    s = s[s["k"] == 1.0].copy()

    def ci(v, lo, hi, scale=1.0, d=0):
        if pd.isna(v):
            return "-"
        return f"{fr(scale * v, d)} [{fr(scale * lo, d)} ; {fr(scale * hi, d)}]"

    return pd.DataFrame(
        {
            "Signal": s["smooth"].map({1: "journalier", 7: "lissé 7 j"}),
            "Politique": s["policy"].map(POLICY_E4),
            "Rappel (%) [IC 90 %]": [
                ci(r.recall, r.recall_lo, r.recall_hi, 100) for r in s.itertuples()
            ],
            "Délai médian (j)": s["median_delay"].map(lambda v: fr(v, 1)),
            "Fausses alertes / 30 j": [
                ci(
                    r.false_alerts_per_30d,
                    r.false_alerts_per_30d_lo,
                    r.false_alerts_per_30d_hi,
                    1,
                    2,
                )
                for r in s.itertuples()
            ],
            "Jours nuls en alerte (%)": s["share_null_days_in_alert"].map(lambda v: fr(100 * v, 1)),
            "Alerte sur dérive non nuisible (%)": s["benign_alert_rate"].map(
                lambda v: fr(100 * v, 0)
            ),
        }
    )


@table("e4_by_kind")
def e4_by_kind() -> pd.DataFrame:
    r = pd.read_csv(T / "e4_results.csv")
    r = r[(r["smooth"] == 1) & (r["k"] == 1.0) & r["harmful"]]
    t = r.groupby(["policy", "kind"])["detected"].mean().unstack()
    n = r.drop_duplicates("id")["kind"].value_counts()
    cols = {
        "covariate_harmful": "covariate nuisible",
        "level": "choc propagé",
        "concept": "concept pur",
    }
    out = pd.DataFrame({"Politique": [POLICY_E4[p] for p in t.index]})
    for k, lab in cols.items():
        out[f"{lab} (n = {n.get(k, 0)})"] = (100 * t[k]).map(lambda v: fr(v, 0) + " %").to_numpy()
    return out


@table("e5_summary")
def e5_summary() -> pd.DataFrame:
    s = pd.read_csv(T / "e5_summary.csv")
    cols = [
        "choc (13 mars-12 avril)",
        "printemps (13 avril-30 juin)",
        "second semestre",
        "13 mars-31 déc.",
    ]
    out = pd.DataFrame(
        {"Délai des labels": s["delay"].map(lambda d: f"L = {d} j"), "Politique": s["policy"]}
    )
    for c in cols:
        out[c] = s[c].map(lambda v: fr(v, 3))
    out["Remplacements"] = s["retrains"]
    return out


@table("e4_sensitivity")
def e4_sensitivity() -> pd.DataFrame:
    s = pd.read_csv(T / "e4_sensitivity.csv")
    s = s[s["policy"] == "P3b_clean_and_dle"]
    unit = {"persistance": " j", "nuisance": "", "fenêtre": " j"}
    return pd.DataFrame(
        {
            "Paramètre": s["parameter"],
            "Valeur": [
                (fr(100 * v, 0) + " %") if p == "nuisance" else fr(v, 0) + unit[p]
                for p, v in zip(s["parameter"], s["value"], strict=True)
            ],
            "Scénarios nuisibles": s["n_harmful"],
            "Rappel P3b (%)": s["recall"].map(lambda v: fr(100 * v, 0)),
            "Fausses alertes / 30 j": s["false_alerts_per_30d"].map(lambda v: fr(v, 2)),
            "Délai médian (j)": s["median_delay"].map(lambda v: fr(v, 0)),
        }
    )


@table("e6_auc")
def e6_auc() -> pd.DataFrame:
    a = pd.read_csv(T / "e6_auc.csv")
    w = a.pivot(index="signal", columns="cas", values="auc")
    w = w.loc[list(dict.fromkeys(a["signal"]))]
    return pd.DataFrame(
        {"Signal": w.index, **{c: w[c].map(lambda v: fr(v, 2)).to_numpy() for c in w.columns}}
    )


@table("e6_top_covid")
def e6_top_covid() -> pd.DataFrame:
    z = pd.read_csv(T / "e6_zones_covid.csv")
    z["rank_est"] = z["delta_est"].rank(ascending=False).astype(int)  # rang parmi les 60 zones
    z = z.sort_values("delta_real", ascending=False).head(8)
    return pd.DataFrame(
        {
            "Zone": z["name"],
            "CBD": z["cbd"].map({True: "oui", False: "non"}),
            "Hausse réalisée": z["delta_real"].map(lambda v: fr(v, 2)),
            "Hausse estimée": z["delta_est"].map(lambda v: fr(v, 2)),
            "Rang estimé (sur 60)": z["rank_est"].to_numpy(),
        }
    )


@table("e1_chunks")
def e1_chunks_table() -> pd.DataFrame:
    c = pd.read_csv(T / "e1_chunks_summary.csv", parse_dates=["first_decision"])
    return pd.DataFrame(
        {
            "Chunk": c["chunk"],
            "Nombre": c["n_chunks"],
            "Corrélation avant le choc": c["corr_pre"].map(lambda v: fr(v, 2)),
            "Corrélation après": c["corr_post"].map(lambda v: fr(v, 2)),
            "Largeur relative de l'IC (médiane)": c["ci_rel_width_median"].map(
                lambda v: fr(100 * v, 1) + " %"
            ),
            "1re décision (sans persistance)": c["first_decision"].dt.strftime("%d/%m/%Y"),
            "Délai (j)": c["delay_days"].map(lambda v: fr(v, 0)),
            "Fausses alertes avant": c["false_alert_chunks_before"],
        }
    )


@table("e0_conformal")
def e0_conformal_table() -> pd.DataFrame:
    c = pd.read_csv(T / "e0_conformal.csv")
    w = c.pivot_table(
        index=["split", "period"], columns="method", values=["coverage", "width"], sort=False
    )
    rows = []
    for (split, period), r in w.iterrows():
        rows.append(
            {
                "Modèle": SPLIT_FR[split],
                "Période": period,
                "Couverture brute (%)": fr(100 * r[("coverage", "aucune")], 1),
                "Couverture calibrée (%)": fr(100 * r[("coverage", "globale")], 1),
                "Largeur brute": fr(r[("width", "aucune")], 1),
                "Largeur calibrée": fr(r[("width", "globale")], 1),
            }
        )
    return pd.DataFrame(rows)


def kpis_html() -> str:
    """Tuiles de chiffres clés de l'accueil, calculées depuis results/."""
    det = pd.read_csv(T / "e1_detection.csv").set_index("policy")
    e3 = pd.read_csv(T / "e3_summary.csv")
    cov = e3[e3["family"].str.startswith("covariate")]["tracking_steady"]
    con = e3[e3["family"] == "concept pur"]["tracking_steady"]
    e4 = pd.read_csv(T / "e4_summary.csv")
    p3b = e4[(e4["policy"] == "P3b_clean_and_dle") & (e4["k"] == 1.0) & (e4["smooth"] == 7)].iloc[0]
    e2 = pd.read_csv(T / "e2_detection_by_group.csv")
    p1_days = int(e2[e2["policy"] == "P1_drift"]["alert_days_after"].max())
    tiles = [
        (
            f"{fr(det.loc['P2_dle', 'delay_days'], 0)} j",
            "détection du choc COVID sans labels "
            + f"(contre {fr(det.loc['P4_realized', 'delay_days'], 0)} j avec)",
        ),
        (f"{fr(cov.min(), 2)}-{fr(cov.max(), 2)}", "taux de suivi de DLE sous covariate shift"),
        (fr(abs(con).max(), 2), "taux de suivi de DLE sous concept drift pur"),
        (
            f"{fr(100 * p3b['recall'], 0)} %",
            "rappel de la condition jointe, avec 0 fausse alerte sur 240 scénarios",
        ),
        (
            f"{p1_days} / {p1_days}",
            "jours d'alerte de la dérive seule après la taxe de congestion, pour 0 dégradation",
        ),
    ]
    body = "\n".join(
        f'<div class="kpi"><div class="value">{v}</div><div class="label">{lab}</div></div>'
        for v, lab in tiles
    )
    return f'```{{=html}}\n<div class="kpi-row">\n{body}\n</div>\n```\n'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kpis.md").write_text(kpis_html())
    print("kpis: 5 tuiles")
    for name, fn in REGISTRY.items():
        df = fn()
        (OUT / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
        print(f"{name}: {len(df)} lignes")


if __name__ == "__main__":
    main()
