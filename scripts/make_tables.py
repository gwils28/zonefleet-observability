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
    "P2_dle": "P2 DLE seul",
    "P3_drift_and_dle": "P3 dérive ET DLE",
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in REGISTRY.items():
        df = fn()
        (OUT / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
        print(f"{name}: {len(df)} lignes")


if __name__ == "__main__":
    main()
