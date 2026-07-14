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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in REGISTRY.items():
        df = fn()
        (OUT / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
        print(f"{name}: {len(df)} lignes")


if __name__ == "__main__":
    main()
