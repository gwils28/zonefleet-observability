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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in REGISTRY.items():
        df = fn()
        (OUT / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
        print(f"{name}: {len(df)} lignes")


if __name__ == "__main__":
    main()
