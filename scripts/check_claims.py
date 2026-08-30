"""Vérifie les chiffres cités dans la prose contre results/tables.

Usage : uv run python scripts/check_claims.py [registre]  (défaut : configs/claims.yml)

Chaque entrée du registre donne le texte cité, les fichiers qui le citent et une
expression Python qui recalcule ce texte depuis les tables. Échec si la valeur recalculée
diffère du texte, ou si un fichier ne contient plus le texte (espaces et retours à la
ligne normalisés). Les expressions sont celles du dépôt, évaluées avec : `t(nom)` (table
de results/tables), `fr(x, d)` (format français) et `pct(x)` (proportion en « 51 % »).
"""

from __future__ import annotations

import re
import sys
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def fr(x: float, digits: int = 2) -> str:
    """Nombre au format français, sans zéro négatif (comme scripts/make_tables.py)."""
    if round(x, digits) == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return f"{fr(100 * x, digits)} %"


@cache
def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check(claims: list[dict], root: Path = ROOT) -> list[str]:
    errors = []
    env = {"t": t, "fr": fr, "pct": pct, "pd": pd}
    for c in claims:
        got = str(eval(c["expr"], env))  # noqa: S307 (expressions du dépôt)
        if got != c["text"]:
            errors.append(f"{c['id']} : recalculé « {got} », cité « {c['text']} »")
        for f in c["files"]:
            if normalize(c["text"]) not in normalize((root / f).read_text(encoding="utf-8")):
                errors.append(f"{c['id']} : « {c['text']} » absent de {f}")
    return errors


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else ROOT / "configs" / "claims.yml"
    claims = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = check(claims)
    for e in errors:
        print(e)
    print(f"{len(claims)} chiffres vérifiés, {len(errors)} écart(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
