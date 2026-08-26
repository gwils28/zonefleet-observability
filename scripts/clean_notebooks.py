"""Retire des notebooks publiés ce qui dépend de la machine et de l'heure d'exécution.

Usage : uv run python scripts/clean_notebooks.py [chemins...]  (défaut : site/notebooks/)
Supprime les métadonnées d'exécution des cellules (horodatages nbconvert) ; les sorties
sont conservées. Deux exécutions successives donnent alors le même fichier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def clean(path: Path) -> bool:
    """Nettoie un notebook ; renvoie vrai s'il a changé."""
    nb = nbformat.read(path, as_version=4)
    changed = False
    for cell in nb.cells:
        if "execution" in cell.get("metadata", {}):
            del cell["metadata"]["execution"]
            changed = True
    if changed:
        nbformat.write(nb, path)
    return changed


def main(argv: list[str]) -> None:
    paths = [Path(a) for a in argv] or sorted((ROOT / "site" / "notebooks").glob("*.ipynb"))
    for p in paths:
        print(f"{p.name} : {'nettoyé' if clean(p) else 'déjà propre'}")


if __name__ == "__main__":
    main(sys.argv[1:])
