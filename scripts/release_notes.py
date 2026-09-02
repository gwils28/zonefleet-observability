"""Extrait du CHANGELOG la section d'une version (notes de release).

Usage : uv run python scripts/release_notes.py 1.0.0 > notes.md
Code de sortie 1 si la version est absente du CHANGELOG.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def section(changelog: str, version: str) -> str | None:
    """Texte entre « ## [version] » et l'en-tête suivant (ou les liens), sans l'en-tête."""
    # fin de section : version suivante, ou définitions de liens en bas de fichier
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)"
    m = re.search(pattern, changelog, re.M | re.S)
    return m.group(1).strip() + "\n" if m else None


def main(argv: list[str]) -> int:
    version = argv[0].removeprefix("v")
    text = section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if text is None:
        print(f"version {version} absente du CHANGELOG", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
