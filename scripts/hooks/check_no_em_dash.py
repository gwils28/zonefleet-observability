"""Hook pre-commit : refuse le tiret cadratin (U+2014) dans les fichiers texte indexés.

Règle typographique du projet (CLAUDE.md) : virgule, deux-points, parenthèses ou tiret
simple à la place.
"""

from __future__ import annotations

import sys
from pathlib import Path

DASH = chr(0x2014)


def main(paths: list[str]) -> int:
    bad = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if DASH in line:
                bad.append(f"{p}:{i}")
    for b in bad:
        print(f"tiret cadratin interdit : {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
