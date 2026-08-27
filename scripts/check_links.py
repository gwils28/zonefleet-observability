"""Vérifie les liens internes du site rendu : fichier cible, ancre, images, textes alternatifs.

Usage : uv run python scripts/check_links.py [dossier]  (défaut : site/_site)
Code de sortie 1 si un lien est cassé. Les liens externes ne sont pas suivis.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BASE = "/zonefleet-observability/"  # préfixe GitHub Pages des liens absolus (page 404)


class Page(HTMLParser):
    """Collecte les identifiants, les liens et les sources d'images d'une page."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.images: list[str] = []
        self.no_alt: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        for key in ("id", "name"):
            if a.get(key):
                self.ids.add(str(a[key]))
        if tag == "a" and a.get("href"):
            self.links.append(str(a["href"]))
        if tag == "img" and a.get("src"):
            self.images.append(str(a["src"]))
            if not (a.get("alt") or "").strip():
                self.no_alt.append(str(a["src"]))


def parse(path: Path) -> Page:
    p = Page()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return p


def is_internal(url: str) -> bool:
    s = urlsplit(url)
    return not s.scheme and not s.netloc and not url.startswith(("mailto:", "javascript:"))


def resolve(site: Path, page: Path, path: str) -> Path:
    """Fichier visé par un chemin relatif à la page, ou absolu sous le préfixe du site."""
    if path.startswith(BASE):
        return (site / path[len(BASE) :]).resolve()
    if path.startswith("/"):
        return (site / path.lstrip("/")).resolve()
    return (page.parent / path).resolve()


def check(site: Path) -> list[str]:
    pages = {p: parse(p) for p in site.rglob("*.html")}
    errors = []
    for path, page in pages.items():
        rel = path.relative_to(site)
        for url in page.no_alt:
            errors.append(f"{rel} : image sans texte alternatif {url}")
        for url in page.images:
            if is_internal(url) and not url.startswith("data:"):
                target = resolve(site, path, unquote(urlsplit(url).path))
                if not target.exists():
                    errors.append(f"{rel} : image introuvable {url}")
        for url in page.links:
            if not is_internal(url):
                continue
            parts = urlsplit(url)
            target = path if not parts.path else resolve(site, path, unquote(parts.path))
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{rel} : cible introuvable {url}")
                continue
            frag = unquote(parts.fragment)
            if frag and target.suffix == ".html":
                ids = pages[target].ids if target in pages else parse(target).ids
                if frag not in ids:
                    errors.append(f"{rel} : ancre absente {url}")
    return sorted(set(errors))


def main(argv: list[str]) -> int:
    site = Path(argv[0]) if argv else ROOT / "site" / "_site"
    errors = check(site.resolve())
    for e in errors:
        print(e)
    print(f"{len(errors)} lien(s) cassé(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
