"""Extraction des notes de version depuis le CHANGELOG."""

from pathlib import Path

from scripts.release_notes import section

CHANGELOG = """# Journal

## [Non publié]

## [1.0.0] - 2026-09-02
Première version.
### Ajouté
- tout

## [0.10.0] - 2026-08-26
- avant

[1.0.0]: https://example.org/compare/v0.10.0...v1.0.0
"""


def test_section_stops_at_next_version():
    s = section(CHANGELOG, "1.0.0")
    assert s.startswith("Première version.") and "avant" not in s and "### Ajouté" in s


def test_missing_version_and_last_section():
    assert section(CHANGELOG, "2.0.0") is None
    assert section(CHANGELOG, "0.10.0") == "- avant\n"


def test_real_changelog_has_current_version():
    import tomllib

    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    assert section(Path("CHANGELOG.md").read_text(encoding="utf-8"), version)
