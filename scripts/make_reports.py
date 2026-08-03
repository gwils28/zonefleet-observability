"""Exemples de rapports quotidiens publiés sur le site (site/_reports/*.md).

Usage : uv run python scripts/make_reports.py
"""

from __future__ import annotations

from zonefleet.cli import daily_report
from zonefleet.config import ROOT

EXAMPLES = {
    "covid_2020-03-23": ("covid", "2020-03-23"),
    "congestion_2025-03-10": ("congestion", "2025-03-10"),
}


def main() -> None:
    out = ROOT / "site" / "_reports"
    out.mkdir(parents=True, exist_ok=True)
    for name, (split, day) in EXAMPLES.items():
        text = daily_report(split, day)
        body = text.split("\n", 1)[1]  # le titre est porté par la page
        (out / f"{name}.md").write_text(body.replace("\n## ", "\n#### "))
        print(name)


if __name__ == "__main__":
    main()
