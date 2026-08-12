"""Référence d'API du package, générée depuis les docstrings (site/reference-api.qmd).

Usage : uv run python scripts/make_api_docs.py
Seules les fonctions et classes publiques (sans tiret bas initial) définies dans chaque
module sont listées, avec leur signature et la première partie de leur docstring.
"""

from __future__ import annotations

import importlib
import inspect

from zonefleet.config import ROOT

MODULES = [
    (
        "Données",
        [
            "zonefleet.config",
            "zonefleet.contract",
            "zonefleet.data",
            "zonefleet.covariates",
            "zonefleet.quality",
        ],
    ),
    (
        "Modèle",
        [
            "zonefleet.features",
            "zonefleet.model",
            "zonefleet.baselines",
            "zonefleet.metrics",
            "zonefleet.conformal",
        ],
    ),
    (
        "Monitoring",
        ["zonefleet.monitoring", "zonefleet.rowscore", "zonefleet.alerting", "zonefleet.scenarios"],
    ),
    (
        "Exploitation",
        ["zonefleet.report", "zonefleet.ops.decisions", "zonefleet.ops.registry", "zonefleet.cli"],
    ),
]


def doc_summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ")


def module_section(name: str) -> list[str]:
    mod = importlib.import_module(name)
    lines = [f"### `{name}`", "", doc_summary(mod), ""]
    members = [
        (n, o)
        for n, o in inspect.getmembers(mod)
        if not n.startswith("_")
        and (inspect.isfunction(o) or inspect.isclass(o))
        and getattr(o, "__module__", None) == name
    ]
    for n, o in members:
        try:
            sig = str(inspect.signature(o)).replace("'", "")  # annotations différées
        except (TypeError, ValueError):
            sig = "(...)"
        kind = "classe" if inspect.isclass(o) else "fonction"
        lines += [f"`{n}{sig}`", f": *{kind}*. {doc_summary(o)}", ""]
    return lines


def main() -> None:
    lines = [
        "---",
        'title: "Référence de l\'API"',
        'description: "Fonctions et classes publiques du package zonefleet, '
        'générées depuis les docstrings."',
        "---",
        "",
        "Page générée par `scripts/make_api_docs.py` : elle suit le code, "
        "elle ne se rédige pas à la main.",
        "",
    ]
    for section, mods in MODULES:
        lines += [f"## {section}", ""]
        for m in mods:
            lines += module_section(m)
    text = "\n".join(line.rstrip() for line in lines).rstrip() + "\n"
    (ROOT / "site" / "reference-api.qmd").write_text(text)
    print(sum(1 for line in lines if line.startswith("`")), "entrées")


if __name__ == "__main__":
    main()
