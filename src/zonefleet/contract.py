"""Data contract des fichiers TLC : validation de schéma et journal des ruptures.

Le contract distingue trois cas :
- violation bloquante (colonne requise absente, type incompatible) ;
- rupture connue et datée (`known_breaks`) : journalisée, attendue ;
- rupture inconnue (nouvelle colonne, changement de type) : journalisée comme alerte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from zonefleet.config import CONFIGS

FILE_RE = re.compile(r"(?P<service>yellow|green)_tripdata_(?P<month>\d{4}-\d{2})\.parquet$")


@dataclass
class ContractReport:
    """Résultat de la validation d'un fichier."""

    file: str
    service: str
    month: str
    violations: list[str] = field(default_factory=list)
    known_breaks: list[str] = field(default_factory=list)
    unknown_breaks: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def load_contract(path: Path | None = None) -> dict:
    """Charge le contract YAML (par défaut `configs/data_contract.yaml`)."""
    return yaml.safe_load((path or CONFIGS / "data_contract.yaml").read_text())


def type_family(t: pa.DataType) -> str:
    """Ramène un type Arrow à une famille stable (integer, float, timestamp, string, null)."""
    if pa.types.is_integer(t):
        return "integer"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_null(t):
        return "null"
    return str(t)


def parse_name(path: Path) -> tuple[str, str]:
    """Extrait (service, mois AAAA-MM) du nom de fichier TLC."""
    m = FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"nom de fichier TLC inattendu : {path.name}")
    return m["service"], m["month"]


def validate_schema(schema: pa.Schema, service: str, month: str, contract: dict) -> ContractReport:
    """Valide un schéma Arrow contre le contract."""
    rep = ContractReport(file="", service=service, month=month)
    names = list(schema.names)
    rep.types = {n: type_family(schema.field(n).type) for n in names}
    pickup = contract["services"][service]["pickup_column"]
    required = {
        "pickup_datetime": pickup,
        "PULocationID": "PULocationID",
        "DOLocationID": "DOLocationID",
    }
    for key, col in required.items():
        expected = contract["required"][key]["type"]
        if col not in names:
            rep.violations.append(f"colonne requise absente : {col}")
        # Un entier stocké en flottant reste exploitable, un autre type non.
        elif rep.types[col] != expected and (expected, rep.types[col]) != ("integer", "float"):
            rep.violations.append(f"{col} : type {rep.types[col]} au lieu de {expected}")
    known = {c.lower() for c in contract["known_optional"]}
    dropoff = pickup.lower().replace("pickup", "dropoff")
    known |= {pickup.lower(), dropoff, "pulocationid", "dolocationid"}
    breaks = {b["column"]: b for b in contract.get("known_breaks", [])}
    for n in names:
        if n in breaks and month >= breaks[n]["since"]:
            rep.known_breaks.append(f"{n} : {breaks[n]['note']}")
        elif n.lower() not in known:
            rep.unknown_breaks.append(f"colonne inconnue : {n}")
    return rep


def validate_file(path: Path, contract: dict | None = None) -> ContractReport:
    """Valide un fichier Parquet TLC (lecture du schéma seulement, sans charger les données)."""
    contract = contract or load_contract()
    service, month = parse_name(path)
    rep = validate_schema(pq.read_schema(path), service, month, contract)
    rep.file = path.name
    return rep


def schema_timeline(paths: list[Path], contract: dict | None = None) -> list[ContractReport]:
    """Valide une liste de fichiers et renvoie les rapports triés par service puis mois."""
    contract = contract or load_contract()
    reps = [validate_file(p, contract) for p in paths]
    return sorted(reps, key=lambda r: (r.service, r.month))


def type_changes(reports: list[ContractReport]) -> list[str]:
    """Détecte les changements de famille de type d'une colonne d'un mois au suivant."""
    out: list[str] = []
    last: dict[tuple[str, str], tuple[str, str]] = {}
    for r in reports:
        for col, fam in r.types.items():
            key = (r.service, col.lower())
            if key in last and last[key][0] != fam and "null" not in (fam, last[key][0]):
                prev, since = last[key]
                out.append(f"{r.service} {col} : {prev} ({since}) -> {fam} ({r.month})")
            last[key] = (fam, r.month)
    return out
