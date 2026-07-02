"""Constantes du protocole (SPEC.md) et chemins du projet.

Tout ce qui fixe une décision de protocole vit ici, pour qu'un seul fichier fasse foi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "site" / "figures"

# Trois horloges (SPEC, section 3)
HORIZON_H = 24  # heures entre émission et cible
LABEL_DELAY_DAYS = 60  # délai de publication TLC
TZ = "America/New_York"

N_ZONES = 60
QUANTILES = (0.1, 0.5, 0.9)
SEASON_H = 168  # saisonnalité hebdomadaire, échelle du MASE


@dataclass(frozen=True)
class Split:
    """Découpage temporel d'une expérience (bornes incluses, au jour près)."""

    name: str
    train: tuple[str, str]
    reference: tuple[str, str]
    analysis: tuple[str, str]

    def period(self, which: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Bornes [début, fin) en horodatage horaire pour `train`, `reference` ou `analysis`."""
        start, end = getattr(self, which)
        return pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)


SPLITS = {
    "covid": Split(
        "covid",
        train=("2019-01-01", "2019-08-31"),
        reference=("2019-09-01", "2019-12-31"),
        analysis=("2020-01-01", "2020-06-30"),
    ),
    "congestion": Split(
        "congestion",
        train=("2024-01-01", "2024-08-31"),
        reference=("2024-09-01", "2024-12-31"),
        analysis=("2025-01-01", "2025-06-30"),
    ),
}

YEARS = (2019, 2020, 2024, 2025)
SERVICES = ("yellow", "green")
CONGESTION_START = pd.Timestamp("2025-01-05")
COVID_EMERGENCY = pd.Timestamp("2020-03-13")
