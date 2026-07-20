"""Enregistre les deux modèles figés du SPEC dans le registry MLflow local.

Usage : uv run python scripts/register_models.py
Le modèle 2024 (E2) devient `champion` : c'est lui que le graphe Dagster surveille.
"""

from __future__ import annotations

import numpy as np

from zonefleet.ops import registry
from zonefleet.pipeline import frozen_model, predictions


def reference_metrics(split: str) -> dict[str, float]:
    ref = predictions(split).query("period == 'reference'")
    err = np.abs(np.log1p(ref["pickups"]) - np.log1p(ref["q50"]))
    cover = ((ref["pickups"] >= ref["q10"]) & (ref["pickups"] <= ref["q90"])).mean()
    return {
        "mae_log_ref": float(err.mean()),
        "mae_ref": float((ref["pickups"] - ref["q50"]).abs().mean()),
        "coverage_80_ref": float(cover),
    }


def main() -> None:
    versions = {}
    for split in ("covid", "congestion"):
        versions[split] = registry.register(
            frozen_model(split), reference_metrics(split), tags={"split": split}
        )
        print(f"{split} : version {versions[split]}")
    registry.set_alias("champion", versions["congestion"])
    print("champion :", versions["congestion"])


if __name__ == "__main__":
    main()
