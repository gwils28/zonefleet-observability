"""Construit le panel zone x heure et la configuration des zones.

Usage : uv run python scripts/build_panel.py
Sorties : data/processed/panel.parquet, configs/zones.yaml, results/tables/schema_*.csv
"""

from __future__ import annotations

import pandas as pd
import yaml

from zonefleet.config import CONFIGS, DATA_PROCESSED, DATA_RAW, RESULTS
from zonefleet.contract import schema_timeline, type_changes
from zonefleet.data import aggregate_all, build_panel, cbd_share, check_panel, tlc_files, top_zones

CBD_THRESHOLD = 0.8  # bimodalité nette : zones CBD > 0,93, autres < 0,6
BLOCKS = [("2019-01-01", "2021-01-01"), ("2024-01-01", "2026-01-01")]


def schema_tables(files, tables) -> None:
    """Journal des schémas mois par mois et des changements de type."""
    reps = schema_timeline(files)
    rows = [
        {
            "file": r.file,
            "service": r.service,
            "month": r.month,
            "ok": r.ok,
            "n_columns": len(r.types),
            "known_breaks": " | ".join(r.known_breaks),
            "unknown_breaks": " | ".join(r.unknown_breaks),
        }
        for r in reps
    ]
    pd.DataFrame(rows).to_csv(tables / "schema_timeline.csv", index=False)
    changes = pd.DataFrame({"change": type_changes(reps)})
    changes.to_csv(tables / "schema_type_changes.csv", index=False)


def zone_meta(zones, share) -> list[dict]:
    """Nom, arrondissement et indicateur CBD de chaque zone retenue."""
    lookup = pd.read_csv(DATA_RAW / "ref" / "taxi_zone_lookup.csv").set_index("LocationID")
    meta = []
    for z in zones:
        s = float(share.loc[z, "cbd_share"]) if z in share.index else 0.0
        meta.append(
            {
                "zone": z,
                "borough": str(lookup.loc[z, "Borough"]),
                "name": str(lookup.loc[z, "Zone"]),
                "cbd_share_2025": round(s, 4),
                "cbd": bool(s > CBD_THRESHOLD),
            }
        )
    return meta


def main() -> None:
    files = tlc_files()
    tables = RESULTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    schema_tables(files, tables)

    long = aggregate_all(files)
    zones = top_zones(long)
    share = cbd_share([f for f in files if "2025" in f.name]).set_index("zone")
    share.reset_index().to_csv(tables / "cbd_share_all_zones.csv", index=False)
    meta = zone_meta(zones, share)
    header = (
        "# 60 zones les plus actives en 2019 (jaune + vert), liste figée (SPEC, section 2).\n"
        f"# cbd : part 2025 des courses avec cbd_congestion_fee > 0 supérieure à {CBD_THRESHOLD}.\n"
    )
    body = yaml.safe_dump({"zones": meta}, allow_unicode=True, sort_keys=False)
    (CONFIGS / "zones.yaml").write_text(header + body)

    panel = pd.concat([build_panel(long, zones, a, b) for a, b in BLOCKS], ignore_index=True)
    issues = check_panel(panel)
    if issues:
        raise SystemExit(f"panel invalide : {issues}")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(DATA_PROCESSED / "panel.parquet", index=False)

    city = long.assign(day=long["ts"].dt.floor("D")).groupby(["day", "service"])["pickups"].sum()
    city.unstack().to_csv(tables / "city_daily_pickups.csv")
    print(f"panel : {panel.shape}, zones CBD : {sum(m['cbd'] for m in meta)}/{len(meta)}")


if __name__ == "__main__":
    main()
