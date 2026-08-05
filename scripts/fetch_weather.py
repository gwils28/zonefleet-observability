"""Télécharge et met en cache la météo horaire locale de Manhattan.

Usage : uv run python scripts/fetch_weather.py
Sortie : data/processed/weather.csv (versionnée, ~2 Mo)
"""

import pandas as pd

from zonefleet.config import DATA_PROCESSED
from zonefleet.covariates import fetch_weather, to_local_naive

BLOCKS = [("2018-12-31", "2022-01-01"), ("2023-12-31", "2026-01-01")]


def main() -> None:
    parts = [to_local_naive(fetch_weather(a, b)) for a, b in BLOCKS]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    df = df.round(2)
    df.to_csv(DATA_PROCESSED / "weather.csv", index=False)
    n_missing = int(df.isna().sum().sum())
    print(df.shape, df["ts"].min(), df["ts"].max(), n_missing, "valeurs manquantes")


if __name__ == "__main__":
    main()
