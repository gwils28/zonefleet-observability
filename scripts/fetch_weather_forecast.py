"""Télécharge et met en cache la prévision météo émise la veille (2024-2025).

Usage : uv run python scripts/fetch_weather_forecast.py
Sortie : data/processed/weather_forecast.csv (versionnée)
"""

from zonefleet.config import DATA_PROCESSED
from zonefleet.covariates import fetch_weather_forecast, to_local_naive


def main() -> None:
    df = to_local_naive(fetch_weather_forecast("2023-12-31", "2026-01-01")).round(2)
    df.to_csv(DATA_PROCESSED / "weather_forecast.csv", index=False)
    n_missing = int(df.isna().sum().sum())
    print(df.shape, df["ts"].min(), df["ts"].max(), n_missing, "valeurs manquantes")


if __name__ == "__main__":
    main()
