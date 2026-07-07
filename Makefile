.PHONY: help data weather panel test lint check experiments figures site all

help:        ## Liste des cibles
	@grep -E '^[a-z]+:.*##' Makefile | sed 's/:.*##/ :/'
data:        ## Télécharge ~3 Go de fichiers TLC et vérifie le manifeste SHA-256
	./scripts/download_data.sh
weather:     ## Met en cache la météo Open-Meteo (data/processed/weather.csv, versionné)
	uv run python scripts/fetch_weather.py
panel:       ## Agrège le panel zone x heure (data/processed/panel.parquet, versionné)
	uv run python scripts/build_panel.py
test:        ## Suite de tests rapide
	uv run pytest
lint:
	uv run ruff check .
check: lint test
e0:          ## E0 : qualité du modèle surveillé (≈ 6 min, MSTL inclus)
	uv run python -m zonefleet.experiments.e0_model
e1:          ## E1 : choc COVID, DLE estimé vs réalisé
	uv run python -m zonefleet.experiments.e1_covid
