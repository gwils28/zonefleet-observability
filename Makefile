.PHONY: help data weather panel test lint check e0 e1 e2 e3 e4 e5 experiments figures notebook registry dagster site all

help:        ## Liste des cibles
	@grep -E '^[a-z0-9]+:.*##' Makefile | sed 's/:.*##/ :/'
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
e2:          ## E2 : tarification de congestion, CBD vs hors CBD
	uv run python -m zonefleet.experiments.e2_congestion
e3:          ## E3 : frontière covariate shift / concept drift (≈ 2 min)
	uv run python -m zonefleet.experiments.e3_frontier
e4:          ## E4 : backtest du moniteur, 240 scénarios (≈ 11 min)
	uv run python -m zonefleet.experiments.e4_backtest
e5:          ## E5 : politiques de réentraînement sur 2020 (≈ 10 min)
	uv run python -m zonefleet.experiments.e5_retraining
registry:    ## Enregistre les modèles figés dans le registry MLflow local (alias champion)
	uv run python scripts/register_models.py
dagster:     ## Interface Dagster du monitoring (http://localhost:3000)
	uv run dagster dev -m zonefleet.ops.definitions
notebook:    ## Régénère et exécute le notebook tutoriel
	uv run python scripts/build_notebook.py
	uv run jupyter nbconvert --to notebook --execute --inplace site/notebooks/tutoriel-dle.ipynb
experiments: e0 e1 e2 e3 e4 e5  ## Toutes les expériences du SPEC
figures:     ## Figures et tables du site (depuis results/)
	uv run python scripts/make_concept_figures.py
	uv run python scripts/make_data_figures.py
	uv run python scripts/make_result_figures.py
	uv run python scripts/make_tables.py
site:        ## Rendu du site Quarto (site/_site)
	quarto render site
all: check experiments figures notebook site
e4s:         ## E4 bis : sensibilité au protocole (relit le cache de E4, ≈ 10 s)
	uv run python -m zonefleet.experiments.e4_sensitivity
e6:          ## E6 : localisation de la dérive par zone (≈ 10 s)
	uv run python -m zonefleet.experiments.e6_localisation
e1c:         ## E1 bis : taille de chunk (jour ou semaine)
	uv run python -m zonefleet.experiments.e1_chunks
