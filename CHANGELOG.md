# Journal des versions

Format inspiré de *Keep a Changelog*. Les dates sont celles de fusion dans `main`.

## [Non publié]

### Ajouté
- Hook pre-commit (ruff, fichiers, YAML, typographie), couverture de tests en CI.
- Modèles d'issue et de PR, guide de contribution.

## [0.6.0] - 2026-07-22
### Ajouté
- E4 bis : sensibilité du backtest à la persistance, au seuil de nuisance et à la
  fenêtre. La persistance de 2 jours est le minimum qui garde la condition jointe sans
  fausse alerte.
- Politiques P1b et P3b dans les tables de E1 et E2 : la dérive nettoyée détecte le
  COVID en un jour sans fausse alerte préalable.

## [0.5.0] - 2026-07-21
### Ajouté
- Synthèse : system design, cas ZoneFleet, notebook tutoriel, accueil à chiffres clés
  générés, journal de bord, références, README.
- CLI `zonefleet` (contract, monitor, alert, run).
- Graphe d'assets Dagster (9 assets, 2 contrôles) et registry MLflow local.

## [0.4.0] - 2026-07-18
### Ajouté
- E4 : backtest du moniteur sur 240 scénarios, détection par front montant, lissage
  7 jours, intervalles bootstrap.
- E5 : cinq politiques de réentraînement sur l'année 2020, labels à J + 1 et J + 60.
- `RowScorer` : scores par ligne réutilisant les modèles NannyML (égalité testée).

## [0.3.0] - 2026-07-16
### Ajouté
- E2 (tarification de congestion) et E3 (frontière covariate shift / concept drift).
- Injections de dérives pures (`scenarios.py`).
### Corrigé
- Télémétrie d'usage de NannyML désactivée.

## [0.2.0] - 2026-07-10
### Ajouté
- Monitoring (dérive univariée et multivariée, DLE, performance réalisée), politiques
  d'alerte, E1 (COVID).
- Site Quarto : veille, concepts, données, modèle.

## [0.1.0] - 2026-07-07
### Ajouté
- Data contract, panel zone x heure, météo et calendrier.
- Forecaster LightGBM à trois quantiles, baselines, E0.
