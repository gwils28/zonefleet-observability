# Journal des versions

Format inspiré de *Keep a Changelog*. Les dates sont celles de fusion dans `main`.

## [Non publié]

## [0.10.0] - 2026-08-26
### Ajouté
- E10 : condition jointe par groupe de zones (CBD, hors CBD), 55 % de rappel sans fausse
  alerte.
- E3 bis : stabilité du taux de suivi sur 5 graines (critère déclaré avant exécution,
  rempli) ; le nanny model s'avère déterministe.
- Notebook « anatomie d'une alerte » (22 mars 2020) ; page Rétrospective.
- Asset check `predictions_complete` ; `zonefleet list`, `zonefleet run` pour toutes les
  extensions, `zonefleet --version`.
- Image Docker de reproduction (vérifiée dans un conteneur vierge).
### Modifié
- PSI journalier vectorisé (`psi_by_day`, égalité stricte testée) : E10 passe d'environ
  40 min à 92 s, résultats identiques à l'octet.
- Typage mypy de tous les modules, sans exclusion ; génération du site déterministe
  (identifiants plotly fixes) ; relecture de 18 pages ; glossaire (29 entrées) et FAQ.
### Corrigé
- Notebooks publiés : message d'aide MLflow (date, chemins locaux) et horodatages
  d'exécution retirés.
- Zéro négatif dans les tables ; seuils de la politique gardée dans le glossaire.
- CI rouge après #91 (ruff sur les notebooks générés).
### Retiré
- Dependabot (PR datées hors période, pandas 3 incompatible avec NannyML 0.13).

## [0.9.0] - 2026-08-10
### Ajouté
- E8 : bugs de données amont ; module `quality`, asset check `data_quality`, sévérité
  « incident données » dans le rapport du matin.
- E9 : vraie prévision météo de la veille (Open-Meteo Previous Runs) ; hypothèse
  optimiste levée.
- E4 quater : dérive saisonnière dans la condition jointe (fausses alertes, non retenue).
- ADR 0007 (qualité avant modèle) et 0008 (calibration glissante).

## [0.8.0] - 2026-08-06
### Ajouté
- E7 : dérive contre base saisonnière (z-score de la moyenne journalière).
- E4 ter : fausses alertes sur trois périodes calmes ; ticket « perte estimée seule ».
- Panel étendu à 2021 ; E5 bis sur la reprise.
- Rapport quotidien de monitoring (CLI, asset Dagster), notebook des coûts d'alerte,
  libellés partagés, CI de rendu du site, favicon, 404, page Méthode.

## [0.7.0] - 2026-07-29
### Ajouté
- E1 bis (taille de chunk), E6 (localisation par zone), calibration conforme fixe et
  glissante (extension hors SPEC), graphiques interactifs.
- Hook pre-commit, couverture de tests (57 % puis 76 %), modèles d'issue et de PR,
  guide de contribution.
### Corrigé
- `e0_model.by_zone` (double sélection de colonnes), tables publiées inchangées.

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
