# ADR 0006 : orchestration par assets Dagster

- Date : 2026-07-18
- Statut : accepté

## Contexte
Sculley et al. décrivent les *pipeline jungles* ; le monitoring enchaîne données,
features, prévisions, signaux, décisions.

## Décision
Graphe d'assets Dagster (données versionnables), contrôles qualité en asset checks,
décisions en fonctions pures testées hors Dagster, registry MLflow pour le champion.

## Conséquences
- Traçabilité et fraîcheur de chaque donnée ; orchestrateur sans logique métier.
- Dépendance supplémentaire (extras `ops`).
