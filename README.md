# ZoneFleet : un modèle ne tombe pas en panne, il dérive

[![tests](https://github.com/gwils28/zonefleet-observability/actions/workflows/tests.yml/badge.svg)](https://github.com/gwils28/zonefleet-observability/actions/workflows/tests.yml)
[![site](https://github.com/gwils28/zonefleet-observability/actions/workflows/publish.yml/badge.svg)](https://gwils28.github.io/zonefleet-observability/)

Observabilité d'un forecaster de demande **zone x heure** (taxis de New York) dont la
vérité terrain arrive avec 24 heures d'horizon puis **60 jours** de délai de publication.
Comment mesurer sa santé **sans attendre les labels**, et où cette mesure devient-elle
un mensonge ?

**Documentation complète (site Quarto) : https://gwils28.github.io/zonefleet-observability/**

## Résultats en bref

| Question | Réponse mesurée | Où |
|---|---|---|
| La performance estimée sans labels (NannyML DLE) suit-elle l'erreur réelle ? | oui sous **covariate shift** (taux de suivi 0,85 à 0,99), **non** sous **concept drift pur** (0,00) | E3 |
| Sur un choc réel ? | COVID : détection sans labels en **9 jours** contre 62 avec ; mais DLE sous-estime l'ampleur d'un facteur 2,7 pendant la transition | E1 |
| Alerter sur la dérive des entrées ? | dérive seule : alarme **permanente** (100 % des jours) ; après la taxe de congestion, 177 jours d'alerte pour un modèle qui ne s'est pas dégradé | E2, E4 |
| Quelle politique d'alerte ? | **dérive nettoyée ET DLE**, lissée 7 jours : 51 % de rappel, **0 fausse alerte** sur 240 scénarios, 3 jours de délai médian | E4 |
| Réentraîner dès l'alerte ? | **non** : sur l'année COVID, le réflexe fait pire que l'inaction ; la politique gardée évite les accidents | E5 |

## Le projet

- **Données** : NYC TLC (taxis jaunes et verts) 2019-2021 et 2024-2025, 120 fichiers
  validés par un *data contract*, agrégés en panel de 60 zones x heure (2,6 M lignes) ;
  météo Open-Meteo ; calendrier US.
- **Modèle surveillé** : LightGBM global à trois quantiles, lags ≥ 24 h, figé après
  entraînement ; bat le seasonal naive dans les 60 zones et MSTL (E0).
- **Monitoring** : dérive univariée (KS, Wasserstein, PSI), multivariée (reconstruction
  PCA), dérive des prédictions, performance estimée (NannyML DLE) et réalisée.
- **Expériences** : E1 COVID, E2 tarification de congestion, E3 dérives pures
  construites, E4 backtest du moniteur (240 scénarios), E5 politiques de réentraînement.
- **Mise en production de démonstration** : graphe d'assets Dagster avec contrôles
  qualité, registry MLflow (champion/challenger), CLI `zonefleet`.

## Démarrer

```bash
uv sync --all-extras
make check                 # ruff + pytest
make e0 e1 e2 e3           # le panel agrégé est versionné : pas besoin des données brutes
make e4 e5                 # ≈ 20 min
make figures site          # figures, tables et site Quarto
zonefleet alert --split congestion
make dagster               # interface Dagster sur http://localhost:3000
```

Les données brutes (3,6 Go) ne sont nécessaires que pour reconstruire le panel :
`make data` les télécharge et vérifie leurs empreintes SHA-256.

## Organisation

```text
SPEC.md                  protocole figé avant les résultats
docs/notes/              déviations au protocole, fiches de veille
configs/                 data contract, zones, manifeste des données
src/zonefleet/           code (données, modèle, monitoring, alertes, scénarios, ops)
src/zonefleet/experiments/   E0 à E5
scripts/                 panel, météo, figures, tables, notebook, registry
results/tables/          tous les chiffres publiés (générés)
site/                    site Quarto (figures et tables générées, jamais à la main)
tests/                   tests (anti-fuite, contrat, pureté des injections, égalité NannyML)
```

## Méthode

Le protocole est figé dans `SPEC.md` ; chaque écart est daté et justifié dans
`docs/notes/implementation.md`. Les chiffres du site sont lus dans `results/`, les
figures et tables sont générées par script. Les résultats imprévus (la taxe de congestion
ne dégrade pas le modèle, la PCA ne voit pas le COVID, un calendrier mensuel bat la
politique de réentraînement gardée) sont rapportés tels quels.

## Licence

MIT. Données : NYC TLC (domaine public), Open-Meteo (CC BY 4.0).
