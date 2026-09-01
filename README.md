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
| La performance estimée sans labels (NannyML DLE) suit-elle l'erreur réelle ? | oui sous **covariate shift** (taux de suivi 0,85 à 0,99), **non** sous **concept drift pur** (0,00) ; stable sur 5 graines d'injection | E3, E3 bis |
| Sur un choc réel ? | COVID : détection sans labels en **9 jours** contre 62 avec ; mais DLE sous-estime l'ampleur d'un facteur 2,7 pendant la transition | E1 |
| Alerter sur la dérive des entrées ? | dérive seule : alarme **permanente** (100 % des jours) ; après la taxe de congestion, 177 jours d'alerte pour un modèle qui ne s'est pas dégradé | E2, E4 |
| Quelle politique d'alerte ? | **dérive nettoyée ET DLE**, lissée 7 jours : 51 % de rappel, **0 fausse alerte** sur 240 scénarios, 3 jours de délai médian | E4 |
| Réentraîner dès l'alerte ? | **non** : sur l'année COVID, le réflexe fait pire que l'inaction ; sur la reprise de 2021, la politique gardée égale un calendrier mensuel avec 9 remplacements au lieu de 23 | E5, E5 bis |
| Les fausses alertes tiennent-elles ailleurs ? | oui : 0 page de la condition jointe sur trois périodes calmes ; DLE seul y voit les dégradations des fêtes | E4 ter |
| Où regarder après une page ? | la perte estimée par zone classe les zones touchées (corrélation de rang 0,74 sur le COVID) ; le PSI par zone se perd dans la saison | E6 |
| Et la saisonnalité ? | un z-score saisonnier supprime la dérive de la température ; pour les lags, il importe la croissance d'une année sur l'autre | E7, E4 quater |
| Et si ce sont les données qui mentent ? | des contrôles de qualité voient 4 bugs injectés dès le 1er jour ; le moniteur de modèle, non | E8 |
| La météo « parfaite » biaise-t-elle tout ? | non : servie sur la vraie prévision de la veille, la MAE bouge de 0,1 % | E9 |
| Alerter à la ville ou par quartier ? | par groupe (CBD, hors CBD) : 55 % de rappel au lieu de 51 %, toujours sans fausse alerte ; les dérives locales manquées sont des concept drifts | E10 |

## Le projet

- **Données** : NYC TLC (taxis jaunes et verts) 2019-2021 et 2024-2025, 120 fichiers
  validés par un *data contract*, agrégés en panel de 60 zones x heure (2,6 M lignes) ;
  météo Open-Meteo ; calendrier US.
- **Modèle surveillé** : LightGBM global à trois quantiles, lags ≥ 24 h, figé après
  entraînement ; bat le seasonal naive dans les 60 zones et MSTL (E0).
- **Monitoring** : dérive univariée (KS, Wasserstein, PSI), multivariée (reconstruction
  PCA), dérive des prédictions, performance estimée (NannyML DLE) et réalisée.
- **Expériences** : E1 COVID, E2 tarification de congestion, E3 dérives pures
  construites, E4 backtest du moniteur (240 scénarios, sensibilité, trois périodes
  calmes), E5 politiques de réentraînement (2020-2021), E6 localisation par zone, E7 base
  saisonnière, E8 bugs de données, E9 vraie prévision météo, E10 alerte par groupe de
  zones ; calibration conforme, stabilité selon les graines (`zonefleet list`).
- **Mise en production de démonstration** : graphe d'assets Dagster avec contrôles
  qualité, registry MLflow (champion/challenger), rapport du matin, CLI `zonefleet`.

## Démarrer

```bash
uv sync --all-extras
make check                 # ruff + pytest
make e0 e1 e2 e3           # le panel agrégé est versionné : pas besoin des données brutes
make e4 e5                 # ≈ 20 min
make figures site          # figures, tables et site Quarto
zonefleet list             # les 17 expériences (SPEC et extensions)
zonefleet run e4-ter       # en lancer une
zonefleet alert --split congestion
make dagster               # interface Dagster sur http://localhost:3000
```

Sans installer Python localement :

```bash
docker build -t zonefleet .
docker run --rm zonefleet make check            # 160 tests, lint, typage
docker run --rm zonefleet make e1               # une expérience sur le panel versionné
```

Les données brutes (3,6 Go) ne sont nécessaires que pour reconstruire le panel :
`make data` les télécharge et vérifie leurs empreintes SHA-256.

## Organisation

```text
SPEC.md                  protocole figé avant les résultats
docs/notes/              déviations au protocole, fiches de veille
configs/                 data contract, zones, manifeste des données
src/zonefleet/           code (données, modèle, monitoring, alertes, scénarios, ops)
src/zonefleet/experiments/   E0 à E5 (SPEC), puis extensions et variantes jusqu'à E10
scripts/                 panel, météo, figures, tables, notebooks, registry, API
results/tables/          tous les chiffres publiés (générés)
site/                    site Quarto (figures et tables générées, jamais à la main)
tests/                   tests (anti-fuite, contrat, pureté des injections, égalité NannyML)
```

## Vérifier

Chaque PR passe le lint, le typage, les 160 tests, la régénération des tables depuis
`results/` (aucune différence admise), le recalcul des 27 chiffres clés cités dans la
prose (`configs/claims.yml`), le rendu du site et la vérification de ses liens, ancres et
textes alternatifs. Avant une version, `make release-check` ajoute la régénération de
toutes les figures.

## Méthode

Le protocole est figé dans `SPEC.md` ; chaque écart est daté et justifié dans
`docs/notes/implementation.md`. Les chiffres du site sont lus dans `results/`, les
figures et tables sont générées par script. Les résultats imprévus (la taxe de congestion
ne dégrade pas le modèle, la PCA ne voit pas le COVID, un calendrier mensuel bat la
politique de réentraînement gardée) sont rapportés tels quels.

## Citer

Voir [`CITATION.cff`](CITATION.cff) (bouton « Cite this repository » sur GitHub).

## Licence

MIT. Données : NYC TLC (domaine public), Open-Meteo (CC BY 4.0).
