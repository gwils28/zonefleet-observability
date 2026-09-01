# Contribuer

## Installer

```bash
uv sync --all-extras
uv run pre-commit install
make check
```

## Règles du dépôt

1. **Le protocole est figé** dans `SPEC.md`. On ne le modifie pas pour accommoder un
   résultat ; toute déviation est datée et justifiée dans `docs/notes/implementation.md`.
2. **Une expérience se déclare avant de s'exécuter** (modèle d'issue « Expérience ») :
   question, mesure, critère, et le résultat qui serait une surprise.
3. **Aucun chiffre tapé à la main** dans `site/` : les tables sont générées par
   `scripts/make_tables.py`, les figures par `scripts/make_*_figures.py`, depuis
   `results/`.
4. **Pas de fuite temporelle** : aucune feature à moins de 24 h de la cible, aucune
   référence de monitoring dans l'échantillon d'entraînement (tests dédiés).
5. **`data/raw/` est en lecture seule** : il se reconstruit avec `make data`.
6. **Typographie** : pas de tiret cadratin (hook pre-commit).
7. **Chaque figure a un texte alternatif** (`fig-alt`, ou `#| fig-alt:` entre guillemets
   dans une cellule de notebook) qui décrit ce qu'elle trace ; `scripts/check_links.py`
   refuse une image sans texte alternatif.
8. **Un chiffre clé cité dans la prose entre au registre** `configs/claims.yml`, avec
   l'expression qui le recalcule depuis `results/tables` (`scripts/check_claims.py`).
9. **Les notebooks publiés sont nettoyés** après exécution (`scripts/clean_notebooks.py`,
   lancé par `make notebook`) : pas d'horodatage ni de chemin local dans les sorties.

## Flux

- Une branche par sujet (`feat/`, `exp/`, `docs/`, `fix/`, `chore/`), une PR qui
  référence son issue, CI verte avant fusion.
- Avant une version : `make release-check` (lint, typage, tests, régénération de toutes
  les figures et tables sans différence, rendu du site, liens et ancres).
- Messages de commit au format Conventional Commits (`feat`, `fix`, `docs`, `test`,
  `exp`, `ci`, `build`, `chore`, `refactor`, `perf`, `data`).

## Publier une version

1. `make release-check` sur un clone propre : aucune différence, aucun écart.
2. Section de la version dans `CHANGELOG.md`, version dans `pyproject.toml` (et
   `uv lock`), `CITATION.cff`, pied de page du site ; PR, CI verte, fusion.
3. Tag annoté sur `main` : `git tag -a vX.Y.Z -m "Version X.Y.Z"` puis `git push origin vX.Y.Z`.
4. Le workflow `release` crée la release GitHub avec la section du CHANGELOG
   (`scripts/release_notes.py`) et y joint deux archives : `results/tables` avec le
   registre des chiffres, et le site rendu.

## Structure du code

| Module | Rôle |
|---|---|
| `contract.py`, `data.py`, `covariates.py` | données : contrat, panel, météo, calendrier |
| `features.py`, `model.py`, `baselines.py`, `metrics.py` | le modèle surveillé |
| `monitoring.py`, `rowscore.py`, `alerting.py` | signaux, scores par ligne, politiques |
| `scenarios.py` | injections de dérives pures |
| `experiments/` | E0 à E5 (et E4 bis) |
| `ops/` | décisions, registry MLflow, graphe Dagster |
| `cli.py` | commande `zonefleet` |
