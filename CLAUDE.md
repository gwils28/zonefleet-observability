# CLAUDE.md (zonefleet-observability)

Question : surveiller un forecaster de demande zone x heure (NYC TLC) dont la vérité
terrain arrive avec H + L de retard ; estimer sa santé sans labels (NannyML DLE, dérive)
et nommer le régime où l'estimation ment. Protocole figé dans `SPEC.md` : ne pas le
modifier pour accommoder un résultat.

## Commandes
- Installation : `uv sync --all-extras` · Tests : `uv run pytest` · Lint : `uv run ruff check .`
- `make check` = lint + tests (rapide). `make data` télécharge ~5 Go dans `data/raw/`.
- `make panel` agrège le panel ; `make e0` ... `make e5` (ou `make experiments`) ;
  `make figures` (figures + tables du site depuis `results/`) ; `make site`.
- CLI : `zonefleet contract | monitor | alert | report | list | run eX` (`run e4-ter`, etc.). Dagster : `make dagster`.
- E4 relit ses résultats sans tout relancer : `python -m zonefleet.experiments.e4_backtest --summary-only`.

## Pièges
- Trois horloges : émission τ, cible t = τ + 24 h, label t + 60 j. Aucun lag < 24 h.
- La référence NannyML doit être hors échantillon pour le forecaster.
- NannyML 0.13 exige pandas < 3 ; kaleido épinglé à 0.2.1 (roue x86_64).
- Horodatages TLC naïfs en heure locale : passage à l'heure d'été géré dans `data.py`.
- `data/raw/` est en lecture seule (hook `protect_data.py`) : passer par `scripts/download_data.sh`.
- `from __future__ import annotations` casse la config Dagster (`ops/definitions.py`).
- NannyML envoie de la télémétrie par défaut : `NML_DISABLE_USAGE_LOGGING=1` (posé dans `monitoring.py`).

## Façon de travailler
- Chiffres publiés dans `site/` : lus dans `results/`, jamais de mémoire.
- Figures produites par `scripts/make_figures.py` dans `site/figures/`, jamais à la main.
- Déviations au plan : `docs/notes/implementation.md`.
- Code : noms en anglais, docstrings et commentaires en français.
- Commits : Conventional Commits (`feat`, `fix`, `docs`, `test`, `exp`, `ci`, `chore`, `refactor`).

## Typographie
- N'utilise JAMAIS le tiret cadratin (U+2014), ni dans le code, ni dans les commentaires,
  ni dans la documentation, ni dans tes réponses.
- À la place, utilise une virgule, deux-points, des parenthèses ou un tiret simple "-".
- Garde-fou : `.claude/hooks/no_em_dash.py` (PreToolUse et Stop).

## Compaction
Conserver : la liste des fichiers modifiés, les commandes de test, les décisions de
modélisation prises et l'état des expériences (`results/`).
