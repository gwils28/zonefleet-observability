# ADR 0002 : la référence du moniteur est hors échantillon

- Date : 2026-07-02
- Statut : accepté

## Contexte
DLE apprend la carte des erreurs du modèle sur une période de référence. Sur les
données d'entraînement, le modèle se trompe moins qu'ailleurs.

## Décision
Chaque expérience a trois périodes : entraînement (janv.-août), **référence hors
échantillon** (sept.-déc.), analysis. Les seuils sont calculés sur la référence.

## Conséquences
- Pas de « dégradation » artificielle dès le premier jour d'analysis.
- La référence est automnale : les analyses d'hiver dérivent sur la météo (assumé,
  filtré par la politique d'alerte).
