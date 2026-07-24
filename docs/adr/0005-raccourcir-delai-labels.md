# ADR 0005 : couvrir l'angle mort en raccourcissant le délai des labels

- Date : 2026-07-20
- Statut : proposé (dépend des données ZoneFleet)

## Contexte
Le concept drift pur est invisible sans labels (E3 : suivi de DLE nul). Avec les labels
publics (L = 60 jours), la fenêtre aveugle dure 61 jours.

## Décision
Brancher la performance réalisée sur les journaux de courses internes (L ≈ 1 jour) ;
compléter par des proxys indépendants des features (demande non servie, attentes).

## Conséquences
- Fenêtre aveugle de 61 à 2 jours ; toutes les politiques de réentraînement deviennent
  réactives (E5).
- Instrumentation de l'application nécessaire.
