# ADR 0001 : un modèle global LightGBM quantile, figé

- Date : 2026-07-06
- Statut : accepté

## Contexte
Il faut un forecaster réaliste à observer, pas le meilleur possible. 60 zones, horizon
24 h, positionnement sous incertitude.

## Décision
Un seul LightGBM global (zone catégorielle) par quantile (0,1 ; 0,5 ; 0,9), sur log1p,
hyperparamètres fixés a priori, sérialisé puis figé.

## Conséquences
- Un seul objet à surveiller ; hétéroscédasticité forte (condition favorable à DLE).
- Pas de recherche d'hyperparamètres : on ne sur-ajuste pas la référence du moniteur.
- Intervalles sous-couvrants (72 %) acceptés et documentés.
