# ADR 0003 : l'astreinte est déclenchée par la condition jointe

- Date : 2026-07-17
- Statut : accepté (E4)

## Contexte
La dérive seule alerte 100 % des jours (toutes features) ou sur 55 % des dérives
bénignes (nettoyée). DLE seul fait 1,2 fausse alerte par mois.

## Décision
PAGE si dérive nettoyée (PCA ou PSI sur `lag_24`/`lag_168`) ET perte estimée lissée
7 jours au-dessus de moyenne + 1 écart-type de la référence, pendant 2 jours. Dérive
seule : TICKET hebdomadaire.

## Conséquences
- Zéro fausse alerte mesurée sur 240 scénarios, 51 % de rappel, 3 jours de délai médian.
- Concept drift non couvert (rappel 0 %) : voir ADR 0005.
