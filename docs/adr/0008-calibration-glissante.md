# ADR 0008 : calibrer les intervalles en glissant sur les labels récents

- Date : 2026-07-29
- Statut : proposé (dépend des labels à J + 1)

## Contexte
L'intervalle [q10, q90] couvre 72 % au lieu de 80 %. Une calibration conforme sur une
référence fixe ne tient ni à travers les fêtes (74 %) ni à travers un choc (56 %).

## Décision
Correction CQR recalculée chaque jour sur les 28 derniers jours de labels publiés ;
l'écart entre couverture réalisée et couverture visée devient un indicateur réalisé.

## Conséquences
- Avec des labels à J + 1 : 80 % retrouvés le 3 avril 2020, trois semaines après le choc ;
  2025 tenu à 80 % toute l'année.
- Avec des labels à J + 60 : retour fin mai puis sur-couverture ; l'intérêt dépend donc
  de l'ADR 0005 (raccourcir le délai des labels).
