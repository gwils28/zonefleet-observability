# ADR 0004 : réentraînement gardé, jamais réflexe

- Date : 2026-07-18
- Statut : accepté (E5), amendé

## Contexte
Sur l'année COVID, le réentraînement déclenché par l'alerte fait pire que l'inaction ;
un réentraînement sur quelques jours post-choc sur-apprend le pire moment.

## Décision
Réentraîner seulement si dérive confirmée ET ≥ 28 jours de labels post-alerte ET
fenêtre non pilotée par le modèle ; challenger promu s'il gagne ≥ 5 % sur des labels
postérieurs. **Amendement** : réentraînement de fond trimestriel, soumis à la même
validation, car un calendrier mensuel a battu la politique gardée en 2020 (L = 1).

## Conséquences
- Pas de remplacement non motivé ; adaptation de fond conservée.
