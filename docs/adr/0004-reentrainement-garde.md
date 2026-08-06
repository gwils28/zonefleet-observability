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

## Mise à jour du 2026-08-06 (E5 bis)
Prolongée à 2021, la simulation montre que la politique gardée rejoint le calendrier
mensuel (MAE log 0,264 contre 0,261) avec 9 remplacements au lieu de 23. L'avantage du
calendrier tenait au choc de 2020. La politique gardée devient la règle par défaut ; le
réentraînement de fond trimestriel est conservé comme filet en cas de choc.
