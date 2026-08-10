# ADR 0007 : les contrôles de qualité des données passent avant le moniteur de modèle

- Date : 2026-08-08
- Statut : accepté (E8)

## Contexte
Un fichier tronqué, une journée dupliquée, un décalage horaire ou une zone muette sont
invisibles pour la perte estimée (E8 : DLE et la condition jointe restent muets) mais
font exploser l'erreur réalisée deux mois plus tard, qui accuserait alors le modèle.

## Décision
Contrôles journaliers sur le panel (volume contre les 4 mêmes jours, profil horaire,
zones muettes), calendrier connu exclu, seuils calibrés sur l'année passée. En échec :
sévérité « incident données », prioritaire sur toute alerte de modèle ; jours exclus de
tout réentraînement.

## Conséquences
- Quatre bugs injectés détectés le premier jour (le deuxième pour le décalage).
- Environ deux fausses alertes de qualité par trimestre (jours atypiques non fériés).
