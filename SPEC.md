# SPEC : observabilité d'un forecaster de demande zone x heure (ZoneFleet)

Protocole figé le 2 juillet 2026, avant tout résultat. Toute déviation est consignée dans
`docs/notes/implementation.md` avec sa raison ; on ne modifie pas ce fichier pour
accommoder un résultat.

## 1. Question

Un forecaster de demande horaire par zone se dégrade en silence quand le monde change.
La vérité terrain arrive avec retard (horizon H, puis délai de publication L).

1. Peut-on mesurer la santé du modèle **pendant la fenêtre aveugle**, sans labels, avec la
   dérive des entrées (univariée, multivariée) et la performance estimée (NannyML DLE) ?
2. **Où cette mesure ment-elle ?** Hypothèse à tester : DLE suit l'erreur sous
   *covariate shift* (P(X) change, P(y|X) stable) et décroche sous *concept drift pur*
   (P(X) stable, P(y|X) change).
3. Quelle **politique d'alerte** (dérive seule, DLE seul, condition jointe, labels
   retardés) offre le meilleur couple rappel x délai de détection pour un taux de fausses
   alertes donné ?

## 2. Données

- NYC TLC Trip Records, taxis jaunes et verts, 2019-2020 et 2024-2025 (Parquet mensuels).
- Agrégation : nombre de prises en charge par `PULocationID` et par heure locale
  (America/New_York), jointure avec la Taxi Zone Lookup Table.
- Panel : les **60 zones** les plus actives en 2019 (liste figée, versionnée dans
  `configs/zones.yaml`), plus un indicateur CBD (zone de tarification de congestion)
  dérivé des données 2025 (part des courses avec `cbd_congestion_fee > 0`).
- Météo horaire Open-Meteo (archive ERA5) pour Manhattan : température, précipitations.
  Hypothèse optimiste assumée : la météo réalisée sert de prévision météo.
- Calendrier : heure, jour de semaine, jours fériés US (paquet `holidays`).
- Data contract versionné (`configs/data_contract.yaml`) : colonnes requises, types,
  plages ; une rupture de schéma (ex. apparition de `cbd_congestion_fee` en 2025) est
  détectée et journalisée, jamais silencieusement absorbée.

## 3. Trois horloges

- Émission τ : instant où la prévision est produite.
- Cible t = τ + H avec **H = 24 h** (positionnement de flotte la veille).
- Label disponible à t + L, avec **L = 60 jours** (délai de publication TLC). La fenêtre
  aveugle d'une prévision dure donc H + L.

Toute feature est connue à τ : aucun lag inférieur à 24 h. Un délai de détection est
rapporté sur l'horloge de décision (jour où l'alerte peut être levée), pas sur la cible.

## 4. Modèle surveillé

- Cible : `log1p(pickups)`. Features : lags 24, 48, 72, 168, 336 h ; moyennes glissantes
  24 h et 168 h terminées à t - 24 h ; heure, jour de semaine, férié ; météo ; zone
  (catégorielle).
- LightGBM global, trois quantiles (0,1 ; 0,5 ; 0,9). La médiane est la prévision ponctuelle.
- Baselines : seasonal naive (lag 168 h), MSTL (statsforecast) sur un sous-ensemble.
- Métriques réalisées : MAE, RMSE, sMAPE, MASE (échelle : seasonal naive 168 h sur
  l'entraînement), pinball moyen, couverture empirique de l'intervalle 80 %.
- Une fois entraîné, le modèle est **figé** : il devient l'objet du monitoring.

## 5. Découpage temporel (référence hors échantillon)

| Expérience | Entraînement | Référence (hors échantillon) | Analyse |
|---|---|---|---|
| E1 COVID | 2019-01-01 à 2019-08-31 | 2019-09-01 à 2019-12-31 | 2020-01-01 à 2020-06-30 |
| E2 congestion | 2024-01-01 à 2024-08-31 | 2024-09-01 à 2024-12-31 | 2025-01-01 à 2025-06-30 |

La référence NannyML est hors échantillon pour le forecaster (sinon DLE est optimiste).

## 6. Monitoring

- Chunks journaliers (60 zones x 24 h = 1 440 lignes) ; vues hebdomadaires pour les figures.
- Dérive univariée : Kolmogorov-Smirnov, Wasserstein (NannyML), PSI (implémentation maison).
- Dérive multivariée : erreur de reconstruction PCA (NannyML `DataReconstructionDriftCalculator`).
- Dérive des prédictions : distribution de la médiane prédite.
- Performance estimée : NannyML DLE, métriques MAE et RMSE (échelle des courses).
- Performance réalisée : `PerformanceCalculator`, révélée a posteriori.

## 7. Expériences

- **E0** qualité du modèle sur la référence (vs seasonal naive et MSTL).
- **E1** choc COVID (naturel) : DLE estimé vs réalisé à travers le choc.
- **E2** tarification de congestion du 5 janvier 2025 (naturel) : zones CBD vs hors CBD.
- **E3** semi-synthétique contrôlé, sur la période d'analyse « calme » (référence prolongée) :
  - covariate shift pur par sous-échantillonnage pondéré de lignes réelles (P(y|X) intact
    par construction) ;
  - concept drift pur : y multiplié sur un sous-ensemble, X gelé (P(X) intact par construction) ;
  - choc de niveau propagé (y et lags modifiés ensemble), cas réaliste mixte.
- **E4** backtest du moniteur : au moins 200 scénarios (type, amplitude, date de début
  aléatoires, dont des scénarios nuls). Politiques : P1 dérive seule, P2 DLE seul,
  P3 dérive ET DLE, P4 labels retardés. Mesures : rappel, délai de détection, taux de
  fausses alertes par jour, précision vis-à-vis des scénarios réellement nuisibles
  (hausse de MAE réalisée supérieure à 10 %).
- **E5** politique de réentraînement après un choc : jamais, réflexe (dès l'alerte, sur
  peu de labels), gardée (dérive confirmée ET au moins 28 jours de labels, puis
  champion/challenger sur une fenêtre de validation).

## 8. Critères de réussite

- Figure « erreur estimée vs réalisée » sur E1 et E3, avec le régime de décrochage nommé.
- E3 : corrélation estimé/réalisé élevée sous covariate shift, nulle sous concept drift
  pur ; sinon, on rapporte l'hypothèse comme réfutée.
- E4 : tableau rappel x délai x fausses alertes par politique, intervalles bootstrap.
- Tous les chiffres du site viennent de `results/` (générés), jamais tapés à la main.

## 9. Hors périmètre

FHVHV (Uber/Lyft), prévision par quantiles conformes, réconciliation hiérarchique,
déploiement cloud réel.
