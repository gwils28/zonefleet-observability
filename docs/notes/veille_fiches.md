# Fiches de veille (sources vérifiées)

Tout chiffre de veille repris sur le site vient de cette fiche, relevée sur la source
primaire. Les incohérences internes des sources sont signalées, pas lissées.

## NannyML : estimation de performance (DLE, CBPE)

Source : https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html

- DLE (Direct Loss Estimation) : un modèle supplémentaire, le *nanny model*, estime la
  perte du modèle surveillé (*child model*) observation par observation.
- Entraînement : (1) perte par observation sur la référence (MAE : |y - f(X)|) ;
  (2) nanny model entraîné sur la référence, entrées = features X et prédiction f(X),
  cible = la perte ; (3) sur l'analysis, perte prédite par observation puis moyennée
  par chunk (RMSE : erreur au carré, racine de la moyenne).
- Algorithme du nanny model : LightGBM ; perte L2 par défaut, L1 possible ;
  hyperparamètres réglables avec FLAML.
- La prédiction du modèle surveillé est une entrée du nanny model.
- Hypothèses listées : (1) pas de concept drift, DLE « will not work under concept
  drift » ; (2) pas de covariate shift vers des régions jamais vues ; (3) bruit
  hétéroscédastique (sinon la vraie performance est constante et toute variation est du
  bruit d'échantillonnage) ; (4) échantillon assez grand.
- La page affirme que DLE gère bien le covariate shift.
- CBPE : classification seulement, s'appuie sur des probabilités calibrées ; mêmes
  hypothèses (pas de concept drift, pas de régions inédites, échantillon suffisant).

## NannyML : dérive multivariée par reconstruction PCA

Source : https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html

- Préparation : imputation, encodage fréquentiel des catégorielles, standardisation.
- PCA ajustée sur la référence, 65 % de variance par défaut (paramétrable), appliquée
  à l'analysis ; décompression par PCA inverse.
- Erreur de reconstruction : distance euclidienne point original / reconstruit,
  moyennée par chunk.
- Seuils : moyenne et écart-type de l'erreur sur les chunks de référence ; alerte à plus
  de 3 écarts-types.
- Exemple « papillon » de la doc : une rotation de 90 degrés (changement de signe de
  corrélation) passe inaperçue des tests KS univariés mais est vue par la PCA.
- Pas de section de limites explicite. Limite que nous ajoutons (test du dépôt) : un
  déplacement le long d'une direction principale n'augmente pas l'erreur de reconstruction.

## DriftGuard (Alam, Rahman, Sadeqa, arXiv:2601.08928, 13 janvier 2026)

Source : https://arxiv.org/abs/2601.08928 et version HTML v1.

- Motivation : le concept drift provoque une dégradation silencieuse (ruptures ou
  surstocks sans alerte) ; pratique courante : surveillance manuelle et réentraînement
  planifié « every 3-6 months ».
- Cinq modules : moteur de prévision de base (Prophet, XGBoost, LSTM comparés ; XGBoost
  retenu, WMAPE 0,048), génération de scénarios de dérive semi-synthétiques, détection
  par ensemble, diagnostic SHAP, réentraînement adaptatif sensible au coût.
- Détection : quatre détecteurs votant en binaire, alerte si au moins trois sur quatre
  sont d'accord : ratio de RMSE glissant, KS (continu) et PSI (catégoriel),
  autoencodeur sur les résidus, CUSUM sur les résidus.
- Diagnostic : écart des valeurs SHAP moyennes entre base et dérive, agrégé sur la
  hiérarchie (SKU, magasin, région).
- Remédiation : fenêtre d'entraînement choisie dans {30, 60, 90, 180} jours ;
  réentraînement des seuls top-K SKU dont la hausse de WMAPE dépasse un seuil ;
  ROI = (gain de coût de stock - coût de calcul) / coût de calcul.
- Données : M5 (Walmart), 10 magasins dans 3 États, 3 049 produits ; dérive injectée à
  partir du jour 1800 sur 20 % des SKU tirés au hasard. Cinq scénarios : saisonnalité
  (-40 % en fêtes), tendance (-25 %), choc de niveau (x0,8), volatilité (variance x3),
  dérive hiérarchique (confinement régional).
- Résultats annoncés : 97,8 % de rappel, latence moyenne 4,2 jours ; détecteur à base
  d'erreur seul : 92 % de rappel en 12 jours ; tests statistiques seuls : 6 jours mais
  plus de faux positifs ; taux de faux positifs 3,2 % (« environ une fois par mois »).
  ROI « jusqu'à 417 ».
- Incohérences internes : l'encadré de l'algorithme 1 annonce 98 % de rappel, 3 % de
  FPR et une latence médiane de 5 jours ; le nombre de séries varie (30 490 en IV-A,
  5 040 SKU en III-A, « 30 000+ » en V-C) ; le ROI de 417 ne se retrouve pas exactement
  à partir des montants publiés (3,7 M$ / 9,6 k$ ≈ 385).
- Limites : dérive injectée (semi-synthétique), pas observée en production ; récupération
  incomplète pour les dérives sévères (« fundamental limitation ») ; démarrage à froid ;
  coût de SHAP ; pas de lien vers du code dans le texte lu.

## Sculley et al. (2015), Hidden Technical Debt in Machine Learning Systems

Source : NeurIPS 28 (2015), p. 2494-2502, PDF des proceedings.

- Auteurs : D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner,
  V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison.
- Thèse : les gains rapides du ML ne sont pas gratuits ; les systèmes réels accumulent
  des coûts de maintenance massifs. « Hidden debt is dangerous because it compounds silently. »
- Sections : 2 Complex Models Erode Boundaries (Entanglement / CACE, Correction Cascades,
  Undeclared Consumers) ; 3 Data Dependencies Cost More than Code Dependencies (Unstable,
  Underutilized, Static Analysis) ; 4 Feedback Loops (Direct, Hidden) ; 5 ML-System
  Anti-Patterns (Glue Code, Pipeline Jungles, Dead Experimental Codepaths, Abstraction
  Debt, Common Smells) ; 6 Configuration Debt ; 7 Dealing with Changes in the External
  World (Fixed Thresholds in Dynamic Systems, Monitoring and Testing : Prediction Bias,
  Action Limits, Up-Stream Producers) ; 8 Other Areas (Data Testing, Reproducibility,
  Process Management, Cultural Debt) ; 9 Conclusions.
- Citations utiles : un système mature peut être « (at most) 5% machine learning code and
  (at least) 95% glue code » ; « Comprehensive live monitoring of system behavior in real
  time combined with automated response is critical for long-term system reliability. »
- Prediction bias : la distribution des prédictions devrait égaler celle des labels
  observés ; diagnostic simple mais utile, à découper par dimensions.
