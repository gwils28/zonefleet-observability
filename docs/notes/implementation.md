# Notes d'implémentation

Journal des déviations au `SPEC.md` et des choix conservateurs, dans l'ordre chronologique.
Chaque entrée : date, décision, raison, impact sur les résultats.

## 2026-07-07 : retrait des features saisonnières `doy_sin`, `doy_cos`

- Décision : les features de jour de l'année sont calculées par `calendar_features` mais
  exclues de `FEATURES`.
- Raison : l'entraînement couvre janvier-août, la référence septembre-décembre. Le modèle
  extrapolait sur des valeurs de saison jamais vues : MAE de référence (E1) 29,5 avec,
  22,7 sans ; couverture 80 % de 0,58 à 0,72. Une feature de saison crée ici un
  *covariate shift artificiel* entre entraînement et référence, exactement le genre de
  dérive que le moniteur détecterait sans qu'elle soit liée au monde réel.
- Impact : conforme au SPEC (section 4 ne listait pas la saison). La saisonnalité annuelle
  passe par les lags 168 et 336 h.

## 2026-07-07 : signal de dérive = PCA OU PSI

- Décision : le signal `drift` des politiques combine l'alerte PCA de NannyML et un PSI
  journalier > 0,25 sur quatre features clés (`lag_24`, `lag_168`, `roll_mean_168`,
  `temperature_2m`).
- Raison : un test synthétique (`test_pca_reconstruction_sees_broken_correlation_but_not_shift_along_it`)
  montre que l'erreur de reconstruction PCA ne voit pas un déplacement le long d'une
  direction principale. Or un choc de niveau déplace tous les lags ensemble, précisément
  le long de la première composante.
- Impact : le signal de dérive devient plus sensible, donc plus bavard ; c'est voulu pour
  mesurer en E4 ce que la condition jointe retire comme bruit.
- Persistance : une alerte exige 2 jours consécutifs en dépassement (toutes politiques).

## 2026-07-15 : artefacts de la dérive univariée relevés en E1 et E2

- `dow` et `is_weekend` : KS constant par chunk journalier (un chunk = un seul jour de
  semaine). Artefact de découpage, pas une dérive. Recommandation : exclure les features
  calendaires déterministes des tests univariés journaliers.
- `roll_mean_168` : PSI > 0,25 sur 73 % des jours de E2 alors que `lag_168` ne dépasse
  que 3 % du temps. Une moyenne glissante de 7 jours, vue sur un chunk d'un jour, a une
  distribution resserrée : le PSI sur les déciles de la référence explose sans
  déplacement. Recommandation : pas de PSI journalier sur une feature lissée à la semaine.
- `temperature_2m` : PSI > 0,25 tous les jours en E1 et E2 (saison). Dérive réelle,
  impact nul.
- Décision : le signal `drift` des expériences E1/E2 n'est **pas** modifié après coup
  (protocole figé). E4 ajoute une variante de politique « dérive nettoyée » (P1b : PCA OU
  PSI sur `lag_24` et `lag_168` seulement), déclarée ici avant son exécution, pour ne pas
  comparer la condition jointe à un homme de paille.

## 2026-07-16 : E3 mesuré contre un contrefactuel, pas en avant/après

- Décision : l'effet d'un scénario est la différence avec le scénario « aucune dérive »
  sur les mêmes jours, et non la différence avant/après l'injection.
- Raison : premier passage, le scénario sans dérive montrait déjà Δ estimé = -0,10 entre
  janvier-février (vague de froid, DLE haut) et mars-juin. L'avant/après mesurait la
  saison autant que l'injection.
- Impact : les taux de suivi deviennent interprétables (0,82 à 0,99 en covariate shift,
  0 en concept drift pur). Le SPEC (section 8) demandait une corrélation ; le taux de
  suivi est rapporté en plus, la corrélation journalière est dans `e3_summary.csv`.

## 2026-07-16 : E4, détection par front montant et lissage sur 7 jours

- Détection comptée seulement sur un **front montant** de l'alerte dans les 28 jours
  suivant l'injection : une alarme déjà allumée avant l'injection ne « détecte » rien.
  Premier passage (20 scénarios) : sans cette règle, P1 affichait un rappel de 1 avec un
  délai de -1 jour alors qu'elle était en alerte 100 % du temps.
- Fausses alertes rapportées de deux façons : événements (allumages) par 30 jours, et
  part des jours sans dérive passés en alerte.
- Dimension ajoutée : signal estimé et réalisé lissé sur 7 jours (moyenne passée) contre
  signal journalier brut ; seuils calculés sur la référence lissée de la même façon.
  Raison : l'erreur estimée de base vaut ~0,29 et le seuil k = 1 vaut 0,34 ; un shift
  nuisible modéré (+10 %) reste sous le bruit jour à jour.
- Grille de k étendue à 0,5. La nuisance d'un scénario est mesurée sur le signal brut,
  indépendamment du lissage.

## 2026-07-17 : E5, deux délais de labels et un résultat imprévu

- Ajout : E5 est rejouée pour L = 1 jour (journaux de l'opérateur) en plus de L = 60 jours
  (publication TLC). Raison : à L = 60, aucune politique ne peut réagir au choc, ce qui
  rendrait la comparaison triviale ; ZoneFleet connaît ses courses le lendemain.
- Réentraînements avec 300 arbres au lieu de 500 (temps de calcul) ; le champion de
  départ reste le modèle figé de E1.
- Résultat imprévu, rapporté tel quel : le calendrier mensuel bat la politique gardée sur
  l'année 2020 (MAE log 0,363 contre 0,411 à L = 1). La recommandation du site en tient
  compte (politique gardée plus réentraînement de fond trimestriel).

## 2026-07-23 : E6, variante a posteriori du PSI par zone

- Le protocole déclaré (issue #37) mesurait le PSI de `lag_168` par zone contre la
  fenêtre précédant l'injection. Résultat : AUC 0,46 pour localiser un choc limité au
  CBD, contre 1,0 pour la perte estimée.
- Diagnostic : la fenêtre de comparaison (janvier-février) et la fenêtre analysée (mars)
  diffèrent par la saison ; le choc x0,6 ramène même les zones CBD vers leur niveau
  d'hiver, ce qui **réduit** leur PSI.
- Variante ajoutée après coup, et étiquetée comme telle partout : PSI rapporté au
  contrefactuel sans dérive. AUC 1,0. Leçon opérationnelle (sans contrefactuel en
  production) : comparer le PSI par zone à une base saisonnière, pas à une référence fixe.

## 2026-07-28 : extension hors SPEC, calibration conforme (E0 bis)

- Ajout documenté comme extension (SPEC, section 9 : conformalisation hors périmètre),
  sans toucher au modèle surveillé ni aux expériences publiées.
- Résultat : la garantie CQR ne tient pas entre septembre-octobre et novembre-décembre
  (74 % au lieu de 80 %) ; elle tient en 2025 (79-81 %) ; elle s'effondre pendant le
  COVID (56 %). Les fiches et la page E0 le rapportent tel quel.

## 2026-08-02 : E7, z-score saisonnier ajouté après coup

- Déclaré (issue #53) : PSI journalier contre une base saisonnière (28 jours, un an plus
  tôt). Un test synthétique, écrit avant l'exécution sur données réelles, a montré que le
  PSI d'une journée contre une fenêtre de 28 jours est mal posé (distributions de
  largeurs incomparables) ; sur données réelles, le PSI de température déclenche 99 %
  des jours avec les deux bases.
- Ajouté et étiqueté « a posteriori » : z-score de la moyenne journalière contre la
  distribution des moyennes journalières de la base saisonnière (|z| > 3). Il ramène les
  alertes de température à 0 % (2020) et 0,6 % (2025), et détecte le COVID dès le 13 mars
  sur `lag_24`.

## 2026-08-04 : E4 ter, périodes calmes supplémentaires (extension)

- Découpages ajoutés hors SPEC (`config.CALM_SPLITS`) : entraînement janv.-avril,
  référence mai-août, analysis sept.-déc., en 2019 et 2024 ; les découpages du SPEC
  (`SPLITS`) sont inchangés, `pipeline` utilise l'union.
- Résultat : P3b ne déclenche sur aucune des trois périodes. Mais en automne, l'erreur
  réalisée dépasse son seuil 22 à 26 % des jours (fêtes), et chaque alerte de DLE seul y
  tombe sur une vraie dégradation. « Calme » signifie ici « sans choc externe ».
- Conséquence intégrée au system design : DLE seul persistant ouvre un ticket.

## 2026-08-05 : extension du panel à 2021

- Le SPEC (section 2) liste 2019-2020 et 2024-2025 ; 2021 est ajoutée pour E5 bis (la
  reprise) et n'entre dans aucune expérience déjà publiée.
- Vérifié : lignes 2019-2020 du panel identiques à l'écart près de 0, météo identique sur
  la période commune, prédictions COVID identiques après rafraîchissement du cache.

## 2026-08-11 : typage statique progressif

- mypy sur `src/zonefleet` : modules métier sans erreur (métriques et PSI acceptent tout
  `ArrayLike`, réparation DST réécrite, registry qui refuse un modèle non entraîné).
- Pilotes d'expérience exclus provisoirement (`ignore_errors`) : 39 erreurs, presque toutes
  dues à la rigueur des stubs pandas sur des `groupby` et des `DataFrame | Series`.
- Vérifié : panel reconstruit identique, 113 tests verts.
