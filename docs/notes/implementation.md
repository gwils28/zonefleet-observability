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
