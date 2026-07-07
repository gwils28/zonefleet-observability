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
