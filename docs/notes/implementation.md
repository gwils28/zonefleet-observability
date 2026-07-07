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
