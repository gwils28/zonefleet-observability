
**Sévérité : RAS** · modèle `zonefleet-demand` v2

#### Signaux du jour

| Signal | Valeur | Seuil | État |
|---|---|---|---|
| perte estimée (MAE log, DLE) | 0,258 | 0,521 | normal |
| dérive nettoyée (PCA ou PSI lags) | non | - | normal |
| perte réalisée (labels) | - | 0,521 | en attente |

#### Zones en tête de perte estimée

| Zone | Perte estimée |
|---|---|
| 129 Jackson Heights | 0,393 |
| 33 Brooklyn Heights | 0,386 |
| 82 Elmhurst | 0,382 |
| 244 Washington Heights South | 0,379 |
| 7 Astoria | 0,365 |

#### Features en dérive (PSI > 0,25)

- `temperature_2m` : 3,20
- `roll_mean_168` : 0,78

#### Recommandation

- Réentraînement : **non** (pas de dérive confirmée : réentraîner serait du bruit).

> Angle mort : un changement de la relation entre entrées et demande (concept drift pur) n'apparaît dans aucun de ces signaux avant l'arrivée des labels.
