#!/usr/bin/env bash
# Télécharge les fichiers TLC (jaune, vert) de 2019-2020 et 2024-2025 et la table des zones,
# puis vérifie les empreintes SHA-256 du manifeste versionné (configs/manifest_sha256.txt).
# Usage : ./scripts/download_data.sh [--update-manifest]
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://d37ci6vzurychx.cloudfront.net
RAW=data/raw
mkdir -p "$RAW/tlc" "$RAW/ref"

lookup="$RAW/ref/taxi_zone_lookup.csv"
[ -s "$lookup" ] || curl -sSfL -o "$lookup" "$BASE/misc/taxi_zone_lookup.csv"
# Contours des zones (shapefile), pour les cartes de la documentation
shp="$RAW/ref/taxi_zones.zip"
[ -s "$shp" ] || curl -sSfL -o "$shp" "$BASE/misc/taxi_zones.zip"
for y in 2019 2020 2024 2025; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    for s in yellow green; do
      f="$RAW/tlc/${s}_tripdata_${y}-${m}.parquet"
      [ -s "$f" ] || curl -sSfL --retry 3 -o "$f" "$BASE/trip-data/${s}_tripdata_${y}-${m}.parquet"
    done
  done
done

if [ "${1:-}" = "--update-manifest" ]; then
  (cd "$RAW" && sha256sum ref/*.csv ref/*.zip tlc/*.parquet) > configs/manifest_sha256.txt
  echo "manifeste mis à jour"
else
  (cd "$RAW" && sha256sum --quiet -c ../../configs/manifest_sha256.txt) && echo "empreintes vérifiées"
fi
