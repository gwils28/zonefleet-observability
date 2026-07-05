"""Agrégation des courses TLC en panel zone x heure.

Les horodatages TLC sont naïfs, en heure locale de New York. Deux artefacts en découlent :
- passage à l'heure d'été (mars) : l'heure 02:00 n'existe pas, elle reçoit presque
  aucune course ; on la remplace par la moyenne de ses voisines ;
- retour à l'heure d'hiver (novembre) : l'heure 01:00 cumule deux heures réelles ; on
  divise son volume par deux.
Les deux heures sont marquées par la colonne `dst_repaired`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from zonefleet.config import DATA_PROCESSED, DATA_RAW, N_ZONES, TZ
from zonefleet.contract import load_contract, parse_name, validate_file

TLC_DIR = DATA_RAW / "tlc"


def tlc_files(years: tuple[int, ...] | None = None) -> list[Path]:
    """Fichiers Parquet TLC présents dans `data/raw/tlc`, filtrés par année."""
    files = sorted(TLC_DIR.glob("*_tripdata_*.parquet"))
    if years:
        files = [f for f in files if int(parse_name(f)[1][:4]) in years]
    return files


def aggregate_file(path: Path, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Compte les prises en charge par zone et par heure pour un fichier mensuel.

    Les courses dont l'horodatage sort du mois du fichier (erreurs d'horloge des
    taximètres, fréquentes dans TLC) sont écartées, ainsi que les zones inconnues (264, 265).
    """
    rep = validate_file(path)
    if not rep.ok:
        raise ValueError(f"{path.name} viole le data contract : {rep.violations}")
    service, month = parse_name(path)
    col = load_contract()["services"][service]["pickup_column"]
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    con = con or duckdb.connect()
    q = f"""
        SELECT CAST(PULocationID AS INTEGER) AS zone,
               date_trunc('hour', {col}) AS ts,
               COUNT(*) AS pickups
        FROM read_parquet('{path.as_posix()}')
        WHERE {col} >= TIMESTAMP '{start}' AND {col} < TIMESTAMP '{end}'
          AND PULocationID BETWEEN 1 AND 263
        GROUP BY 1, 2
    """
    df = con.execute(q).df()
    df["service"] = service
    return df


def aggregate_all(files: list[Path]) -> pd.DataFrame:
    """Agrège tous les fichiers en une table longue (zone, ts, service, pickups)."""
    con = duckdb.connect()
    parts = [aggregate_file(f, con) for f in files]
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    return out


def cbd_share(files: list[Path]) -> pd.DataFrame:
    """Part des courses avec `cbd_congestion_fee > 0` par zone de prise en charge.

    Une zone située dans la zone de tarification a une part proche de 1 : toute course
    qui y commence paie. Calculé sur les fichiers 2025 qui portent la colonne.
    """
    con = duckdb.connect()
    rows = []
    for f in files:
        service, month = parse_name(f)
        if month < "2025-01" or "cbd_congestion_fee" not in validate_file(f).types:
            continue
        rows.append(
            con.execute(f"""
            SELECT CAST(PULocationID AS INTEGER) AS zone,
                   SUM(CASE WHEN cbd_congestion_fee > 0 THEN 1 ELSE 0 END) AS with_fee,
                   COUNT(*) AS trips
            FROM read_parquet('{f.as_posix()}')
            WHERE PULocationID BETWEEN 1 AND 263
            GROUP BY 1
        """).df()
        )
    df = pd.concat(rows).groupby("zone", as_index=False)[["with_fee", "trips"]].sum()
    df["cbd_share"] = df["with_fee"] / df["trips"]
    return df


def top_zones(long: pd.DataFrame, year: int = 2019, n: int = N_ZONES) -> list[int]:
    """Les n zones au plus grand volume (jaune + vert) sur l'année de référence."""
    sub = long[long["ts"].dt.year == year]
    tot = sub.groupby("zone")["pickups"].sum().sort_values(ascending=False)
    return sorted(int(z) for z in tot.index[:n])


def dst_hours(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Heures locales naïves inexistantes (printemps) et ambiguës (automne) de l'index."""
    years = sorted(set(index.year))
    spring, fall = [], []
    for y in years:
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D", tz=TZ)
        offs = days.map(lambda d: d.utcoffset())
        for i in range(1, len(days)):
            if offs[i] > offs[i - 1]:
                spring.append(pd.Timestamp(days[i - 1].date()) + pd.Timedelta(hours=2))
            elif offs[i] < offs[i - 1]:
                fall.append(pd.Timestamp(days[i - 1].date()) + pd.Timedelta(hours=1))
    keep = lambda xs: pd.DatetimeIndex([x for x in xs if x in index])  # noqa: E731
    return keep(spring), keep(fall)


def build_panel(long: pd.DataFrame, zones: list[int], start: str, end: str) -> pd.DataFrame:
    """Panel complet zone x heure (zéros explicites) avec réparation des heures DST.

    Colonnes : zone, ts, pickups (jaune + vert), yellow, green, dst_repaired.
    `end` est exclu.
    """
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    sub = long[long["zone"].isin(zones) & (long["ts"] >= idx[0]) & (long["ts"] <= idx[-1])]
    wide = sub.pivot_table(index=["zone", "ts"], columns="service", values="pickups", aggfunc="sum")
    full = pd.MultiIndex.from_product([zones, idx], names=["zone", "ts"])
    wide = wide.reindex(full).fillna(0.0)
    for s in ("yellow", "green"):
        if s not in wide:
            wide[s] = 0.0
    panel = wide[["yellow", "green"]].astype(float).reset_index()
    panel["dst_repaired"] = False
    spring, fall = dst_hours(idx)
    panel = panel.set_index(["zone", "ts"]).sort_index()
    for s in ("yellow", "green"):
        col = panel[s].unstack("zone")
        one = pd.Timedelta(hours=1)
        for h in spring:
            nxt = col.loc[h + one] if h + one in col.index else col.loc[h - one]
            col.loc[h] = 0.5 * (col.loc[h - one] + nxt)
        for h in fall:
            col.loc[h] = col.loc[h] / 2.0
        panel[s] = col.stack().reorder_levels(["zone", "ts"]).sort_index()
    panel.loc[(slice(None), spring.append(fall)), "dst_repaired"] = True
    panel = panel.reset_index()
    panel["pickups"] = panel["yellow"] + panel["green"]
    return panel[["zone", "ts", "pickups", "yellow", "green", "dst_repaired"]]


def check_panel(panel: pd.DataFrame, max_per_hour: int | None = None) -> list[str]:
    """Contrôles de cohérence du panel (règle `panel` du data contract)."""
    max_per_hour = max_per_hour or load_contract()["panel"]["max_pickups_per_zone_hour"]
    issues = []
    if panel.duplicated(["zone", "ts"]).any():
        issues.append("doublons zone x heure")
    if (panel["pickups"] < 0).any():
        issues.append("volumes négatifs")
    if (panel["pickups"] > max_per_hour).any():
        issues.append(f"volume > {max_per_hour} courses par zone et par heure")
    n = panel.groupby("zone")["ts"].count()
    if n.nunique() != 1:
        issues.append("zones de longueurs différentes")
    return issues


def load_panel(path: Path | None = None) -> pd.DataFrame:
    """Charge le panel agrégé (`data/processed/panel.parquet`)."""
    return pd.read_parquet(path or DATA_PROCESSED / "panel.parquet")


def log1p_demand(x: pd.Series | np.ndarray) -> np.ndarray:
    """Transformation de la demande (intermittente sur les zones peu actives)."""
    return np.log1p(np.asarray(x, dtype=float))
