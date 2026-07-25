"""Tests de l'agrégation sur de vrais fichiers Parquet synthétiques (dossier temporaire)."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from zonefleet import data as D


def write_tlc(path, service, month, rows):
    col = "tpep_pickup_datetime" if service == "yellow" else "lpep_pickup_datetime"
    drop = col.replace("pickup", "dropoff")
    ts = pd.to_datetime([r[0] for r in rows])
    table = pa.table(
        {
            col: pa.array(ts, pa.timestamp("us")),
            drop: pa.array(ts + pd.Timedelta(minutes=10), pa.timestamp("us")),
            "PULocationID": pa.array([r[1] for r in rows], pa.int64()),
            "DOLocationID": pa.array([1] * len(rows), pa.int64()),
        }
    )
    f = path / f"{service}_tripdata_{month}.parquet"
    pq.write_table(table, f)
    return f


def test_aggregate_file_counts_and_filters(tmp_path):
    f = write_tlc(
        tmp_path,
        "yellow",
        "2019-03",
        [
            ("2019-03-01 08:10", 161),
            ("2019-03-01 08:50", 161),
            ("2019-03-01 09:05", 161),
            ("2019-02-28 23:59", 161),  # hors du mois du fichier : écarté
            ("2019-03-02 10:00", 264),  # zone inconnue : écartée
        ],
    )
    out = D.aggregate_file(f).set_index("ts")
    assert out.loc["2019-03-01 08:00", "pickups"] == 2
    assert out["pickups"].sum() == 3
    assert (out["service"] == "yellow").all()


def test_aggregate_file_rejects_contract_violation(tmp_path):
    table = pa.table(
        {"tpep_pickup_datetime": pa.array(pd.to_datetime(["2019-03-01"]), pa.timestamp("us"))}
    )
    f = tmp_path / "yellow_tripdata_2019-03.parquet"
    pq.write_table(table, f)
    with pytest.raises(ValueError, match="contract"):
        D.aggregate_file(f)


def test_aggregate_all_and_tlc_files(tmp_path, monkeypatch):
    write_tlc(tmp_path, "yellow", "2019-03", [("2019-03-01 08:10", 7)])
    write_tlc(tmp_path, "green", "2020-03", [("2020-03-01 08:10", 7)])
    monkeypatch.setattr(D, "TLC_DIR", tmp_path)
    assert len(D.tlc_files()) == 2
    assert [f.name for f in D.tlc_files((2020,))] == ["green_tripdata_2020-03.parquet"]
    long = D.aggregate_all(D.tlc_files())
    assert set(long["service"]) == {"yellow", "green"}
    assert pd.api.types.is_datetime64_any_dtype(long["ts"])
