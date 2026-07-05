"""Tests du data contract sur des schémas Arrow synthétiques."""

import pyarrow as pa
import pytest

from zonefleet.contract import load_contract, parse_name, type_changes, validate_schema


@pytest.fixture(scope="module")
def contract():
    return load_contract()


def schema(**cols):
    return pa.schema([(k, v) for k, v in cols.items()])


def base(**extra):
    return schema(
        tpep_pickup_datetime=pa.timestamp("us"),
        tpep_dropoff_datetime=pa.timestamp("us"),
        PULocationID=pa.int64(),
        DOLocationID=pa.int64(),
        **extra,
    )


def test_valid_schema_has_no_violation(contract):
    rep = validate_schema(base(fare_amount=pa.float64()), "yellow", "2019-01", contract)
    assert rep.ok and not rep.unknown_breaks


def test_missing_required_column_is_blocking(contract):
    s = schema(tpep_pickup_datetime=pa.timestamp("us"), PULocationID=pa.int64())
    rep = validate_schema(s, "yellow", "2019-01", contract)
    assert not rep.ok
    assert any("DOLocationID" in v for v in rep.violations)


def test_integer_stored_as_float_is_tolerated(contract):
    s = base()
    s = s.set(s.get_field_index("PULocationID"), pa.field("PULocationID", pa.float64()))
    assert validate_schema(s, "yellow", "2019-01", contract).ok


def test_string_location_is_a_violation(contract):
    s = base()
    s = s.set(s.get_field_index("PULocationID"), pa.field("PULocationID", pa.string()))
    assert not validate_schema(s, "yellow", "2019-01", contract).ok


def test_known_break_is_logged_only_after_its_date(contract):
    after = validate_schema(base(cbd_congestion_fee=pa.float64()), "yellow", "2025-02", contract)
    assert after.known_breaks and not after.unknown_breaks
    before = validate_schema(base(cbd_congestion_fee=pa.float64()), "yellow", "2024-02", contract)
    assert before.unknown_breaks


def test_unknown_column_is_flagged(contract):
    rep = validate_schema(base(surge_multiplier=pa.float64()), "yellow", "2024-05", contract)
    assert rep.ok and rep.unknown_breaks == ["colonne inconnue : surge_multiplier"]


def test_type_change_detection(contract):
    a = validate_schema(base(passenger_count=pa.float64()), "yellow", "2020-12", contract)
    b = validate_schema(base(passenger_count=pa.int64()), "yellow", "2024-01", contract)
    assert type_changes([a, b]) == ["yellow passenger_count : float (2020-12) -> integer (2024-01)"]


def test_parse_name():
    from pathlib import Path

    assert parse_name(Path("green_tripdata_2025-03.parquet")) == ("green", "2025-03")
    with pytest.raises(ValueError):
        parse_name(Path("fhvhv_tripdata_2025-03.parquet"))
