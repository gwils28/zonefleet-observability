"""Tests de la CLI (analyse des arguments, sans calcul lourd)."""

import pytest

from zonefleet.cli import EXPERIMENTS, build_parser, main


def test_parser_subcommands():
    ap = build_parser()
    assert ap.parse_args(["monitor", "--split", "covid"]).split == "covid"
    assert ap.parse_args(["run", "e3"]).experiment == "e3"
    with pytest.raises(SystemExit):
        ap.parse_args(["run", "e9"])


def test_every_experiment_module_has_main():
    import importlib

    for mod in EXPERIMENTS.values():
        assert callable(importlib.import_module(mod).main)


def test_contract_without_raw_files_exits_2(monkeypatch, capsys):
    monkeypatch.setattr("zonefleet.data.tlc_files", lambda years=None: [])
    assert main(["contract"]) == 2
    assert "make data" in capsys.readouterr().err
