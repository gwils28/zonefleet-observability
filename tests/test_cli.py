"""Tests de la CLI (analyse des arguments, sans calcul lourd)."""

import pandas as pd
import pytest

from zonefleet import cli
from zonefleet.cli import ALL_EXPERIMENTS, EXPERIMENTS, build_parser, main


def test_parser_subcommands():
    ap = build_parser()
    assert ap.parse_args(["monitor", "--split", "covid"]).split == "covid"
    assert ap.parse_args(["run", "e3"]).experiment == "e3"
    assert ap.parse_args(["run", "e4-ter"]).experiment == "e4-ter"
    with pytest.raises(SystemExit):
        ap.parse_args(["run", "e11"])


def test_every_experiment_module_has_main():
    import importlib

    for mod in ALL_EXPERIMENTS.values():
        assert callable(importlib.import_module(mod).main)


def test_contract_without_raw_files_exits_2(monkeypatch, capsys):
    monkeypatch.setattr("zonefleet.data.tlc_files", lambda years=None: [])
    assert main(["contract"]) == 2
    assert "make data" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.startswith("zonefleet ")


def test_spec_experiments_are_listed_first(capsys):
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(ALL_EXPERIMENTS)
    assert [ln.split()[0] for ln in lines[: len(EXPERIMENTS)]] == list(EXPERIMENTS)
    assert all(ln.split()[1] == "ext." for ln in lines[len(EXPERIMENTS) :])


def test_describe_reads_docstring_without_import():
    assert cli.describe("zonefleet.experiments.e3_frontier").startswith("E3 :")
    assert cli.describe("zonefleet.module_inexistant") == "?"


def _fake_signals(page: bool) -> pd.DataFrame:
    idx = pd.date_range("2020-03-01", periods=5, freq="D", name="date")
    return pd.DataFrame(
        {
            "drift_clean": [False, False, page, page, page],
            "perf_est": [False, False, page, page, page],
            "estimated": 0.3,
            "threshold": 0.4,
            "perf_real": None,
        },
        index=idx,
    )


def test_monitor_writes_csv(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_signals", lambda split: _fake_signals(False))
    assert main(["monitor", "--split", "covid"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("date,") and out.count("\n") == 6


def test_run_dispatches_to_module_main(monkeypatch):
    called = []

    class Fake:
        @staticmethod
        def main():
            called.append(True)

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: Fake)
    assert main(["run", "e10"]) == 0 and called == [True]


def test_report_prints_markdown(monkeypatch, capsys):
    monkeypatch.setattr(cli, "daily_report", lambda split, day: f"# Rapport {split} {day}")
    assert main(["report", "--split", "covid", "--date", "2020-03-23"]) == 0
    assert capsys.readouterr().out.strip() == "# Rapport covid 2020-03-23"


@pytest.mark.parametrize(("page", "code", "word"), [(True, 1, "page"), (False, 0, "aucune")])
def test_alert_exit_code_follows_severity(monkeypatch, capsys, page, code, word):
    monkeypatch.setattr(cli, "_signals", lambda split: _fake_signals(page))
    assert main(["alert"]) == code
    assert capsys.readouterr().out.strip() == word
