"""Interface en ligne de commande.

Exemples :
    zonefleet contract                  # valide les fichiers TLC contre le data contract
    zonefleet monitor --split covid     # signaux journaliers (CSV sur la sortie standard)
    zonefleet alert --split congestion  # sévérité du dernier jour : page, ticket, aucune
    zonefleet run e3                    # lance une expérience du SPEC
"""

from __future__ import annotations

import argparse
import importlib
import sys

EXPERIMENTS = {
    "e0": "zonefleet.experiments.e0_model",
    "e1": "zonefleet.experiments.e1_covid",
    "e2": "zonefleet.experiments.e2_congestion",
    "e3": "zonefleet.experiments.e3_frontier",
    "e4": "zonefleet.experiments.e4_backtest",
    "e5": "zonefleet.experiments.e5_retraining",
}


def cmd_contract(_: argparse.Namespace) -> int:
    from zonefleet.contract import schema_timeline
    from zonefleet.data import tlc_files

    files = tlc_files()
    if not files:
        print("aucun fichier dans data/raw/tlc : lancer `make data`", file=sys.stderr)
        return 2
    reps = schema_timeline(files)
    bad = [r for r in reps if not r.ok]
    unknown = [r for r in reps if r.unknown_breaks]
    print(f"{len(reps)} fichiers, {len(bad)} en violation, {len(unknown)} avec rupture inconnue")
    for r in bad:
        print(f"  VIOLATION {r.file} : {'; '.join(r.violations)}")
    for r in unknown:
        print(f"  INCONNU   {r.file} : {'; '.join(r.unknown_breaks)}")
    return 1 if bad else 0


def _signals(split: str):
    from zonefleet.alerting import daily_signals
    from zonefleet.monitoring import monitor
    from zonefleet.pipeline import predictions

    p = predictions(split)
    return daily_signals(monitor(p[p.period == "reference"], p[p.period == "analysis"]))


def cmd_monitor(args: argparse.Namespace) -> int:
    _signals(args.split).round(4).to_csv(sys.stdout, index_label="date")
    return 0


def cmd_alert(args: argparse.Namespace) -> int:
    from zonefleet.ops.decisions import PAGE, alert_severity

    sev = alert_severity(_signals(args.split))
    print(sev)
    return 1 if sev == PAGE else 0


def cmd_run(args: argparse.Namespace) -> int:
    importlib.import_module(EXPERIMENTS[args.experiment]).main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zonefleet", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("contract", help="valide les fichiers TLC").set_defaults(fn=cmd_contract)
    for name, fn, helptext in [
        ("monitor", cmd_monitor, "signaux journaliers d'une expérience"),
        ("alert", cmd_alert, "sévérité d'alerte du dernier jour"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--split", choices=["covid", "congestion"], default="congestion")
        p.set_defaults(fn=fn)
    p = sub.add_parser("run", help="lance une expérience")
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.set_defaults(fn=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
