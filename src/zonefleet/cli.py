"""Interface en ligne de commande.

Exemples :
    zonefleet contract                  # valide les fichiers TLC contre le data contract
    zonefleet monitor --split covid     # signaux journaliers (CSV sur la sortie standard)
    zonefleet alert --split congestion  # sévérité du dernier jour : page, ticket, aucune
    zonefleet report --date 2020-03-23 --split covid   # rapport du matin
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


def daily_report(split: str, day: str | None = None) -> str:
    """Rapport Markdown d'un jour (par défaut le dernier jour de l'analysis)."""
    import pandas as pd

    from zonefleet.alerting import daily_signals
    from zonefleet.features import zone_table
    from zonefleet.monitoring import monitor
    from zonefleet.pipeline import predictions
    from zonefleet.report import build
    from zonefleet.rowscore import RowScorer

    p = predictions(split)
    ref, an = p[p.period == "reference"], p[p.period == "analysis"]
    out = monitor(ref, an)
    sig = daily_signals(out)
    target = pd.Timestamp(day) if day else sig.index[-1]
    today = an[an["ts"].dt.floor("D") == target]
    rows = RowScorer().fit(ref).score_rows(today)
    rows["zone"] = today["zone"].astype(int).to_numpy()
    zl = rows.groupby("zone")["est"].mean().reset_index()
    zl = zl.merge(zone_table()[["zone", "name"]], on="zone", how="left")
    from zonefleet import quality as Q
    from zonefleet.data import load_panel

    panel = load_panel()
    year = panel[
        (panel["ts"] >= target - pd.Timedelta(days=400))
        & (panel["ts"] < target + pd.Timedelta(days=1))
    ]
    history = year[year["ts"] < target - pd.Timedelta(days=30)]
    checks = Q.daily_checks(year, Q.calibrate(Q.daily_checks(history)))
    quality = checks.loc[target] if target in checks.index else None
    version = "1" if split == "covid" else "2"
    return build(target, sig, zl, out["psi"], model_version=version, quality=quality)


def cmd_report(args: argparse.Namespace) -> int:
    print(daily_report(args.split, args.date))
    return 0


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
    p = sub.add_parser("report", help="rapport quotidien de monitoring (Markdown)")
    p.add_argument("--split", choices=["covid", "congestion"], default="congestion")
    p.add_argument("--date", default=None, help="AAAA-MM-JJ (défaut : dernier jour)")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("run", help="lance une expérience")
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.set_defaults(fn=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
