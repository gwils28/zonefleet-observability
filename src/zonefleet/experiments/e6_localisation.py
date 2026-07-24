"""E6 : localiser la dérive par zone (diagnostic à la DriftGuard, sans labels).

Usage : uv run python -m zonefleet.experiments.e6_localisation
Sorties : results/tables/e6_*.csv

Les scores par ligne de `RowScorer` (perte estimée, erreur réalisée, valeur de
`lag_168`) sont agrégés par zone et par fenêtre :
- **réel** : COVID (13 mars-12 avril 2020 contre 6 janvier-12 mars) et taxe de congestion
  (février-juin 2025 contre la référence) ;
- **injecté** : choc de niveau propagé et concept drift limités aux 32 zones CBD
  (facteur 0,6, à partir du 3 mars 2025) ; on mesure l'AUC du classement des 60 zones
  par chaque signal, la vérité étant « zone CBD ».
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from zonefleet.config import SPLITS
from zonefleet.covariates import load_weather
from zonefleet.data import load_panel
from zonefleet.features import zone_table
from zonefleet.monitoring import psi
from zonefleet.pipeline import TABLES, frozen_model, predictions
from zonefleet.rowscore import RowScorer
from zonefleet.scenarios import concept_drift, level_shock

ONSET = pd.Timestamp("2025-03-03")
FACTOR = 0.6
WINDOW = pd.Timedelta(days=14)


def zone_scores(scorer: RowScorer, df: pd.DataFrame) -> pd.DataFrame:
    """Scores par ligne, avec la zone."""
    rows = scorer.score_rows(df)
    rows["zone"] = df["zone"].astype(int).to_numpy()
    return rows


def zone_delta(
    rows: pd.DataFrame, ref_rows: pd.DataFrame, after: tuple, before: tuple | None = None
):
    """Par zone : hausse de perte estimée et réalisée, PSI de lag_168, entre deux fenêtres.

    `before` absent : la fenêtre de comparaison est la référence entière.
    """
    a = rows[(rows["date"] >= after[0]) & (rows["date"] < after[1])]
    b = (
        ref_rows
        if before is None
        else rows[(rows["date"] >= before[0]) & (rows["date"] < before[1])]
    )
    ga, gb = a.groupby("zone"), b.groupby("zone")
    out = pd.DataFrame(
        {
            "est_before": gb["est"].mean(),
            "est_after": ga["est"].mean(),
            "real_before": gb["real"].mean(),
            "real_after": ga["real"].mean(),
        }
    )
    out["delta_est"] = out["est_after"] - out["est_before"]
    out["delta_real"] = out["real_after"] - out["real_before"]
    out["psi_lag_168"] = [
        psi(gb.get_group(z)["lag_168"].to_numpy(), ga.get_group(z)["lag_168"].to_numpy())
        for z in out.index
    ]
    return out.join(zone_table().set_index("zone")[["name", "borough", "cbd"]])


def auc(truth: pd.Series, score: pd.Series) -> float:
    return float(roc_auc_score(truth.astype(int), score))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    real_rows = []

    # --- réel : COVID
    p20 = predictions("covid")
    sc20 = RowScorer().fit(p20[p20.period == "reference"])
    r20 = zone_scores(sc20, p20[p20.period == "analysis"])
    ref20 = zone_scores(sc20, p20[p20.period == "reference"])
    covid = zone_delta(r20, ref20, ("2020-03-13", "2020-04-13"), ("2020-01-06", "2020-03-13"))
    covid.round(4).to_csv(TABLES / "e6_zones_covid.csv", index_label="zone")
    real_rows.append(
        {
            "cas": "COVID",
            "corr_est_real": covid["delta_est"].corr(covid["delta_real"]),
            "spearman_est_real": covid["delta_est"].corr(covid["delta_real"], method="spearman"),
        }
    )

    # --- réel : congestion
    p25 = predictions("congestion")
    ref25 = p25[p25.period == "reference"]
    an25 = p25[p25.period == "analysis"].reset_index(drop=True)
    sc25 = RowScorer().fit(ref25)
    ref_rows = zone_scores(sc25, ref25)
    base_rows = zone_scores(sc25, an25)
    cong = zone_delta(base_rows, ref_rows, ("2025-02-01", "2025-07-01"))
    cong.round(4).to_csv(TABLES / "e6_zones_congestion.csv", index_label="zone")
    real_rows.append(
        {
            "cas": "congestion",
            "corr_est_real": cong["delta_est"].corr(cong["delta_real"]),
            "spearman_est_real": cong["delta_est"].corr(cong["delta_real"], method="spearman"),
        }
    )
    pd.DataFrame(real_rows).round(4).to_csv(TABLES / "e6_real_correlations.csv", index=False)

    # --- injecté : CBD seulement
    start, end = SPLITS["congestion"].period("analysis")
    window = (ONSET, ONSET + WINDOW)
    before = (pd.Timestamp("2025-01-06"), ONSET)
    cases = {
        "choc propagé CBD": level_shock(
            load_panel(),
            load_weather(),
            frozen_model("congestion"),
            start,
            end,
            ONSET,
            FACTOR,
            subset="cbd",
            zones=zone_table(),
        ),
        "concept pur CBD": concept_drift(an25, ONSET, FACTOR, subset="cbd"),
    }
    base = zone_delta(base_rows, ref_rows, window, before)
    auc_rows = []
    for name, df in cases.items():
        z = zone_delta(zone_scores(sc25, df.reset_index(drop=True)), ref_rows, window, before)
        z["delta_est_vs_base"] = z["delta_est"] - base["delta_est"]
        z["delta_real_vs_base"] = z["delta_real"] - base["delta_real"]
        # Ajouté après coup (voir docs/notes) : le PSI brut est saturé par la saison.
        z["psi_vs_base"] = z["psi_lag_168"] - base["psi_lag_168"]
        z.round(4).to_csv(TABLES / f"e6_zones_{name.replace(' ', '_')}.csv", index_label="zone")
        for sig, col in [
            ("perte estimée (DLE)", "delta_est_vs_base"),
            ("PSI lag_168", "psi_lag_168"),
            ("PSI lag_168 vs contrefactuel (a posteriori)", "psi_vs_base"),
            ("erreur réalisée (labels)", "delta_real_vs_base"),
        ]:
            score = z[col].abs() if col != "psi_lag_168" else z[col]
            auc_rows.append({"cas": name, "signal": sig, "auc": auc(z["cbd"], score)})
    aucs = pd.DataFrame(auc_rows)
    aucs.round(4).to_csv(TABLES / "e6_auc.csv", index=False)
    print(pd.DataFrame(real_rows).round(3).to_string())
    print(aucs.round(3).to_string())
    print(
        covid.sort_values("delta_real", ascending=False)[["name", "cbd", "delta_est", "delta_real"]]
        .head(8)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
