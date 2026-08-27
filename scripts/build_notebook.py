"""Construit le notebook tutoriel « DLE pas à pas » (site/notebooks/tutoriel-dle.ipynb).

Usage : uv run python scripts/build_notebook.py [nom.ipynb ...]
        uv run jupyter nbconvert --to notebook --execute --inplace site/notebooks/<nom>.ipynb
Le notebook est exécuté localement ; le site le rend avec ses sorties, sans le relancer.
"""

from __future__ import annotations

import nbformat as nbf

from zonefleet.config import ROOT

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# DLE pas à pas : estimer l'erreur d'un forecaster sans vérité terrain

Ce notebook rejoue, en petit, le cœur du projet :

1. charger le forecaster figé de 2024 et ses prévisions ;
2. ajuster **NannyML DLE** sur la référence (sept.-déc. 2024) ;
3. estimer l'erreur de l'analysis (janv.-juin 2025) **sans lire les labels** ;
4. injecter un *covariate shift* puis un *concept drift* et voir lequel DLE détecte.

Prérequis : `make panel` (ou le panel versionné) puis `make e0`, qui met en cache le
modèle et ses prévisions.""",
    ),
    (
        "code",
        """import os
os.environ["NML_DISABLE_USAGE_LOGGING"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zonefleet import viz
from zonefleet.monitoring import estimate_performance, realized_performance
from zonefleet.pipeline import predictions
from zonefleet.scenarios import concept_drift, covariate_shift

viz.setup()
pred = predictions("congestion")
ref = pred[pred.period == "reference"]
an = pred[pred.period == "analysis"].reset_index(drop=True)
print(f"référence : {len(ref):,} lignes · analysis : {len(an):,} lignes")""",
    ),
    (
        "md",
        """## 1. L'hétéroscédasticité que DLE va exploiter

DLE ne sert à rien si le modèle se trompe partout pareil. Regardons l'erreur relative
(MAE sur log1p) selon le niveau récent de la demande (`lag_168`) dans la référence.""",
    ),
    (
        "code",
        """#| fig-alt: "Barres de l'erreur relative moyenne par décile de lag_168 dans la référence : environ 0,44 au premier décile, décroissante jusqu'à environ 0,17 aux deux derniers."
err = np.abs(np.log1p(ref["pickups"]) - np.log1p(ref["q50"]))
decile = pd.qcut(ref["lag_168"], 10, labels=[f"D{i}" for i in range(1, 11)])
by_decile = err.groupby(decile, observed=True).mean()
ax = by_decile.plot.bar(color=viz.BLUE, figsize=(7, 3))
ax.set_ylabel("MAE log")
ax.set_xlabel("décile de lag_168 (niveau récent)")
ax.set_title("Le modèle se trompe 2,5 fois plus dans les situations de faible niveau")
plt.show()
by_decile.round(3)""",
    ),
    (
        "md",
        """## 2. DLE sur l'analysis réelle

`estimate_performance` ajuste le nanny model sur la référence et estime l'analysis jour
par jour. Les labels de l'analysis ne lui sont **pas** transmis ; on les révèle
ensuite pour comparer.""",
    ),
    (
        "code",
        """#| fig-alt: "Courbes journalières de l'erreur estimée et réalisée sur l'analysis 2025 sans dérive, avec le seuil NannyML vers 0,52 : les deux courbes oscillent entre 0,2 et 0,5 ; l'estimée approche le seuil en janvier et fin février."
est = estimate_performance(ref, an, scale="log", metrics=("mae",))
real = realized_performance(an, scale="log").query("metric == 'mae_log'")
cmp = est.merge(real, on=["date", "metric"])
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(cmp["date"], cmp["estimated"], color=viz.BLUE, label="estimée (sans labels)")
ax.plot(cmp["date"], cmp["realized"], color=viz.ORANGE, label="réalisée")
ax.plot(cmp["date"], cmp["threshold"], color=viz.MUTED, ls="--", lw=1, label="seuil NannyML")
ax.legend(loc="upper right")
ax.set_title("Analysis 2025 sans dérive injectée")
plt.show()
print("corrélation journalière :", round(cmp["estimated"].corr(cmp["realized"]), 2))""",
    ),
    (
        "md",
        """## 3. Un covariate shift nuisible

À partir du 3 mars, chaque jour est rééchantillonné vers les situations de faible niveau
(poids exp(-z), z = `lag_168` standardisé). Chaque ligne reste une vraie ligne : la
relation X → y n'a pas changé, seule la distribution des entrées bouge.""",
    ),
    (
        "code",
        """#| fig-alt: "Courbes journalières de l'erreur estimée et réalisée en 2025 avec un trait au 3 mars : après l'injection d'un covariate shift nuisible, les deux montent ensemble vers 0,4."
onset = pd.Timestamp("2025-03-03")
stats = (ref["lag_168"].mean(), ref["lag_168"].std())
an_cov = covariate_shift(an, onset, kappa=1.5, ref_stats=stats, seed=0)

def compare(df, title):
    e = estimate_performance(ref, df, scale="log", metrics=("mae",))
    r = realized_performance(df, scale="log").query("metric == 'mae_log'")
    c = e.merge(r, on=["date", "metric"])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(c["date"], c["estimated"], color=viz.BLUE, label="estimée (sans labels)")
    ax.plot(c["date"], c["realized"], color=viz.ORANGE, label="réalisée")
    ax.axvline(onset, color=viz.INK2, lw=1)
    ax.legend(loc="upper left")
    ax.set_title(title)
    plt.show()
    post = c[c["date"] >= onset]
    return post[["estimated", "realized"]].mean().round(3)

compare(an_cov, "Covariate shift nuisible : DLE suit")""",
    ),
    (
        "md",
        """## 4. Un concept drift pur

Même date, mais cette fois la demande est divisée par deux **sans toucher aux
features** : le modèle et DLE voient exactement les mêmes entrées qu'avant.""",
    ),
    (
        "code",
        """#| fig-alt: "Courbes journalières de l'erreur estimée et réalisée en 2025 avec un trait au 3 mars : après un concept drift pur, la réalisée saute vers 0,6 à 0,7 tandis que l'estimée reste vers 0,25."
an_con = concept_drift(an, onset, factor=0.5)
compare(an_con, "Concept drift pur : DLE ne voit rien")""",
    ),
    (
        "md",
        """## À retenir

- DLE transforme la fenêtre aveugle en fenêtre floue : il voit les dérives qui déplacent
  le modèle vers des régions où il se trompait déjà davantage.
- Il ne voit **pas** un changement de la relation X → y. Pour un forecaster à lags, tout
  choc brutal commence par là, jusqu'à ce que les lags l'absorbent.
- La mesure complète, sur 13 scénarios puis 240, est dans les pages
  [E3](../experiences/e3-frontiere.qmd) et [E4](../experiences/e4-backtest.qmd).""",
    ),
]


COST_CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# Choisir une politique d'alerte par son coût

E4 compare six politiques sur 240 scénarios. Aucune ne domine sur tous les critères :
il faut **arbitrer** entre fausses pages, dégradations manquées et retard. Ce notebook
rend l'arbitrage explicite : on donne un coût à chaque type d'erreur, et on cherche la
politique et le seuil k qui minimisent le coût attendu par scénario.

Données : `results/tables/e4_results.csv` (une ligne par scénario, politique, seuil k et
lissage). Rien n'est recalculé.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zonefleet import viz
from zonefleet.config import RESULTS
from zonefleet.labels import POLICY_COLOR, POLICY_SHORT

viz.setup()
r = pd.read_csv(RESULTS / "tables" / "e4_results.csv")
r = r[r["policy"].isin(POLICY_COLOR)]
print(r.groupby("policy").size().to_string())""",
    ),
    (
        "md",
        """## Le modèle de coût

Par scénario :

- **fausse page** : chaque allumage d'alerte sur un jour sans dérive coûte `c_fa`
  (une nuit d'astreinte, de la confiance perdue dans le moniteur) ;
- **dégradation manquée** : un scénario nuisible non détecté dans les 28 jours coûte
  `c_miss` (des semaines de flotte mal positionnée) ;
- **retard** : chaque jour entre l'injection et la décision coûte `c_day`.

Les valeurs par défaut sont des ordres de grandeur à remplacer par ceux de ZoneFleet.""",
    ),
    (
        "code",
        """def expected_cost(r, c_fa=1.0, c_miss=10.0, c_day=0.3):
    d = r.copy()
    missed = d["harmful"] & ~d["detected"]
    late = np.where(d["harmful"] & d["detected"], d["delay_days"].clip(lower=0), 0.0)
    d["cost"] = c_fa * d["false_alert_events"] + c_miss * missed + c_day * late
    return d.groupby(["policy", "smooth", "k"])["cost"].mean().reset_index()


costs = expected_cost(r)
best = costs.sort_values("cost").head(8)
best.assign(policy=best["policy"].map(POLICY_SHORT)).round(3)""",
    ),
    (
        "md",
        """## Le choix dépend du prix d'une fausse page

On fait varier le coût d'une fausse page de 0,1 à 20 (le coût d'une dégradation manquée
reste à 10) et on regarde quelle politique gagne.""",
    ),
    (
        "code",
        """#| fig-alt: "Points du coût attendu minimal en fonction du coût d'une fausse page, en échelle log de 0,1 à 20, colorés par politique gagnante : la dérive nettoyée gagne jusqu'à environ 0,7, avec un coût croissant de 1,40 à 1,67 ; au-delà, la condition jointe gagne avec un coût constant d'environ 1,70."
grid = np.geomspace(0.1, 20, 30)
winners = []
for c in grid:
    cc = expected_cost(r, c_fa=c).sort_values("cost").iloc[0]
    winners.append((c, cc["policy"], cc["smooth"], cc["k"], cc["cost"]))
w = pd.DataFrame(winners, columns=["c_fa", "policy", "smooth", "k", "cost"])
fig, ax = plt.subplots(figsize=(9, 3.4))
for pol, col in POLICY_COLOR.items():
    sub = w[w["policy"] == pol]
    ax.scatter(sub["c_fa"], sub["cost"], color=col, s=30, label=POLICY_SHORT[pol])
ax.set_xscale("log")
ax.set_xlabel("coût d'une fausse page (coût d'une dégradation manquée = 10)")
ax.set_ylabel("coût attendu minimal")
ax.set_title("Politique optimale selon le prix d'une fausse page")
ax.legend(loc="upper left", fontsize=8.5)
plt.show()
w.drop_duplicates("policy").round(3)""",
    ),
    (
        "md",
        """## Lecture

- Tant qu'une fausse page coûte moins d'environ **0,75** (soit 7,5 % d'une dégradation
  manquée), la **dérive nettoyée seule** gagne : elle détecte le plus, quitte à alerter
  plus souvent.
- Au-delà, la **condition jointe nettoyée** devient optimale : elle ne fait jamais de
  fausse page et garde un délai de quelques jours.
- Les **labels** (P4) ne gagnent jamais : leur délai de deux mois coûte trop cher, même
  avec un coût journalier de retard modeste.

Le choix d'une politique d'astreinte n'est donc pas une question de « meilleure
méthode » mais de **prix relatif des erreurs**, à fixer avec le métier. Pour ZoneFleet,
où une nuit d'astreinte inutile coûte plus que quelques pourcents d'une semaine de flotte
mal placée, la condition jointe est le bon choix.""",
    ),
]


ALERT_CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# Anatomie d'une alerte : le 22 mars 2020

Le 13 mars 2020, New York déclare l'état d'urgence. Ce notebook rejoue, jour par jour, ce
que le moniteur ZoneFleet aurait vu, sans aucun label, jusqu'à la première page ; puis ce
que les labels révèlent deux mois plus tard.

Prérequis : le cache des prévisions (`make e0`).""",
    ),
    (
        "code",
        """import os
os.environ["NML_DISABLE_USAGE_LOGGING"] = "1"

import pandas as pd

from zonefleet.alerting import daily_signals, policy_flags
from zonefleet.monitoring import monitor
from zonefleet.pipeline import predictions

p = predictions("covid")
ref, an = p[p.period == "reference"], p[p.period == "analysis"]
out = monitor(ref, an)
sig = daily_signals(out)
flags = policy_flags(sig)
window = sig.loc["2020-03-09":"2020-03-25"]
cols = ["psi_clean_max", "pca_alert", "drift_clean", "estimated", "threshold", "perf_est"]
window[cols].round(3)""",
    ),
    (
        "md",
        """## Ce qu'on lit dans ce tableau

- Le **PSI des lags** dépasse 0,25 dès le 14 mars : la dérive nettoyée s'allume
  (la demande d'hier ne ressemble plus à celle de l'automne).
- La **PCA** ne voit presque rien (une seule journée, le 23) : les lags chutent ensemble,
  le long de la première composante.
- La **perte estimée** monte lentement et ne franchit son seuil que le 19, puis durablement
  à partir du 22 : il faut que les lags aient absorbé le choc pour que DLE le voie.""",
    ),
    (
        "code",
        """cols = ["P1b_drift_clean", "P2_dle", "P3b_clean_and_dle"]
flags.loc["2020-03-09":"2020-03-25", cols]""",
    ),
    (
        "md",
        """La condition jointe nettoyée (P3b) s'allume le **23 mars** (jour cible) : la décision
est possible la veille, le **22 mars**, neuf jours après l'état d'urgence. La dérive
nettoyée seule (P1b) aurait alerté dès le 14 mars, mais elle alerte aussi sur des dérives
bénignes (E4) : elle ouvre un ticket, pas une page.""",
    ),
    (
        "code",
        """from zonefleet.cli import daily_report

print(daily_report("covid", "2020-03-23"))""",
    ),
    ("md", """## Deux mois plus tard : ce que disent les labels"""),
    (
        "code",
        """#| fig-alt: "Courbes journalières du 9 mars au 15 avril 2020, avec un trait au 13 mars : l'erreur réalisée monte de 0,2 à 1,75 le 22 mars puis redescend vers 0,75 ; l'erreur estimée ne dépasse pas 0,53."
late = sig.loc["2020-03-09":"2020-04-15", ["estimated", "realized"]]
ax = late.rename(columns={"estimated": "estimée", "realized": "réalisée"}).plot(
    figsize=(9, 3.2), color=["#2a78d6", "#eb6834"]
)
ax.axvline(pd.Timestamp("2020-03-13"), color="#52514e", lw=1)
ax.set_ylabel("MAE log")
ax.set_title("Estimée le jour même, réalisée connue deux mois plus tard")
late.loc["2020-03-22":"2020-03-28"].round(3)""",
    ),
    (
        "md",
        """L'erreur réelle de ces journées est de 1,4 à 1,7, soit près de trois à quatre fois
l'estimation :
la page était juste sur le **fait** (le modèle se dégrade) et fausse sur l'**ampleur**.
C'est exactement la frontière mesurée dans E3 : pendant la transition, le choc ressemble à
un concept drift, que la perte estimée ne peut voir qu'en partie.""",
    ),
]


def write(cells, name: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"]["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    out = ROOT / "site" / "notebooks" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(out)


NOTEBOOKS = {
    "tutoriel-dle.ipynb": CELLS,
    "couts-alerte.ipynb": COST_CELLS,
    "anatomie-alerte.ipynb": ALERT_CELLS,
}


def main(names: list[str]) -> None:
    """Écrit les notebooks demandés (tous par défaut), sans sorties : les exécuter ensuite."""
    for name in names or list(NOTEBOOKS):
        write(NOTEBOOKS[name], name)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
