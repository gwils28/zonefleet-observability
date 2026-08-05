"""Libellés et couleurs de rôle partagés par les tables, figures et graphiques du site.

Une politique ou une famille de dérive porte le même nom et la même couleur partout.
"""

from __future__ import annotations

from zonefleet import viz

POLICY_LONG = {
    "P1_drift": "P1 dérive seule",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE seul",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b dérive nettoyée ET DLE",
    "P4_realized": "P4 labels retardés",
}

POLICY_SHORT = {
    "P1_drift": "P1 dérive",
    "P1b_drift_clean": "P1b dérive nettoyée",
    "P2_dle": "P2 DLE",
    "P3_drift_and_dle": "P3 dérive ET DLE",
    "P3b_clean_and_dle": "P3b dérive nettoyée ET DLE",
    "P4_realized": "P4 labels (D + 60 j)",
}

POLICY_COLOR = {
    "P1b_drift_clean": viz.AQUA,
    "P2_dle": viz.BLUE,
    "P3b_clean_and_dle": viz.VIOLET,
    "P4_realized": viz.ORANGE,
}

FAMILY_COLOR = {
    "covariate nuisible": viz.BLUE,
    "covariate bénin": viz.AQUA,
    "concept pur": viz.MAGENTA,
    "choc propagé": viz.YELLOW,
}

RETRAIN_COLOR = {
    "jamais": viz.NEUTRAL,
    "calendrier": viz.BLUE,
    "réflexe": viz.YELLOW,
    "réflexe court": viz.MAGENTA,
    "gardé": viz.VIOLET,
}

SPLIT_LONG = {"covid": "E1 (2019-2020)", "congestion": "E2 (2024-2025)"}
