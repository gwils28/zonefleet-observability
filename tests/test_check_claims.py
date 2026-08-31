"""Vérificateur des chiffres cités : sur un mini-dépôt."""

import pandas as pd
import pytest

import scripts.check_claims as C


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"policy": ["a", "b"], "recall": [0.506, 0.552]}).to_csv(
        tmp_path / "tables" / "res.csv", index=False
    )
    (tmp_path / "page.md").write_text(
        "Le rappel atteint\n51 % sans fausse alerte.", encoding="utf-8"
    )
    monkeypatch.setattr(C, "TABLES", tmp_path / "tables")
    C.t.cache_clear()
    yield tmp_path
    C.t.cache_clear()


def claim(text, expr="pct(t('res').set_index('policy').loc['a', 'recall'])"):
    return {"id": "x", "text": text, "files": ["page.md"], "expr": expr}


def test_matching_claim_passes_across_line_breaks(repo):
    assert C.check([claim("51 %")], root=repo) == []


def test_recomputed_value_differs(repo):
    errors = C.check([claim("55 %")], root=repo)
    assert any("recalculé « 51 % »" in e for e in errors)
    assert any("absent de page.md" in e for e in errors)


def test_french_format_helpers():
    assert C.fr(-0.0004, 3) == "0,000" and C.fr(1234.56, 1) == "1 234,6"
    assert C.pct(0.5517) == "55 %"
