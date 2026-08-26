"""Vérificateur de liens du site rendu, sur un mini-site."""

from scripts.check_links import check


def _site(tmp_path, body: str):
    (tmp_path / "a").mkdir()
    (tmp_path / "index.html").write_text(
        '<h2 id="résultats">R</h2><img src="fig.png">' + body, encoding="utf-8"
    )
    (tmp_path / "fig.png").write_bytes(b"x")
    (tmp_path / "a" / "page.html").write_text('<p id="ici">p</p>', encoding="utf-8")
    return tmp_path


def test_valid_links_pass(tmp_path):
    body = (
        '<a href="a/page.html#ici">1</a><a href="#résultats">2</a>'
        '<a href="/zonefleet-observability/a/page.html">3</a><a href="https://x.org">4</a>'
    )
    assert check(_site(tmp_path, body)) == []


def test_broken_anchor_file_and_image_are_reported(tmp_path):
    body = '<a href="a/page.html#ailleurs">1</a><a href="b.html">2</a><img src="manque.png">'
    errors = check(_site(tmp_path, body))
    assert len(errors) == 3
    assert any("ancre absente" in e for e in errors)
    assert any("cible introuvable" in e for e in errors)
    assert any("image introuvable" in e for e in errors)
