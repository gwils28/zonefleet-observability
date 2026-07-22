"""Tests du hook typographique."""

from scripts.hooks.check_no_em_dash import DASH, main


def test_detects_em_dash(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text(f"ligne correcte\nligne {DASH} fautive\n", encoding="utf-8")
    assert main([str(f)]) == 1
    assert "a.md:2" in capsys.readouterr().out


def test_accepts_clean_files_and_binaries(tmp_path):
    ok = tmp_path / "b.py"
    ok.write_text("x = 1  # tiret simple - correct\n", encoding="utf-8")
    binf = tmp_path / "c.bin"
    binf.write_bytes(b"\xff\xfe\x00")
    assert main([str(ok), str(binf)]) == 0
