import io

from airfoil_surrogate.banner import PALETTES, build_banner, print_banner, render_text
from airfoil_surrogate.cli import main


def test_render_text_has_six_rows():
    rows = render_text("AERO-AUTO")
    assert len(rows) == 6
    assert "█" in rows[0]


def test_banner_variants():
    full = build_banner(color=False, width=120)
    assert "█" in full and "\033[" not in full
    assert "moving ground" not in full                                   # 날개 그림 없음
    assert "\033[" in build_banner(color=True, width=120)
    assert len(build_banner(color=False, width=40).splitlines()) == 1   # 좁은 터미널
    assert build_banner(unicode=False).isascii()                         # cp949 등


def test_banner_skipped_when_not_tty(monkeypatch):
    monkeypatch.delenv("AFS_NO_BANNER", raising=False)
    buf = io.StringIO()
    assert print_banner(buf) is False and buf.getvalue() == ""
    assert print_banner(buf, force=True) is True and buf.getvalue()


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: afs" in capsys.readouterr().out


def test_palette_colors_differ():
    sunset = build_banner(palette="sunset", width=120)
    ocean = build_banner(palette="ocean", width=120)
    assert sunset != ocean
    assert all(f"38;5;{c}m" in sunset for c in PALETTES["sunset"])


def test_help_colors_only_on_terminal(capsys, monkeypatch):
    import re

    from airfoil_surrogate.console import colorize_help

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert main(["--no-banner"]) == 0
    plain = capsys.readouterr().out
    assert "\033[" not in plain  # 파이프·테스트에서는 색 없음

    colored = colorize_help(plain, {"run", "check"})
    assert re.sub(r"\033\[[0-9;]*m", "", colored) == plain  # 색만 넣고 글자·정렬은 그대로
    assert "\033[1;93musage:" in colored and "\033[1;96moptions:" in colored
    assert "\033[1;92mrun\033[0m" in colored and "\033[1;93m--no-clear" in colored
    assert "\033[1;92mafs\033[0m" in colored


def test_clear_screen_only_on_terminal(monkeypatch):
    from airfoil_surrogate.console import clear_screen

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("AFS_NO_CLEAR", raising=False)
    assert clear_screen(io.StringIO()) is False
    if __import__("os").name != "nt":
        tty = Tty()
        assert clear_screen(tty) is True and "\033[2J" in tty.getvalue()
    monkeypatch.setenv("AFS_NO_CLEAR", "1")
    assert clear_screen(Tty()) is False
