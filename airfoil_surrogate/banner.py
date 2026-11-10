"""시작 배너: 큰 글씨 로고. 실행할 때마다 색 조합이 바뀝니다.

외부 라이브러리 없이 글꼴을 직접 들고 있습니다. 터미널(TTY)일 때만 보이고,
파이프·리다이렉트·테스트에서는 출력하지 않습니다.

    AFS_NO_BANNER=1   배너 끄기 (또는 --no-banner)
    NO_COLOR=1              색 끄기 (https://no-color.org)
"""

from __future__ import annotations

import os
import random
import shutil
import sys

from . import __version__

# ANSI Shadow 스타일 글꼴 (글자마다 6줄)
_FONT = {
    "A": [" █████╗ ",
          "██╔══██╗",
          "███████║",
          "██╔══██║",
          "██║  ██║",
          "╚═╝  ╚═╝"],
    "E": ["███████╗",
          "██╔════╝",
          "█████╗  ",
          "██╔══╝  ",
          "███████╗",
          "╚══════╝"],
    "R": ["██████╗ ",
          "██╔══██╗",
          "██████╔╝",
          "██╔══██╗",
          "██║  ██║",
          "╚═╝  ╚═╝"],
    "O": [" ██████╗ ",
          "██╔═══██╗",
          "██║   ██║",
          "██║   ██║",
          "╚██████╔╝",
          " ╚═════╝ "],
    "U": ["██╗   ██╗",
          "██║   ██║",
          "██║   ██║",
          "██║   ██║",
          "╚██████╔╝",
          " ╚═════╝ "],
    "T": ["████████╗",
          "╚══██╔══╝",
          "   ██║   ",
          "   ██║   ",
          "   ██║   ",
          "   ╚═╝   "],
    "-": ["      ",
          "      ",
          "█████╗",
          "╚════╝",
          "      ",
          "      "],
    "I": ["██╗",
          "██║",
          "██║",
          "██║",
          "██║",
          "╚═╝"],
    "F": ["███████╗",
          "██╔════╝",
          "█████╗  ",
          "██╔══╝  ",
          "██║     ",
          "╚═╝     "],
    "L": ["██╗     ",
          "██║     ",
          "██║     ",
          "██║     ",
          "███████╗",
          "╚══════╝"],
    "S": ["███████╗",
          "██╔════╝",
          "███████╗",
          "╚════██║",
          "███████║",
          "╚══════╝"],
    " ": ["  "] * 6,
}

_TAGLINE = "단일 익형 CFD 데이터셋 · CL/CD 회귀 모델"

# 큰 글씨 6줄의 위→아래 색 (256색 번호). 실행할 때마다 하나를 무작위로 고릅니다.
PALETTES = {
    "ocean":  [51, 45, 39, 33, 27, 21],      # 하늘색 → 파랑
    "sunset": [226, 220, 214, 208, 202, 196],  # 노랑 → 빨강
    "forest": [191, 155, 119, 83, 47, 35],   # 연두 → 초록
    "neon":   [219, 213, 207, 201, 165, 129],  # 분홍 → 보라
    "aurora": [87, 86, 85, 84, 120, 156],    # 청록 → 연두
    "flame":  [229, 223, 217, 211, 205, 199],  # 살구 → 진분홍
    "steel":  [255, 252, 249, 246, 243, 240],  # 흰색 → 회색
}
_DIM = "38;5;244"


def render_text(text: str) -> list[str]:
    """글꼴에 있는 글자로 큰 글씨 6줄을 만듭니다."""
    rows = [""] * 6
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for i in range(6):
            rows[i] += glyph[i]
    return [r.rstrip() for r in rows]


def _supports_unicode(stream) -> bool:
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "█╗═✈·다".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False
    if os.name == "nt":
        os.system("")  # Windows 콘솔에서 ANSI 이스케이프 처리 켜기
    return True


def build_banner(*, color: bool = True, unicode: bool = True, width: int = 80,
                 palette: str | None = None) -> str:
    """배너 문자열. palette를 안 주면 PALETTES에서 무작위로 고릅니다.
    폭이 좁거나 유니코드를 못 쓰면 한 줄짜리로 줄입니다."""
    def paint(code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if color else s

    version = f"v{__version__}"
    if not unicode:
        return f"== AIRFOIL-SURROGATE {version} :: airfoil CFD dataset / CL-CD surrogate models ==\n"

    gradient = PALETTES[palette] if palette else random.choice(list(PALETTES.values()))
    accent = f"1;38;5;{gradient[1]}"
    title = render_text("AIRFOIL")
    if width < max(len(r) for r in title) + 2:
        return paint(accent, "✈ AIRFOIL-SURROGATE") + f" {version}  {paint(_DIM, _TAGLINE)}\n"

    lines = [""]
    for row, c in zip(title, gradient):
        lines.append(" " + paint(f"1;38;5;{c}", row))
    lines.append("")
    lines.append("  " + paint(accent, _TAGLINE) + "  " + paint(_DIM, version))
    lines.append("")
    return "\n".join(lines) + "\n"


def print_banner(stream=None, *, force: bool = False) -> bool:
    """TTY일 때만 배너를 출력합니다. 출력했으면 True."""
    stream = stream or sys.stdout
    if not force:
        if os.environ.get("AFS_NO_BANNER"):
            return False
        if not (hasattr(stream, "isatty") and stream.isatty()):
            return False
    width = shutil.get_terminal_size((80, 24)).columns
    stream.write(build_banner(color=_use_color(stream), unicode=_supports_unicode(stream), width=width))
    stream.flush()
    return True
