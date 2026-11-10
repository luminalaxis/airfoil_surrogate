"""터미널 표시: 시작할 때 화면 지우기, 색을 넣은 도움말.

둘 다 터미널(TTY)일 때만 동작하고, 파이프·리다이렉트·테스트에서는 그대로 둔다.

    AFS_NO_CLEAR=1   화면 지우기 끄기 (또는 --no-clear)
    NO_COLOR=1             색 끄기 (https://no-color.org)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from .banner import _use_color

# 어두운·밝은 배경 모두에서 잘 보이는 굵은 밝은 색
USAGE = "1;93"      # 밝은 노랑
HEADING = "1;96"    # 밝은 청록
COMMAND = "1;92"    # 밝은 초록
OPTION = "1;93"     # 밝은 노랑
METAVAR = "96"      # 청록
TEXT = ""           # 설명 글은 터미널 기본 글자색 (밝은 배경에서도 보이도록)
DESCRIPTION = "1"   # 굵게


def _paint(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if s and code else s


def clear_screen(stream=None) -> bool:
    """TTY면 화면(과 스크롤 기록)을 지운다. 지웠으면 True."""
    stream = stream or sys.stdout
    if os.environ.get("AFS_NO_CLEAR"):
        return False
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False
    if os.name == "nt":
        # 고정된 인자의 내장 명령만 실행 (사용자 입력 없음). 옛 콘솔은 ANSI 지우기를 모를 수 있어 cls 사용
        subprocess.run(["cmd", "/c", "cls"], check=False)
    else:
        stream.write("\033[H\033[2J\033[3J")
        stream.flush()
    return True


_OPTION_LINE = re.compile(r"^(  )(-\S.*?)(\s{2,}|$)(.*)$")
_COMMAND_LINE = re.compile(r"^(    )(\S+)(\s{2,}|$)(.*)$")
_FLAG = re.compile(r"(?<![\w-])(--?[\w-]+)")
_UPPER = re.compile(r"\b([A-Z][A-Z0-9_/]+)\b")


def _paint_invocation(text: str) -> str:
    """'-c CONFIG, --config CONFIG' → 플래그와 메타변수를 따로 칠한다."""
    out = []
    for part in re.split(r"(, )", text):
        if part == ", ":
            out.append(part)
            continue
        m = re.match(r"^(-\S+)(.*)$", part)
        out.append(_paint(OPTION, m.group(1)) + _paint(METAVAR, m.group(2)) if m else _paint(METAVAR, part))
    return "".join(out)


def colorize_help(text: str, commands: set[str] | None = None) -> str:
    """argparse가 정렬까지 끝낸 도움말 텍스트에 색을 넣는다 (폭 계산에 영향 없음)."""
    commands = commands or set()
    lines, section, in_usage = [], "", False
    for i, line in enumerate(text.splitlines()):
        if line.startswith("usage: "):
            in_usage = True
            body = line[len("usage: "):]
            lines.append(_paint(USAGE, "usage:") + " " + _paint_usage(body))
            continue
        if in_usage and line.startswith(" ") and line.strip():
            lines.append(_paint_usage(line))
            continue
        in_usage = False
        if line and not line.startswith(" ") and line.endswith(":"):
            section = line
            lines.append(_paint(HEADING, line))
            continue
        if line and not line.startswith(" "):
            lines.append(_paint(DESCRIPTION, line))  # 설명 문단
            continue
        m = _OPTION_LINE.match(line)
        if m:
            lines.append(m.group(1) + _paint_invocation(m.group(2)) + m.group(3) + _paint(TEXT, m.group(4)))
            continue
        m = _COMMAND_LINE.match(line)
        if m and m.group(2) in commands:
            lines.append(m.group(1) + _paint(COMMAND, m.group(2)) + m.group(3) + _paint(TEXT, m.group(4)))
            continue
        if line.strip().startswith("{") and line.strip().endswith("}"):
            lines.append(_paint_choices(line))
            continue
        lines.append(_paint(TEXT, line) if line.strip() and section else line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _paint_choices(line: str) -> str:
    def repl(m):
        names = m.group(1).split(",")
        return "{" + ",".join(_paint(COMMAND, n) for n in names) + "}"
    return re.sub(r"\{([^{}]*)\}", repl, line)


def _paint_usage(text: str) -> str:
    text = _paint_choices(text) if "{" in text else text
    text = _FLAG.sub(lambda m: _paint(OPTION, m.group(1)), text)
    return re.sub(r"(?:^|(?<=\s))(afs(?: [a-z][\w-]*)?)(?=\s|$)", lambda m: _paint(COMMAND, m.group(1)), text, count=1)


class ColorArgumentParser(argparse.ArgumentParser):
    """터미널로 내보낼 때만 도움말·사용법에 색을 넣는 ArgumentParser (하위 명령도 같은 클래스)."""

    def _commands(self) -> set[str]:
        names: set[str] = set()
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                names |= set(action.choices)
        return names

    def _emit(self, text: str, file) -> None:
        file = file or sys.stdout
        if _use_color(file):
            text = colorize_help(text, self._commands())
        self._print_message(text, file)

    def print_help(self, file=None) -> None:
        self._emit(self.format_help(), file)

    def print_usage(self, file=None) -> None:
        self._emit(self.format_usage(), file)
