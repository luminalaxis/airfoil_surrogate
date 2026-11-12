"""저널/입력 파일 템플릿 치환.

``${변수}`` 또는 ``${변수:포맷}`` (예: ``${aoa:.2f}``) 형식만 치환한다.
Fluent Scheme/TUI 문법과 충돌하지 않도록 중괄호가 있는 형태만 인식한다.
"""

from __future__ import annotations

import re

from .config import ConfigError

_PLACEHOLDER = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def render(text: str, variables: dict, source: str = "템플릿") -> str:
    missing: set[str] = set()

    def replace(match: re.Match) -> str:
        key, spec = match.group(1), match.group(2)
        if key not in variables:
            missing.add(key)
            return match.group(0)
        value = variables[key]
        if not spec:
            return str(value)
        try:
            return format(value, spec)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"{source}의 ${{{key}:{spec}}}: 값 {value!r}에 쓸 수 없는 형식입니다 ({exc})") from None

    result = _PLACEHOLDER.sub(replace, text)
    if missing:
        raise ConfigError(f"{source}에서 정의되지 않은 변수: {', '.join(sorted(missing))}")
    return result
