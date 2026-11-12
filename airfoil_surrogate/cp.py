"""표면 압력계수 Cp: Fluent ASCII export 파일을 읽어 고정된 x/c 지점에 보간한다.

저널의 ${cp_export} (기본: /file/export/ascii "cp.txt" main () yes pressure-coefficient () no) 가 쓰는
파일 형식: 머리줄(nodenumber, x-coordinate, y-coordinate, pressure-coefficient) + 노드별 값 (쉼표 또는 공백 구분).

메쉬는 회전하지 않으므로(받음각은 유입 방향) 좌표는 그대로 물체 좌표 = x/c × chord.
윗면/아랫면은 형상 좌표(Selig 순서)에서 가장 가까운 점이 앞전 앞쪽인지 뒤쪽인지로 나눈다.
결과 열 이름: cp_u_0.050 (윗면 x/c = 0.05), cp_l_0.050 (아랫면).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


def read_fluent_ascii(path: Path) -> dict[str, np.ndarray]:
    header: list[str] | None = None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if not parts:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            if header is None:
                header = [p.lower() for p in parts]
    if header is None or not rows:
        return {}
    width = min(len(header), min(len(r) for r in rows))
    data = np.array([r[:width] for r in rows if len(r) >= width])
    return {name: data[:, i] for i, name in enumerate(header[:width])}


def column_names(stations: list[float]) -> list[str]:
    return [f"cp_{side}_{x:.3f}" for side in ("u", "l") for x in stations]


def interpolate_cp(path: Path, stations: list[float], section_xy: np.ndarray, le_index: int,
                   chord: float) -> dict[str, float]:
    """section_xy: 코드 1 기준 형상 좌표 (Selig 순서)."""
    data = read_fluent_ascii(path)
    xk = next((k for k in data if k.startswith("x-coord")), None)
    yk = next((k for k in data if k.startswith("y-coord")), None)
    ck = next((k for k in data if "pressure-coef" in k), None)
    out = {name: math.nan for name in column_names(stations)}
    if not (xk and yk and ck):
        return out
    pts = np.column_stack([data[xk], data[yk]]) / chord
    cp = data[ck]
    d = np.linalg.norm(pts[:, None, :] - section_xy[None, :, :], axis=2)
    upper = d.argmin(1) <= le_index
    for side, mask in (("u", upper), ("l", ~upper)):
        if mask.sum() < 2:
            continue
        x, c = pts[mask, 0], cp[mask]
        order = np.argsort(x)
        x, c = x[order], c[order]
        for s in stations:
            if x[0] - 1e-9 <= s <= x[-1] + 1e-9:
                out[f"cp_{side}_{s:.3f}"] = float(np.interp(s, x, c))
    return out
