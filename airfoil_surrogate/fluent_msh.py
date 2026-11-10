"""2D Fluent 메쉬 파일(.msh, ASCII) 쓰기/읽기.

면(face) 방향 규칙 (Fluent 2D): 면 n0→n1 방향으로 진행할 때 c0는 왼쪽, c1은 오른쪽 셀.
경계면은 c1 = 0 이고 c0가 유체 셀이다. 인덱스는 모두 16진수, 1부터 시작.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Fluent 경계 조건 타입 코드
BC_TYPES = {
    "interior": 2,
    "wall": 3,
    "pressure-inlet": 4,
    "pressure-outlet": 5,
    "symmetry": 7,
    "pressure-far-field": 9,
    "velocity-inlet": 10,
}
_TRI, _QUAD = 1, 3  # Fluent 셀 요소 타입

NODE_ZONE, FLUID_ZONE, INTERIOR_ZONE = 1, 2, 3


@dataclass
class Boundary:
    name: str
    bc_type: str
    edges: list[tuple[int, int]]  # 0-based 노드 인덱스 쌍


def _ccw(nodes: np.ndarray, cell: tuple[int, ...]) -> tuple[int, ...]:
    pts = nodes[list(cell)]
    x, y = pts[:, 0], pts[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    return cell if area2 > 0 else tuple(reversed(cell))


def build_faces(nodes: np.ndarray, cells: list[tuple[int, ...]]):
    """셀 목록에서 면(edge)을 만든다. 반환: {정렬된 노드쌍: [n0, n1, c0, c1]} (셀 번호 1부터)."""
    faces: dict[tuple[int, int], list[int]] = {}
    for ci, cell in enumerate(cells, start=1):
        cell = _ccw(nodes, cell)
        for k in range(len(cell)):
            a, b = cell[k], cell[(k + 1) % len(cell)]
            key = (a, b) if a < b else (b, a)
            face = faces.get(key)
            if face is None:
                # 반시계 셀의 변 a→b 기준으로 셀은 왼쪽 → c0
                faces[key] = [a, b, ci, 0]
            elif face[3] == 0 and face[0] == b and face[1] == a:
                face[3] = ci
            else:
                raise ValueError(f"비정상 메쉬: 변 {key}를 공유하는 셀이 2개를 초과하거나 방향이 꼬였습니다.")
    return faces


def write_msh(path: Path, nodes: np.ndarray, cells: list[tuple[int, ...]],
              boundaries: list[Boundary], fluid_name: str = "fluid",
              title: str = "airfoil_surrogate 2D mesh") -> dict:
    nodes = np.asarray(nodes, float)[:, :2]
    faces = build_faces(nodes, cells)

    interior = [f for f in faces.values() if f[3] != 0]
    zone_faces: list[tuple[int, str, str, list]] = [(INTERIOR_ZONE, "interior", f"interior-{fluid_name}", interior)]
    assigned: set[tuple[int, int]] = set()
    for zid, b in enumerate(boundaries, start=INTERIOR_ZONE + 1):
        flist = []
        for a, c in b.edges:
            key = (a, c) if a < c else (c, a)
            face = faces.get(key)
            if face is None or face[3] != 0:
                raise ValueError(f"경계 '{b.name}'의 변 {key}가 메쉬 경계면이 아닙니다.")
            if key in assigned:
                raise ValueError(f"변 {key}가 여러 경계에 중복 지정되었습니다.")
            assigned.add(key)
            flist.append(face)
        zone_faces.append((zid, b.bc_type, b.name, flist))
    unassigned = sum(1 for k, f in faces.items() if f[3] == 0 and k not in assigned)
    if unassigned:
        raise ValueError(f"경계 지정이 안 된 경계면이 {unassigned}개 있습니다.")

    n_nodes, n_cells, n_faces = len(nodes), len(cells), len(faces)
    types = [_TRI if len(c) == 3 else _QUAD for c in cells]
    out = [f'(0 "{title}")', "(2 2)",
           f"(10 (0 1 {n_nodes:x} 0 2))",
           f"(12 (0 1 {n_cells:x} 0))",
           f"(13 (0 1 {n_faces:x} 0))",
           f"(10 ({NODE_ZONE:x} 1 {n_nodes:x} 1 2)(",
           *(f"{x:.12e} {y:.12e}" for x, y in nodes),
           "))"]
    if len(set(types)) == 1:
        out.append(f"(12 ({FLUID_ZONE:x} 1 {n_cells:x} 1 {types[0]:x}))")
    else:
        out.append(f"(12 ({FLUID_ZONE:x} 1 {n_cells:x} 1 0)(")
        out += [" ".join(str(t) for t in types[i:i + 40]) for i in range(0, n_cells, 40)]
        out.append("))")

    first = 1
    for zid, bc, _, flist in zone_faces:
        if not flist:
            continue
        last = first + len(flist) - 1
        out.append(f"(13 ({zid:x} {first:x} {last:x} {BC_TYPES[bc]:x} 2)(")
        out += [f"{n0 + 1:x} {n1 + 1:x} {c0:x} {c1:x}" for n0, n1, c0, c1 in flist]
        out.append("))")
        first = last + 1

    out.append(f"(45 ({FLUID_ZONE} fluid {fluid_name})())")
    for zid, bc, name, flist in zone_faces:
        if flist:
            out.append(f"(45 ({zid} {bc} {name})())")

    path.write_text("\n".join(out) + "\n", encoding="ascii")
    return {"nodes": n_nodes, "cells": n_cells, "faces": n_faces,
            "triangles": types.count(_TRI), "quads": types.count(_QUAD),
            "zones": {name: len(fl) for _, _, name, fl in zone_faces}}


# ---------------------------------------------------------------- 검증용 읽기

def read_msh(path: Path) -> dict:
    """write_msh가 쓴 형식을 다시 읽는다 (검증/테스트용)."""
    lines = path.read_text(encoding="ascii").splitlines()
    nodes, faces, zones, names = [], [], {}, {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("(10 (") and line.endswith("(") and not line.startswith("(10 (0 "):
            i += 1
            while lines[i] != "))":
                nodes.append([float(v) for v in lines[i].split()])
                i += 1
        elif line.startswith("(13 (") and line.endswith("("):
            zid = int(line[5:].split()[0], 16)
            zones[zid] = []
            i += 1
            while lines[i] != "))":
                n0, n1, c0, c1 = (int(v, 16) for v in lines[i].split())
                faces.append((n0, n1, c0, c1, zid))
                zones[zid].append(len(faces) - 1)
                i += 1
        elif line.startswith("(45 ("):
            zid, bc, name = line[5:].split(")")[0].split()
            names[int(zid)] = (bc, name)
        i += 1
    return {"nodes": np.array(nodes), "faces": faces, "zones": zones, "names": names}


def check_msh(path: Path) -> dict:
    """Fluent 규칙(c0 왼쪽)으로 셀 면적을 다시 계산해 모든 셀이 닫히고 면적이 양수인지 확인."""
    data = read_msh(path)
    nodes, faces = data["nodes"], data["faces"]
    n_cells = max(max(f[2], f[3]) for f in faces)
    area = np.zeros(n_cells + 1)
    count = np.zeros(n_cells + 1, int)
    for n0, n1, c0, c1, _ in faces:
        (x0, y0), (x1, y1) = nodes[n0 - 1], nodes[n1 - 1]
        cross = 0.5 * (x0 * y1 - x1 * y0)  # 원점 기준 삼각형 부호 면적
        area[c0] += cross
        count[c0] += 1
        if c1:
            area[c1] -= cross
            count[c1] += 1
        elif c0 == 0:
            raise ValueError("c0가 0인 면이 있습니다.")
    area, count = area[1:], count[1:]
    oq = orthogonal_quality(nodes, faces, n_cells) if not (area <= 0).any() else np.zeros(n_cells)
    return {
        "cells": n_cells,
        "min_area": float(area.min()),
        "total_area": float(area.sum()),
        "bad_cells": int((area <= 0).sum()),
        "open_cells": int((count < 3).sum()),
        "min_orthogonal_quality": float(oq.min()),
        "cells_oq_below_0.1": int((oq < 0.1).sum()),
        "names": data["names"],
    }


def orthogonal_quality(nodes: np.ndarray, faces: list, n_cells: int) -> np.ndarray:
    """셀별 직교 품질 (Fluent Mesh > Quality의 Orthogonal Quality와 같은 정의, 0~1, 1이 최고).

    셀의 면마다 바깥 방향 면 벡터 A와 (셀 중심 → 면 중심) f, (셀 중심 → 이웃 셀 중심) c 사이의
    cos 값 A·f/(|A||f|), A·c/(|A||c|) 중 가장 작은 값. Fluent는 0.01 미만이면 오류, 0.1 미만은 주의를 권한다.
    """
    f = np.asarray([fc[:4] for fc in faces], dtype=np.int64)
    n0, n1, c0, c1 = f[:, 0] - 1, f[:, 1] - 1, f[:, 2], f[:, 3]
    p0, p1 = nodes[n0], nodes[n1]
    cross = p0[:, 0] * p1[:, 1] - p1[:, 0] * p0[:, 1]
    # 셀 면적·도심 (c0는 면 n0→n1의 왼쪽, c1은 오른쪽)
    area = np.zeros(n_cells + 1)
    cx = np.zeros(n_cells + 1)
    cy = np.zeros(n_cells + 1)
    for cell, sign in ((c0, 1.0), (c1, -1.0)):
        np.add.at(area, cell, sign * 0.5 * cross)
        np.add.at(cx, cell, sign * (p0[:, 0] + p1[:, 0]) * cross / 6)
        np.add.at(cy, cell, sign * (p0[:, 1] + p1[:, 1]) * cross / 6)
    area[0] = 1.0
    cen = np.column_stack([cx / area, cy / area])
    d = p1 - p0
    a0 = np.column_stack([d[:, 1], -d[:, 0]])  # c0 바깥 방향 (오른쪽)
    fmid = 0.5 * (p0 + p1)

    def cos(a, b):
        return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)

    oq = np.ones(n_cells + 1)
    np.minimum.at(oq, c0, cos(a0, fmid - cen[c0]))
    inner = c1 > 0
    np.minimum.at(oq, c1[inner], cos(-a0[inner], fmid[inner] - cen[c1[inner]]))
    cc = cos(a0[inner], cen[c1[inner]] - cen[c0[inner]])
    np.minimum.at(oq, c0[inner], cc)
    np.minimum.at(oq, c1[inner], cc)
    return np.clip(oq[1:], 0.0, 1.0)
