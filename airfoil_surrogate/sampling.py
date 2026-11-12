"""케이스 목록 생성: 라틴 하이퍼큐브 샘플링 (m, p, t, α) → cases.csv, 검증용 목록 → cases_validation.csv.

    [sampling]
    seed = 2026
    mode = "nested"          # nested: 형상 (m, p, t)을 LHS로 n_shapes개 뽑고, 형상마다 α를 alpha_per_shape개
                             #         층화 추출 (형상당 메쉬 1개 공유, 형상 단위 분할 가능)
                             # joint : (m, p, t, α) 4차원 LHS로 n_cases개 (케이스마다 형상이 다름 → 메쉬 n_cases개)
    n_shapes = 120
    alpha_per_shape = 6
    m = [0.0, 6.0]           # 최대 캠버 [% c]
    p = [2.0, 6.0]           # 최대 캠버 위치 [c/10]
    t = [6.0, 18.0]          # 최대 두께 [% c]
    alpha = [0.0, 12.0]      # 받음각 [deg]. 15°(실속 후)를 넣을지는 팀 결정
    assignees = ["A", "B", "C", "D"]
    decimals = 4

    [sampling.validation]
    shapes = ["0012", "2412", "4412", "0015", "2415", "4415"]
    alpha = [0, 4, 8]

출력 열: case_id, shape_id, m, p, t, alpha, assignee (shape_id는 메쉬 공유·형상 단위 분할에 씀).
담당자는 형상 단위로 번갈아 배정한다 (같은 형상의 α는 한 사람이 → 메쉬를 한 번만 만든다).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .config import ConfigError, ProjectConfig

CASE_COLUMNS = ("case_id", "shape_id", "m", "p", "t", "alpha", "assignee")


@dataclass
class ValidationSet:
    shapes: list = field(default_factory=lambda: ["0012", "2412", "4412", "0015", "2415", "4415"])
    alpha: list = field(default_factory=lambda: [0.0, 4.0, 8.0])


@dataclass
class SamplingConfig:
    seed: int = 2026
    mode: str = "nested"
    n_shapes: int = 120
    alpha_per_shape: int = 6
    n_cases: int = 720
    m: list = field(default_factory=lambda: [0.0, 6.0])
    p: list = field(default_factory=lambda: [2.0, 6.0])
    t: list = field(default_factory=lambda: [6.0, 18.0])
    alpha: list = field(default_factory=lambda: [0.0, 12.0])
    assignees: list = field(default_factory=lambda: ["member1", "member2", "member3", "member4"])
    decimals: int = 4
    validation: ValidationSet = field(default_factory=ValidationSet)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SamplingConfig":
        data = dict(data or {})
        val = data.pop("validation", {})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"[sampling]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        unknown = set(val) - {f.name for f in fields(ValidationSet)}
        if unknown:
            raise ConfigError(f"[sampling.validation]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        obj = cls(**data, validation=ValidationSet(**val))
        if obj.mode not in ("nested", "joint"):
            raise ConfigError(f"[sampling] mode는 nested 또는 joint여야 합니다: {obj.mode!r}")
        for name in ("m", "p", "t", "alpha"):
            lo, hi = getattr(obj, name)
            if not lo <= hi:
                raise ConfigError(f"[sampling] {name} 범위는 [최소, 최대]여야 합니다: {getattr(obj, name)}")
        if obj.m[0] < 0 or obj.t[0] <= 0 or (obj.m[1] > 0 and obj.p[0] <= 0):
            raise ConfigError("[sampling] m ≥ 0, t > 0, 캠버가 있으면 p > 0 이어야 합니다 (p = 0은 대칭익 전용).")
        if not obj.assignees:
            raise ConfigError("[sampling] assignees(담당자)를 한 명 이상 지정하세요.")
        return obj


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """[0, 1)^d 라틴 하이퍼큐브 n점: 각 차원을 n등분한 칸마다 정확히 한 점 (칸 안 위치는 균일 난수)."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def _scale(u: np.ndarray, rng_: list) -> np.ndarray:
    lo, hi = rng_
    return lo + u * (hi - lo)


def generate_cases(sc: SamplingConfig) -> list[dict]:
    rng = np.random.default_rng(sc.seed)
    r = sc.decimals
    rows = []
    if sc.mode == "nested":
        u = latin_hypercube(sc.n_shapes, 3, rng)
        shapes = np.column_stack([_scale(u[:, 0], sc.m), _scale(u[:, 1], sc.p), _scale(u[:, 2], sc.t)])
        k = sc.alpha_per_shape
        for i, (m, p, t) in enumerate(shapes):
            # 형상마다 α 범위를 k칸으로 나눠 칸마다 하나 (1차원 LHS) → 모든 형상이 받음각 범위 전체를 덮음
            a = _scale((np.arange(k) + rng.random(k)) / k, sc.alpha)
            for alpha in a:
                rows.append({"shape_id": f"s{i + 1:04d}", "m": m, "p": p, "t": t, "alpha": alpha})
    else:
        u = latin_hypercube(sc.n_cases, 4, rng)
        for i in range(sc.n_cases):
            rows.append({"shape_id": f"s{i + 1:04d}", "m": _scale(u[i, 0], sc.m), "p": _scale(u[i, 1], sc.p),
                         "t": _scale(u[i, 2], sc.t), "alpha": _scale(u[i, 3], sc.alpha)})
    for row in rows:
        for key in ("m", "p", "t", "alpha"):
            row[key] = round(float(row[key]), r)
    _assign(rows, sc.assignees)
    rows.sort(key=lambda x: (x["shape_id"], x["alpha"]))
    for i, row in enumerate(rows, start=1):
        row["case_id"] = f"c{i:05d}"
    return [{k: row[k] for k in CASE_COLUMNS} for row in rows]


def validation_cases(sc: SamplingConfig) -> list[dict]:
    """검증용 목록: NACA 코드 형상 × α (Airfoil2Vec·XFOIL 비교). 형상 이름 = naca<코드>."""
    rows = []
    for code in sc.validation.shapes:
        code = str(code).upper().removeprefix("NACA").strip()
        if len(code) != 4 or not code.isdigit():
            raise ConfigError(f"[sampling.validation] shapes는 NACA 4자리 코드여야 합니다: {code!r}")
        m, p, t = int(code[0]), int(code[1]), int(code[2:])
        for alpha in sc.validation.alpha:
            rows.append({"shape_id": f"naca{code}", "m": m, "p": p, "t": t, "alpha": float(alpha)})
    _assign(rows, sc.assignees)
    for row in rows:
        row["case_id"] = f"v_{row['shape_id']}_a{row['alpha']:g}"
    return [{k: row[k] for k in CASE_COLUMNS} for row in rows]


def _assign(rows: list[dict], assignees: list[str]) -> None:
    """형상 단위로 담당자를 번갈아 배정 (형상 수가 고르게 나뉨)."""
    order = sorted({r["shape_id"] for r in rows})
    who = {s: assignees[i % len(assignees)] for i, s in enumerate(order)}
    for r in rows:
        r["assignee"] = who[r["shape_id"]]


def write_cases(path: Path, rows: list[dict], meta: dict, force: bool = False) -> Path:
    if path.exists() and not force:
        old = path.read_text(encoding="utf-8-sig")
        new = _csv_text(rows)
        if old != new:
            raise ConfigError(f"{path.name}이 이미 있고 내용이 다릅니다 (팀원 결과가 섞이지 않게 덮어쓰지 않음). "
                              "의도한 변경이면 --force.")
        return path
    path.write_text(_csv_text(rows), encoding="utf-8")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _csv_text(rows: list[dict]) -> str:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CASE_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def make_cases(cfg: ProjectConfig, force: bool = False) -> dict:
    """[sampling] 설정으로 cases.csv와 cases_validation.csv([cases] csv / csv_validation 경로)를 쓴다."""
    import tomllib

    sc = SamplingConfig.from_dict(cfg.sampling)
    with cfg.path.open("rb") as f:
        cases_tbl = tomllib.load(f).get("cases", {})
    out = {}
    meta = {"generator": "afs make-cases", "sampling": {k: v for k, v in asdict(sc).items()}}
    for key, rows in (("csv", generate_cases(sc)), ("csv_validation", validation_cases(sc))):
        if not cases_tbl.get(key):
            continue
        path = (cfg.root / cases_tbl[key]).resolve()
        write_cases(path, rows, {**meta, "list": key, "n_cases": len(rows),
                                 "n_shapes": len({r["shape_id"] for r in rows})}, force)
        per = {}
        for r in rows:
            per.setdefault(r["assignee"], set()).add(r["shape_id"])
        out[key] = {"path": path, "rows": rows,
                    "by_assignee": {a: (len(s), sum(1 for r in rows if r["assignee"] == a)) for a, s in per.items()}}
    if not out:
        raise ConfigError("[cases] csv (와 csv_validation) 경로가 없습니다.")
    return out
