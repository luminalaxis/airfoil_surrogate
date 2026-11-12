"""케이스 매트릭스 생성 및 분류."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, ProjectConfig, safe_name
from .expr import evaluate
from .meshing import MeshJob, make_job, mesh_name


@dataclass
class Case:
    name: str
    group: str
    params: dict
    dir: Path
    mesh: MeshJob | None = None


def _coerce(value: str):
    value = value.strip()
    if len(value) > 1 and value[0] == "0" and value.isdigit():
        return value  # "0012" 같은 NACA 코드는 문자열 그대로
    for conv in (int, float):
        try:
            number = conv(value)
        except ValueError:
            continue
        # "nan", "inf", "infinity" 같은 이름(익형 이름 등)은 숫자로 바꾸지 않는다
        return number if math.isfinite(number) else value
    return value


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise ConfigError(f"케이스 CSV를 찾을 수 없습니다: {path}")
    # utf-8-sig: 엑셀에서 저장한 CSV의 BOM 처리
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = [
                {k.strip(): _coerce(v) for k, v in row.items()
                 if k and isinstance(v, str) and v.strip()}  # 열보다 값이 많은 행의 남는 값(list)은 무시
                for row in csv.DictReader(f)
            ]
    except UnicodeDecodeError as exc:
        raise ConfigError(f"케이스 CSV를 UTF-8로 읽을 수 없습니다 (엑셀에서 'CSV UTF-8'로 저장): {path} ({exc})") from None
    return [r for r in rows if r]


def _sweep_rows(sweep: dict[str, list]) -> list[dict]:
    if not sweep:
        return [{}]
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*sweep.values())]


def _format(pattern: str, params: dict, what: str) -> str:
    try:
        return pattern.format(**params)
    except KeyError as exc:
        raise ConfigError(f"{what} 패턴 {pattern!r}에 없는 변수 {exc}가 쓰였습니다.") from exc


def _expand(cfg: ProjectConfig):
    """(params, 이름 패턴, 그룹 패턴) 생성.

    [[cases.set]]가 있으면 세트마다 (공통 [cases.sweep] × 세트 sweep)을 펼쳐 합친다.
    값 우선순위: [constants] < 세트 constants < CSV 행 < 공통 sweep < 세트 sweep.
    세트 label은 케이스 변수 case_set 으로 쓸 수 있다.
    """
    base_rows = _read_csv(cfg.cases_csv) if cfg.cases_csv else [{}]
    sets = cfg.case_sets or [{}]
    for s in sets:
        extra = {"case_set": s["label"]} if "label" in s else {}
        for row, common, own in itertools.product(base_rows, _sweep_rows(cfg.sweep),
                                                  _sweep_rows(s.get("sweep", {}))):
            params = {**cfg.constants, **s.get("constants", {}), **extra, **row, **common, **own}
            yield params, s.get("name", cfg.case_name), s.get("group", cfg.group)


def build_cases(cfg: ProjectConfig) -> list[Case]:
    """CSV 행 × 스윕 조합(케이스 세트별)으로 케이스 목록을 만든다."""
    cases: list[Case] = []
    seen: dict[str, int] = {}

    for idx, (params, name_pattern, group_pattern) in enumerate(_expand(cfg), start=1):
        for key, expression in cfg.derived.items():
            params[key] = evaluate(expression, params)

        name = params.pop("name", None)
        if name is None:
            name = _format(name_pattern, params, "케이스 이름") if name_pattern else f"case_{idx:03d}"
        name = safe_name(str(name))
        group = safe_name(_format(group_pattern, params, "그룹")) if group_pattern else ""

        if name in seen:
            raise ConfigError(
                f"케이스 이름 {name!r}이 중복됩니다 ({seen[name]}번째, {idx}번째). "
                "[cases] name 패턴에 구분 변수를 추가하세요."
            )
        seen[name] = idx

        case_dir = cfg.output_dir / group / name if group else cfg.output_dir / name
        cases.append(Case(name=name, group=group, params=params, dir=case_dir))

    _set_flow_angle(cases, cfg.results.x_param)
    if cfg.mesh is not None:
        _attach_meshes(cases, cfg)
    return cases


def _set_flow_angle(cases: list[Case], x_param: str) -> None:
    """Fluent 유입 각도 flow_angle [deg] = 받음각. 저널은 ${flow_angle}을 쓴다."""
    for c in cases:
        a = c.params.get(x_param)
        if isinstance(a, (int, float)) and "flow_angle" not in c.params:
            c.params["flow_angle"] = a


def _attach_meshes(cases: list[Case], cfg: ProjectConfig) -> None:
    """형상이 같은 케이스는 메쉬 하나를 공유한다.

    케이스 변수 reynolds가 형상 안에서 다르면(예: 속도 스윕) 가장 큰 값으로 첫 셀 높이를 잡아
    모든 조건에서 y+ 가 목표 이하가 되게 한다.
    """
    meshes_dir = cfg.output_dir / "_meshes"
    max_re: dict[str, float] = {}
    for c in cases:
        if "reynolds" in c.params and "main_airfoil" in c.params:
            key = mesh_name(c.params)
            max_re[key] = max(max_re.get(key, 0.0), float(c.params["reynolds"]))
    first: dict[str, Case] = {}
    for c in cases:
        params = c.params
        if "reynolds" in params and "main_airfoil" in params:
            key = mesh_name(params)
            params = {**params, "reynolds": max_re[key]}
        c.mesh = make_job(params, cfg.mesh, meshes_dir)
        # 이름이 같은 메쉬는 한 번만 만들어 공유하므로, 이름이 같은데 형상이나 메쉬 설정이 다르면
        # 엉뚱한 메쉬로 계산된다 (예: 케이스 변수로 surface_cells 등을 바꾸면서 mesh_level을 안 준 경우)
        other = first.setdefault(c.mesh.name, c)
        if other is not c and (other.mesh.settings != c.mesh.settings
                               or other.mesh.geometry_params != c.mesh.geometry_params):
            raise ConfigError(
                f"메쉬 이름 {c.mesh.name}이 같은데 형상이나 메쉬 설정이 다른 케이스가 있습니다 "
                f"({other.name}, {c.name}). 케이스마다 [mesh] 항목(refine, surface_cells 등)을 바꾸려면 "
                "케이스 변수 mesh_level로 메쉬를 구분하세요.")


def filter_cases(cases: list[Case], patterns: list[str] | None) -> list[Case]:
    """--only 옵션: 케이스/그룹 이름에 대한 와일드카드 필터."""
    if not patterns:
        return cases
    from fnmatch import fnmatch

    return [
        c for c in cases
        if any(fnmatch(c.name, p) or fnmatch(c.group, p) for p in patterns)
    ]


def partition_cases(cases: list[Case], part: str | None) -> list[Case]:
    """--part K/N 옵션: 형상(메쉬, 없으면 그룹) 단위로 N등분해 K번째 몫만 남긴다.

    같은 형상의 받음각·Re 케이스는 같은 몫에 들어가서 메쉬를 한 사람만 만든다.
    형상 이름 순서대로 번갈아 배정하므로 팀원별 케이스 수가 거의 같다.
    """
    if not part:
        return cases
    try:
        k, n = (int(v) for v in part.split("/"))
    except ValueError:
        raise ConfigError(f"--part 형식은 K/N 입니다 (예: 1/4): {part!r}") from None
    if not 1 <= k <= n:
        raise ConfigError(f"--part K/N 에서 K는 1~N 이어야 합니다: {part!r}")

    def unit(c: Case) -> str:
        return c.mesh.name if c.mesh is not None else (c.group or c.name)

    units = sorted({unit(c) for c in cases})
    mine = {u for i, u in enumerate(units) if i % n == k - 1}
    return [c for c in cases if unit(c) in mine]
