"""프로젝트 설정(project.toml) 로딩."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_ERROR_PATTERNS = [
    "Error:",
    "Divergence detected",
    "floating point exception",
]


class ConfigError(ValueError):
    pass


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\s]+')


def safe_name(text: str) -> str:
    """파일/폴더 이름으로 쓸 수 없는 문자를 _로 바꾼다."""
    return _UNSAFE_CHARS.sub("_", text).strip("._") or "case"


@dataclass
class SolverConfig:
    kind: str = "fluent"  # "fluent" 또는 "custom"
    executable: str | None = None  # 비우면 AWP_ROOTxxx 환경변수로 자동 탐색
    mode: str = "3ddp"  # Fluent: 2d, 2ddp, 3d, 3ddp
    processes: int = 4
    gui: bool = False
    graphics: bool = False  # Fluent 이미지 저장 시 true (Windows: -hidden, Linux: -gu -driver null)
    extra_args: list[str] = field(default_factory=list)
    command: list[str] | None = None  # kind="custom"일 때 명령어 템플릿
    timeout_minutes: float | None = None


@dataclass
class ResultsConfig:
    report_files: list[str] = field(default_factory=lambda: ["*.out"])
    average_last: int = 1  # 마지막 N개 반복의 평균을 결과값으로 사용
    average_last_oscillating: int | None = None  # 진동(oscillating) 판정 케이스만 마지막 N개 평균 (비우면 average_last)
    error_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_ERROR_PATTERNS))
    summary_file: str = "summary.csv"
    x_param: str = "aoa"  # 그래프 x축 / 받음각 변수 이름
    wind_axes: bool = True  # cx, cy 리포트를 받음각으로 회전해 cl, cd 계산
    objective: str = "max_ld"  # 형상 순위 기준: max_ld (최대 양항비) 또는 max_cl (최대 양력계수)
    split_by: list[str] = field(default_factory=list)  # 조건별로 나눠 순위/그래프/선별 (예: ["speed_kmh"])


@dataclass
class ProjectConfig:
    path: Path
    name: str
    template: Path
    output_dir: Path
    case_name: str
    group: str | None
    cases_csv: Path | None
    sweep: dict[str, list]
    case_sets: list[dict]  # [[cases.set]]: 세트별 label/constants/sweep/name/group (없으면 단일 세트)
    constants: dict
    derived: dict[str, str]
    max_parallel: int
    mesh: dict | None
    solver: SolverConfig
    results: ResultsConfig
    convergence: dict | None = None  # [convergence] 수렴 판정 기준 (convergence.py)
    ml: dict | None = None           # [ml] 데이터셋 분할·학습 설정 (dataset.py, learn.py)

    @property
    def root(self) -> Path:
        return self.path.parent


def _section(cls, data, where: str):
    """[solver]/[results] 표를 데이터클래스로. 알 수 없는 항목(오타)은 traceback 대신 ConfigError."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}는 표(table)여야 합니다.")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        import difflib

        hints = [f"{k} (→ {m[0]}?)" if (m := difflib.get_close_matches(k, known, 1)) else k for k in sorted(unknown)]
        raise ConfigError(f"{where}에 알 수 없는 항목: {', '.join(hints)}")
    return cls(**data)


def _check_number(value, where: str, integer: bool = False, minimum: float | None = None,
                  allow_none: bool = False):
    if value is None and allow_none:
        return
    ok = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not ok:
        raise ConfigError(f"{where}는 {'정수' if integer else '숫자'}여야 합니다: {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}는 {minimum} 이상이어야 합니다: {value!r}")


def _check_str_list(value, where: str) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}는 문자열 목록이어야 합니다: {value!r}")


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"설정 파일 문법 오류 ({path.name}): {exc}") from None
    except UnicodeDecodeError as exc:
        raise ConfigError(f"설정 파일을 UTF-8로 읽을 수 없습니다 ({path.name}): {exc}") from None

    root = path.parent
    project = data.get("project", {})
    cases = data.get("cases", {})

    if "template" not in project:
        raise ConfigError("[project] template (저널 템플릿 경로)을 지정해야 합니다.")

    solver = _section(SolverConfig, data.get("solver", {}), "[solver]")
    _check_number(solver.processes, "[solver] processes", integer=True, minimum=1)
    _check_number(solver.timeout_minutes, "[solver] timeout_minutes", minimum=0, allow_none=True)
    _check_str_list(solver.extra_args, "[solver] extra_args")
    if solver.command is not None:
        _check_str_list(solver.command, "[solver] command")
    if solver.kind not in ("fluent", "custom"):
        raise ConfigError(f"[solver] kind는 'fluent' 또는 'custom'이어야 합니다: {solver.kind!r}")
    if solver.kind == "custom" and not solver.command:
        raise ConfigError("[solver] kind='custom'이면 command를 지정해야 합니다.")

    results = _section(ResultsConfig, data.get("results", {}), "[results]")
    _check_number(results.average_last, "[results] average_last", integer=True, minimum=1)
    _check_number(results.average_last_oscillating, "[results] average_last_oscillating", integer=True,
                  minimum=1, allow_none=True)
    _check_str_list(results.report_files, "[results] report_files")
    _check_str_list(results.error_patterns, "[results] error_patterns")
    _check_str_list(results.split_by, "[results] split_by")
    if results.objective not in ("max_ld", "max_cl"):
        raise ConfigError(f"[results] objective는 'max_ld' 또는 'max_cl'이어야 합니다: {results.objective!r}")
    max_parallel = project.get("max_parallel", 1)
    _check_number(max_parallel, "[project] max_parallel", integer=True, minimum=1)

    cases_csv = cases.get("csv")
    sweep = cases.get("sweep", {})
    case_sets = cases.get("set", [])
    if not cases_csv and not sweep and not case_sets:
        raise ConfigError("[cases] csv, [cases.sweep], [[cases.set]] 중 하나는 지정해야 합니다.")
    for where, sw in [("[cases.sweep]", sweep)] + [
        (f"[[cases.set]] {s.get('label', i + 1)}", s.get("sweep", {})) for i, s in enumerate(case_sets)
    ]:
        for key, values in sw.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"{where} sweep의 {key}는 값 목록이어야 합니다.")
    for i, s in enumerate(case_sets):
        unknown = set(s) - {"label", "constants", "sweep", "name", "group"}
        if unknown:
            raise ConfigError(f"[[cases.set]] {i + 1}번째에 알 수 없는 항목: {', '.join(sorted(unknown))}")

    return ProjectConfig(
        path=path,
        name=project.get("name", root.name),
        template=(root / project["template"]).resolve(),
        output_dir=(root / project.get("output_dir", "runs")).resolve(),
        case_name=cases.get("name", ""),
        group=cases.get("group"),
        cases_csv=(root / cases_csv).resolve() if cases_csv else None,
        sweep=sweep,
        case_sets=case_sets,
        constants=data.get("constants", {}),
        derived=data.get("derived", {}),
        max_parallel=max_parallel,
        mesh=data.get("mesh"),
        solver=solver,
        results=results,
        convergence=data.get("convergence"),
        ml=data.get("ml"),
    )
