"""프로젝트 설정(project.toml) 로딩."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
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
    results_file: str = "results.csv"  # 팀 공통 스키마 결과 (results.py RESULT_COLUMNS)
    cp_stations: list[float] = field(default_factory=list)  # 표면 Cp를 보간할 x/c 지점 (비우면 Cp 저장 안 함)
    cp_file: str = "cp.txt"            # 저널이 내보내는 표면 Cp 파일 (Fluent ASCII export)
    # 저널 ${cp_export} 자리에 들어가는 Fluent 명령. 버전마다 프롬프트가 다르면 GUI 저널 녹화로 바꾸세요
    cp_export_command: str = '/file/export/ascii "cp.txt" main () yes pressure-coefficient () no'
    solver_version: str = ""           # 로그에서 Fluent 버전을 못 찾을 때 기록할 값


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
    flow: object | None = None       # [flow] 유동 조건·모드 (flow.FlowConfig)
    sampling: dict | None = None     # [sampling] 라틴 하이퍼큐브 케이스 생성 (sampling.py)
    validation: dict | None = None   # [validation] Airfoil2Vec·XFOIL 검증 (validate.py)
    base_output_dir: Path | None = None  # 모드 꼬리표를 붙이기 전 output_dir (메쉬는 여기 _meshes에 공유)
    overrides: dict = field(default_factory=dict)  # 명령줄 --re-mode 등

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def meshes_dir(self) -> Path:
        """형상별 메쉬 폴더. 모드별 결과 폴더와 달리 난류 모델끼리 공유한다 (이름에 Re·메쉬 모드 꼬리표)."""
        return (self.base_output_dir or self.output_dir) / "_meshes"

    def with_overrides(self, **overrides) -> "ProjectConfig":
        """같은 설정 파일을 다른 모드로 다시 읽는다 (예: re_mode="validation", mesh_mode="fine")."""
        return load_config(self.path, {**self.overrides, **overrides})


FLOW_OVERRIDE_KEYS = ("re_mode", "mesh_mode", "turbulence_model", "case_list")


def _split_mesh_modes(mesh: dict | None, mode: str | None) -> dict | None:
    """[mesh] 공통 항목 + [mesh.light] / [mesh.fine] 중 선택한 모드의 항목."""
    if mesh is None:
        return None
    base = {k: v for k, v in mesh.items() if not isinstance(v, dict)}
    presets = {k: v for k, v in mesh.items() if isinstance(v, dict)}
    if not presets:
        return base
    if mode is None:
        raise ConfigError(f"[mesh]에 모드별 설정({', '.join(presets)})이 있는데 [flow] mesh_mode가 없습니다.")
    if mode not in presets:
        raise ConfigError(f"[mesh.{mode}] 설정이 없습니다 (있는 모드: {', '.join(presets)}).")
    return {**base, **presets[mode]}


def load_config(path: str | Path, overrides: dict | None = None) -> ProjectConfig:
    """overrides: 명령줄에서 덮어쓸 [flow] 모드 (re_mode, mesh_mode, turbulence_model, case_list)."""
    path = Path(path).resolve()
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    if not path.is_file():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)

    root = path.parent
    project = data.get("project", {})
    cases = data.get("cases", {})

    if "template" not in project:
        raise ConfigError("[project] template (저널 템플릿 경로)을 지정해야 합니다.")

    solver = SolverConfig(**data.get("solver", {}))
    if solver.kind not in ("fluent", "custom"):
        raise ConfigError(f"[solver] kind는 'fluent' 또는 'custom'이어야 합니다: {solver.kind!r}")
    if solver.kind == "custom" and not solver.command:
        raise ConfigError("[solver] kind='custom'이면 command를 지정해야 합니다.")

    results = ResultsConfig(**data.get("results", {}))

    constants = dict(data.get("constants", {}))
    flow = None
    if "flow" in data:
        from .flow import FlowConfig

        flow = FlowConfig.from_dict(data["flow"], overrides)
        flow_consts = flow.constants()
        clash = sorted(set(flow_consts) & set(constants))
        if clash:
            raise ConfigError(f"[constants]와 [flow]에 같은 값이 있습니다: {', '.join(clash)}. [flow] 한곳에서만 정하세요.")
        constants = {**constants, **flow_consts}
    elif overrides:
        raise ConfigError(f"--{next(iter(overrides)).replace('_', '-')} 옵션은 [flow] 섹션이 있는 설정에서만 씁니다.")

    cases_csv = cases.get("csv")
    if flow is not None and flow.effective_case_list == "validation":
        cases_csv = cases.get("csv_validation")
        if not cases_csv:
            raise ConfigError("검증 케이스 목록을 쓰려면 [cases] csv_validation 을 지정하세요.")
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

    base_out = (root / project.get("output_dir", "runs")).resolve()
    mesh_mode = flow.mesh_mode if flow is not None else None
    return ProjectConfig(
        path=path,
        name=project.get("name", root.name),
        template=(root / project["template"]).resolve(),
        output_dir=base_out / flow.tag if flow is not None else base_out,
        case_name=cases.get("name", ""),
        group=cases.get("group"),
        cases_csv=(root / cases_csv).resolve() if cases_csv else None,
        sweep=sweep,
        case_sets=case_sets,
        constants=constants,
        derived=data.get("derived", {}),
        max_parallel=int(project.get("max_parallel", 1)),
        mesh=_split_mesh_modes(data.get("mesh"), mesh_mode),
        solver=solver,
        results=results,
        convergence=data.get("convergence"),
        ml=data.get("ml"),
        flow=flow,
        sampling=data.get("sampling"),
        validation=data.get("validation"),
        base_output_dir=base_out,
        overrides=overrides,
    )
