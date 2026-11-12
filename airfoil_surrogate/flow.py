"""유동 조건과 실행 모드 ([flow] 섹션).

팀원 4명이 같은 조건으로 돌리도록 유속·점성·Re, 난류 모델, 메쉬 모드, Re 모드를 한곳에 모은다.

    [flow]
    re_mode = "production"          # production (Re 1e5, 본 데이터) / validation (Re 4.66e6, Airfoil2Vec 비교)
    mesh_mode = "light"             # light (본 데이터) / fine (검증)
    turbulence_model = "kOmegaSST"  # kOmegaSST (Fluent SST k-ω) / kOmegaSSTLM (Fluent Transition SST)
    chord = 0.1                     # [m]
    nu = 1.4607e-5                  # 동점성 [m²/s] (Fluent 기본 공기 μ/ρ = 1.7894e-5 / 1.225)

    [flow.production]               # 모드별로 위 값을 덮어쓴다. velocity 또는 reynolds 중 하나만
    reynolds = 1.0e5
    y_plus = 1.0
    wall_treatment = "low_re"

    [flow.validation]
    chord = 1.0
    reynolds = 4.66e6
    y_plus = 50
    wall_treatment = "wall_function"
    naca_te = "sharp"

계산해서 케이스 변수(저널 ${...}, [derived], 메쉬)로 넘기는 값: velocity, reynolds, mach, density, viscosity,
aref(= chord × span), lref(= chord), k_in, omega_in, intermittency_in, retheta_in, base_case 등 (constants()).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

from .config import ConfigError

TURBULENCE_MODELS = {"kOmegaSST": "SST k-omega", "kOmegaSSTLM": "Transition SST (gamma-ReTheta)"}
RE_MODES = ("production", "validation")
MESH_MODES = ("light", "fine")
CASE_LISTS = ("main", "validation")
WALL_TREATMENTS = ("low_re", "wall_function")
TE_SHAPES = ("sharp", "blunt")
C_MU = 0.09


def retheta_t(tu_percent: float) -> float:
    """Langtry–Menter 상관식: 자유류 난류강도 Tu[%]에서 천이 운동량 두께 Re_θt (압력 구배 0)."""
    tu = max(tu_percent, 0.027)  # 상관식 하한 (Tu → 0이면 발산)
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / tu**2
    return 331.5 * (tu - 0.5658) ** -0.671


@dataclass
class FlowConfig:
    re_mode: str = "production"
    mesh_mode: str = "light"
    turbulence_model: str = "kOmegaSST"
    case_list: str = ""              # main / validation (비우면 production → main, validation → validation)
    chord: float = 0.1               # [m]
    span: float = 1.0                # 2D 깊이 [m] (Fluent reference depth)
    rho: float = 1.225               # [kg/m³] 베이스 케이스 재질과 같아야 함
    nu: float = 1.7894e-5 / 1.225    # [m²/s]
    velocity: float | None = None    # [m/s] velocity 또는 reynolds 중 하나만
    reynolds: float | None = None
    y_plus: float = 1.0              # 메쉬 목표 y+
    wall_treatment: str = "low_re"   # low_re (y+ ≈ 1, 벽까지 해석) / wall_function (y+ 30~200)
    turbulence_intensity: float = 0.01   # 유입 난류강도 I (0.01 = 1%)
    length_scale: float = 0.01           # 유입 난류 길이척도 l [c]
    naca_te: str = "blunt"           # NACA 4자리 뒷전: sharp (a4 = −0.1036) / blunt (−0.1015)
    moment_ref: float = 0.25         # 모멘트 기준점 x/c (CM = c/4)
    speed_of_sound: float = 340.29
    iterations: int | None = None    # 모드별 최대 반복 수를 다르게 줄 때
    base_cases: dict = field(default_factory=dict)  # 난류 모델 → 베이스 케이스 파일
    modes: dict = field(default_factory=dict)       # [flow.production], [flow.validation] 원본

    @classmethod
    def from_dict(cls, data: dict, overrides: dict | None = None) -> "FlowConfig":
        data = dict(data)
        modes = {k: data.pop(k) for k in list(data) if k in RE_MODES}
        for name, table in modes.items():
            if not isinstance(table, dict):
                raise ConfigError(f"[flow.{name}]은 표(table)여야 합니다.")
        known = {f.name for f in fields(cls)} - {"modes"}
        overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
        re_mode = overrides.get("re_mode", data.get("re_mode", "production"))
        if re_mode not in RE_MODES:
            raise ConfigError(f"[flow] re_mode는 {' / '.join(RE_MODES)} 중 하나여야 합니다: {re_mode!r}")
        layer = {**modes.get(re_mode, {}), **overrides}
        merged = {**data, **layer}
        # 모드 표가 velocity와 reynolds 중 하나를 주면, 공통 값의 다른 하나는 버린다
        for a, b in (("velocity", "reynolds"), ("reynolds", "velocity")):
            if a in layer and b not in layer:
                merged.pop(b, None)
        unknown = set(merged) - known
        if unknown:
            raise ConfigError(f"[flow]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        obj = cls(**merged, modes=modes)
        obj.re_mode = re_mode
        obj.validate()
        return obj

    # ---------------------------------------------------------------- 검증
    def validate(self) -> None:
        def one_of(name, value, allowed):
            if value not in allowed:
                raise ConfigError(f"[flow] {name}는 {' / '.join(allowed)} 중 하나여야 합니다: {value!r}")

        one_of("mesh_mode", self.mesh_mode, MESH_MODES)
        one_of("turbulence_model", self.turbulence_model, tuple(TURBULENCE_MODELS))
        one_of("wall_treatment", self.wall_treatment, WALL_TREATMENTS)
        one_of("naca_te", self.naca_te, TE_SHAPES)
        if self.case_list:
            one_of("case_list", self.case_list, CASE_LISTS)
        for name in ("chord", "span", "rho", "nu", "turbulence_intensity", "length_scale", "y_plus"):
            if not getattr(self, name) > 0:
                raise ConfigError(f"[flow] {name}은 0보다 커야 합니다: {getattr(self, name)}")
        if (self.velocity is None) == (self.reynolds is None):
            raise ConfigError(f"[flow] ({self.re_mode}) velocity와 reynolds 중 정확히 하나만 지정하세요 "
                              "(나머지는 chord, nu로 계산).")
        if self.wall_treatment == "low_re" and self.y_plus > 5:
            raise ConfigError(f"[flow] wall_treatment = low_re 인데 y_plus = {self.y_plus:g} 입니다. "
                              "벽까지 해석하려면 y+ ≈ 1 (5 이하)로 두세요.")
        if self.wall_treatment == "wall_function" and not 30 <= self.y_plus <= 300:
            raise ConfigError(f"[flow] wall_treatment = wall_function 인데 y_plus = {self.y_plus:g} 입니다. "
                              "벽함수 영역은 y+ 30~200 입니다.")
        if self.turbulence_model == "kOmegaSSTLM" and self.wall_treatment != "low_re":
            raise ConfigError("[flow] kOmegaSSTLM(Transition SST)은 경계층 내부를 풀어야 해서 y+ ≈ 1 "
                              "(wall_treatment = low_re)에서만 씁니다.")

    # ---------------------------------------------------------------- 계산 값
    @property
    def U(self) -> float:
        return self.velocity if self.velocity is not None else self.reynolds * self.nu / self.chord

    @property
    def Re(self) -> float:
        return self.reynolds if self.reynolds is not None else self.velocity * self.chord / self.nu

    @property
    def mach(self) -> float:
        return self.U / self.speed_of_sound

    @property
    def effective_case_list(self) -> str:
        return self.case_list or ("main" if self.re_mode == "production" else "validation")

    def turbulence_inlet(self) -> dict:
        """난류강도 I와 길이척도 l에서 유입 k, ω (그리고 Transition SST의 γ, Re_θt).

        k = 1.5 (U I)², ω = √k / (C_μ^¼ l),  l = length_scale × chord
        """
        k = 1.5 * (self.U * self.turbulence_intensity) ** 2
        ell = self.length_scale * self.chord
        omega = math.sqrt(k) / (C_MU**0.25 * ell)
        return {
            "k_in": k, "omega_in": omega, "length_scale_m": ell,
            "turbulent_viscosity_ratio": k / omega / self.nu,
            "intermittency_in": 1.0,
            "retheta_in": retheta_t(100 * self.turbulence_intensity),
        }

    @property
    def tag(self) -> str:
        """결과 폴더 이름: 모드가 다른 결과가 섞이지 않게 output_dir/<tag>/ 에 저장."""
        return f"{self.re_mode}_{self.turbulence_model}_{self.mesh_mode}_{self.effective_case_list}"

    @property
    def mesh_tag(self) -> str:
        """메쉬 이름 꼬리표: 메쉬는 난류 모델과 무관하므로 Re 모드(y+)와 메쉬 모드로만 구분."""
        return f"{self.re_mode[:3]}-{self.mesh_mode}"

    def model_journal_lines(self) -> str:
        """난류 모델별 저널 줄 (${model_lines}). Transition SST만 유입 간헐도(γ)를 갱신한다."""
        if self.turbulence_model == "kOmegaSSTLM":
            return (f'/define/named-expressions/edit "intermittency_in" definition '
                    f'"{self.turbulence_inlet()["intermittency_in"]:g}" q')
        return "; kOmegaSST: 천이 모델 변수 없음"

    def constants(self) -> dict:
        inlet = self.turbulence_inlet()
        out = {
            "re_mode": self.re_mode, "mesh_mode": self.mesh_mode, "turb_model": self.turbulence_model,
            "case_list": self.effective_case_list, "mesh_tag": self.mesh_tag,
            "fluent_model": TURBULENCE_MODELS[self.turbulence_model],
            "wall_treatment": self.wall_treatment, "naca_te": self.naca_te,
            "chord": self.chord, "span": self.span, "density": self.rho, "nu": self.nu,
            "viscosity": self.rho * self.nu,
            "velocity": self.U, "reynolds": self.Re, "mach": self.mach,
            "aref": self.chord * self.span, "lref": self.chord, "y_plus": self.y_plus,
            "turbulence_intensity": self.turbulence_intensity, "moment_ref": self.moment_ref,
            "k_in": inlet["k_in"], "omega_in": inlet["omega_in"],
            "intermittency_in": inlet["intermittency_in"], "retheta_in": inlet["retheta_in"],
            "model_lines": self.model_journal_lines(),
        }
        if self.iterations is not None:
            out["iterations"] = int(self.iterations)
        if self.base_cases:
            if self.turbulence_model not in self.base_cases:
                raise ConfigError(f"[flow] base_cases에 {self.turbulence_model}용 베이스 케이스가 없습니다.")
            out["base_case"] = self.base_cases[self.turbulence_model]
        return out

    def warnings(self) -> list[str]:
        out = []
        if self.mach > 0.3:
            out.append(f"Mach {self.mach:.2f} > 0.3: 밀도 일정(비압축) 해석이라 압축성 효과는 빠집니다 "
                       "(계수는 Re만의 함수로 계산됨). 코드를 키워 유속을 낮추는 것을 권장합니다.")
        return out

    def describe(self) -> str:
        i = self.turbulence_inlet()
        return (f"{self.re_mode} / {self.turbulence_model} / mesh {self.mesh_mode} / cases {self.effective_case_list}: "
                f"Re {self.Re:,.0f}, U {self.U:.3f} m/s, c {self.chord:g} m, Mach {self.mach:.3f}, "
                f"y+ {self.y_plus:g} ({self.wall_treatment}), I {100 * self.turbulence_intensity:g}%, "
                f"l {self.length_scale:g}c → k {i['k_in']:.4g} m²/s², ω {i['omega_in']:.4g} 1/s "
                f"(μt/μ {i['turbulent_viscosity_ratio']:.3g})")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("modes")
        return d
