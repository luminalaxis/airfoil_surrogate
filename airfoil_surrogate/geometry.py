"""2D 단일 익형 형상: NACA 4자리 / 좌표 파일.

좌표계: 코드 = 1, 앞전 = 원점, x는 뒷전 방향 (내부 계산은 코드 기준 무차원).
점 순서는 Selig 형식(뒷전 윗면 → 앞전 → 뒷전 아랫면)으로 통일한다.

입력 변수 (둘 중 하나):
  - m, p, t      : NACA 4자리 실수 파라미터 (라틴 하이퍼큐브 샘플링용). 단위는 NACA 자리수와 같다:
                   m = 최대 캠버 [% c], p = 최대 캠버 위치 [c/10], t = 최대 두께 [% c].  NACA 2412 = (2, 4, 12)
  - main_airfoil : NACA 4자리 코드(예: "2412") 또는 좌표 파일 경로
  - naca_te      : (선택) NACA 뒷전 sharp (a4 = −0.1036, 닫힘) / blunt (a4 = −0.1015, 표준식, 기본값)
  - main_chord   : (선택) 코드 길이. 메쉬는 코드 기준 무차원으로 만들고 [mesh] chord로 스케일한다
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError


@dataclass
class Element:
    name: str
    xy: np.ndarray  # (N, 2), Selig 순서
    le_index: int

    @property
    def sharp_te(self) -> bool:
        return bool(np.allclose(self.xy[0], self.xy[-1], atol=1e-9))

    @property
    def te_upper(self) -> np.ndarray:
        return self.xy[0]

    @property
    def te_lower(self) -> np.ndarray:
        return self.xy[-1]

    def transformed(self, scale=1.0, angle_deg=0.0, pivot=(0.0, 0.0), shift=(0.0, 0.0)) -> "Element":
        """pivot 기준 scale, 시계방향 angle_deg 회전 후 shift 만큼 이동."""
        a = np.radians(-angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        xy = (self.xy - pivot) * scale @ rot.T + pivot + shift
        return Element(self.name, xy, self.le_index)


@dataclass
class Geometry:
    elements: list[Element]
    info: dict  # 실제 gap/overlap, 최소 간격 등


# ---------------------------------------------------------------- 단면 생성

TE_COEFFICIENT = {"sharp": -0.1036, "blunt": -0.1015}  # NACA 두께식 마지막 계수


def naca4_mpt(m: float, p: float, t: float, n: int = 160, te: str = "blunt") -> np.ndarray:
    """NACA 4자리 좌표 (Selig 순서), 실수 파라미터.

    m: 최대 캠버 [% c], p: 최대 캠버 위치 [c/10], t: 최대 두께 [% c] (NACA 2412 = 2, 4, 12).
    te: "sharp"(a4 = −0.1036, 뒷전이 닫힘) / "blunt"(a4 = −0.1015, 표준식).
    점은 코사인 분포로 앞전·뒷전에 모은다. m = 0 또는 p = 0이면 대칭익(캠버선 0)으로 처리해
    캠버 식의 m/p², m/(1−p)² 에서 0으로 나누지 않는다.
    """
    if te not in TE_COEFFICIENT:
        raise ConfigError(f"뒷전 형상은 'sharp' 또는 'blunt'여야 합니다: {te!r}")
    m, p, t = float(m) / 100, float(p) / 10, float(t) / 100
    if not 0 < t < 0.5:
        raise ConfigError(f"두께 t는 0~50% 사이여야 합니다: {100 * t:g}")
    if m < 0:
        raise ConfigError(f"캠버 m은 0 이상이어야 합니다: {100 * m:g}")
    symmetric = m == 0 or p == 0
    if m > 0 and p == 0:
        raise ConfigError(f"캠버가 있는데(m = {100 * m:g}%) 최대 캠버 위치 p가 0입니다. 대칭익이면 m = 0으로 주세요.")
    if not symmetric and not 0 < p < 1:
        raise ConfigError(f"최대 캠버 위치 p는 0~10 (c/10 단위) 사이여야 합니다: {10 * p:g}")
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))  # 코사인 분포: 앞전/뒷전 집중
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + TE_COEFFICIENT[te] * x**4)
    if symmetric:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lower = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    xy = np.vstack([upper[::-1], lower[1:]])
    if te == "sharp":
        xy[-1] = xy[0] = 0.5 * (xy[0] + xy[-1])  # 반올림 오차(~1e-17)까지 정확히 닫음
    return xy


def naca4(code: str, n: int = 160, closed_te: bool = False) -> np.ndarray:
    """NACA 4자리 코드 문자열 (예: "2412"). closed_te=True면 뾰족한 뒷전."""
    if not re.fullmatch(r"\d{4}", code):
        raise ConfigError(f"NACA 4자리 코드가 아닙니다: {code!r}")
    return naca4_mpt(int(code[0]), int(code[1]), int(code[2:]), n, "sharp" if closed_te else "blunt")


def naca_code(m: float, p: float, t: float) -> str:
    """(m, p, t) → 가장 가까운 NACA 4자리 코드 (검증 데이터와 이름 맞추기용)."""
    return f"{round(float(m)):d}{round(float(p)):d}{round(float(t)):02d}"


def read_airfoil_file(path: Path) -> np.ndarray:
    """Selig 또는 Lednicer 형식 .dat 파일."""
    if not path.is_file():
        raise ConfigError(f"익형 좌표 파일을 찾을 수 없습니다: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    if len(rows) < 10:
        raise ConfigError(f"좌표 점이 너무 적습니다: {path}")
    pts = np.array(rows)
    if pts[0, 0] > 1.5 and pts[0, 1] > 1.5:  # Lednicer: 첫 줄이 (윗면 점 수, 아랫면 점 수)
        nu = int(pts[0, 0])
        upper, lower = pts[1:1 + nu], pts[1 + nu:]
        pts = np.vstack([upper[::-1], lower[1:] if np.allclose(lower[0], upper[0]) else lower])
    return pts


def normalize(xy: np.ndarray) -> tuple[np.ndarray, int]:
    """앞전을 원점, 시위를 1, 시위선을 x축에 맞춘다. (좌표, 앞전 인덱스) 반환."""
    te = 0.5 * (xy[0] + xy[-1])
    le_idx = int(np.argmax(np.linalg.norm(xy - te, axis=1)))
    le = xy[le_idx]
    chord_vec = te - le
    chord = np.linalg.norm(chord_vec)
    ang = np.arctan2(chord_vec[1], chord_vec[0])
    rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    out = (xy - le) @ rot.T / chord
    # Selig 순서(윗면 먼저)가 아니면 뒤집기
    if out[: le_idx + 1, 1].mean() < out[le_idx:, 1].mean():
        out = out[::-1].copy()
        le_idx = len(out) - 1 - le_idx
    # 중복 점 제거 (뒷전 닫힘 점은 유지)
    keep = np.ones(len(out), bool)
    keep[1:] = np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9
    keep[-1] = True
    le_idx = int(keep[:le_idx].sum())
    return out[keep], le_idx


CUSP_WEDGE_DEG = 1.0  # 뾰족한 뒷전 바로 앞 점 쌍의 쐐기각이 이보다 작으면 칼날형(cusp)으로 보고 정리
TE_GAP_MIN = 5e-4     # 뒷전 두 끝점 간격이 이보다 작으면 [c] 뾰족한 뒷전으로 닫음


def close_tiny_te_gap(xy: np.ndarray) -> np.ndarray:
    """좌표 파일의 뒷전 끝점이 반올림 등으로 아주 조금 벌어져 있으면(예: S1221은 0.00001c) 뭉툭한 뒷전
    면이 거의 0 길이가 되어 그 면의 셀이 찌그러진다. 두 끝점을 중점으로 모아 뾰족한 뒷전으로 만든다."""
    gap = float(np.linalg.norm(xy[0] - xy[-1]))
    if 0 < gap < TE_GAP_MIN:
        xy = xy.copy()
        xy[0] = xy[-1] = 0.5 * (xy[0] + xy[-1])
    return xy


def split_te_wedge(xy: np.ndarray) -> float:
    """뒷전 바로 앞 윗면·아랫면 점이 이루는 쐐기각 [deg] (Selig 순서, 시위 = x축)."""
    te, u, lo = xy[0], xy[1], xy[-2]
    return float(np.degrees(np.arctan2(u[1] - lo[1], te[0] - min(u[0], lo[0]))))


def open_cusp_te(xy: np.ndarray, le_idx: int, max_pairs: int = 3) -> tuple[np.ndarray, int]:
    """칼날형(cusp) 뒷전 정리.

    뒷전이 닫힌 좌표에서 뒷전 바로 앞의 윗면·아랫면 점이 거의 겹치면(쐐기각 < CUSP_WEDGE_DEG, 예: S4180은
    0.2°) 두 곡선이 메쉬 단계에서 서로 교차해 gmsh가 끝없이 재시도한다. 그런 점 쌍을 지워 뒷전을
    그 다음 점들이 이루는 쐐기로 만든다 (최대 max_pairs쌍, 바뀌는 길이는 보통 시위의 0.5% 이하).
    """
    if not np.allclose(xy[0], xy[-1], atol=1e-9):
        return xy, le_idx
    for _ in range(max_pairs):
        if le_idx < 4 or len(xy) - le_idx < 5 or split_te_wedge(xy) >= CUSP_WEDGE_DEG:
            break
        xy = np.vstack([xy[:1], xy[2:-2], xy[-1:]])
        le_idx -= 1
    return xy, le_idx


def load_section(spec, root: Path, name: str, te: str = "blunt") -> Element:
    spec = str(spec).strip()
    code = spec.upper().removeprefix("NACA").strip()
    xy = naca4(code, closed_te=te == "sharp") if re.fullmatch(r"\d{4}", code) \
        else read_airfoil_file((root / spec).resolve())
    return _element(xy, name)


def _element(xy: np.ndarray, name: str) -> Element:
    xy, le_idx = normalize(xy)
    xy, le_idx = open_cusp_te(close_tiny_te_gap(xy), le_idx)
    return Element(name, xy, le_idx)


# ---------------------------------------------------------------- 거리 계산

def _point_segments_distance(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """점들(M,2)에서 폴리라인 poly(N,2) 선분들까지의 최소 거리 (M,)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.maximum((ab**2).sum(1), 1e-30)
    ap = p[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(2) / denom[None], 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=2).min(1)


def _closed(poly: np.ndarray) -> np.ndarray:
    return poly if np.allclose(poly[0], poly[-1]) else np.vstack([poly, poly[:1]])


def _inside(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """점들(M,2)이 닫힌 다각형 내부인지 (ray casting)."""
    poly = _closed(poly)
    x, y = p[:, 0][:, None], p[:, 1][:, None]
    x1, y1, x2, y2 = poly[:-1, 0], poly[:-1, 1], poly[1:, 0], poly[1:, 1]
    cond = (y1 > y) != (y2 > y)
    xint = x1 + (y - y1) * (x2 - x1) / np.where(y2 == y1, 1e-30, y2 - y1)
    return ((cond & (x < xint)).sum(1) % 2) == 1


def signed_distance(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    d = _point_segments_distance(np.atleast_2d(p), _closed(poly))
    return np.where(_inside(np.atleast_2d(p), poly), -d, d)


def body_clearance(a: np.ndarray, b: np.ndarray) -> float:
    """두 물체 사이 최소 거리. 겹치면 음수."""
    if _inside(a, b).any() or _inside(b, a).any():
        return -1.0
    return float(min(_point_segments_distance(a, _closed(b)).min(),
                     _point_segments_distance(b, _closed(a)).min()))


# ---------------------------------------------------------------- 형상 조립

GEOMETRY_KEYS = ("main_airfoil", "m", "p", "t", "naca_te", "main_chord", "chord")
NACA_PARAM_KEYS = ("m", "p", "t")


def geometry_params(params: dict) -> dict:
    """케이스 변수 중 형상(메쉬)을 결정하는 것만 추린다."""
    out = {k: params[k] for k in GEOMETRY_KEYS if k in params and params[k] not in ("", None)}
    if all(k in out for k in NACA_PARAM_KEYS):
        out.pop("main_airfoil", None)  # (m, p, t)가 있으면 그것이 형상
    if "naca_te" in out and not _is_naca(out):
        out.pop("naca_te")             # 좌표 파일 형상에는 의미 없음
    return out


def _is_naca(params: dict) -> bool:
    if all(k in params for k in NACA_PARAM_KEYS):
        return True
    spec = str(params.get("main_airfoil", "")).upper().removeprefix("NACA").strip()
    return bool(re.fullmatch(r"\d{4}", spec))


def has_geometry(params: dict) -> bool:
    return "main_airfoil" in params or all(k in params for k in NACA_PARAM_KEYS)


def section_from_params(params: dict, root: Path, name: str = "main") -> Element:
    """케이스 변수에서 단면: (m, p, t) 실수 파라미터 또는 main_airfoil (NACA 코드/좌표 파일)."""
    te = str(params.get("naca_te") or "blunt")
    if all(k in params and params[k] not in ("", None) for k in NACA_PARAM_KEYS):
        return _element(naca4_mpt(params["m"], params["p"], params["t"], te=te), name)
    if "main_airfoil" not in params:
        raise ConfigError("형상 변수가 없습니다: (m, p, t) 또는 main_airfoil 을 지정하세요.")
    return load_section(params["main_airfoil"], root, name, te=te)


def build_geometry(params: dict, root: Path) -> Geometry:
    """단일 익형 형상 (요소 1개: main)."""
    main = section_from_params(params, root, "main")
    info = {"elements": 1, "main_chord": float(params.get("main_chord", 1.0))}
    return Geometry([main], info)
