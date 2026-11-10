"""CST(Class-Shape Transformation, Kulfan) 형상 계수 피팅과 단면 형상 지표.

윗면/아랫면을 각각

    y(x) = C(x) · Σ_{i=0}^{n-1} A_i · K_i · x^i · (1-x)^(n-1-i)  +  x · y_te
    C(x) = x^0.5 · (1-x)^1.0   (둥근 앞전, 뾰족한 뒷전)

로 표현한다. n = 면당 계수 수(기본 8 → 윗면 8 + 아랫면 8 = 16개). y_te는 뒷전 두께의 절반에 해당하는
면별 뒷전 높이로, 뭉툭한 뒷전(NACA 표준식)을 맞추기 위한 항이다(형상 계수와 별도 특성으로 기록).

좌표는 geometry.normalize와 같이 앞전 = 원점, 시위 = 1, 뒷전 중점이 x축 위에 오도록 맞춘 뒤 피팅한다.
아랫면 계수는 보통 음수다. 대칭 익형이면 아랫면 계수 = −윗면 계수.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import load_section

N1, N2 = 0.5, 1.0


def bernstein(x: np.ndarray, n_coef: int) -> np.ndarray:
    """(len(x), n_coef) 번스타인 기저 K_i x^i (1-x)^(n-1-i)."""
    n = n_coef - 1
    x = np.asarray(x, float)[:, None]
    i = np.arange(n_coef)[None, :]
    k = np.array([math.comb(n, j) for j in range(n_coef)], float)[None, :]
    return k * x**i * (1 - x) ** (n - i)


def class_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x**N1 * (1 - x) ** N2


def split_surfaces(xy: np.ndarray, le_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Selig 순서 좌표를 (윗면, 아랫면)으로 나눈다. 둘 다 앞전 → 뒷전(x 증가) 순서."""
    upper = xy[: le_index + 1][::-1]
    lower = xy[le_index:]
    return upper, lower


@dataclass
class CSTFit:
    upper: np.ndarray   # 윗면 계수 (n,)
    lower: np.ndarray   # 아랫면 계수 (n,)
    te_upper: float     # 윗면 뒷전 높이 [c]
    te_lower: float     # 아랫면 뒷전 높이 [c]
    rms: float          # 피팅 잔차 RMS [c]
    max_err: float      # 피팅 잔차 최대 [c]

    @property
    def n(self) -> int:
        return len(self.upper)

    def evaluate(self, x) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        c = class_function(x)
        b = bernstein(x, self.n)
        return c * (b @ self.upper) + x * self.te_upper, c * (b @ self.lower) + x * self.te_lower

    def coefficients(self) -> np.ndarray:
        return np.concatenate([self.upper, self.lower])

    def features(self, prefix: str = "cst") -> dict[str, float]:
        """예: cst_u0..cst_u7, cst_l0..cst_l7"""
        out = {f"{prefix}_u{i}": float(v) for i, v in enumerate(self.upper)}
        out.update({f"{prefix}_l{i}": float(v) for i, v in enumerate(self.lower)})
        return out


def _fit_surface(pts: np.ndarray, n_coef: int) -> tuple[np.ndarray, float]:
    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = pts[:, 1]
    y_te = float(y[-1])
    a = class_function(x)[:, None] * bernstein(x, n_coef)
    coef, *_ = np.linalg.lstsq(a, y - x * y_te, rcond=None)
    return coef, y_te


def fit_cst(xy: np.ndarray, le_index: int, n_coef: int = 8) -> CSTFit:
    """정규화된 Selig 순서 좌표(normalize 결과)에 면당 n_coef개 CST 계수를 최소제곱으로 맞춘다."""
    if n_coef < 2:
        raise ValueError("CST 계수는 면당 2개 이상이어야 합니다.")
    upper, lower = split_surfaces(xy, le_index)
    au, te_u = _fit_surface(upper, n_coef)
    al, te_l = _fit_surface(lower, n_coef)
    fit = CSTFit(au, al, te_u, te_l, 0.0, 0.0)
    res = []
    for pts, k in ((upper, 0), (lower, 1)):
        y_fit = fit.evaluate(pts[:, 0])[k]
        res.append(pts[:, 1] - y_fit)
    res = np.concatenate(res)
    fit.rms = float(np.sqrt(np.mean(res**2)))
    fit.max_err = float(np.max(np.abs(res)))
    return fit


def fit_section(spec, root: Path, n_coef: int = 8) -> CSTFit:
    """main_airfoil 형식("2412", "airfoils/s1223.dat")의 단면에 CST를 맞춘다."""
    el = load_section(spec, root, "main")
    return fit_cst(el.xy, el.le_index, n_coef)


# ---------------------------------------------------------------- 단면 지표

def _surface_interp(pts: np.ndarray, x: np.ndarray) -> np.ndarray:
    order = np.argsort(pts[:, 0], kind="stable")
    px, py = pts[order, 0], pts[order, 1]
    px, idx = np.unique(px, return_index=True)
    return np.interp(x, px, py[idx])


def section_stats(xy: np.ndarray, le_index: int, n: int = 400) -> dict[str, float]:
    """두께·캠버 지표와 얇은 익형 이론의 무양력 받음각.

    α0 = −(1/π) ∫₀^π (dy_c/dx)(cos θ − 1) dθ,  x = (1 − cos θ)/2
    """
    upper, lower = split_surfaces(xy, le_index)
    theta = np.linspace(0.0, np.pi, n + 1)
    tm = 0.5 * (theta[1:] + theta[:-1])          # 구간 중점 (적분용)
    x_nodes = 0.5 * (1 - np.cos(theta))
    yc_nodes = 0.5 * (_surface_interp(upper, x_nodes) + _surface_interp(lower, x_nodes))
    dyc = np.diff(yc_nodes) / np.diff(x_nodes)   # 구간 중점에서의 기울기
    alpha0 = -np.sum(dyc * (np.cos(tm) - 1) * np.diff(theta)) / np.pi

    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yu, yl = _surface_interp(upper, x), _surface_interp(lower, x)
    t, yc = yu - yl, 0.5 * (yu + yl)
    i_t, i_c = int(np.argmax(t)), int(np.argmax(np.abs(yc)))
    return {
        "thickness": float(t[i_t]),
        "thickness_x": float(x[i_t]),
        "camber": float(yc[i_c]),
        "camber_x": float(x[i_c]),
        "te_thickness": float(upper[-1, 1] - lower[-1, 1]),
        "alpha0_thin_deg": float(np.degrees(alpha0)),
    }
