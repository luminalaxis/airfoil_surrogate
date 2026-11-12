"""케이스별 수렴 상태 판정: converged(수렴) / oscillating(진동) / diverged(발산).

판정 근거 (summary.csv의 conv_* 열로 함께 기록):
  1. diverged  : 솔버 로그에 발산 메시지(Divergence detected, floating point exception 등),
                 리포트 값이 NaN/inf, 또는 |계수|가 max_abs_coef를 넘음
  2. converged : 마지막 window 반복에서 CL·CD 진폭(최대 − 최소)이 허용치 이하이거나,
                 Fluent가 잔차 기준을 만족해 스스로 멈춤("solution is converged")
  incomplete : 발산은 아니지만 정답으로 믿을 수 없는 경우. 실행이 정상 종료되지 않았거나(시간 초과, 종료 코드,
               로그 오류, 중단, 실행 중) 반복 수가 min_iters보다 적음. 학습에는 쓰지 않는다(train_states에 없음).
  3. oscillating: 그 밖의 경우. 값이 주기적으로 흔들리는 경우와 아직 한쪽으로 계속 변하는 경우(drift)를
                 모두 포함한다. conv_reason에 "amplitude"/"drift"로 구분해 남긴다.
  결과 파일이 없으면 no_data.

허용치는 CL은 max(cl_tol, cl_rel_tol·|평균 CL|), CD는 max(cd_tol, cd_rel_tol·평균 CD).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .config import ConfigError

CONVERGED, OSCILLATING, DIVERGED, NO_DATA = "converged", "oscillating", "diverged", "no_data"
INCOMPLETE = "incomplete"
# 이 실행 상태(status.json)의 결과는 반복 도중에 끊긴 값일 수 있어 정답으로 쓰지 않는다
INCOMPLETE_RUN_STATES = ("failed", "interrupted", "running", "unknown")
DIVERGENCE_PATTERNS = ("diverg", "floating point exception")
RESIDUALS = ("continuity", "x-velocity", "y-velocity", "energy", "k", "omega", "epsilon", "nut",
             "intermit", "retheta")
_CONVERGED_PATTERN = re.compile(r"solution is converged", re.IGNORECASE)


@dataclass
class ConvergenceConfig:
    window: int = 200           # 마지막 N 반복으로 판정
    cl_tol: float = 0.005       # CL 진폭 허용치 (절대)
    cl_rel_tol: float = 0.01    # CL 진폭 허용치 (|평균 CL| 대비)
    cd_tol: float = 1e-4        # CD 진폭 허용치 (절대)
    cd_rel_tol: float = 0.02    # CD 진폭 허용치 (평균 CD 대비)
    max_abs_coef: float = 10.0  # 계수 크기가 이보다 크면 발산
    min_iters: int = 50         # 반복 수가 이보다 적으면 incomplete (초기 과도 구간만 있는 결과)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ConvergenceConfig":
        data = data or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"[convergence]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        obj = cls(**data)
        for f in fields(cls):
            v = getattr(obj, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ConfigError(f"[convergence] {f.name}는 0 이상의 숫자여야 합니다: {v!r}")
        return obj


def coefficient_history(headers: list[str], rows: list[list[float]],
                        angle_deg: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    """리포트 파일에서 반복별 (cl, cd) 이력. cl/cd 열이 없으면 cx/cy를 유입 각도로 회전한다."""
    if not rows:
        return None
    lower = [h.lower() for h in headers]
    data = np.asarray(rows, float)
    if "cl" in lower and "cd" in lower:
        return data[:, lower.index("cl")], data[:, lower.index("cd")]
    if "cx" in lower and "cy" in lower and angle_deg is not None:
        a = math.radians(angle_deg)
        cx, cy = data[:, lower.index("cx")], data[:, lower.index("cy")]
        return (cy * math.cos(a) - cx * math.sin(a)), cx * math.cos(a) + cy * math.sin(a)
    return None


def solver_log_info(*log_paths: Path) -> dict:
    """Fluent 콘솔 로그(표준 출력 로그, transcript)에서 마지막 잔차, 수렴 메시지, 발산 메시지를 읽는다.

    Windows에서 -hidden으로 실행하면 표준 출력에는 거의 아무것도 나오지 않으므로 저널의
    /file/start-transcript 파일도 함께 읽는다. 없는 파일은 건너뛴다.
    """
    info: dict = {}
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in log_paths if p.is_file()]
    if not texts:
        return info
    text = "\n".join(texts)
    if _CONVERGED_PATTERN.search(text):
        info["solver_converged"] = True
    columns: list[str] | None = None
    last: dict[str, float] = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        low = line.lower()
        if tokens[0] == "iter" and "continuity" in low:
            columns = tokens[1:]
            continue
        if columns and tokens[0].isdigit():
            values = {}
            for name, tok in zip(columns, tokens[1:]):
                try:
                    values[name] = float(tok)
                except ValueError:
                    break
            if "continuity" in values:
                last = values
        if any(p in low for p in ("divergence detected", "floating point exception")):
            info.setdefault("divergence_message", line.strip())
    for name, value in last.items():
        if name in RESIDUALS:
            info[f"res_{name}"] = value
    return info


def classify(history: tuple[np.ndarray, np.ndarray] | None, cfg: ConvergenceConfig,
             status: dict | None = None, log_info: dict | None = None) -> dict:
    """수렴 상태와 근거 지표를 dict로 반환 (summary.csv 열 이름 그대로)."""
    status, log_info = status or {}, log_info or {}
    out: dict = {}
    message = str(status.get("message", "")).lower()
    diverged_msg = log_info.get("divergence_message") or (
        message if any(p in message for p in DIVERGENCE_PATTERNS) else None)

    if history is None or len(history[0]) == 0:
        if diverged_msg:
            return {"conv_state": DIVERGED, "conv_reason": "solver: " + diverged_msg[:80]}
        return {"conv_state": NO_DATA, "conv_reason": ""}

    cl, cd = history
    tail = slice(-max(2, cfg.window), None)
    cl_w, cd_w = cl[tail], cd[tail]
    out["conv_iters"] = len(cl)
    finite = np.all(np.isfinite(cl_w)) and np.all(np.isfinite(cd_w))
    if finite:
        out.update(
            conv_cl_std=float(np.std(cl_w)), conv_cd_std=float(np.std(cd_w)),
            conv_cl_amp=float(np.ptp(cl_w)), conv_cd_amp=float(np.ptp(cd_w)),
        )
        if len(cl_w) > 2:
            t = np.arange(len(cl_w))
            out["conv_cl_drift"] = float(np.polyfit(t, cl_w, 1)[0] * (len(cl_w) - 1))
    if diverged_msg:
        return {**out, "conv_state": DIVERGED, "conv_reason": "solver: " + diverged_msg[:80]}
    if not finite:
        return {**out, "conv_state": DIVERGED, "conv_reason": "non-finite coefficient"}
    if max(np.max(np.abs(cl_w)), np.max(np.abs(cd_w))) > cfg.max_abs_coef or np.mean(cd_w) <= 0:
        return {**out, "conv_state": DIVERGED, "conv_reason": "coefficient out of range"}

    run_state = status.get("state")
    if run_state in INCOMPLETE_RUN_STATES:
        reason = str(status.get("message", "")).strip()
        return {**out, "conv_state": INCOMPLETE,
                "conv_reason": f"run {run_state}" + (f": {reason[:70]}" if reason else "")}
    if len(cl) < cfg.min_iters:
        return {**out, "conv_state": INCOMPLETE, "conv_reason": f"too few iterations ({len(cl)} < {cfg.min_iters})"}

    cl_lim = max(cfg.cl_tol, cfg.cl_rel_tol * abs(float(np.mean(cl_w))))
    cd_lim = max(cfg.cd_tol, cfg.cd_rel_tol * abs(float(np.mean(cd_w))))
    if out["conv_cl_amp"] <= cl_lim and out["conv_cd_amp"] <= cd_lim:
        return {**out, "conv_state": CONVERGED, "conv_reason": "amplitude within tolerance"}
    if log_info.get("solver_converged"):
        return {**out, "conv_state": CONVERGED, "conv_reason": "solver residual criteria met"}
    drift = abs(out.get("conv_cl_drift", 0.0))
    kind = "drift" if drift > 0.5 * out["conv_cl_amp"] else "amplitude"
    ratio = max(out["conv_cl_amp"] / cl_lim, out["conv_cd_amp"] / cd_lim)
    return {**out, "conv_state": OSCILLATING, "conv_reason": f"{kind} {ratio:.1f}x tolerance"}
