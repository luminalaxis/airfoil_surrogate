"""학습한 모델로 새 익형·조건의 CL, CD 예측 (afs predict).

입력 CSV (utf-8, 엑셀 CSV UTF-8 가능)
  airfoil   익형: NACA 4자리 코드("2412") 또는 좌표 파일 경로(project.toml 기준, 예: "airfoils/s1223.dat")
            (main_airfoil 열 이름도 받음)
  aoa       받음각 [deg]   ([results] x_param 이름도 받음)
  reynolds  레이놀즈수     ([ml] re_key 이름도 받음)
  그 밖의 열은 출력에 그대로 붙는다 (예: name).

출력 CSV (기본: runs/ml/<실행>/predict_<입력 파일 이름>.csv)
  <모델>_cl, <모델>_cd (= 10^log10_cd), <모델>_log10_cd, 90% 예측 구간 *_lo/*_hi (구간이 있는 모델만)
  extrapolation  학습 데이터 범위 밖인 입력 (aoa, log10_re, 형상 계수 이름). 비어 있지 않으면 외삽이므로
                 예측과 구간을 믿기 어렵다. n_extrapolated = 범위 밖 입력 수.
  cst_max_err    형상의 CST 피팅 최대 오차 [c] (크면 형상 계수가 형상을 제대로 나타내지 못함)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


from .config import ConfigError, ProjectConfig
from .dataset import MAX_FIT_ERROR, MLConfig, _write_csv, shape_features
from .learn import ML_DIR, MODELS_DIR, TARGETS, apply_calibration, load_models

AIRFOIL_KEYS = ("airfoil", "main_airfoil")


def _pick(row: dict, keys: tuple[str, ...], what: str, line: int) -> str:
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k].strip()
    raise ConfigError(f"입력 {line}행: {what} 열({' 또는 '.join(dict.fromkeys(keys))})이 비어 있습니다.")


def _number(text: str, what: str, line: int) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"입력 {line}행: {what} 값이 숫자가 아닙니다: {text!r}") from None
    if not math.isfinite(value):
        raise ConfigError(f"입력 {line}행: {what} 값이 유한한 숫자가 아닙니다: {text!r}")
    return value


def read_inputs(path: Path, cfg: ProjectConfig, ml: MLConfig) -> list[dict]:
    if not path.is_file():
        raise ConfigError(f"입력 CSV를 찾을 수 없습니다: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            raw = [{(k or "").strip(): v for k, v in r.items() if isinstance(v, str)} for r in csv.DictReader(f)]
    except UnicodeDecodeError as exc:
        raise ConfigError(f"입력 CSV를 UTF-8로 읽을 수 없습니다: {path} ({exc})") from None
    rows = []
    for line, r in enumerate(raw, start=2):
        if not any(v.strip() for v in r.values()):
            continue
        airfoil = _pick(r, AIRFOIL_KEYS, "익형", line)
        aoa = _number(_pick(r, ("aoa", cfg.results.x_param), "받음각", line), "받음각", line)
        re = _number(_pick(r, ("reynolds", ml.re_key), "레이놀즈수", line), "레이놀즈수", line)
        if re <= 0:
            raise ConfigError(f"입력 {line}행: 레이놀즈수는 0보다 커야 합니다: {re}")
        extra = {k: v for k, v in r.items() if k not in {*AIRFOIL_KEYS, "aoa", "reynolds", cfg.results.x_param, ml.re_key}}
        rows.append({**extra, "airfoil": airfoil, "aoa": aoa, "reynolds": re, "log10_re": math.log10(re)})
    if not rows:
        raise ConfigError(f"입력 CSV에 예측할 행이 없습니다: {path}")
    return rows


def predict(cfg: ProjectConfig, run: str, input_path: Path, models: list[str] | None = None,
            out_path: Path | None = None) -> dict:
    ml = MLConfig.from_dict(cfg.ml)
    run_dir = cfg.output_dir / ML_DIR / run
    meta, loaded = load_models(run_dir, models)
    feature_set: str = meta["feature_set"]
    n_coef = int(feature_set.removeprefix("cst")) // 2

    rows = read_inputs(input_path, cfg, ml)
    shapes: dict[str, dict] = {}
    warnings = []
    for spec in dict.fromkeys(r["airfoil"] for r in rows):
        feats, fits = shape_features(spec, cfg.root, [n_coef])
        shapes[spec] = {**feats, "cst_max_err": fits[n_coef].max_err}
        if fits[n_coef].max_err > MAX_FIT_ERROR:
            warnings.append(f"{spec}: CST 최대 피팅 오차 {fits[n_coef].max_err:.4f}c (형상 계수가 형상을 잘 나타내지 못함)")
    for r in rows:
        r.update(shapes[r["airfoil"]])

    ranges = meta.get("train_ranges", {})
    n_out = 0
    for r in rows:
        tol = {c: 1e-9 * max(1.0, abs(lo), abs(hi)) for c, (lo, hi) in ranges.items()}
        outside = [c for c, (lo, hi) in ranges.items() if not lo - tol[c] <= r[c] <= hi + tol[c]]
        r["extrapolation"] = ",".join(outside)
        r["n_extrapolated"] = len(outside)
        n_out += bool(outside)
    if n_out:
        warnings.append(f"{n_out}개 행이 학습 데이터 범위 밖 입력을 포함합니다 (extrapolation 열). "
                        "이런 외삽 예측과 구간은 믿기 어렵습니다.")

    shape_cols = set(next(iter(shapes.values())))
    extra = [k for k in rows[0] if k not in shape_cols and k not in
             ("airfoil", "aoa", "reynolds", "log10_re", "extrapolation", "n_extrapolated")]
    out_rows = [{k: r.get(k) for k in (*extra, "airfoil", "aoa", "reynolds", "extrapolation", "n_extrapolated",
                                       "cst_max_err", "thickness", "camber", "alpha0_thin_deg")} for r in rows]
    for name, (model, cal) in loaded.items():
        preds = apply_calibration(model.predict(rows), cal)
        for i, o in enumerate(out_rows):
            for t in TARGETS:
                o[f"{name}_{t}"] = float(preds[t][i])
                if f"{t}_lo" in preds:
                    o[f"{name}_{t}_lo"], o[f"{name}_{t}_hi"] = float(preds[f"{t}_lo"][i]), float(preds[f"{t}_hi"][i])
            o[f"{name}_cd"] = 10 ** o[f"{name}_log10_cd"]
            if f"{name}_log10_cd_lo" in o:
                o[f"{name}_cd_lo"], o[f"{name}_cd_hi"] = 10 ** o[f"{name}_log10_cd_lo"], 10 ** o[f"{name}_log10_cd_hi"]
            o[f"{name}_l/d"] = o[f"{name}_cl"] / o[f"{name}_cd"]

    out_path = out_path or run_dir / f"predict_{input_path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, out_rows)
    return {"path": out_path, "rows": out_rows, "models": list(loaded), "warnings": warnings,
            "n_extrapolated": n_out, "models_dir": run_dir / MODELS_DIR}

