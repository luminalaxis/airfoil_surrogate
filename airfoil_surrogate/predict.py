"""학습된 회귀 모델로 새 에어포일의 CL, CD 예측 (afs predict).

    afs predict --naca 2412 4415                 NACA 4자리 코드
    afs predict --mpt 2.3 4.1 11.7               NACA 실수 파라미터 (m [%c], p [c/10], t [%c])
    afs predict --airfoil my_foil.dat            좌표 파일 (Selig / Lednicer)
    afs predict --random 10                      학습 범위 안에서 무작위 NACA (m, p, t) 10개
    옵션: --alpha 0 8 15 (기본: 학습에 쓴 받음각), --run <runs/ml 아래 이름>, --source xfoil, --out 파일.csv

afs train이 runs/ml/<이름>/models/ 에 저장한 LightGBM(lgbm_*.txt)과 MLP 앙상블(mlp.pt)을 불러 쓴다.
모델마다 예측값(cl, cd)과 90% 예측 구간(*_lo, *_hi)을 낸다.

입력 특성은 학습 때와 같아야 한다 (run.json의 features):
  - mpt 모델  : (m, p, t). 좌표 파일을 넣으면 최대 두께·최대 캠버·캠버 위치로 (m, p, t)를 근사한다 (NACA 4자리가
                아닌 형상은 근사일 뿐이므로 note 열에 표시)
  - cst 모델  : 형상 좌표에서 CST 계수를 피팅 (임의 형상 가능, 단 학습 형상과 다른 계열이면 외삽)
학습 데이터 범위를 벗어난 입력(외삽)과 학습에 없던 받음각(받음각 사이 보간)은 note 열에 표시한다.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import ConfigError, ProjectConfig
from .cst import section_stats
from .dataset import MLConfig, shape_features
from .geometry import NACA_PARAM_KEYS, section_from_params
from .learn import ML_DIR, MODELS_DIR, PI_TAG, TARGETS, LGBMModel, MLPEnsemble

SYMMETRIC_CAMBER = 5e-4  # 최대 캠버가 이보다 작으면 [c] 대칭익으로 본다 (m = p = 0)


def default_run(cfg: ProjectConfig, source: str = "fluent") -> str:
    ml = MLConfig.from_dict(cfg.ml)
    return f"{ml.features}_h64x64" + ("" if source == "fluent" else f"_{source}")


def load_run(cfg: ProjectConfig, run: str) -> tuple[dict, dict]:
    """(run.json 내용, {모델 이름: 모델})."""
    run_dir = cfg.output_dir / ML_DIR / run
    info_path, models_dir = run_dir / "run.json", run_dir / MODELS_DIR
    if not info_path.is_file():
        have = sorted(p.parent.name for p in (cfg.output_dir / ML_DIR).glob("*/run.json"))
        raise ConfigError(f"학습 결과가 없습니다: {run_dir}  (먼저 afs train"
                          + (f". 있는 결과: {', '.join(have)} → --run 으로 지정" if have else "") + ")")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cols = info["features"]
    models = {}
    if (models_dir / f"lgbm_{TARGETS[0]}_mean.txt").is_file():
        models["lgbm"] = LGBMModel.load(models_dir, cols)
    if (models_dir / "mlp.pt").is_file():
        models["mlp"] = MLPEnsemble.load(models_dir, cols)
    if not models:
        raise ConfigError(f"저장된 모델이 없습니다: {models_dir}  (afs train 을 다시 실행하세요. "
                          "이전 버전으로 학습한 결과에는 모델 파일이 없습니다.)")
    return info, models


# ---------------------------------------------------------------- 입력 형상

def _naca_shape(m: float, p: float, t: float, label: str | None = None) -> dict:
    return {"label": label or f"m{m:g}_p{p:g}_t{t:g}", "geom": {"m": m, "p": p, "t": t}, "note": []}


def shapes_from_args(cfg: ProjectConfig, info: dict, naca=(), mpt=(), airfoils=(), n_random: int = 0,
                     seed: int = 0) -> list[dict]:
    te = (cfg.constants or {}).get("naca_te", "blunt")
    shapes = []
    for code in naca:
        code = str(code).upper().removeprefix("NACA").strip()
        if len(code) != 4 or not code.isdigit():
            raise ConfigError(f"--naca 는 4자리 코드여야 합니다: {code!r}")
        shapes.append(_naca_shape(int(code[0]), int(code[1]), int(code[2:]), f"naca{code}"))
    for m, p, t in mpt:
        shapes.append(_naca_shape(float(m), float(p), float(t)))
    for path in airfoils:
        shapes.append({"label": Path(path).stem, "geom": {"main_airfoil": str(Path(path).resolve())}, "note": []})
    if n_random:
        shapes += random_shapes(info, n_random, seed)
    for s in shapes:
        if "m" in s["geom"]:
            s["geom"]["naca_te"] = te
    if not shapes:
        raise ConfigError("예측할 형상이 없습니다: --naca, --mpt, --airfoil, --random 중 하나를 지정하세요.")
    return shapes


def random_shapes(info: dict, n: int, seed: int = 0) -> list[dict]:
    """학습 데이터 범위 안에서 무작위 NACA (m, p, t). mpt 모델이 아니면 예제 기본 범위를 쓴다."""
    rng = np.random.default_rng(seed)
    fr = info.get("feature_ranges", {})
    lo_hi = {"m": fr.get("m", [0.0, 6.0]), "p": fr.get("p", [2.0, 6.0]), "t": fr.get("t", [6.0, 18.0])}
    lo_hi["p"] = [max(lo_hi["p"][0], 1.0), max(lo_hi["p"][1], 1.0)]  # 캠버가 있는 형상의 p는 0이 될 수 없음
    out = []
    for i in range(n):
        m, p, t = (round(float(rng.uniform(*lo_hi[k])), 2) for k in NACA_PARAM_KEYS)
        if m == 0:
            p = 0.0
        out.append(_naca_shape(m, p, t, f"random{i + 1:02d}_m{m:g}_p{p:g}_t{t:g}"))
    return out


def _mpt_from_section(el) -> tuple[float, float, float]:
    """좌표에서 NACA 4자리 파라미터 근사: 최대 두께, 최대 캠버와 그 위치."""
    st = section_stats(el.xy, el.le_index)
    if abs(st["camber"]) < SYMMETRIC_CAMBER:
        return 0.0, 0.0, 100 * st["thickness"]
    return 100 * abs(st["camber"]), 10 * st["camber_x"], 100 * st["thickness"]


def shape_feature_row(shape: dict, cols: list[str], root: Path) -> dict:
    """형상 하나의 입력 특성 (받음각·Re 제외)."""
    geom = shape["geom"]
    row: dict = {}
    if any(c in NACA_PARAM_KEYS for c in cols):
        if all(k in geom for k in NACA_PARAM_KEYS):
            row.update({k: float(geom[k]) for k in NACA_PARAM_KEYS})
        else:
            el = section_from_params(geom, root)
            m, p, t = _mpt_from_section(el)
            row.update(m=m, p=p, t=t)
            shape["note"].append(f"좌표를 NACA 4자리로 근사 (m {m:.2f}, p {p:.2f}, t {t:.2f})")
    orders = sorted({int(c.split("_")[0][3:]) // 2 for c in cols if c.startswith("cst")})
    if orders:
        feats, _ = shape_features(geom, root, orders)
        row.update(feats)
    else:
        el = section_from_params(geom, root)  # 형상이 유효한지 확인 (잘못된 m, p, t는 여기서 오류)
        row["alpha0_thin_deg"] = section_stats(el.xy, el.le_index)["alpha0_thin_deg"]
    return row


# ---------------------------------------------------------------- 예측

def _notes(row: dict, cols: list[str], info: dict) -> list[str]:
    notes = []
    for c, (lo, hi) in info.get("feature_ranges", {}).items():
        if c not in cols or c in ("aoa",):
            continue
        v, tol = row[c], 1e-9 + 1e-6 * max(abs(lo), abs(hi))
        if v < lo - tol or v > hi + tol:
            notes.append(f"{c} {v:.3g} 학습 범위 [{lo:.3g}, {hi:.3g}] 밖 (외삽)")
    alphas = info.get("train_alphas") or []
    if alphas:
        a = row["aoa"]
        if a < min(alphas) - 1e-9 or a > max(alphas) + 1e-9:
            notes.append(f"α {a:g}° 학습 범위 [{min(alphas):g}, {max(alphas):g}] 밖 (외삽)")
        elif len(alphas) <= 20 and min(abs(a - x) for x in alphas) > 1e-6:
            notes.append(f"α {a:g}°는 학습에 없던 받음각 (학습: {', '.join(f'{x:g}' for x in alphas)}°) → 보간")
    return notes


def predict(cfg: ProjectConfig, run: str, shapes: list[dict], alphas: list[float] | None = None,
            reynolds: float | None = None) -> dict:
    info, models = load_run(cfg, run)
    cols = info["features"]
    alphas = [float(a) for a in (alphas or info.get("train_alphas") or [])]
    if not alphas:
        raise ConfigError("받음각을 지정하세요: --alpha 0 8 15")
    if reynolds is None:
        trained = info.get("reynolds") or []
        reynolds = cfg.flow.Re if cfg.flow is not None else (trained[0] if trained else None)
    if "log10_re" in cols and reynolds is None:
        raise ConfigError("이 모델은 Re가 입력입니다: --re 로 지정하세요.")

    rows = []
    for shape in shapes:
        base = shape_feature_row(shape, cols, cfg.root)
        for a in alphas:
            row = {**base, "aoa": a, "shape": shape["label"]}
            if reynolds is not None:
                row.update(reynolds=float(reynolds), log10_re=math.log10(reynolds))
            row["_notes"] = shape["note"] + _notes(row, cols, info)
            rows.append(row)
    trained_re = info.get("reynolds") or []
    re_note = None
    if reynolds is not None and "log10_re" not in cols and trained_re and \
            abs(reynolds - trained_re[0]) > 0.01 * trained_re[0]:
        re_note = (f"이 모델은 Re {trained_re[0]:,.0f} 한 조건으로만 학습했습니다. "
                   f"Re {reynolds:,.0f}에 대한 예측이 아닙니다.")

    out_rows = [{"shape": r["shape"], **{k: r[k] for k in NACA_PARAM_KEYS if k in r}, "alpha": r["aoa"]}
                for r in rows]
    for name, model in models.items():
        p = model.predict(rows)
        for i, o in enumerate(out_rows):
            o[f"{name}_cl"] = float(p["cl"][i])
            o[f"{name}_cd"] = float(10 ** p["log10_cd"][i])
            o[f"{name}_cl_lo{PI_TAG}"], o[f"{name}_cl_hi{PI_TAG}"] = float(p["cl_lo"][i]), float(p["cl_hi"][i])
            o[f"{name}_cd_lo{PI_TAG}"] = float(10 ** p["log10_cd_lo"][i])
            o[f"{name}_cd_hi{PI_TAG}"] = float(10 ** p["log10_cd_hi"][i])
    if len(models) > 1:  # 두 모델의 차이가 크면 믿기 어려운 예측
        names = list(models)
        for o in out_rows:
            o["cl_model_spread"] = float(np.ptp([o[f"{n}_cl"] for n in names]))
            o["cd_model_spread_rel"] = float(np.ptp([o[f"{n}_cd"] for n in names])
                                             / np.mean([o[f"{n}_cd"] for n in names]))
    for o, r in zip(out_rows, rows):
        o["note"] = "; ".join(r["_notes"])
    return {"rows": out_rows, "models": list(models), "run": run, "features": cols, "alphas": alphas,
            "reynolds": reynolds, "re_note": re_note, "info": info}


def write_predictions(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for r in rows:
        columns += [k for k in r if k not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return path
