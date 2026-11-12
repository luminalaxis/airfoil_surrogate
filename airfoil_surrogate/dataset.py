"""회귀 모델 학습용 데이터셋 만들기 (afs dataset).

케이스 결과(summary와 같은 값) + 형상 특성(CST 계수, 두께·캠버) + 형상 단위 분할을 합쳐
runs/dataset/ 에 저장한다.

  dataset.csv   케이스 1건 = 1행. 입력(CST 계수, aoa, log10_re), 목표(cl, log10_cd),
                수렴 상태(conv_state), 평균에 쓴 반복 수(avg_iters), y+(yplus_max, yplus_avg),
                실속 부근 여부(near_stall), 분할(split).
                [ml] tsst_summary가 있으면 같은 케이스의 Transition SST 정답과 차이
                (cl_tsst, log10_cd_tsst, d_cl_tsst, d_log10_cd_tsst = T-SST − SST)
  airfoils.csv  형상별 CST 계수(면당 8개 → cst16_*, 면당 4개 → cst8_*), 피팅 오차, 두께·캠버,
                얇은 익형 이론 무양력각 α0, 분할
  split.json    형상 → 분할 (train / val / test / test_ood). 한 번 만들면 고정되고,
                나중에 추가된 형상만 새로 배정한다. 다시 섞으려면 --resplit.
  cst_fits.png  피팅 오차가 큰 형상의 원래 좌표와 CST 재구성 비교

분할 규칙
  - 같은 형상의 모든 받음각·Re 케이스는 같은 분할에 들어간다 (형상 단위 분할).
  - holdout_families 계열은 학습·검증에 쓰지 않고 test_ood(처음 보는 형상)로만 쓴다.
  - 나머지는 family별로 split 비율(기본 70/15/15)에 맞춰 train / val / test로 나눈다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .cases import Case
from .config import ConfigError, ProjectConfig
from .cst import CSTFit, fit_cst, section_stats
from .geometry import NACA_PARAM_KEYS, geometry_params, section_from_params
from .results import summarize_case

DATASET_DIR = "dataset"
SPLITS = ("train", "val", "test", "test_ood")
MAX_FIT_ERROR = 0.005  # CST 최대 피팅 오차 경고 기준 [c]


@dataclass
class MLConfig:
    holdout_families: list = field(default_factory=list)
    split: list = field(default_factory=lambda: [0.70, 0.15, 0.15])
    seed: int = 2026
    near_stall_window: float = 2.0
    train_states: list = field(default_factory=lambda: ["converged", "oscillating"])
    cst_orders: list = field(default_factory=lambda: [8, 4])  # 면당 계수 수 → cst16_*, cst8_*
    shape_key: str = "airfoil"     # 형상 이름 변수 (없으면 main_airfoil)
    family_key: str = "family"
    re_key: str = "reynolds"
    tsst_summary: str = ""         # Transition SST로 다시 푼 케이스의 summary.csv (예: runs_tsst/summary.csv)
    require_converged: bool = False  # true면 converged == False인 케이스를 학습·평가에서 뺀다 (건수는 보고)

    @classmethod
    def from_dict(cls, data: dict | None) -> "MLConfig":
        data = data or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"[ml]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        obj = cls(**data)
        if len(obj.split) != 3 or abs(sum(obj.split) - 1) > 1e-6 or min(obj.split) < 0:
            raise ConfigError(f"[ml] split은 합이 1인 비율 3개여야 합니다: {obj.split}")
        return obj


def dataset_dir(cfg: ProjectConfig) -> Path:
    return cfg.output_dir / DATASET_DIR


def shape_id(params: dict, ml: MLConfig) -> str:
    value = params.get(ml.shape_key) or params.get("main_airfoil")
    if value in (None, ""):
        raise ConfigError(f"형상 이름 변수 {ml.shape_key!r} 또는 main_airfoil이 케이스에 없습니다.")
    return str(value)


# ---------------------------------------------------------------- 형상 특성

def shape_features(geom: dict, root: Path, orders: list[int]) -> tuple[dict, dict[int, CSTFit]]:
    """geom: 형상 변수 ((m, p, t[, naca_te]) 또는 main_airfoil)."""
    el = section_from_params(geom, root, "main")
    feats = section_stats(el.xy, el.le_index)
    fits = {}
    for n in orders:
        fit = fit_cst(el.xy, el.le_index, n)
        fits[n] = fit
        feats.update(fit.features(f"cst{2 * n}"))
        feats[f"cst{2 * n}_rms"] = fit.rms
        feats[f"cst{2 * n}_max_err"] = fit.max_err
    return feats, fits


# ---------------------------------------------------------------- 분할

def assign_splits(shapes: dict[str, str], ml: MLConfig, previous: dict[str, str] | None = None) -> dict[str, str]:
    """shapes: {형상: family}. previous에 있는 형상은 그대로 두고 새 형상만 배정한다."""
    previous = dict(previous or {})
    rng = np.random.default_rng(ml.seed)
    out: dict[str, str] = {}
    by_family: dict[str, list[str]] = {}
    for name, fam in sorted(shapes.items()):
        if fam in ml.holdout_families:
            out[name] = "test_ood"
        elif name in previous and previous[name] in SPLITS[:3]:
            out[name] = previous[name]
        else:
            by_family.setdefault(fam, []).append(name)
    for fam, names in sorted(by_family.items()):
        # 이 family에서 이미 배정된 수를 고려해 목표 비율에 모자란 분할부터 채운다
        counts = {s: sum(1 for n, f in shapes.items() if f == fam and out.get(n) == s) for s in SPLITS[:3]}
        order = rng.permutation(len(names))
        for i in order:
            total = sum(counts.values()) + 1
            deficit = {s: ml.split[k] * total - counts[s] for k, s in enumerate(SPLITS[:3])}
            pick = max(SPLITS[:3], key=lambda s: (deficit[s], s == "train"))
            out[names[i]] = pick
            counts[pick] += 1
    return out


def _load_split(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("assignments", {})


# ---------------------------------------------------------------- 데이터셋

def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def yplus_columns(summary_row: dict) -> dict:
    """리포트 파일의 y+ 열(Fluent 리포트 이름 yplus-max, yplus-avg 등) → yplus_max, yplus_avg."""
    out = {}
    for key, value in summary_row.items():
        name = str(key).lower().replace("-", "_").replace(" ", "_")
        if name.startswith("yplus") and isinstance(value, float) and math.isfinite(value):
            out[name] = value
    return out


def load_tsst(path: Path, usable: set[str]) -> dict[str, dict]:
    """Transition SST 결과(summary.csv) → {케이스: {conv_state_tsst, cl_tsst, log10_cd_tsst}}.

    수렴 상태가 usable이 아니거나 값이 없으면 상태만 남긴다.
    """
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            state = raw.get("conv_state") or ""
            if not raw.get("case") or state in ("", "no_data"):
                continue
            entry = {"conv_state_tsst": state}
            try:
                cl, cd = float(raw.get("cl", "")), float(raw.get("cd", ""))
            except ValueError:
                cl = cd = math.nan
            if state in usable and math.isfinite(cl) and math.isfinite(cd) and cd > 0:
                entry.update(cl_tsst=cl, log10_cd_tsst=math.log10(cd))
            out[raw["case"]] = entry
    return out


def join_tsst(rows: list[dict], tsst: dict[str, dict], usable: set[str]) -> list[dict]:
    """SST 정답과 Transition SST 정답의 차이(d_cl_tsst, d_log10_cd_tsst = T-SST − SST). 비교된 행 목록 반환."""
    joined = []
    for r in rows:
        t = tsst.get(r["case"])
        if t is None:
            continue
        r.update(t)
        if "cl_tsst" in t and r["conv_state"] in usable and r["cl"] is not None and r["log10_cd"] is not None:
            r["d_cl_tsst"] = t["cl_tsst"] - r["cl"]
            r["d_log10_cd_tsst"] = t["log10_cd_tsst"] - r["log10_cd"]
            joined.append(r)
    return joined


def mark_near_stall(rows: list[dict], window: float, usable: set[str]) -> None:
    """형상·Re별 CL 최대 받음각(aoa_clmax)과 그 ± window 안인지(near_stall)를 기록한다.

    CL 최대는 학습에 쓸 수 있는 수렴 상태(usable)의 케이스로만 정한다.
    """
    best: dict[tuple, tuple[float, float]] = {}
    for r in rows:
        if r["conv_state"] in usable and r["cl"] is not None:
            key = (r["shape"], r["reynolds"])
            if key not in best or r["cl"] > best[key][0]:
                best[key] = (r["cl"], r["aoa"])
    for r in rows:
        b = best.get((r["shape"], r["reynolds"]))
        r["aoa_clmax"] = b[1] if b else ""
        r["near_stall"] = int(b is not None and abs(r["aoa"] - b[1]) <= window + 1e-9)


def build_dataset(cfg: ProjectConfig, cases: list[Case], resplit: bool = False) -> dict:
    ml = MLConfig.from_dict(cfg.ml)
    out_dir = dataset_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes: dict[str, dict] = {}
    for c in cases:
        sid = shape_id(c.params, ml)
        info = shapes.setdefault(sid, {"shape": sid, "main_airfoil": c.params.get("main_airfoil", sid),
                                       "family": str(c.params.get(ml.family_key, "")),
                                       "_geom": geometry_params(c.params)})
        for k in NACA_PARAM_KEYS:  # (m, p, t) 입력 피처
            if k in c.params:
                info[k] = float(c.params[k])

    split_path = out_dir / "split.json"
    previous = {} if resplit else _load_split(split_path)
    splits = assign_splits({k: v["family"] for k, v in shapes.items()}, ml, previous)
    moved = [k for k, v in previous.items() if k in splits and splits[k] != v]
    split_path.write_text(json.dumps({
        "seed": ml.seed, "ratios": ml.split, "holdout_families": ml.holdout_families,
        "assignments": dict(sorted(splits.items())),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    warnings, fits = [], {}
    for sid, info in shapes.items():
        info["split"] = splits[sid]
        try:
            feats, fits[sid] = shape_features(info["_geom"], cfg.root, ml.cst_orders)
        except ConfigError as exc:
            warnings.append(f"{sid}: 형상을 읽지 못함 ({exc})")
            info["error"] = str(exc)
            continue
        info.update(feats)
        main_n = max(ml.cst_orders)
        if feats[f"cst{2 * main_n}_max_err"] > MAX_FIT_ERROR:
            warnings.append(f"{sid}: CST 최대 피팅 오차 {feats[f'cst{2 * main_n}_max_err']:.4f}c "
                            "(좌표 점이 성기거나 앞전이 뾰족할 수 있음, cst_fits.png 확인)")

    rows = []
    for c in cases:
        r = summarize_case(cfg, c)
        sid = shape_id(c.params, ml)
        info = shapes[sid]
        if "error" in info:
            continue
        cl, cd = _num(r.get("cl")), _num(r.get("cd"))
        re = _num(c.params.get(ml.re_key))
        aoa = _num(c.params.get(cfg.results.x_param))
        if re is None or aoa is None:
            raise ConfigError(f"{c.name}: 받음각({cfg.results.x_param}) 또는 Re({ml.re_key}) 값이 숫자가 아닙니다.")
        row = {
            "case": c.name, "shape": sid, "family": info["family"], "split": info["split"],
            "aoa": aoa, "reynolds": re, "log10_re": math.log10(re),
            "state": r.get("state", ""), "conv_state": r.get("conv_state", "no_data"),
            "conv_cl_std": r.get("conv_cl_std", ""), "conv_cd_std": r.get("conv_cd_std", ""),
            "cl": cl, "cd": cd,
            "log10_cd": math.log10(cd) if cd is not None and cd > 0 else None,
            "avg_iters": r.get("avg_iters", ""),
            "converged": int(bool(r.get("converged")) and r.get("state") == "done"),
            **yplus_columns(r),
        }
        row.update({k: info[k] for k in NACA_PARAM_KEYS if k in info})
        if cl is None or row["log10_cd"] is None:
            row["conv_state"] = row["conv_state"] if row["conv_state"] == "diverged" else "no_data"
        row.update({k: v for k, v in info.items()
                    if k.startswith("cst") or k in ("thickness", "camber", "alpha0_thin_deg", "te_thickness")})
        rows.append(row)
    usable = set(ml.train_states)
    mark_near_stall(rows, ml.near_stall_window, usable)

    labeled = [r for r in rows if r["conv_state"] != "no_data"]
    tsst = None
    if ml.tsst_summary:
        tsst_path = (cfg.root / ml.tsst_summary).resolve()
        if tsst_path.is_file():
            tsst = _tsst_stats(join_tsst(labeled, load_tsst(tsst_path, usable), usable))
        else:
            warnings.append(f"Transition SST 결과가 없습니다: {tsst_path}  "
                            "(Transition SST 프로젝트에서 run 후 collect)")
    yplus = [r["yplus_max"] for r in labeled if r.get("yplus_max") is not None and r["conv_state"] in usable]
    if labeled and not any(k.startswith("yplus") for r in labeled for k in r):
        warnings.append("y+ 리포트가 없습니다. 베이스 케이스에 yplus-max, yplus-avg 리포트를 추가하세요 "
                        "(docs/FLUENT_BASE_CASE_AIRFOIL.md 9단계).")
    _write_csv(out_dir / "dataset.csv", labeled)
    _write_csv(out_dir / "airfoils.csv", [{k: v for k, v in s.items() if k != "_geom"}
                                          for s in sorted(shapes.values(), key=lambda s: (s["split"], s["shape"]))])
    plot = _plot_worst_fits(shapes, fits, cfg.root, out_dir / "cst_fits.png")

    counts = {s: {"shapes": sum(1 for v in shapes.values() if v["split"] == s),
                  "cases": sum(1 for r in labeled if r["split"] == s)} for s in SPLITS}
    states: dict[str, int] = {}
    for r in rows:
        states[r["conv_state"]] = states.get(r["conv_state"], 0) + 1
    if moved:
        warnings.append(f"분할이 바뀐 형상 {len(moved)}개 (holdout_families 변경 등): {', '.join(moved[:5])}")
    return {"dir": out_dir, "rows": labeled, "counts": counts, "states": states,
            "near_stall": sum(r["near_stall"] for r in labeled), "warnings": warnings, "plot": plot,
            "yplus": {"n": len(yplus), "median": float(np.median(yplus)), "max": float(np.max(yplus))} if yplus else None,
            "tsst": tsst}


def _tsst_stats(joined: list[dict]) -> dict:
    """SST vs Transition SST 차이 요약 (Re별)."""
    stats = {"n": len(joined), "by_re": {}}
    for re in sorted({r["reynolds"] for r in joined}):
        sub = [r for r in joined if r["reynolds"] == re]
        dcl = np.abs([r["d_cl_tsst"] for r in sub])
        dcd = np.abs([r["d_log10_cd_tsst"] for r in sub])
        stats["by_re"][re] = {"n": len(sub), "median_abs_d_cl": float(np.median(dcl)), "max_abs_d_cl": float(dcl.max()),
                              "max_d_cl_case": sub[int(dcl.argmax())]["case"],
                              "median_abs_d_log10_cd": float(np.median(dcd)), "max_abs_d_log10_cd": float(dcd.max()),
                              "max_d_log10_cd_case": sub[int(dcd.argmax())]["case"]}
    return stats


def _write_csv(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def _plot_worst_fits(shapes: dict, fits: dict[str, dict[int, CSTFit]], root: Path, path: Path,
                     n: int = 6) -> Path | None:
    if not fits:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = max(next(iter(fits.values())))
    worst = sorted(fits, key=lambda s: -fits[s][order].max_err)[:n]
    fig, axes = plt.subplots(len(worst), 1, figsize=(8, 1.8 * len(worst)), squeeze=False)
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 300)))
    for ax, sid in zip(axes[:, 0], worst):
        el = section_from_params(shapes[sid]["_geom"], root, "main")
        ax.plot(el.xy[:, 0], el.xy[:, 1], "k.", ms=2, label="coordinates")
        for k, style in zip(sorted(fits[sid], reverse=True), ("-", "--")):
            yu, yl = fits[sid][k].evaluate(x)
            ax.plot(x, yu, style, lw=1, color=f"C{k % 10}", label=f"CST {2 * k}")
            ax.plot(x, yl, style, lw=1, color=f"C{k % 10}")
        ax.set_aspect("equal")
        ax.set_title(f"{sid}  max err {fits[sid][order].max_err:.4f}c", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def load_dataset(path: Path) -> list[dict]:
    """dataset.csv를 읽는다 (숫자 열은 float, 빈 값은 None)."""
    if not path.is_file():
        raise ConfigError(f"데이터셋이 없습니다: {path}  (먼저 afs dataset 실행)")
    text_cols = {"case", "shape", "family", "split", "state", "conv_state", "conv_state_tsst"}
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k in text_cols:
                    row[k] = v
                elif v in ("", None):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows
