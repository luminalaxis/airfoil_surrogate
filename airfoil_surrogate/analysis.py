"""학습 결과 분석: 실행 간 비교(afs compare)와 경계 사례 선정(afs edge-cases).

compare  (→ runs/ml/_compare/<이름>.csv)
  여러 실행의 predictions.csv에서 모든 실행에 공통인 케이스만 골라 같은 분할(test, test_ood)·
  같은 조건(all, attached, near_stall, post_stall)·같은 지표(MAE, 90% 예측 구간 포함률)로 다시 계산한다.
  --ood-train-fraction으로 처음 보는 형상 일부를 학습에 넣은 실행과 비교해도 남은 형상끼리만 비교된다.
  기준 = 첫 번째 실행의 같은 모델 (실행을 하나만 주면 그 실행의 첫 번째 모델).
  d_mae_* = MAE − 기준 MAE (음수 = 개선). 모든 모델에 같은 형상 재표집을 쓰는 짝지은 부트스트랩 95% 신뢰구간.

edge-cases  (→ runs/ml/<실행>/edge_cases_<모델>/)
  오차 점수 = |CL 오차| / CL MAE + |log10 CD 오차| / log10 CD MAE (그 모델의 평가 세트 평균 오차 대비 배수의 합)
  범주마다 최악(오차 점수 최대)과 대표(오차 점수가 범주 중앙값에 가장 가까움) 사례를 하나씩 고른다.
    stall_pre      실속 직전: 받음각 범위 안에서 실속하는 형상·Re의 CL 최대 받음각 이하 (near_stall)
    stall_post     실속 직후: CL 최대 받음각 초과 (near_stall)
    attached_test  부착 유동(CL 최대 받음각 − 창 미만), 평가 세트(test)
    attached_ood   부착 유동, 처음 보는 형상(test_ood)
    turbulence     SST와 Transition SST의 정답 차이가 큰 순서로 2개 ([ml] tsst_summary 필요)
  → 총 10개. 같은 형상은 되도록 한 번만 고른다.
  케이스별 그래프: 같은 형상·Re의 CL-α, log10 CD-α (CFD 정답, 모델 예측과 90% 예측 구간, Transition SST)
"""

from __future__ import annotations

import numpy as np

from .config import ConfigError, ProjectConfig, safe_name
from .dataset import MLConfig, _write_csv, load_dataset
from .learn import EVAL_SPLITS, ML_DIR, MODEL_NAMES, PI_TAG, TARGETS, ShapeBootstrap, flow_regime

FINAL_SPLITS = ("test", "test_ood")  # 최종 보고용 (val은 조기 종료에 쓰임)
CONDITIONS = ("all", "attached", "near_stall", "post_stall")
COMPARE_DIR = "_compare"
CATEGORIES = ("stall_pre", "stall_post", "turbulence", "attached_test", "attached_ood")
PICKS = ("typical", "worst", "largest_diff")
N_TURBULENCE = 2
CARRY = ("conv_cl_std", "conv_cd_std", "conv_state_tsst", "cl_tsst", "log10_cd_tsst", "d_cl_tsst", "d_log10_cd_tsst")


def load_predictions(cfg: ProjectConfig, run: str) -> list[dict]:
    path = cfg.output_dir / ML_DIR / run / "predictions.csv"
    if not path.is_file():
        raise ConfigError(f"학습 결과가 없습니다: {path}  (먼저 afs train --name {run})")
    rows = load_dataset(path)
    if not rows:
        raise ConfigError(f"예측 결과가 비어 있습니다: {path}")
    return rows


def models_in(rows: list[dict]) -> list[str]:
    return [m for m in MODEL_NAMES if f"{m}_cl" in rows[0]]


def in_condition(row: dict, cond: str, usable: set[str]) -> bool:
    if row["conv_state"] not in usable:
        return False
    return cond == "all" or flow_regime(row) == cond


# ---------------------------------------------------------------- 실행 간 비교

def _same_label(a: dict, b: dict) -> bool:
    """두 실행에서 같은 케이스의 분할과 정답이 같은지 (사이에 데이터셋을 다시 만들었으면 다를 수 있음)."""
    return a["eval_split"] == b["eval_split"] and all(
        a[t] is not None and b[t] is not None and abs(a[t] - b[t]) <= 1e-9 * max(1.0, abs(a[t])) for t in TARGETS)


def compare(cfg: ProjectConfig, runs: list[str], models: list[str] | None = None, name: str | None = None) -> dict:
    if len(set(runs)) != len(runs):
        raise ConfigError("같은 실행 이름이 두 번 들어갔습니다.")
    ml = MLConfig.from_dict(cfg.ml)
    usable = set(ml.train_states)
    data = {run: {r["case"]: r for r in load_predictions(cfg, run)} for run in runs}
    available = {run: models_in(list(d.values())) for run, d in data.items()}
    wanted = models or [m for m in MODEL_NAMES if any(m in a for a in available.values())]
    entries = [(run, m) for run in runs for m in wanted if m in available[run]]
    if not entries:
        raise ConfigError(f"비교할 모델이 없습니다 (요청: {', '.join(wanted)}).")

    first = data[runs[0]]
    common = set(first).intersection(*(data[run] for run in runs[1:]))
    mismatch = sorted(c for c in common if not all(_same_label(first[c], data[run][c]) for run in runs[1:]))
    bad = set(mismatch)
    cases = [c for c in first if c in common and c not in bad and first[c]["eval_split"] in FINAL_SPLITS]

    def reference(run: str, m: str) -> tuple[str, str] | None:
        ref = (runs[0], m) if len(runs) > 1 else entries[0]
        return ref if ref != (run, m) and ref in entries else None

    out = []
    for split in FINAL_SPLITS:
        for cond in CONDITIONS:
            sel = [c for c in cases if first[c]["eval_split"] == split and in_condition(first[c], cond, usable)]
            if not sel:
                continue
            boot = ShapeBootstrap([first[c]["shape"] for c in sel], f"{split}|{cond}", ml.seed)
            truth = {t: np.array([first[c][t] for c in sel]) for t in TARGETS}
            err, cover, width = {}, {}, {}
            for run, m in entries:
                rows = [data[run][c] for c in sel]
                err[run, m] = {t: np.abs(np.array([r[f"{m}_{t}"] for r in rows]) - truth[t]) for t in TARGETS}
                for t in TARGETS:
                    lo, hi = [r.get(f"{m}_{t}_lo") for r in rows], [r.get(f"{m}_{t}_hi") for r in rows]
                    if all(v is not None for v in lo + hi):
                        lo, hi = np.array(lo), np.array(hi)
                        cover[run, m, t] = ((truth[t] >= lo) & (truth[t] <= hi)).astype(float)
                        width[run, m, t] = float(np.mean(hi - lo))
            for run, m in entries:
                ref = reference(run, m)
                entry = {"split": split, "condition": cond, "n": len(sel), "n_shapes": boot.n_shapes,
                         "run": run, "model": m, "reference": "/".join(ref) if ref else ""}
                for t in TARGETS:
                    entry.update(boot.mean_ci(f"mae_{t}", err[run, m][t]))
                    if ref:
                        entry.update(boot.mean_ci(f"d_mae_{t}", err[run, m][t] - err[ref][t]))
                    else:
                        entry.update(dict.fromkeys((f"d_mae_{t}", f"d_mae_{t}_lo95", f"d_mae_{t}_hi95")))
                for t in TARGETS:
                    if (run, m, t) in cover:
                        entry[f"width{PI_TAG}_{t}"] = width[run, m, t]
                        entry.update(boot.mean_ci(f"cover{PI_TAG}_{t}", cover[run, m, t]))
                out.append(entry)
    if not out:
        raise ConfigError("모든 실행에 공통인 평가 케이스(test, test_ood)가 없습니다.")
    out_dir = cfg.output_dir / ML_DIR / COMPARE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _write_csv(out_dir / f"{safe_name(name or '_vs_'.join(runs))}.csv", out)
    return {"path": path, "rows": out, "entries": entries, "n_cases": len(cases), "mismatch": mismatch,
            "not_common": {run: len(d) - len(common) for run, d in data.items()}}


# ---------------------------------------------------------------- 경계 사례

def edge_cases(cfg: ProjectConfig, run: str, model: str = "mlp", plots: bool = True) -> dict:
    ml = MLConfig.from_dict(cfg.ml)
    usable = set(ml.train_states)
    rows = load_predictions(cfg, run)
    if f"{model}_cl" not in rows[0]:
        raise ConfigError(f"{run}에 {model} 예측이 없습니다 (있는 모델: {', '.join(models_in(rows))}).")
    pool = [r for r in rows if r["eval_split"] in FINAL_SPLITS and r["conv_state"] in usable]
    if not pool:
        raise ConfigError(f"{run}: 평가 세트(test, test_ood)에 케이스가 없습니다.")

    def err(r: dict, t: str) -> float:
        return r[f"{model}_{t}"] - r[t]

    scale = {t: float(np.mean([abs(err(r, t)) for r in pool])) or 1.0 for t in TARGETS}

    def score(r: dict) -> float:
        return sum(abs(err(r, t)) / scale[t] for t in TARGETS)

    def tsst_score(r: dict) -> float | None:
        if r.get("d_cl_tsst") is None or r.get("d_log10_cd_tsst") is None:
            return None
        return abs(r["d_cl_tsst"]) / scale["cl"] + abs(r["d_log10_cd_tsst"]) / scale["log10_cd"]

    top: dict[tuple, float] = {}
    for r in rows:
        if r["conv_state"] in usable:
            key = (r["shape"], r["reynolds"])
            top[key] = max(top.get(key, -np.inf), r["aoa"])

    def stalls(r: dict) -> bool:  # CL 최대 다음 받음각이 범위 안에 있음 = 실속이 관측됨
        return r["aoa_clmax"] is not None and top[r["shape"], r["reynolds"]] > r["aoa_clmax"]

    groups = {
        "stall_pre": [r for r in pool if r["near_stall"] and stalls(r) and r["aoa"] <= r["aoa_clmax"]],
        "stall_post": [r for r in pool if r["near_stall"] and r["aoa"] > r["aoa_clmax"]],
        "attached_test": [r for r in pool if flow_regime(r) == "attached" and r["eval_split"] == "test"],
        "attached_ood": [r for r in pool if flow_regime(r) == "attached" and r["eval_split"] == "test_ood"],
    }
    picks: list[tuple[str, str, dict]] = []
    used_shapes: set[str] = set()
    used_cases: set[str] = set()

    def take(category: str, kind: str, cands: list[dict], key) -> None:
        fresh = [r for r in cands if r["case"] not in used_cases]
        free = [r for r in fresh if r["shape"] not in used_shapes] or fresh
        if free:
            r = min(free, key=key)
            used_shapes.add(r["shape"])
            used_cases.add(r["case"])
            picks.append((category, kind, r))

    # 고르는 순서: 최악 사례 → 난류 모델 민감 → 대표 사례 (같은 형상을 피할 때 앞쪽이 우선)
    for cat, cands in groups.items():
        take(cat, "worst", cands, key=lambda r: -score(r))
    turb = [r for r in rows if r["eval_split"] in EVAL_SPLITS and r["conv_state"] in usable
            and tsst_score(r) is not None]
    for _ in range(N_TURBULENCE):
        take("turbulence", "largest_diff", turb, key=lambda r: -tsst_score(r))
    for cat, cands in groups.items():
        if cands:
            med = float(np.median([score(r) for r in cands]))
            take(cat, "typical", cands, key=lambda r: abs(score(r) - med))
    picks.sort(key=lambda p: (CATEGORIES.index(p[0]), PICKS.index(p[1])))

    carry = [k for k in rows[0] if k in CARRY or k.startswith("yplus")]
    out = []
    for no, (cat, kind, r) in enumerate(picks, start=1):
        e = {"no": no, "category": cat, "pick": kind, "case": r["case"], "shape": r["shape"],
             "family": r["family"], "eval_split": r["eval_split"], "reynolds": r["reynolds"], "aoa": r["aoa"],
             "aoa_clmax": r["aoa_clmax"], "conv_state": r["conv_state"], "error_score": score(r)}
        for t in TARGETS:
            e.update({t: r[t], f"pred_{t}": r[f"{model}_{t}"], f"err_{t}": err(r, t)})
            lo, hi = r.get(f"{model}_{t}_lo"), r.get(f"{model}_{t}_hi")
            if lo is not None and hi is not None:
                e.update({f"{t}_lo": lo, f"{t}_hi": hi, f"in_pi{PI_TAG}_{t}": int(lo <= r[t] <= hi)})
        e["tsst_diff_score"] = tsst_score(r)
        e.update({k: r.get(k) for k in carry})
        out.append(e)

    out_dir = cfg.output_dir / ML_DIR / run / f"edge_cases_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):  # 이전 선정 결과 그래프
        old.unlink()
    if plots:
        from .plots import plot_edge_case

        curves: dict[tuple, list[dict]] = {}
        for r in rows:
            curves.setdefault((r["shape"], r["reynolds"]), []).append(r)
        for e, (cat, kind, r) in zip(out, picks):
            path = out_dir / f"{e['no']:02d}_{cat}_{kind}_{safe_name(r['case'])}.png"
            title = (f"#{e['no']} {cat} / {kind}: {r['case']} [{r['eval_split']}, {r['conv_state']}]\n"
                     f"CL err {e['err_cl']:+.3f}, log10 CD err {e['err_log10_cd']:+.3f}, "
                     f"error score {e['error_score']:.1f} (1 = mean error of {model})")
            plot_edge_case(curves[r["shape"], r["reynolds"]], r, model, path, title, pi_label=f"{PI_TAG}% PI")
            e["figure"] = path.name
    csv_path = _write_csv(out_dir / "edge_cases.csv", out)
    return {"dir": out_dir, "csv": csv_path, "rows": out, "scale": scale,
            "missing": [c for c in CATEGORIES if not any(p[0] == c for p in picks)],
            "has_tsst": bool(turb)}
