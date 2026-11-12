"""해석툴(XFOIL) 해석과 ANSYS Fluent 해석 비교.

  1. afs xfoil            케이스 목록 전체(형상 × α)를 XFOIL로 해석 → <runs>/xfoil_<re_mode>_<목록>/results.csv
                          (팀 공통 결과 스키마, turb_model = "xfoil", 수렴 못 한 α는 converged = False로 기록)
  2. afs compare-solvers  같은 case_id끼리 XFOIL vs ANSYS(Fluent) 비교 → <Fluent 결과 폴더>/compare_xfoil/
                          compare.csv (케이스별 ΔCL, ΔCD/CD, ΔCM), summary.json (α별·전체), parity_*.png
  3. afs dataset --source fluent|xfoil → afs train --source ... 로 MLP·LightGBM 학습

Re, 코드 길이, 형상, α는 Fluent와 같은 [flow]·케이스 목록에서 가져온다.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .config import ConfigError, ProjectConfig
from .results import RESULT_COLUMNS, _write_csv
from .validate import XfoilConfig, _from_dict, read_results, run_xfoil

COMPARE_DIR = "compare_xfoil"


def xfoil_dir(cfg: ProjectConfig) -> Path:
    if cfg.flow is None:
        raise ConfigError("XFOIL 해석에는 [flow] 설정(Re, 코드 길이)이 필요합니다.")
    return (cfg.base_output_dir or cfg.output_dir) / f"xfoil_{cfg.flow.re_mode}_{cfg.flow.effective_case_list}"


def xfoil_config(cfg: ProjectConfig) -> XfoilConfig:
    return _from_dict(XfoilConfig, (cfg.validation or {}).get("xfoil"), "validation.xfoil")


def run_xfoil_cases(cfg: ProjectConfig, cases, log=print) -> dict:
    """형상마다 XFOIL을 한 번 실행해 그 형상의 모든 α를 푼다. 결과는 공통 스키마 results.csv."""
    from .geometry import section_from_params

    x = xfoil_config(cfg)
    re = cfg.flow.Re
    shapes: dict[str, list] = {}
    for c in cases:
        shapes.setdefault(str(c.params.get("shape_id") or c.group or c.name), []).append(c)
    out_dir = xfoil_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, polars = [], []
    for i, (sid, cs) in enumerate(sorted(shapes.items()), start=1):
        el = section_from_params(cs[0].params, cfg.root)
        alphas = sorted({float(c.params[cfg.results.x_param]) for c in cs})
        start = time.monotonic()
        points = run_xfoil(el.xy, sid, re, alphas, x)
        dt = time.monotonic() - start
        by_alpha = {round(p["alpha"], 3): p for p in points}
        missing = [a for a in alphas if round(a, 3) not in by_alpha]
        log(f"  [{i}/{len(shapes)}] {sid:<14} α {', '.join(f'{a:g}' for a in alphas)}  "
            + ("수렴" if not missing else f"미수렴 α {', '.join(f'{a:g}' for a in missing)}") + f"  {dt:.1f}s")
        for c in cs:
            a = float(c.params[cfg.results.x_param])
            p = by_alpha.get(round(a, 3))
            row = {k: "" for k in RESULT_COLUMNS}
            row.update(case_id=c.params.get("case_id", c.name), m=c.params.get("m", ""), p=c.params.get("p", ""),
                       t=c.params.get("t", ""), alpha=a, Re=re, turb_model="xfoil", mesh_mode="",
                       converged=p is not None, n_iter=x.iterations, runtime_s=round(dt / len(cs), 3),
                       assignee=c.params.get("assignee", ""), solver_version=f"XFOIL (Ncrit {x.ncrit:g})")
            if p is not None:
                row.update(CL=p["CL"], CD=p["CD"], CM=p["CM"])
                polars.append({"case_id": row["case_id"], "shape_id": sid, **p})
            rows.append(row)
    path = _write_csv(out_dir / "results.csv", rows, list(RESULT_COLUMNS))
    _write_csv(out_dir / "polars.csv", polars)
    return {"path": path, "rows": len(rows), "converged": sum(1 for r in rows if r["converged"]),
            "shapes": len(shapes), "Re": re}


def compare_solvers(cfg: ProjectConfig, fluent: Path | None = None, xfoil: Path | None = None,
                    plots: bool = True) -> dict:
    """같은 case_id의 Fluent(ANSYS)와 XFOIL 결과 비교. 차이 = XFOIL − Fluent (Fluent 기준 상대 오차)."""
    fluent = fluent or cfg.output_dir / cfg.results.results_file
    xfoil = xfoil or xfoil_dir(cfg) / "results.csv"
    for path, how in ((fluent, "afs run → afs collect"), (xfoil, "afs xfoil")):
        if not path.is_file():
            raise ConfigError(f"결과가 없습니다: {path}  (먼저 {how})")
    f_rows = {r["case_id"]: r for r in read_results(fluent)}
    x_rows = {r["case_id"]: r for r in read_results(xfoil)}
    common = sorted(set(f_rows) & set(x_rows))
    if not common:
        raise ConfigError("Fluent와 XFOIL 결과에 같은 case_id가 없습니다 (같은 케이스 목록으로 실행했는지 확인).")
    rows = []
    for cid in common:
        f, x = f_rows[cid], x_rows[cid]
        row = {"case_id": cid, "m": f.get("m"), "p": f.get("p"), "t": f.get("t"), "alpha": f.get("alpha"),
               "converged_fluent": f.get("converged"), "converged_xfoil": x.get("converged")}
        for q in ("CL", "CD", "CM"):
            a, b = f.get(q), x.get(q)
            row[f"{q}_fluent"], row[f"{q}_xfoil"] = a, b
            if isinstance(a, float) and isinstance(b, float):
                row[f"d{q}"] = b - a
                row[f"d{q}_rel"] = (b - a) / a if abs(a) > 1e-12 else math.nan
        row["both_converged"] = bool(f.get("converged") and x.get("converged") and "dCL" in row and "dCD" in row)
        rows.append(row)
    good = [r for r in rows if r["both_converged"]]
    summary = {"fluent": str(fluent), "xfoil": str(xfoil), "cases": len(rows), "both_converged": len(good),
               "fluent_not_converged": sum(1 for r in rows if not r["converged_fluent"]),
               "xfoil_not_converged": sum(1 for r in rows if not r["converged_xfoil"]),
               "all": _stats(good), "by_alpha": {}}
    for a in sorted({r["alpha"] for r in good}):
        summary["by_alpha"][f"{a:g}"] = _stats([r for r in good if r["alpha"] == a])
    out = cfg.output_dir / COMPARE_DIR
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "compare.csv", rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figs = _plot(good, out) if plots and good else []
    return {"dir": out, "summary": summary, "rows": rows, "plots": figs}


def _stats(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    if not rows:
        return out
    for q in ("CL", "CD", "CM"):
        pairs = [(r[f"{q}_fluent"], r[f"{q}_xfoil"]) for r in rows
                 if isinstance(r.get(f"{q}_fluent"), float) and isinstance(r.get(f"{q}_xfoil"), float)]
        if not pairs:
            continue
        a, b = np.array(pairs).T
        d = b - a
        rel = np.abs(d[np.abs(a) > 1e-12] / a[np.abs(a) > 1e-12])
        ss = float(np.sum((a - a.mean()) ** 2))
        out[q] = {"mae": float(np.mean(np.abs(d))), "max_abs": float(np.max(np.abs(d))),
                  "bias": float(np.mean(d)), "mean_abs_rel": float(np.mean(rel)) if rel.size else math.nan,
                  "r2": 1 - float(np.sum(d**2)) / ss if ss > 0 else math.nan}
    return out


def _plot(rows: list[dict], out: Path) -> list[Path]:
    from .plots import _save, plt

    alphas = sorted({r["alpha"] for r in rows})
    figs = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, q in zip(axes, ("CL", "CD", "CM")):
        for k, a in enumerate(alphas):
            sub = [r for r in rows if r["alpha"] == a and isinstance(r.get(f"{q}_xfoil"), float)]
            ax.scatter([r[f"{q}_fluent"] for r in sub], [r[f"{q}_xfoil"] for r in sub], s=14,
                       color=f"C{k}", label=f"α = {a:g}°")
        vals = [v for r in rows for v in (r.get(f"{q}_fluent"), r.get(f"{q}_xfoil")) if isinstance(v, float)]
        if vals:
            lo, hi = min(vals), max(vals)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"{q} (ANSYS Fluent)")
        ax.set_ylabel(f"{q} (XFOIL)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("XFOIL vs ANSYS Fluent (same case)", fontsize=10)
    figs.append(_save(fig, out / "parity.png"))
    return figs
