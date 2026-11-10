"""메쉬 독립성(격자 수렴) 확인: 같은 케이스를 메쉬 3단계(케이스 변수 mesh_level)로 풀어 Richardson 외삽과
GCI(Grid Convergence Index)를 계산한다.

절차: Celik et al. (2008), "Procedure for Estimation and Reporting of Uncertainty Due to Discretization
in CFD Applications", J. Fluids Eng. 130(7), 078001.
  - 대표 셀 크기 h = (1/N)^(1/2) (2D, 같은 해석 영역이므로 면적은 상수로 빠짐), 1 = 조밀, 3 = 성김
  - 관측 차수 p: p = |ln|e32/e21| + q(p)| / ln r21,  q(p) = ln((r21^p − s)/(r32^p − s)),  s = sign(e32/e21)
  - 외삽값 φ_ext = (r21^p φ1 − φ2)/(r21^p − 1)
  - GCI_조밀 = 1.25 |(φ1 − φ2)/φ1| / (r21^p − 1),  GCI_보통 = 1.25 |(φ2 − φ3)/φ2| / (r32^p − 1)
  - 점근 구간 확인: GCI_보통 / (r21^p · GCI_조밀) ≈ 1
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import fields
from pathlib import Path

from .cases import Case
from .config import ConfigError, ProjectConfig, safe_name
from .meshing import INFO_FILE, MeshSettings

STUDY_DIR = "mesh_study"
LEVEL_KEY = "mesh_level"


def gci3(phi: tuple[float, float, float], h: tuple[float, float, float]) -> dict:
    """메쉬 3단계 결과(φ1 조밀, φ2 보통, φ3 성김)와 대표 셀 크기 h로 관측 차수, 외삽값, GCI를 계산한다."""
    (f1, f2, f3), (h1, h2, h3) = phi, h
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"r21": r21, "r32": r32}
    if r21 <= 1 or r32 <= 1:
        return {**out, "convergence": "invalid", "note": "셀 수가 단계마다 늘어나야 합니다 (조밀 > 보통 > 성김)"}
    if e21 == 0 and e32 == 0:
        return {**out, "convergence": "exact", "p": math.nan, "phi_ext": f1,
                "gci_fine": 0.0, "gci_medium": 0.0, "err_medium": 0.0, "asymptotic": math.nan}
    if e21 == 0:
        return {**out, "convergence": "flat", "note": "조밀·보통 결과가 같아 차수를 정할 수 없음",
                "p": math.nan, "phi_ext": f1, "gci_fine": 0.0, "err_medium": 0.0}
    ratio = e32 / e21
    s = math.copysign(1.0, ratio)
    if abs(ratio) <= 1:
        # 조밀해질수록 차이가 줄지 않음 → 발산(또는 점근 구간 밖). 차수·외삽값은 의미 없음
        kind = "divergent" if s > 0 else "oscillatory-divergent"
        return {**out, "convergence": kind, "p": math.nan, "phi_ext": math.nan,
                "gci_fine": math.nan, "gci_medium": math.nan,
                "err_medium": abs(e32) / abs(f2) if f2 else math.nan,
                "note": "메쉬를 조밀하게 해도 차이가 줄지 않음: 더 조밀한 단계가 필요하거나 반복 수렴 오차가 큼"}
    p = abs(math.log(abs(ratio))) / math.log(r21)
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(ratio)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = max(p_new, 1e-6)
    a21, a32 = r21 ** p, r32 ** p
    phi_ext = (a21 * f1 - f2) / (a21 - 1)
    gci_fine = 1.25 * abs(e21 / f1) / (a21 - 1) if f1 else math.nan
    gci_medium = 1.25 * abs(e32 / f2) / (a32 - 1) if f2 else math.nan
    return {**out, "convergence": "monotonic" if s > 0 else "oscillatory", "p": p, "phi_ext": phi_ext,
            "gci_fine": gci_fine, "gci_medium": gci_medium,
            "err_medium": abs((phi_ext - f2) / phi_ext) if phi_ext else math.nan,
            "asymptotic": gci_medium / (a21 * gci_fine) if gci_fine else math.nan}


def mesh_dependent_keys(cfg: ProjectConfig) -> set[str]:
    """메쉬 단계에 따라 달라지는 변수: mesh_level, [mesh] 항목 이름(refine 등), 그 값으로 계산한 [derived] 변수."""
    dep = {LEVEL_KEY} | ({f.name for f in fields(MeshSettings)} - {"chord", "reynolds"})
    changed = True
    while changed:
        changed = False
        for key, expression in cfg.derived.items():
            if key in dep:
                continue
            names = {n.id for n in ast.walk(ast.parse(expression, mode="eval")) if isinstance(n, ast.Name)}
            if names & dep:
                dep.add(key)
                changed = True
    return dep


def _quantities(rows: list[dict], wanted: list[str] | None) -> list[str]:
    if wanted:
        return wanted
    keys = ["cl", "cd"]
    for r in rows:
        keys += [k for k in r if (k.startswith("cl-") or k.startswith("cd-")) and k not in keys]
    return keys


def mesh_study(cfg: ProjectConfig, cases: list[Case], quantities: list[str] | None = None,
               plots: bool = True) -> dict:
    from .results import summarize_case

    if not any(LEVEL_KEY in c.params for c in cases):
        raise ConfigError(f"케이스 변수 {LEVEL_KEY}가 없습니다. 메쉬 단계를 [cases.sweep] {LEVEL_KEY} = [1, 2, 3] 처럼 "
                          "주고, [derived]에서 refine 같은 [mesh] 항목을 단계에 따라 바꾸세요 (예제 mesh_study.toml).")
    dep = mesh_dependent_keys(cfg)
    rows = []
    for c in cases:
        row = summarize_case(cfg, c)
        if row.get("state") != "done" or c.mesh is None or not (c.mesh.dir / INFO_FILE).is_file():
            continue
        info = json.loads((c.mesh.dir / INFO_FILE).read_text(encoding="utf-8"))
        row["_cells"] = int(info["cells"])
        row["_key"] = tuple(sorted((k, repr(v)) for k, v in c.params.items() if k not in dep))
        rows.append(row)
    if not rows:
        raise ConfigError("끝난 케이스가 없습니다. 메쉬 단계별 케이스를 먼저 run 하세요.")

    keys = _quantities(rows, quantities)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["_key"], []).append(r)
    varying = [k for k in dict.fromkeys(k for key in groups for k, _ in key)
               if len({dict(key).get(k) for key in groups}) > 1]

    from .geometry import geometry_params

    results, incomplete = [], []
    for key, rs in groups.items():
        params = {k: eval_repr(v) for k, v in key}
        shown = varying
        if "geometry" in rs[0]:
            # 형상 이름에 이미 들어간 변수(받음각 제외)는 라벨에서 뺀다
            geo = set(geometry_params(params)) - {cfg.results.x_param}
            shown = [k for k in varying if k not in geo]
        label = ", ".join(f"{k}={params.get(k)}" for k in shown) or "all"
        if "geometry" in rs[0]:
            label = f"{_strip_level(rs[0]['geometry'], rs[0].get(LEVEL_KEY))} [{label}]"
        levels = sorted(rs, key=lambda r: -r["_cells"])
        if len({r["_cells"] for r in levels}) < 3:
            incomplete.append({"group": label, "levels": len(levels)})
            continue
        fine, medium, coarse = levels[:3]
        h = tuple((1.0 / r["_cells"]) ** 0.5 for r in (fine, medium, coarse))
        for q in keys:
            vals = [r.get(q) for r in (fine, medium, coarse)]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                continue
            g = gci3(tuple(vals), h)
            results.append({
                "group": label, "quantity": q,
                "level_fine": fine.get(LEVEL_KEY), "level_medium": medium.get(LEVEL_KEY),
                "level_coarse": coarse.get(LEVEL_KEY),
                "cells_fine": fine["_cells"], "cells_medium": medium["_cells"], "cells_coarse": coarse["_cells"],
                "phi_fine": vals[0], "phi_medium": vals[1], "phi_coarse": vals[2],
                "conv_state_fine": fine.get("conv_state", ""), "conv_state_medium": medium.get("conv_state", ""),
                "conv_state_coarse": coarse.get("conv_state", ""), **g,
                "case_medium": medium["case"],
            })
    out = cfg.output_dir / STUDY_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gci.csv"
    _write_csv(path, results)
    figs = _plot(results, out) if plots and results else []
    return {"csv": path, "rows": results, "incomplete": incomplete, "plots": figs, "dir": out}


def eval_repr(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return text


def _strip_level(geometry: str, level) -> str:
    from .meshing import _fmt  # mesh_name과 같은 형식 (예: 2.0 → "2")

    suffix = f"_m{_fmt(level)}" if level is not None else ""
    return geometry[: -len(suffix)] if suffix and geometry.endswith(suffix) else geometry


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for r in rows:
        columns += [k for k in r if k not in columns]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns or ["group"])
        w.writeheader()
        w.writerows(rows)


def _plot(results: list[dict], out: Path) -> list[Path]:
    """그룹마다 φ - 셀 크기(h ∝ N^-1/2) 그래프. 외삽값(h = 0)을 함께 표시."""
    from .plots import _save, plt

    saved = []
    by_group: dict[str, list[dict]] = {}
    for r in results:
        by_group.setdefault(r["group"], []).append(r)
    for group, rs in by_group.items():
        rs = [r for r in rs if r["quantity"] in ("cl", "cd")] or rs[:2]
        fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 3.8), squeeze=False)
        for ax, r in zip(axes[0], rs):
            n = [r["cells_fine"], r["cells_medium"], r["cells_coarse"]]
            h = [(n[0] / x) ** 0.5 for x in n]  # 조밀 메쉬 기준 상대 셀 크기
            phi = [r["phi_fine"], r["phi_medium"], r["phi_coarse"]]
            ax.plot(h, phi, "o-", color="tab:blue", label="CFD")
            for x, y, lv in zip(h, phi, (r["level_fine"], r["level_medium"], r["level_coarse"])):
                ax.annotate(f"m{lv}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
            if isinstance(r.get("phi_ext"), float) and math.isfinite(r["phi_ext"]):
                ax.plot([0], [r["phi_ext"]], "s", color="tab:red", label="extrapolated (h=0)")
            gci = r.get("gci_medium")
            note = f"p={r['p']:.2f}, GCI(m{r['level_medium']})={gci * 100:.2f}%" if isinstance(gci, float) and math.isfinite(gci) else r["convergence"]
            ax.set_title(f"{r['quantity']}: {note}", fontsize=9)
            ax.set_xlabel("relative cell size h / h_fine")
            ax.set_xlim(left=-0.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(group, fontsize=9)
        saved.append(_save(fig, out / f"gci_{safe_name(group)[:120]}.png"))
    return saved
