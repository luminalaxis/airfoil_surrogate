"""파이프라인 검증 (afs validate).

1단계: Airfoil2Vec 대비 (Re 4.66×10⁶, afs validate airfoil2vec)
  - Hugging Face의 metadata 파일(parquet)만 받는다 (전체 423 GB는 받지 않음). 로컬 파일 경로도 가능.
  - 대칭 변환: 우리 NACA abcd의 +α 결과를 그쪽 NACAabcd_inverted(위아래 뒤집은 형상)의 −α와 비교.
      CL_ours = −CL_theirs,  CD_ours = CD_theirs,  CM_ours = −CM_theirs
  - 판정 기준을 미리 고정: |ΔCL| ≤ max(cl_rel_tol·|CL_ref|, cl_abs_tol), |ΔCD| ≤ max(cd_rel_tol·CD_ref, cd_abs_tol)
    (기본 CL ±3%, CD ±10%. α = 0 대칭익처럼 CL_ref ≈ 0이면 절대 허용치 cl_abs_tol을 쓴다)
  - 같은 케이스를 light / fine 메쉬로 각각 돌린 결과가 있으면 둘 다 판정하고, light − fine 차이(경량 메쉬 오차)도 낸다.

2단계: 저Re 물리 검증 (Re 10⁵, afs validate xfoil)
  - XFOIL을 subprocess로 자동 호출 (같은 Re, Ncrit = 9) → 검증용 형상 × α 극선
  - SST와 SST-LM(Transition SST) 결과를 각각 XFOIL과 비교 → 난류 모델 선정 근거

결과: <output_dir>/validation/*.csv, *.json
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .config import ConfigError, ProjectConfig

VALIDATION_DIR = "validation"


def _from_dict(cls, data: dict | None, where: str):
    data = dict(data or {})
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"[{where}]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
    return cls(**data)


def validation_dir(cfg: ProjectConfig) -> Path:
    out = (cfg.base_output_dir or cfg.output_dir) / VALIDATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_results(path: Path) -> list[dict]:
    """results.csv (팀 공통 스키마) → 숫자는 float, converged는 bool."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k == "converged":
                    row[k] = str(v).strip().lower() in ("true", "1")
                elif k in ("case_id", "turb_model", "mesh_mode", "assignee", "solver_version"):
                    row[k] = v
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = None if v in ("", None) else v
            rows.append(row)
    return rows


def _code(row: dict) -> str:
    from .geometry import naca_code

    return naca_code(row["m"], row["p"], row["t"])


def _within(err: float, ref: float, rel: float, abs_tol: float) -> bool:
    return abs(err) <= max(rel * abs(ref), abs_tol)


# ================================================================ 1단계: Airfoil2Vec

COLUMN_CANDIDATES = {
    "name": ("airfoil", "airfoil_name", "name", "foil", "shape", "geometry", "naca"),
    "alpha": ("alpha", "aoa", "angle_of_attack", "alpha_deg", "angle"),
    "cl": ("cl", "c_l", "lift_coefficient", "lift"),
    "cd": ("cd", "c_d", "drag_coefficient", "drag"),
    "cm": ("cm", "c_m", "moment_coefficient", "cm_c4", "moment"),
    "reynolds": ("re", "reynolds", "reynolds_number", "rey"),
}


@dataclass
class Airfoil2VecConfig:
    metadata: str = ""                  # 로컬 metadata 파일 (.parquet / .csv). 있으면 이것을 씀
    repo_id: str = ""                   # Hugging Face 데이터셋 저장소 (metadata 파일 하나만 내려받음)
    filename: str = "metadata.parquet"  # 저장소 안의 metadata 파일 경로
    columns: dict = field(default_factory=dict)  # 열 이름 지정 (name, alpha, cl, cd, cm, reynolds). 비우면 자동 탐색
    name_format: str = "NACA{code}_inverted"     # 그쪽 형상 이름 ({code} = 우리 형상의 NACA 4자리)
    inverted: bool = True               # 뒤집힌 형상과 비교 (CL, CM 부호 반전, α 부호 반전)
    reynolds: float = 4.66e6
    re_rel_tol: float = 0.02            # 그쪽 Re 열이 있으면 이 범위 안의 행만
    alpha_tol: float = 0.05             # α 일치 허용 [deg]
    cl_rel_tol: float = 0.03            # 판정: CL ±3%
    cl_abs_tol: float = 0.01            #       (CL_ref ≈ 0일 때) |ΔCL| ≤ 0.01
    cd_rel_tol: float = 0.10            #       CD ±10%
    cd_abs_tol: float = 1e-4
    cm_abs_tol: float = 0.01            # CM은 참고 판정 (절대 오차)


def _load_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigError("parquet 파일을 읽으려면 pyarrow가 필요합니다: pip install -e \".[validation]\"") from exc
    return pq.read_table(path).to_pylist()


def fetch_metadata(cfg: ProjectConfig, a2v: Airfoil2VecConfig) -> Path:
    """metadata 파일 경로. 로컬 파일이 없으면 Hugging Face에서 그 파일 하나만 받는다."""
    if a2v.metadata:
        path = (cfg.root / a2v.metadata).resolve()
        if path.is_file():
            return path
        if not a2v.repo_id:
            raise ConfigError(f"Airfoil2Vec metadata 파일이 없습니다: {path}")
    if not a2v.repo_id:
        raise ConfigError("[validation.airfoil2vec]에 metadata(로컬 경로) 또는 repo_id(Hugging Face)를 지정하세요.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ConfigError("Hugging Face에서 받으려면 huggingface_hub가 필요합니다: pip install -e \".[validation]\" "
                          "(또는 metadata 파일을 직접 받아 [validation.airfoil2vec] metadata에 경로 지정)") from exc
    local_dir = cfg.root / "validation_data"
    path = Path(hf_hub_download(repo_id=a2v.repo_id, filename=a2v.filename, repo_type="dataset",
                                local_dir=str(local_dir)))
    return path


def _resolve_columns(rows: list[dict], given: dict) -> dict[str, str | None]:
    if not rows:
        raise ConfigError("Airfoil2Vec metadata가 비어 있습니다.")
    available = {k.lower(): k for k in rows[0]}
    out: dict[str, str | None] = {}
    for key, cands in COLUMN_CANDIDATES.items():
        if key in given:
            if given[key] not in rows[0]:
                raise ConfigError(f"[validation.airfoil2vec] columns.{key} = {given[key]!r} 열이 없습니다. "
                                  f"있는 열: {', '.join(rows[0])}")
            out[key] = given[key]
            continue
        out[key] = next((available[c] for c in cands if c in available), None)
    missing = [k for k in ("name", "alpha", "cl", "cd") if out[k] is None]
    if missing:
        raise ConfigError(f"Airfoil2Vec metadata에서 {', '.join(missing)} 열을 찾지 못했습니다. "
                          f"[validation.airfoil2vec] columns에 지정하세요. 있는 열: {', '.join(rows[0])}")
    return out


def load_reference(path: Path, a2v: Airfoil2VecConfig) -> dict[str, list[dict]]:
    """형상 이름(소문자) → [{alpha, cl, cd, cm}]. Re 열이 있으면 a2v.reynolds 근처만."""
    rows = _load_table(path)
    col = _resolve_columns(rows, a2v.columns)
    out: dict[str, list[dict]] = {}
    for r in rows:
        try:
            if col["reynolds"] and r.get(col["reynolds"]) not in (None, ""):
                re_ = float(r[col["reynolds"]])
                if abs(re_ - a2v.reynolds) > a2v.re_rel_tol * a2v.reynolds:
                    continue
            entry = {"alpha": float(r[col["alpha"]]), "cl": float(r[col["cl"]]), "cd": float(r[col["cd"]]),
                     "cm": float(r[col["cm"]]) if col["cm"] and r.get(col["cm"]) not in (None, "") else math.nan}
        except (TypeError, ValueError):
            continue
        out.setdefault(str(r[col["name"]]).strip().lower(), []).append(entry)
    return out


def compare_reference(ours: list[dict], ref: dict[str, list[dict]], a2v: Airfoil2VecConfig) -> list[dict]:
    rows = []
    sign = -1.0 if a2v.inverted else 1.0
    for r in ours:
        if r.get("m") is None or r.get("alpha") is None:
            continue
        code = _code(r)
        name = a2v.name_format.format(code=code).lower()
        target = sign * r["alpha"]
        cands = [e for e in ref.get(name, []) if abs(e["alpha"] - target) <= a2v.alpha_tol]
        row = {"mesh_mode": r.get("mesh_mode", ""), "case_id": r["case_id"], "naca": code, "alpha": r["alpha"],
               "reference": name, "converged": r.get("converged")}
        if not cands:
            rows.append({**row, "matched": False})
            continue
        e = min(cands, key=lambda e: abs(e["alpha"] - target))
        cl_ref, cd_ref, cm_ref = sign * e["cl"], e["cd"], sign * e["cm"]
        row.update(matched=True, alpha_ref=e["alpha"], CL=r.get("CL"), CL_ref=cl_ref, CD=r.get("CD"), CD_ref=cd_ref,
                   CM=r.get("CM"), CM_ref=cm_ref)
        if isinstance(r.get("CL"), float) and isinstance(r.get("CD"), float):
            dcl, dcd = r["CL"] - cl_ref, r["CD"] - cd_ref
            row.update(dCL=dcl, dCL_rel=dcl / cl_ref if cl_ref else math.nan,
                       dCD=dcd, dCD_rel=dcd / cd_ref if cd_ref else math.nan,
                       CL_pass=_within(dcl, cl_ref, a2v.cl_rel_tol, a2v.cl_abs_tol),
                       CD_pass=_within(dcd, cd_ref, a2v.cd_rel_tol, a2v.cd_abs_tol))
            if isinstance(r.get("CM"), float) and math.isfinite(cm_ref):
                row.update(dCM=r["CM"] - cm_ref, CM_pass=abs(r["CM"] - cm_ref) <= a2v.cm_abs_tol)
            row["pass"] = bool(row["CL_pass"] and row["CD_pass"])
        rows.append(row)
    return rows


def _summary(rows: list[dict], group: str) -> dict:
    out = {}
    for g in sorted({r[group] for r in rows}):
        sub = [r for r in rows if r[group] == g]
        judged = [r for r in sub if "pass" in r]
        out[g] = {
            "cases": len(sub), "matched": sum(1 for r in sub if r.get("matched")), "judged": len(judged),
            "pass": sum(1 for r in judged if r["pass"]),
            "CL_pass": sum(1 for r in judged if r["CL_pass"]), "CD_pass": sum(1 for r in judged if r["CD_pass"]),
            "mean_abs_dCL": _mean([abs(r["dCL"]) for r in judged]),
            "mean_abs_dCD_rel": _mean([abs(r["dCD_rel"]) for r in judged if math.isfinite(r["dCD_rel"])]),
            "max_abs_dCD_rel": _max([abs(r["dCD_rel"]) for r in judged if math.isfinite(r["dCD_rel"])]),
            "not_converged": sum(1 for r in sub if r.get("converged") is False),
        }
    return out


def _mean(v):
    return float(np.mean(v)) if v else math.nan


def _max(v):
    return float(np.max(v)) if v else math.nan


def mesh_mode_difference(by_mode: dict[str, list[dict]]) -> list[dict]:
    """같은 케이스의 light − fine 차이 (경량 메쉬 오차). fine을 기준으로 상대 오차."""
    if "light" not in by_mode or "fine" not in by_mode:
        return []
    fine = {r["case_id"]: r for r in by_mode["fine"]}
    out = []
    for r in by_mode["light"]:
        f = fine.get(r["case_id"])
        if f is None:
            continue
        row = {"case_id": r["case_id"], "alpha": r.get("alpha"), "n_cells_light": r.get("n_cells"),
               "n_cells_fine": f.get("n_cells"), "converged_light": r.get("converged"), "converged_fine": f.get("converged")}
        for q in ("CL", "CD", "CM"):
            a, b = r.get(q), f.get(q)
            if isinstance(a, float) and isinstance(b, float):
                row[f"{q}_light"], row[f"{q}_fine"], row[f"d{q}"] = a, b, a - b
                row[f"d{q}_rel"] = (a - b) / b if b else math.nan
        out.append(row)
    return out


def _mode_results(cfg: ProjectConfig, re_mode: str, case_list: str, turb: str | None = None,
                  mesh_modes=("light", "fine")) -> dict[str, tuple[Path, list[dict]]]:
    out = {}
    for mm in mesh_modes:
        over = {"re_mode": re_mode, "mesh_mode": mm, "case_list": case_list}
        if turb:
            over["turbulence_model"] = turb
        try:
            sub = cfg.with_overrides(**over)
        except ConfigError:
            continue
        path = sub.output_dir / sub.results.results_file
        if path.is_file():
            out[mm] = (path, read_results(path))
    return out


def validate_airfoil2vec(cfg: ProjectConfig, results: list[Path] | None = None, metadata: str | None = None) -> dict:
    if cfg.flow is None:
        raise ConfigError("검증에는 [flow] 설정이 필요합니다.")
    a2v = _from_dict(Airfoil2VecConfig, (cfg.validation or {}).get("airfoil2vec"), "validation.airfoil2vec")
    if metadata:
        a2v.metadata = metadata
    if results:
        inputs = {}
        for p in results:
            rows = read_results(p)
            mode = rows[0].get("mesh_mode", p.parent.name) if rows else p.parent.name
            inputs[f"{mode}" if mode not in inputs else f"{mode}:{p}"] = (p, rows)
    else:
        inputs = _mode_results(cfg, "validation", "validation")
    if not inputs:
        raise ConfigError("검증 결과(results.csv)가 없습니다. 먼저\n"
                          "  afs --re-mode validation --mesh-mode light run\n"
                          "  afs --re-mode validation --mesh-mode fine run\n"
                          "을 실행하거나 --results로 파일을 지정하세요.")
    meta_path = fetch_metadata(cfg, a2v)
    ref = load_reference(meta_path, a2v)
    rows = []
    for mode, (_, ours) in inputs.items():
        for r in ours:
            r.setdefault("mesh_mode", mode)
        rows += compare_reference(ours, ref, a2v)
    out = validation_dir(cfg)
    _write(out / "airfoil2vec_report.csv", rows)
    diff = mesh_mode_difference({m: rs for m, (_, rs) in inputs.items()})
    if diff:
        _write(out / "mesh_light_vs_fine.csv", diff)
    summary = {"metadata": str(meta_path), "criteria": {"CL": f"±{a2v.cl_rel_tol:.0%} (최소 {a2v.cl_abs_tol})",
                                                         "CD": f"±{a2v.cd_rel_tol:.0%} (최소 {a2v.cd_abs_tol})"},
               "by_mesh_mode": _summary(rows, "mesh_mode"), "light_vs_fine": _diff_summary(diff),
               "settings": asdict(a2v)}
    (out / "airfoil2vec_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    return {"rows": rows, "summary": summary, "dir": out, "inputs": {m: str(p) for m, (p, _) in inputs.items()},
            "reference_shapes": len(ref)}


def _diff_summary(diff: list[dict]) -> dict:
    if not diff:
        return {}
    out = {"cases": len(diff)}
    for q in ("CL", "CD", "CM"):
        d = [abs(r[f"d{q}"]) for r in diff if f"d{q}" in r]
        rel = [abs(r[f"d{q}_rel"]) for r in diff if f"d{q}_rel" in r and math.isfinite(r[f"d{q}_rel"])]
        out[q] = {"mean_abs": _mean(d), "max_abs": _max(d), "mean_abs_rel": _mean(rel), "max_abs_rel": _max(rel)}
    return out


def validate_mesh_modes(cfg: ProjectConfig) -> dict:
    """현재 Re 모드·케이스 목록에서 light와 fine 결과의 차이 (afs validate mesh)."""
    if cfg.flow is None:
        raise ConfigError("검증에는 [flow] 설정이 필요합니다.")
    inputs = _mode_results(cfg, cfg.flow.re_mode, cfg.flow.effective_case_list, cfg.flow.turbulence_model)
    diff = mesh_mode_difference({m: rs for m, (_, rs) in inputs.items()})
    if not diff:
        raise ConfigError("light와 fine 결과가 둘 다 있는 케이스가 없습니다 (--mesh-mode light / fine 로 각각 run).")
    out = validation_dir(cfg)
    path = out / f"mesh_light_vs_fine_{cfg.flow.re_mode}_{cfg.flow.effective_case_list}.csv"
    _write(path, diff)
    return {"path": path, "summary": _diff_summary(diff)}


# ================================================================ 2단계: XFOIL

@dataclass
class XfoilConfig:
    executable: str = "xfoil"
    ncrit: float = 9.0
    iterations: int = 200
    panels: int = 200
    mach: float = 0.0
    timeout: float = 60.0      # 형상 하나 [s]
    n_points: int = 160        # XFOIL에 넘기는 좌표 점 수 (면당)
    cl_abs_tol: float = 0.05   # 모델 비교 표의 참고 판정
    cd_rel_tol: float = 0.20
    alpha_step: float = 1.0    # 요청한 α까지 이 간격으로 차례로 풀어 간다 (높은 α 수렴용, 0이면 요청한 α만)


def marching_alphas(alphas: list[float], step: float) -> list[float]:
    """0°에서 요청한 α까지 step 간격으로 차례로 푸는 α 목록 (요청한 α 포함). XFOIL은 앞 해를 초기값으로 쓴다."""
    want = sorted({float(a) for a in alphas})
    if step <= 0:
        return want
    seq = set(want)
    for a in want:
        k = 1
        while k * step < abs(a) - 1e-9:
            seq.add(math.copysign(k * step, a))
            k += 1
    return sorted(seq)


def xfoil_script(coord_file: str, polar_file: str, re: float, alphas: list[float], x: XfoilConfig) -> str:
    lines = ["PLOP", "G F", "", f"LOAD {coord_file}", "PPAR", f"N {x.panels}", "", "", "OPER",
             f"VISC {re:.6g}", f"MACH {x.mach:g}", "VPAR", f"N {x.ncrit:g}", "", f"ITER {x.iterations}",
             "PACC", polar_file, ""]
    # 0°에서 멀어지는 순서로 풀어야 앞 해를 초기값으로 써서 잘 수렴한다
    for a in sorted(alphas, key=lambda v: (abs(v), v)):
        lines.append(f"ALFA {a:g}")
    lines += ["PACC", "", "QUIT", ""]
    return "\n".join(lines)


def parse_polar(path: Path) -> list[dict]:
    """XFOIL PACC 극선 파일: '------' 줄 다음부터 alpha CL CD CDp CM Top_Xtr Bot_Xtr."""
    if not path.is_file():
        return []
    rows, started = [], False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("------"):
            started = True
            continue
        if not started:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            v = [float(p) for p in parts[:7]]
        except ValueError:
            continue
        rows.append({"alpha": v[0], "CL": v[1], "CD": v[2], "CDp": v[3], "CM": v[4],
                     "xtr_top": v[5] if len(v) > 5 else math.nan, "xtr_bot": v[6] if len(v) > 6 else math.nan})
    return rows


def run_xfoil(xy: np.ndarray, name: str, re: float, alphas: list[float], x: XfoilConfig) -> list[dict]:
    exe = shutil.which(x.executable) or (x.executable if Path(x.executable).is_file() else None)
    if exe is None:
        raise ConfigError(f"XFOIL 실행 파일을 찾지 못했습니다: {x.executable!r}. "
                          "[validation.xfoil] executable에 경로를 지정하세요.")
    with tempfile.TemporaryDirectory(prefix="afs_xfoil_") as tmp:
        tmp = Path(tmp)
        lines = [name] + [f"{px:.7f} {py:.7f}" for px, py in xy]
        (tmp / "foil.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        script = xfoil_script("foil.dat", "polar.txt", re, marching_alphas(alphas, x.alpha_step), x)
        try:
            subprocess.run([exe], input=script, text=True, cwd=tmp, capture_output=True, timeout=x.timeout)
        except subprocess.TimeoutExpired:
            pass  # 수렴 못 한 α에서 멈추는 경우가 있어 그때까지 저장된 극선은 쓴다
        want = [float(a) for a in alphas]
        return [r for r in parse_polar(tmp / "polar.txt") if any(abs(r["alpha"] - a) < 1e-3 for a in want)]


def _shapes_and_alphas(cases) -> dict[str, dict]:
    shapes: dict[str, dict] = {}
    for c in cases:
        sid = str(c.params.get("shape_id") or c.params.get("main_airfoil") or c.name)
        s = shapes.setdefault(sid, {"params": c.params, "alphas": set()})
        s["alphas"].add(float(c.params["alpha"]))
    return shapes


def validate_xfoil(cfg: ProjectConfig, models=("kOmegaSST", "kOmegaSSTLM")) -> dict:
    """검증용 형상 × α를 XFOIL로 풀고, SST / SST-LM CFD 결과와 비교한다 (production Re)."""
    from .cases import build_cases
    from .geometry import section_from_params

    if cfg.flow is None:
        raise ConfigError("검증에는 [flow] 설정이 필요합니다.")
    x = _from_dict(XfoilConfig, (cfg.validation or {}).get("xfoil"), "validation.xfoil")
    base = cfg.with_overrides(re_mode="production", case_list="validation", turbulence_model="kOmegaSST")
    re = base.flow.Re
    shapes = _shapes_and_alphas(build_cases(base))
    polars = []
    for sid, s in sorted(shapes.items()):
        el = section_from_params(s["params"], cfg.root)
        for r in run_xfoil(el.xy, sid, re, sorted(s["alphas"]), x):
            polars.append({"shape_id": sid, "naca": _code(s["params"]) if "m" in s["params"] else "", "Re": re,
                           "ncrit": x.ncrit, **r})
    out = validation_dir(cfg)
    _write(out / "xfoil_polars.csv", polars)
    xf = {(p["shape_id"], round(p["alpha"], 3)): p for p in polars}

    compare, by_model = [], {}
    for turb in models:
        try:
            sub = cfg.with_overrides(re_mode="production", case_list="validation", turbulence_model=turb)
        except ConfigError as exc:
            by_model[turb] = {"error": str(exc)}
            continue
        path = sub.output_dir / sub.results.results_file
        if not path.is_file():
            by_model[turb] = {"error": f"결과 없음: {path}"}
            continue
        sid_of = {c.params["case_id"]: str(c.params.get("shape_id")) for c in build_cases(sub)}
        for r in read_results(path):
            p = xf.get((sid_of.get(r["case_id"], ""), round(float(r["alpha"]), 3)))
            row = {"turb_model": turb, "case_id": r["case_id"], "shape_id": sid_of.get(r["case_id"], ""),
                   "alpha": r["alpha"], "converged": r.get("converged"), "xfoil": p is not None}
            if p is not None and isinstance(r.get("CL"), float) and isinstance(r.get("CD"), float):
                row.update(CL=r["CL"], CL_xfoil=p["CL"], dCL=r["CL"] - p["CL"], CD=r["CD"], CD_xfoil=p["CD"],
                           dCD_rel=(r["CD"] - p["CD"]) / p["CD"], CM=r.get("CM"), CM_xfoil=p["CM"],
                           xtr_top_xfoil=p["xtr_top"], xtr_bot_xfoil=p["xtr_bot"])
                if isinstance(r.get("CM"), float):
                    row["dCM"] = r["CM"] - p["CM"]
                row["pass"] = abs(row["dCL"]) <= x.cl_abs_tol and abs(row["dCD_rel"]) <= x.cd_rel_tol
            compare.append(row)
        sub_rows = [c for c in compare if c["turb_model"] == turb and "dCL" in c and c["converged"]]
        by_model[turb] = {
            "compared": len(sub_rows),
            "mean_abs_dCL": _mean([abs(c["dCL"]) for c in sub_rows]),
            "mean_abs_dCD_rel": _mean([abs(c["dCD_rel"]) for c in sub_rows]),
            "mean_abs_dCM": _mean([abs(c["dCM"]) for c in sub_rows if "dCM" in c]),
            "pass": sum(1 for c in sub_rows if c["pass"]),
        }
    _write(out / "xfoil_compare.csv", compare)
    scored = {m: v for m, v in by_model.items() if v.get("compared")}
    recommended = None
    if scored:
        # 점수 = |ΔCL| 평균 / 0.1 + |ΔCD/CD| 평균 (CL 0.1 오차 ≈ CD 100% 오차로 같은 무게)
        recommended = min(scored, key=lambda m: scored[m]["mean_abs_dCL"] / 0.1 + scored[m]["mean_abs_dCD_rel"])
    summary = {"Re": re, "ncrit": x.ncrit, "shapes": len(shapes), "xfoil_points": len(polars),
               "by_model": by_model, "recommended": recommended}
    (out / "xfoil_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    return {"summary": summary, "dir": out, "polars": polars, "compare": compare}


def _write(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for r in rows:
        columns += [k for k in r if k not in columns]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns or ["empty"])
        w.writeheader()
        w.writerows(rows)
    return path
