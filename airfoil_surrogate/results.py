"""Fluent 리포트 파일(.out) 파싱 및 결과 요약 CSV 생성."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .cases import Case
from .config import ProjectConfig, safe_name
from .convergence import OSCILLATING, ConvergenceConfig, classify, coefficient_history, solver_log_info
from .meshing import INFO_FILE, mesh_name
from .runner import LOG_NAME, TRANSCRIPT_NAME, read_status

_QUOTED = re.compile(r'"([^"]*)"')

# 팀 공통 결과 스키마 (results.csv, 한 행 = 한 케이스). 4명이 같은 열로 내고 afs merge-results로 합친다
RESULT_COLUMNS = (
    "case_id", "m", "p", "t", "alpha",            # 입력
    "Re", "turb_model", "mesh_mode",              # 조건
    "CL", "CD", "CM",                             # 출력
    "converged", "n_iter", "runtime_s",           # 수렴 정보
    "n_cells", "yplus_max", "yplus_mean",         # 메쉬 품질
    "assignee", "solver_version",                 # 추적용
)
_FLUENT_VERSION = re.compile(r"Fluent\s+(?:Release\s+)?(20\d\d\s*R\d|\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_AWP_VERSION = re.compile(r"[\\/]v(\d)(\d)(\d)[\\/]", re.IGNORECASE)
_MESH_QUALITY = {
    "fluent_min_orthogonal_quality": re.compile(r"Minimum Orthogonal Quality\s*=\s*([-+\d.eE]+)"),
    "fluent_max_aspect_ratio": re.compile(r"Maximum Aspect Ratio\s*=\s*([-+\d.eE]+)"),
}
_INDEX_COLUMNS = {"iteration", "time step", "flow-time", "flow time"}


def parse_report_file(path: Path) -> tuple[list[str], list[list[float]]]:
    """Fluent report-file 형식을 읽는다.

    예::

        "cl-rfile"
        "Iteration" "cl" "cd"
        ("Iteration" "cl" "cd")
        1 0.123 0.0456
    """
    headers: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(tok) for tok in line.split()]
        except ValueError:
            names = _QUOTED.findall(line)
            if len(names) > 1:
                headers = names
            continue
        if headers and len(values) == len(headers):
            rows.append(values)
    return headers, rows


def summarize_report(path: Path, average_last: int = 1) -> dict[str, float]:
    return summarize_rows(*parse_report_file(path), average_last)


def summarize_rows(headers: list[str], rows: list[list[float]], average_last: int = 1) -> dict[str, float]:
    if not rows:
        return {}
    tail = rows[-max(1, average_last):]
    summary = {}
    for col, name in enumerate(headers):
        if name.lower() in _INDEX_COLUMNS:
            continue
        summary[name] = sum(r[col] for r in tail) / len(tail)
    summary["iter_done"] = rows[-1][0]
    return summary


def _wind_axes(row: dict, aoa_key: str) -> None:
    """물체축 힘계수(cx, cy)를 받음각으로 회전해 양력/항력계수(cl, cd)를 만든다.

    양력 방향 = (−sin α, cos α), 항력 방향 = (cos α, sin α): 유입 속도 (U cos α, U sin α)와 같은 α로 돌린다.
    리포트 이름이 cx/cy 로 시작하면 같은 접미사끼리 짝을 짓는다 (예: cx-main, cy-main → cl-main, cd-main).
    받음각이 바뀌어도 Fluent 쪽 force vector를 매번 바꿀 필요가 없다. 베이스 케이스에 고정 방향 cl/cd 리포트가
    함께 있으면 회전값과 비교해 다르면 force_dir_check = "mismatch" (양력·항력 방향을 안 돌린 흔한 버그).
    """
    angle = row.get("flow_angle", row.get(aoa_key))
    if not isinstance(angle, (int, float)):
        return
    a = math.radians(angle)
    for key in [k for k in row if k.lower().startswith("cx")]:
        suffix = key[2:]
        ky = next((k for k in row if k.lower() == "cy" + suffix.lower()), None)
        cx, cy = row[key], row.get(ky)
        if ky is None or not isinstance(cx, float) or not isinstance(cy, float):
            continue
        cl, cd = cy * math.cos(a) - cx * math.sin(a), cx * math.cos(a) + cy * math.sin(a)
        old_cl, old_cd = row.get("cl" + suffix), row.get("cd" + suffix)
        if isinstance(old_cl, float) and isinstance(old_cd, float):
            ok = abs(old_cl - cl) <= 1e-3 * max(1.0, abs(cl)) and abs(old_cd - cd) <= 1e-4 + 1e-3 * abs(cd)
            row["force_dir_check" + suffix] = "ok" if ok else "mismatch"
        row["cl" + suffix] = cl
        row["cd" + suffix] = cd


def summarize_case(cfg: ProjectConfig, case: Case) -> dict:
    status = read_status(case)
    row = {"case": case.name, "group": case.group, **case.params}
    if case.mesh is not None:
        row["mesh"] = case.mesh.name
        # 형상 이름 (받음각 제외): 순위·비교는 이 이름으로
        row["geometry"] = mesh_name({k: v for k, v in case.params.items() if k != cfg.results.x_param})
    row["state"] = status.get("state", "pending")
    row["elapsed_min"] = round(status["elapsed_sec"] / 60, 2) if status.get("elapsed_sec") else ""
    files = sorted({p for pattern in cfg.results.report_files for p in case.dir.glob(pattern)})
    reports = [(path, *parse_report_file(path)) for path in files]
    angle = row.get("flow_angle", row.get(cfg.results.x_param))
    history = next((h for h in (coefficient_history(headers, data, angle)
                                for _, headers, data in reports) if h is not None), None)
    conv: dict = {}
    if row["state"] != "pending":
        log_info = solver_log_info(case.dir / LOG_NAME, case.dir / TRANSCRIPT_NAME)
        conv = classify(history, ConvergenceConfig.from_dict(cfg.convergence), status, log_info)
        conv.update({k: v for k, v in log_info.items() if k.startswith("res_")})
    # 결과값 = 마지막 N 반복 평균. 진동 케이스는 average_last_oscillating(예: 500)으로 더 길게 평균
    window = cfg.results.average_last
    if conv.get("conv_state") == OSCILLATING and cfg.results.average_last_oscillating:
        window = cfg.results.average_last_oscillating
    for path, headers, data in reports:
        for key, value in summarize_rows(headers, data, window).items():
            if key == "iter_done":
                row["iter_done"] = max(row.get("iter_done", 0), int(value))
                continue
            column = key if key not in row else f"{path.stem}.{key}"
            row[column] = value
        if data:
            row["avg_iters"] = max(row.get("avg_iters", 0), min(max(1, window), len(data)))
    row.update(conv)
    if cfg.results.wind_axes:
        _wind_axes(row, cfg.results.x_param)
    if isinstance(row.get("cl"), float) and isinstance(row.get("cd"), float) and row["cd"]:
        row["l/d"] = row["cl"] / row["cd"]
    _moment(row)
    _forces_per_span(row)
    _yplus(row)
    row["runtime_s"] = status.get("elapsed_sec", "")
    if row["state"] != "pending":
        row["solver_version"] = solver_version(cfg, case, status)
        row.update(fluent_mesh_quality(case.dir / TRANSCRIPT_NAME))
    if case.mesh is not None:
        row.update(mesh_info_columns(case.mesh.dir / INFO_FILE))
    if cfg.results.cp_stations and case.mesh is not None:
        row.update(_cp_columns(cfg, case))
    return row


def _moment(row: dict) -> None:
    """피칭 모멘트 계수 CM (기준점 x/c = moment_ref, 기본 c/4, 머리 들림 +).

    베이스 케이스의 cmz 리포트: 원점(앞전, 코드와 무관) 기준, 축 (0, 0, 1) → 반시계 +.
    c/4로 옮기면 M_ref = M_0 − x_ref·c·F_y 이고 머리 들림(시계 방향)을 +로 바꾸면 CM = x_ref·cy − cmz.
    리포트 이름이 cm이면 이미 기준점·부호를 맞춘 값으로 보고 그대로 쓴다.
    """
    if isinstance(row.get("cm"), float):
        return
    cmz, cy = row.get("cmz"), row.get("cy")
    if isinstance(cmz, float) and isinstance(cy, float):
        row["cm"] = float(row.get("moment_ref", 0.25)) * cy - cmz


def _yplus(row: dict) -> None:
    """Fluent 리포트 yplus-max, yplus-avg → yplus_max, yplus_mean."""
    for src, dst in (("yplus-max", "yplus_max"), ("yplus-avg", "yplus_mean"), ("yplus_avg", "yplus_mean")):
        if isinstance(row.get(src), float) and dst not in row:
            row[dst] = row[src]


def solver_version(cfg: ProjectConfig, case: Case, status: dict) -> str:
    """솔버 버전: 로그의 "Fluent 2026 R1" → 실행 파일 경로의 v261 → [results] solver_version → 솔버 종류."""
    for path in (case.dir / LOG_NAME, case.dir / TRANSCRIPT_NAME):
        if path.is_file():
            m = _FLUENT_VERSION.search(path.read_text(encoding="utf-8", errors="replace")[:200_000])
            if m:
                return f"Fluent {m.group(1)}"
    cmd = status.get("command") or []
    if cfg.solver.kind == "fluent" and cmd:
        m = _AWP_VERSION.search(str(cmd[0]))
        if m:
            return f"Fluent 20{m.group(1)}{m.group(2)} R{m.group(3)}"
    return cfg.results.solver_version or cfg.solver.kind


def fluent_mesh_quality(transcript: Path) -> dict:
    """저널의 /mesh/quality 출력 (Minimum Orthogonal Quality, Maximum Aspect Ratio)."""
    if not transcript.is_file():
        return {}
    text = transcript.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pattern in _MESH_QUALITY.items():
        m = pattern.search(text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def mesh_info_columns(info_path: Path) -> dict:
    if not info_path.is_file():
        return {}
    import json

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out = {"n_cells": info.get("cells", "")}
    for src, dst in (("quality_flag", "mesh_flag"), ("max_aspect_ratio", "mesh_max_aspect_ratio"),
                     ("max_non_orthogonality", "mesh_max_non_orthogonality"),
                     ("min_orthogonal_quality", "mesh_min_orthogonal_quality")):
        if src in info:
            out[dst] = info[src]
    return out


def _cp_columns(cfg: ProjectConfig, case: Case) -> dict:
    from .cp import column_names, interpolate_cp
    from .geometry import section_from_params

    path = case.dir / cfg.results.cp_file
    if not path.is_file():
        return {k: "" for k in column_names(cfg.results.cp_stations)}
    el = section_from_params(case.mesh.geometry_params, cfg.root)
    return interpolate_cp(path, cfg.results.cp_stations, el.xy, el.le_index, case.mesh.settings.chord)


def result_row(cfg: ProjectConfig, row: dict) -> dict:
    """summary 행 → 팀 공통 스키마 행 (RESULT_COLUMNS, Cp를 켰으면 cp_* 열 추가)."""
    x = cfg.results.x_param

    def num(key):
        v = row.get(key)
        return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else ""

    out = {
        "case_id": row.get("case_id", row["case"]), "m": row.get("m", ""), "p": row.get("p", ""),
        "t": row.get("t", ""), "alpha": row.get(x, ""),
        "Re": row.get("reynolds", ""), "turb_model": row.get("turb_model", row.get("turbulence_model", "")),
        "mesh_mode": row.get("mesh_mode", ""),
        "CL": num("cl"), "CD": num("cd"), "CM": num("cm"),
        "converged": bool(row.get("converged", False)) and row.get("state") == "done",
        "n_iter": row.get("iter_done", ""), "runtime_s": row.get("runtime_s", ""),
        "n_cells": row.get("n_cells", ""), "yplus_max": num("yplus_max"), "yplus_mean": num("yplus_mean"),
        "assignee": row.get("assignee", ""), "solver_version": row.get("solver_version", ""),
    }
    if cfg.results.cp_stations:
        from .cp import column_names

        out.update({k: num(k) for k in column_names(cfg.results.cp_stations)})
    return out


def merge_results(paths: list[Path], out: Path) -> dict:
    """팀원별 results.csv를 하나로. 열이 스키마와 다르거나 같은 case_id가 다른 값으로 두 번 나오면 오류."""
    from .config import ConfigError

    rows, seen, header = [], {}, None
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if list(cols[:len(RESULT_COLUMNS)]) != list(RESULT_COLUMNS):
                raise ConfigError(f"{path}: 열이 공통 스키마와 다릅니다.\n  기대: {', '.join(RESULT_COLUMNS)}\n"
                                  f"  실제: {', '.join(cols)}")
            if header is None:
                header = cols
            elif cols != header:
                raise ConfigError(f"{path}: 다른 파일과 열이 다릅니다 (Cp 지점 설정이 다른지 확인).")
            for r in reader:
                key = (r["case_id"], r["turb_model"], r["mesh_mode"], r["Re"])
                if key in seen:
                    if seen[key] != r:
                        raise ConfigError(f"case_id {r['case_id']}가 {path.name}에서 다른 값으로 또 나옵니다.")
                    continue
                seen[key] = r
                rows.append(r)
    rows.sort(key=lambda r: (r["turb_model"], r["mesh_mode"], r["case_id"]))
    _write_csv(out, rows)
    return {"path": out, "rows": len(rows), "files": len(paths),
            "converged": sum(1 for r in rows if r["converged"] == "True")}


def _forces_per_span(row: dict) -> None:
    """케이스 변수 velocity[m/s], chord[m]가 있으면 단위 폭(1 m)당 양력/항력 [N/m]도 계산.

    코드 길이가 다른 형상끼리 실제 힘을 비교할 때 사용 (계수는 각자의 코드로 무차원화돼 있음).
    """
    v, c = row.get("velocity"), row.get("chord")
    if not isinstance(v, (int, float)) or not isinstance(c, (int, float)):
        return
    rho = row.get("density", 1.225)
    q = 0.5 * rho * v**2
    for coef, name in (("cl", "lift_N_per_m"), ("cd", "drag_N_per_m")):
        if isinstance(row.get(coef), float):
            row[name] = q * c * row[coef]


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    columns = list(columns or [])
    for row in rows:
        columns += [k for k in row if k not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: 엑셀에서 바로 열어도 한글이 깨지지 않도록
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def geometry_keys(cfg: ProjectConfig, cases: list[Case]) -> list[str]:
    """형상을 구분하는 변수: 케이스마다 값이 다른 변수 중 받음각(과 받음각에서 정한 flow_angle)/파생 변수를 뺀 것."""
    keys = []
    for k in cases[0].params:
        if k in (cfg.results.x_param, "flow_angle", "case_id", "assignee") or k in cfg.derived:
            continue
        if len({repr(c.params.get(k)) for c in cases}) > 1:
            keys.append(k)
    return keys


def rank_geometries(rows: list[dict], keys: list[str], x_param: str, objective: str,
                    split_by: list[str] = ()) -> list[dict]:
    """형상별 최대 Cl/L/D. split_by가 있으면 그 조건(예: 속도)마다 따로 순위를 매긴다."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("state") == "done" and isinstance(r.get("cl"), float):
            groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    ranking = []
    for values, rs in groups.items():
        best_cl = max(rs, key=lambda r: r["cl"])
        entry = {**dict(zip(keys, values)), "n_aoa": len(rs),
                 "cl_max": best_cl["cl"], f"{x_param}@cl_max": best_cl.get(x_param)}
        with_ld = [r for r in rs if isinstance(r.get("l/d"), float)]
        if with_ld:
            best_ld = max(with_ld, key=lambda r: r["l/d"])
            entry.update({"ld_max": best_ld["l/d"], f"{x_param}@ld_max": best_ld.get(x_param),
                          "cl@ld_max": best_ld["cl"], "cd@ld_max": best_ld.get("cd")})
        if "geometry" in rs[0]:
            entry["geometry"] = rs[0]["geometry"]
        ranking.append(entry)
    sort_key = "ld_max" if objective == "max_ld" else "cl_max"
    ranking.sort(key=lambda e: (tuple(e.get(k) for k in split_by), -e.get(sort_key, float("-inf"))))
    counter: dict[tuple, int] = {}
    for e in ranking:
        cond = tuple(e.get(k) for k in split_by)
        counter[cond] = counter.get(cond, 0) + 1
        e["rank"] = counter[cond]
    return ranking


def collect(cfg: ProjectConfig, cases: list[Case], plots: bool = True) -> dict:
    """summary.csv, ranking.csv, 케이스별 수렴 그래프, 요약 그래프 생성."""
    rows = [summarize_case(cfg, c) for c in cases]
    out_dir = cfg.output_dir
    result = {"summary": _write_csv(out_dir / cfg.results.summary_file, rows), "ranking": None,
              "top": [], "plots": [], "warnings": []}
    done = [r for r in rows if r.get("state") != "pending"]
    team = [result_row(cfg, r) for r in done]
    if team or (out_dir / cfg.results.results_file).is_file():
        result["results"] = _write_csv(out_dir / cfg.results.results_file, team, list(RESULT_COLUMNS))
        result["n_results"] = len(team)
        result["n_converged"] = sum(1 for r in team if r["converged"])
    bad_dir = [r["case"] for r in rows if r.get("force_dir_check") == "mismatch"]
    if bad_dir:
        result["warnings"].append(f"베이스 케이스의 cl/cd 리포트 방향이 받음각과 맞지 않는 케이스 {len(bad_dir)}개 "
                                  f"(예: {bad_dir[0]}). 결과는 cx/cy를 α로 회전한 값을 씁니다.")
    bad_mesh = sorted({r.get("mesh", "") for r in rows if r.get("mesh_flag") == "bad"})
    if bad_mesh:
        result["warnings"].append(f"불량 판정 메쉬 {len(bad_mesh)}개 (mesh_info.json quality_reasons): "
                                  + ", ".join(bad_mesh[:5]))

    keys = geometry_keys(cfg, cases)
    split_by = [k for k in cfg.results.split_by if k in keys]
    keys = [k for k in keys if k not in split_by] + split_by
    ranking = rank_geometries(rows, keys, cfg.results.x_param, cfg.results.objective, split_by)
    if ranking:
        result["ranking"] = _write_csv(out_dir / "ranking.csv", ranking)
        result["top"] = [e for e in ranking if e["rank"] <= 3] if split_by else ranking[:5]

    if plots:
        from .plots import plot_convergence, plot_summary
        states = {r["case"]: r.get("conv_state") for r in rows}
        for case in cases:
            files = sorted({p for pat in cfg.results.report_files for p in case.dir.glob(pat)})
            if files:
                state = states.get(case.name)
                title = f"{case.name} [{state}]" if state else case.name
                plot_convergence(files, case.dir / "convergence.png", title=title)
        geo_keys = [k for k in keys if k not in split_by]
        cl_label = "Cl"
        if split_by:
            # 조건(속도 등)별 폴더에 그래프를 따로 저장
            conds = sorted({tuple(r.get(k) for k in split_by) for r in rows})
            for cond in conds:
                sub = [r for r in rows if tuple(r.get(k) for k in split_by) == cond]
                label = safe_name(", ".join(f"{k}={v}" for k, v in zip(split_by, cond)))
                result["plots"] += plot_summary(sub, out_dir / "plots" / label, cfg.results.x_param, geo_keys, cl_label)
            if len(split_by) == 1 and ranking:
                from .plots import plot_condition_trend
                path = plot_condition_trend(ranking, split_by[0], geo_keys, cfg.results.objective,
                                            out_dir / "plots" / f"by_{split_by[0]}.png")
                if path:
                    result["plots"].append(path)
        else:
            result["plots"] = plot_summary(rows, out_dir / "plots", cfg.results.x_param, geo_keys, cl_label)
    return result
