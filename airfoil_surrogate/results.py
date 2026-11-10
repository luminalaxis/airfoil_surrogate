"""Fluent 리포트 파일(.out) 파싱 및 결과 요약 CSV 생성."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .cases import Case
from .config import ProjectConfig, safe_name
from .convergence import OSCILLATING, ConvergenceConfig, classify, coefficient_history, solver_log_info
from .meshing import mesh_name
from .runner import LOG_NAME, TRANSCRIPT_NAME, read_status

_QUOTED = re.compile(r'"([^"]*)"')
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

    리포트 이름이 cx/cy 로 시작하면 같은 접미사끼리 짝을 짓는다 (예: cx-main, cy-main → cl-main, cd-main).
    받음각이 바뀌어도 Fluent 쪽 force vector를 매번 바꿀 필요가 없다.
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
        row["cl" + suffix] = (cy * math.cos(a) - cx * math.sin(a))
        row["cd" + suffix] = cx * math.cos(a) + cy * math.sin(a)


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
    _forces_per_span(row)
    return row


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


def _write_csv(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
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
        if k in (cfg.results.x_param, "flow_angle") or k in cfg.derived:
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
              "top": [], "plots": []}

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
