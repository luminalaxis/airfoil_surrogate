"""Fluent 실행 전 점검 (afs check).

Fluent 없이 확인할 수 있는 것을 모두 본다: 실행할 케이스, 저널 치환, 베이스 케이스 파일, Fluent 실행 파일,
메쉬(있음·최신·실패), 세트별 경계 이름 일치, 리포트 파일 이름, Windows 경로 문제(한글·길이·OneDrive),
디스크와 실행 시간 추정. 아무것도 만들거나 바꾸지 않는다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .cases import Case
from .config import ConfigError, ProjectConfig
from .template import render

OK, INFO, WARN, TODO, ERROR = "OK", "참고", "경고", "할 일", "오류"
LEVEL_ORDER = (ERROR, TODO, WARN, INFO, OK)

# 결과 크기 추정 (파일럿 결과가 있으면 실제 크기를 쓴다)
EST_CASE_DATA_BYTES_PER_CELL = 200    # result.cas.h5 + result.dat.h5
EST_IMAGE_BYTES = 400_000             # save-picture 한 장
EST_MESH_DIR_BYTES = 8_000_000        # mesh.msh + 이미지 (만들어진 메쉬가 없을 때)
WINDOWS_PATH_LIMIT = 250              # MAX_PATH(260) 여유
CASE_OUTPUT_NAMES = ("img_velocity.png", "coef-rfile.out", "convergence.png", "result.cas.h5")

_READ_CASE = re.compile(r'^\s*/?file/read-case(?:-data)?\s+"([^"]+)"', re.IGNORECASE | re.MULTILINE)
_REPORT_FILE = re.compile(r'report-files/edit\s+"[^"]+"\s+file-name\s+"([^"]+)"', re.IGNORECASE)


@dataclass
class Item:
    level: str
    title: str
    detail: str = ""


def _active_lines(journal: str) -> list[str]:
    """주석(;)이 아닌 저널 줄."""
    return [ln for ln in journal.splitlines() if ln.strip() and not ln.lstrip().startswith(";")]


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def run_checks(cfg: ProjectConfig, cases: list[Case], parallel: int | None = None,
               check_solver: bool = True) -> list[Item]:
    from .runner import read_status

    items: list[Item] = []
    add = lambda *a: items.append(Item(*a))  # noqa: E731

    # ---------------------------------------------------------------- 실행할 케이스
    run_cases = list(cases)
    add(OK, f"실행할 케이스 {len(cases)}개")
    statuses = {c.name: read_status(c) for c in run_cases}
    todo_cases = [c for c in run_cases if statuses[c.name].get("state") != "done"]
    done = len(run_cases) - len(todo_cases)
    if done:
        add(INFO, f"이미 끝난 케이스 {done}개", f"남은 케이스 {len(todo_cases)}개")

    # ---------------------------------------------------------------- 저널
    template_text = cfg.template.read_text(encoding="utf-8") if cfg.template.is_file() else None
    journals: dict[str, str] = {}
    if template_text is None:
        add(ERROR, "저널 템플릿 없음", str(cfg.template))
    else:
        from .runner import case_variables

        failures = {}
        for c in run_cases:
            try:
                journals[c.name] = render(template_text, case_variables(cfg, c), cfg.template.name)
            except ConfigError as exc:
                failures.setdefault(str(exc), []).append(c.name)
        if failures:
            for msg, names in failures.items():
                add(ERROR, f"저널 치환 실패 {len(names)}개 케이스", f"{msg} (예: {names[0]})")
        else:
            add(OK, f"저널 치환: {len(journals)}개 케이스 모두 성공", cfg.template.name)
        _journal_checks(cfg, template_text, add)

    # ---------------------------------------------------------------- 베이스 케이스
    if journals:
        needed: dict[Path, list[Case]] = {}
        by_name = {c.name: c for c in run_cases}
        for name, text in journals.items():
            for raw in _READ_CASE.findall(text):
                path = Path(raw)
                if not path.is_absolute():
                    path = by_name[name].dir / path
                needed.setdefault(path, []).append(by_name[name])
        missing = {p: cs for p, cs in needed.items() if not p.is_file()}
        for path, cs in sorted(needed.items()):
            label = _set_label(cs)
            if path in missing:
                add(TODO, f"베이스 케이스 만들기: {path.name}",
                    f"{label}케이스 {len(cs)}개가 읽음. 베이스 케이스 문서(docs/FLUENT_BASE_CASE.md, 단일 익형은 "
                    f"FLUENT_BASE_CASE_AIRFOIL.md) 순서로 만들어 {path.parent} 에 저장")
            else:
                add(OK, f"베이스 케이스 있음: {path.name}", f"{label}케이스 {len(cs)}개")

    # ---------------------------------------------------------------- Fluent 실행 파일
    if check_solver and cfg.solver.kind == "fluent":
        from .runner import find_fluent

        if cfg.solver.executable:
            exe = Path(cfg.solver.executable)
            add(OK if exe.is_file() else ERROR, "Fluent 실행 파일" + ("" if exe.is_file() else " 없음"), str(exe))
        else:
            try:
                add(OK, "Fluent 실행 파일 (AWP_ROOT 자동 탐색)", find_fluent())
            except ConfigError:
                add(ERROR, "Fluent 실행 파일을 찾지 못함",
                    "AWP_ROOTxxx 환경변수에도 PATH에도 없습니다. ANSYS가 설치된 PC에서 실행하거나 "
                    "project.toml의 [solver] executable에 fluent.exe 경로를 지정하세요.")
        cores = os.cpu_count() or 1
        want = cfg.solver.processes * max(1, parallel or cfg.max_parallel)
        if want > cores:
            add(WARN, f"코어 부족: 동시 실행 × processes = {want} > 코어 {cores}개",
                "케이스마다 느려지므로 -j 또는 [solver] processes 를 줄이세요.")

    # ---------------------------------------------------------------- 메쉬
    mesh_cells: list[int] = []
    if cfg.mesh is not None:
        mesh_cells = _mesh_checks(cfg, run_cases, add)

    # ---------------------------------------------------------------- 경로 (Windows + Fluent)
    _path_checks(cfg, run_cases, add)

    # ---------------------------------------------------------------- 디스크와 시간
    _estimates(cfg, run_cases, todo_cases, statuses, journals, mesh_cells, parallel, add)
    return items


def _config_arg(cfg: ProjectConfig) -> str:
    return "" if cfg.path.name == "project.toml" else f"-c {cfg.path.name} "


def _span(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:g}" if lo == hi else f"{lo:g}~{hi:g}"


def _hours(h: float) -> str:
    return f"{h:.1f}시간" if h < 10 else f"{h:.0f}시간({h / 24:.1f}일)"


def _set_label(cases: list[Case]) -> str:
    sets = sorted({str(c.params.get("case_set")) for c in cases if c.params.get("case_set") is not None})
    return f"[{', '.join(sets)}] " if sets else ""


def _journal_checks(cfg: ProjectConfig, template_text: str, add) -> None:
    active = _active_lines(template_text)
    joined = "\n".join(active).lower()
    if "start-transcript" not in joined:
        add(WARN, "저널에 /file/start-transcript 가 없음",
            "Windows에서 -hidden으로 실행하면 Fluent 출력이 solver.log에 남지 않아 오류·발산을 못 잡습니다. "
            "예제 저널처럼 첫 줄에 /file/start-transcript \"transcript.txt\" 를 넣으세요.")
    patterns = [p.lower() for p in cfg.results.error_patterns]
    bad = [ln.strip() for ln in template_text.splitlines()
           if ln.lstrip().startswith(";") and any(p in ln.lower() for p in patterns)]
    if bad:
        add(INFO, f"저널 주석 {len(bad)}줄에 오류 검사 문구가 있음 (주석 줄은 검사에서 빼므로 문제는 없음)", bad[0][:80])
    if cfg.mesh is not None and "${mesh_file}" not in template_text:
        add(WARN, "저널에 ${mesh_file} 가 없음", "형상별 메쉬로 교체하지 않으면 모든 케이스가 같은 메쉬로 계산됩니다.")
    pictures = "save-picture" in joined
    if pictures and not cfg.solver.graphics:
        add(WARN, "저널이 이미지를 저장하는데 [solver] graphics = false",
            "-g 모드에서는 save-picture가 실패해 케이스가 실패로 표시될 수 있습니다. graphics = true 또는 이미지 줄 삭제.")
    report = _REPORT_FILE.search(template_text)
    if report:
        name = report.group(1)
        if not any(fnmatch(name, pat) for pat in cfg.results.report_files):
            add(ERROR, f"리포트 파일 {name} 이 [results] report_files 패턴과 안 맞음",
                f"패턴 {cfg.results.report_files}: collect가 결과를 못 읽습니다.")
        else:
            add(OK, f"리포트 파일 {name} ([results] report_files {cfg.results.report_files})")
    iters = cfg.constants.get("iterations")
    if isinstance(iters, int):
        window = (cfg.convergence or {}).get("window", 200)
        avg = max(cfg.results.average_last, cfg.results.average_last_oscillating or 0)
        if avg >= iters or window >= iters:
            add(WARN, f"평균·수렴 판정 구간이 반복 수({iters})보다 김",
                f"average_last {avg}, [convergence] window {window}")


def _mesh_checks(cfg: ProjectConfig, run_cases: list[Case], add) -> list[int]:
    from .meshing import ERROR_FILE, INFO_FILE

    jobs = {c.mesh.name: c.mesh for c in run_cases if c.mesh is not None}
    current = [j for j in jobs.values() if j.is_current()]
    failed = {j.name: (j.dir / ERROR_FILE).read_text(encoding="utf-8", errors="replace")
              for j in jobs.values() if (j.dir / ERROR_FILE).is_file()}
    missing = [j for j in jobs.values() if not j.is_current() and j.name not in failed]
    if missing:
        try:
            import gmsh  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - gmsh 공유 라이브러리 오류 등
            add(ERROR, "gmsh를 불러올 수 없음", f"{type(exc).__name__}: {exc}. pip install gmsh "
                "(Linux에서는 libGLU 같은 그래픽 라이브러리도 필요)")
        add(TODO, f"메쉬 만들기: {len(missing)}개 (있음 {len(current)}개)",
            f"afs {_config_arg(cfg)}mesh  (-j 로 동시 개수 지정, 기본 CPU 코어 수·최대 8)")
    else:
        add(OK, f"메쉬 {len(current)}개 모두 있음 (최신 설정)")
    if failed:
        first = next(iter(failed.items()))
        add(WARN, f"메쉬 생성에 실패한 형상 {len(failed)}개 (그 형상의 케이스는 실패로 표시됨)",
            f"{first[0]}: {first[1][:160]}")

    cells, zones_by_set, oq = [], {}, {}
    for job in current:
        info = json.loads((job.dir / INFO_FILE).read_text(encoding="utf-8"))
        cells.append(int(info.get("cells", 0)))
        if info.get("min_orthogonal_quality") is not None:
            oq[job.name] = float(info["min_orthogonal_quality"])
    for c in run_cases:
        if c.mesh is None or not c.mesh.is_current():
            continue
        key = str(c.params.get("base_case") or c.params.get("case_set") or "")
        zones_by_set.setdefault(key, {})
        if c.mesh.name not in zones_by_set[key]:
            info = json.loads((c.mesh.dir / INFO_FILE).read_text(encoding="utf-8"))
            zones_by_set[key][c.mesh.name] = tuple(sorted(info.get("zones", {})))
    for key, meshes in sorted(zones_by_set.items()):
        variants = sorted(set(meshes.values()))
        label = key or "전체"
        if len(variants) > 1:
            add(ERROR, f"{label}: 메쉬마다 경계 이름이 다름 (Fluent replace-mesh 실패)",
                " / ".join(", ".join(v) for v in variants))
        else:
            sample = next(iter(meshes))
            add(OK, f"{label}: 메쉬 {len(meshes)}개 경계 이름 같음",
                f"{', '.join(variants[0])}  (베이스 케이스용 예: {sample})")
    if cells:
        add(INFO, f"셀 수 {min(cells):,} ~ {max(cells):,} (중앙값 {int(statistics.median(cells)):,})")
    if oq:
        worst = min(oq, key=oq.get)
        low = [n for n, q in oq.items() if q < 0.1]
        # Fluent: 0.01 미만은 계산이 불안정(사실상 오류), 0.1 이상 권장. 셀 몇 개가 0.05~0.1인 것은 흔하고 괜찮다
        level = ERROR if oq[worst] < 0.01 else WARN if oq[worst] < 0.05 else INFO if low else OK
        add(level, f"직교 품질(Fluent Orthogonal Quality) 최소 {oq[worst]:.3f}"
            + (f", 0.1 미만 셀이 있는 메쉬 {len(low)}개" if low else ""),
            f"메쉬 {len(oq)}개 기준, 가장 낮은 메쉬 {worst}. Fluent는 0.01 미만이면 계산이 불안정하고 0.1 이상을 권장 "
            "(0.1 미만 셀 수는 mesh_info.json의 cells_oq_below_0.1)")
    hs = {round(j.settings.first_cell_height() * j.settings.chord * 1e6, 2) for j in jobs.values()}
    res = [j.settings.reynolds for j in jobs.values()]
    if hs:
        ys = sorted({round(j.settings.y_plus / j.settings.refine, 2) for j in jobs.values()})
        re_txt = f"{min(res):,.0f}" if min(res) == max(res) else f"{min(res):,.0f}~{max(res):,.0f}"
        add(INFO, f"첫 셀 높이 {_span(hs)} µm",
            f"목표 y+ {', '.join(f'{y:g}' for y in ys)} (평판 마찰 추정), 형상별 최고 Re {re_txt} 기준")
    return cells


def _path_checks(cfg: ProjectConfig, run_cases: list[Case], add) -> None:
    root = str(cfg.root)
    if any(ord(ch) > 127 for ch in root):
        add(WARN, "프로젝트 경로에 영문이 아닌 문자가 있음",
            f"{root}\nFluent는 한글 등이 들어간 경로의 파일을 못 읽는 경우가 많습니다. C:\\aero 처럼 영문 경로로 옮기세요.")
    if "onedrive" in root.lower():
        add(WARN, "OneDrive 폴더 안에서 실행",
            "동기화가 결과 파일을 잠가 실행이 실패하거나 느려질 수 있습니다. 동기화하지 않는 폴더를 권장합니다.")
    longest = ""
    for c in run_cases:
        for name in CASE_OUTPUT_NAMES:
            p = str(c.dir / name)
            if len(p) > len(longest):
                longest = p
        if c.mesh is not None:
            p = str(c.mesh.dir / "mesh_domain.png")
            if len(p) > len(longest):
                longest = p
    if len(longest) > WINDOWS_PATH_LIMIT:
        add(WARN, f"가장 긴 파일 경로 {len(longest)}자 (Windows 기본 한도 260자)",
            f"{longest}\n프로젝트를 더 짧은 경로로 옮기거나 [cases] name/group 패턴을 줄이세요.")
    elif longest:
        add(OK, f"가장 긴 파일 경로 {len(longest)}자 (Windows 한도 260자 이내)")


def _estimates(cfg: ProjectConfig, run_cases: list[Case], todo_cases: list[Case], statuses: dict,
               journals: dict, mesh_cells: list[int], parallel: int | None, add) -> None:
    from .meshing import MESH_FILE

    jobs = {c.mesh.name: c.mesh for c in run_cases if c.mesh is not None}
    sample = next(iter(journals.values()), "")
    active = "\n".join(_active_lines(sample)).lower()
    writes_data = "write-case-data" in active or "write-data" in active
    n_pictures = active.count("save-picture")
    iters = cfg.constants.get("iterations", 1000)
    iters = iters if isinstance(iters, (int, float)) else 1000

    # 디스크: 끝난 케이스가 있으면 실제 폴더 크기 중앙값, 없으면 셀 수로 추정
    done_dirs = [c.dir for c in run_cases if statuses[c.name].get("state") == "done"]
    if done_dirs:
        per_case = statistics.median(_dir_size(d) for d in done_dirs[:50])
        basis = f"끝난 케이스 폴더 크기 중앙값 {_fmt_bytes(per_case)}"
    else:
        cells = statistics.median(mesh_cells) if mesh_cells else 80_000
        per_case = (iters * 200 + iters * 150 + n_pictures * EST_IMAGE_BYTES
                    + (cells * EST_CASE_DATA_BYTES_PER_CELL if writes_data else 0))
        basis = (f"케이스당 약 {_fmt_bytes(per_case)} 추정"
                 + (f" (result 파일 = 셀 {int(cells):,}개 × {EST_CASE_DATA_BYTES_PER_CELL} B 가정)" if writes_data else ""))
    made = [j.dir for j in jobs.values() if j.mesh_file.is_file()]
    per_mesh = statistics.median(_dir_size(d) for d in made[:20]) if made else EST_MESH_DIR_BYTES
    mesh_bytes = per_mesh * (len(jobs) - len(made))
    need = per_case * len(todo_cases) + mesh_bytes
    free = shutil.disk_usage(_existing_parent(cfg.output_dir)).free
    level = WARN if need > 0.8 * free else INFO
    tip = ("  저널의 /file/write-case-data 줄을 지우면(주석 처리) 크게 줄어듭니다 — 필요한 케이스만 나중에 --only 로 다시 실행."
           if writes_data and level == WARN else "")
    add(level, f"디스크: 약 {_fmt_bytes(need)} 필요 / 남은 공간 {_fmt_bytes(free)}",
        f"남은 케이스 {len(todo_cases)}개, {basis}" + (f", 새 메쉬 {_fmt_bytes(mesh_bytes)}" if mesh_bytes else "") + tip)

    # 실행 시간: 끝난 케이스가 있으면 중앙값, 없으면 몇 가지 가정
    jobs_n = max(1, parallel or cfg.max_parallel)
    times = [s["elapsed_sec"] for s in statuses.values() if s.get("state") == "done" and s.get("elapsed_sec")]
    if times:
        per = statistics.median(times)
        hours = per * len(todo_cases) / jobs_n / 3600
        add(INFO, f"실행 시간: 약 {_hours(hours)}",
            f"남은 {len(todo_cases)}개 × 케이스당 {per / 60:.1f}분(끝난 {len(times)}개 중앙값) ÷ 동시 {jobs_n}개")
    elif todo_cases:
        guesses = ", ".join(f"{m}분이면 {_hours(m * len(todo_cases) / jobs_n / 60)}" for m in (3, 5, 10))
        add(INFO, f"실행 시간: 케이스 {len(todo_cases)}개 ÷ 동시 {jobs_n}개",
            f"케이스당 {guesses}. 파일럿 케이스를 돌린 뒤 다시 check 하면 실제 시간으로 계산합니다.")


def format_items(items: list[Item]) -> str:
    lines = []
    for it in items:
        lines.append(f"[{it.level}] {it.title}")
        for d in it.detail.splitlines():
            lines.append(f"       {d}")
    counts = {lv: sum(1 for it in items if it.level == lv) for lv in LEVEL_ORDER}
    lines.append("")
    lines.append("요약: " + ", ".join(f"{lv} {n}" for lv, n in counts.items() if n and lv != OK)
                 + ("" if counts[ERROR] or counts[TODO] else "  → Fluent 실행 준비 완료"))
    return "\n".join(lines)
