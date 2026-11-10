"""케이스 폴더 준비 및 솔버 배치 실행."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .cases import Case
from .config import ConfigError, ProjectConfig
from .template import render

IS_WINDOWS = os.name == "nt"
JOURNAL_NAME = "run.jou"
LOG_NAME = "solver.log"
# 저널의 /file/start-transcript 파일. Windows에서 -hidden으로 실행하면 Fluent 콘솔 출력이 표준 출력으로
# 나오지 않아서, 오류·발산·잔차는 이 파일에서 읽는다.
TRANSCRIPT_NAME = "transcript.txt"
# 다시 실행할 때 지우는 이전 결과 (Fluent가 덮어쓰기 확인을 물으면 배치 저널이 꼬이거나 멈춘다)
STALE_OUTPUTS = ("*.png", TRANSCRIPT_NAME, "*.cas.h5", "*.dat.h5", "*.cas.gz", "*.dat.gz")
MESH_WORKERS_DEFAULT_MAX = 8  # mesh -j 를 주지 않았을 때 동시에 만들 메쉬 수의 상한 (메모리 고려)
STATUS_NAME = "status.json"
PARAMS_NAME = "case.json"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _posix(path: Path) -> str:
    # Fluent는 Windows에서도 슬래시 경로를 받아들이고, 역슬래시는 이스케이프 문제를 만든다.
    return path.resolve().as_posix()


def case_variables(cfg: ProjectConfig, case: Case) -> dict:
    variables = {
        **case.params,
        "case_name": case.name,
        "group": case.group,
        "case_dir": _posix(case.dir),
        "project_dir": _posix(cfg.root),
        "nprocs": cfg.solver.processes,
    }
    if case.mesh is not None:
        variables["mesh_file"] = _posix(case.mesh.mesh_file)
        variables["mesh_name"] = case.mesh.name
    return variables


def ensure_meshes(cfg: ProjectConfig, cases: list[Case], force: bool = False,
                  workers: int | None = None) -> dict[str, str]:
    """케이스에 필요한 메쉬를 생성한다 (같은 형상은 한 번만). 반환: {메쉬 이름: 오류 메시지}.

    workers개 작업 프로세스로 동시에 만든다 (기본: CPU 코어 수, 최대 MESH_WORKERS_DEFAULT_MAX개. 프로세스당
    메모리 약 300 MB). 형상마다 시간 제한([mesh] timeout)이
    적용되고, 실패하거나 시간을 넘긴 형상만 실패 처리한 뒤 나머지를 계속 만든다.
    """
    from .meshing import MeshWorker, generate_mesh, single_thread_blas

    jobs = {c.mesh.name: c.mesh for c in cases if c.mesh is not None}
    todo = [j for j in jobs.values() if force or not j.is_current()]
    errors: dict[str, str] = {}
    if not jobs:
        return errors
    n = max(1, min(workers or min(os.cpu_count() or 1, MESH_WORKERS_DEFAULT_MAX), len(todo) or 1))
    log(f"메쉬: 형상 {len(jobs)}개 중 {len(todo)}개 생성 필요"
        + (f" (동시 {n}개)" if len(todo) > 1 and n > 1 else ""))
    if not todo:
        return errors

    local = threading.local()
    mesh_workers: list[MeshWorker] = []
    lock = threading.Lock()
    stop = threading.Event()

    def build(job):
        # 스레드마다 작업 프로세스 하나를 계속 쓴다 (gmsh는 프로세스 안에서 동시에 두 개를 돌릴 수 없음)
        worker = getattr(local, "worker", None)
        if worker is None:
            worker = local.worker = MeshWorker()
            with lock:
                mesh_workers.append(worker)
        start = time.monotonic()
        try:
            info = generate_mesh(job, cfg.root, worker=worker, stop=stop)
            return job, info, None, time.monotonic() - start
        except ConfigError as exc:
            return job, None, str(exc), time.monotonic() - start

    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=n)
    try:
        with single_thread_blas():
            futures = [pool.submit(build, job) for job in todo]
            for i, fut in enumerate(as_completed(futures), start=1):
                job, info, err, dt = fut.result()
                if err is not None:
                    errors[job.name] = err
                    log(f"  [{i}/{len(todo)}] {job.name:<45} 실패 - {err}")
                else:
                    oq = info.get("min_orthogonal_quality")
                    log(f"  [{i}/{len(todo)}] {job.name:<45} {info['cells']:>8,} cells  "
                        f"품질 {info['min_quality']:.2f}" + (f"  직교 {oq:.2f}" if oq is not None else "")
                        + f"  {dt:4.1f}s")
    except BaseException:
        # Ctrl+C 등: 남은 형상은 취소하고, 만드는 중인 작업 프로세스는 바로 끝낸다
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for w in mesh_workers:
            w.close(kill=True)
        raise
    pool.shutdown(wait=True)
    for w in mesh_workers:
        w.close()
    total = time.monotonic() - started
    log(f"메쉬 {len(todo) - len(errors)}개 생성, 실패 {len(errors)}개 ({total / 60:.1f}분)")
    return errors


def find_fluent() -> str:
    """AWP_ROOTxxx 환경변수(ANSYS 설치 시 자동 등록)에서 최신 버전 Fluent를 찾는다."""
    roots = sorted(
        ((int(m.group(1)), v) for k, v in os.environ.items() if (m := re.fullmatch(r"AWP_ROOT(\d+)", k))),
        reverse=True,
    )
    for _, root in roots:
        exe = (
            Path(root, "fluent", "ntbin", "win64", "fluent.exe")
            if IS_WINDOWS
            else Path(root, "fluent", "bin", "fluent")
        )
        if exe.is_file():
            return str(exe)
    found = shutil.which("fluent")
    if found:
        return found
    raise ConfigError(
        "Fluent 실행 파일을 찾지 못했습니다. project.toml의 [solver] executable에 경로를 지정하세요.\n"
        r"  예) C:\Program Files\ANSYS Inc\v261\fluent\ntbin\win64\fluent.exe"
    )


def build_command(cfg: ProjectConfig, case: Case) -> list[str]:
    s = cfg.solver
    if s.kind == "custom":
        variables = {
            **case_variables(cfg, case),
            "exe": s.executable or "",
            "journal": JOURNAL_NAME,
            "python": sys.executable,
        }
        return [render(part, variables, "[solver] command") for part in s.command]

    exe = s.executable or find_fluent()
    cmd = [exe, s.mode, f"-t{s.processes}", "-i", JOURNAL_NAME]
    if s.graphics and not s.gui:
        # GUI 없이 그래픽 기능만 살려 save-picture 가능하게
        cmd += ["-hidden"] if IS_WINDOWS else ["-gu", "-driver", "null"]
    elif not s.gui:
        cmd.append("-g")
    if IS_WINDOWS:
        cmd.append("-wait")  # Windows 런처가 솔버 종료까지 기다리도록
    return cmd + list(s.extra_args)


def prepare_case(cfg: ProjectConfig, case: Case, template_text: str | None = None) -> Path:
    if template_text is None:
        template_text = cfg.template.read_text(encoding="utf-8")
    case.dir.mkdir(parents=True, exist_ok=True)
    journal = render(template_text, case_variables(cfg, case), str(cfg.template.name))
    journal_path = case.dir / JOURNAL_NAME
    journal_path.write_text(journal, encoding="utf-8")
    (case.dir / PARAMS_NAME).write_text(
        json.dumps({"name": case.name, "group": case.group, "params": case.params},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return journal_path


def read_status(case: Case) -> dict:
    path = case.dir / STATUS_NAME
    if not path.is_file():
        return {"state": "pending"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": "unknown"}


def _write_status(case: Case, **status) -> None:
    (case.dir / STATUS_NAME).write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    # Fluent는 런처 아래로 cortex/MPI 프로세스를 띄우므로 트리 전체를 종료해야 한다.
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# transcript에 다시 찍힌 저널 주석 줄 (앞에 "> " 나 "/file> " 같은 TUI 프롬프트가 붙을 수 있음)
_ECHOED_COMMENT = re.compile(r"^\s*(?:[\w/.-]*>\s*)*;")


def _scan_errors(log_paths: list[Path], patterns: list[str]) -> str | None:
    """로그 파일들(표준 출력 로그, Fluent transcript)에서 처음 나오는 오류 줄. 저널 주석 줄은 건너뛴다."""
    if not patterns:
        return None
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if _ECHOED_COMMENT.match(line):
                continue
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    return line.strip()
    return None


def _remove_stale_outputs(cfg: ProjectConfig, case: Case) -> str | None:
    """이전 실행 결과를 지운다. 지우지 못한 파일이 있으면(다른 프로그램이 열고 있음 등) 그 메시지."""
    for pattern in [*cfg.results.report_files, *STALE_OUTPUTS]:
        for old in case.dir.glob(pattern):
            try:
                old.unlink()
            except OSError as exc:
                return f"이전 결과 파일을 지울 수 없습니다 ({old.name}: {exc}). 파일을 연 프로그램을 닫으세요."
    return None


@dataclass
class RunResult:
    case: Case
    state: str
    message: str = ""
    elapsed: float = 0.0


def run_case(cfg: ProjectConfig, case: Case, template_text: str | None = None,
             force: bool = False) -> RunResult:
    if not force and read_status(case).get("state") == "done":
        return RunResult(case, "skipped", "이미 완료됨")

    if case.mesh is not None and not case.mesh.mesh_file.is_file():
        from .meshing import ERROR_FILE

        error_file = case.mesh.dir / ERROR_FILE
        reason = error_file.read_text(encoding="utf-8") if error_file.is_file() else "메쉬 파일 없음"
        msg = f"메쉬 실패 ({case.mesh.name}): {reason}"
        case.dir.mkdir(parents=True, exist_ok=True)
        _write_status(case, state="failed", message=msg)
        return RunResult(case, "failed", msg)

    # 이전 실행 결과 정리 (리포트, 이미지, transcript, result.cas.h5 등). Fluent가 기존 파일 덮어쓰기 확인을
    # 물으면 그다음 저널 줄이 답으로 읽혀 배치가 꼬이거나 멈춘다.
    case.dir.mkdir(parents=True, exist_ok=True)
    if msg := _remove_stale_outputs(cfg, case):
        _write_status(case, state="failed", message=msg)
        return RunResult(case, "failed", msg)
    prepare_case(cfg, case, template_text)
    cmd = build_command(cfg, case)
    log_path = case.dir / LOG_NAME
    timeout = cfg.solver.timeout_minutes * 60 if cfg.solver.timeout_minutes else None

    _write_status(case, state="running", command=cmd, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        popen_kwargs = {"start_new_session": True} if not IS_WINDOWS else {}
        try:
            proc = subprocess.Popen(cmd, cwd=case.dir, stdout=log_file,
                                    stderr=subprocess.STDOUT, **popen_kwargs)
        except OSError as exc:
            _write_status(case, state="failed", command=cmd, message=f"실행 실패: {exc}")
            return RunResult(case, "failed", f"실행 실패: {exc}")
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.wait()
            elapsed = time.monotonic() - start
            msg = f"시간 초과 ({cfg.solver.timeout_minutes}분)"
            _write_status(case, state="failed", command=cmd, message=msg, elapsed_sec=round(elapsed, 1))
            return RunResult(case, "failed", msg, elapsed)
        except KeyboardInterrupt:
            _kill_tree(proc)
            _write_status(case, state="interrupted", command=cmd)
            raise

    elapsed = time.monotonic() - start
    if returncode != 0:
        state, msg = "failed", f"종료 코드 {returncode}"
    elif err := _scan_errors([log_path, case.dir / TRANSCRIPT_NAME], cfg.results.error_patterns):
        state, msg = "failed", f"로그 오류: {err}"
    else:
        state, msg = "done", ""
    _write_status(case, state=state, command=cmd, returncode=returncode,
                  message=msg, elapsed_sec=round(elapsed, 1))
    return RunResult(case, state, msg, elapsed)


def run_all(cfg: ProjectConfig, cases: list[Case], force: bool = False,
            max_parallel: int | None = None) -> list[RunResult]:
    template_text = cfg.template.read_text(encoding="utf-8")
    ensure_meshes(cfg, [c for c in cases if force or read_status(c).get("state") != "done"])
    workers = max(1, max_parallel or cfg.max_parallel)
    total = len(cases)
    results: list[RunResult] = []

    def report(i: int, r: RunResult) -> None:
        extra = f" - {r.message}" if r.message else ""
        log(f"[{i}/{total}] {r.case.name:<30} {r.state:<8} {r.elapsed/60:6.1f}분{extra}")

    if workers == 1:
        for i, case in enumerate(cases, start=1):
            log(f"[{i}/{total}] {case.name} 실행 중...")
            r = run_case(cfg, case, template_text, force)
            report(i, r)
            results.append(r)
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, cfg, c, template_text, force): c for c in cases}
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            report(i, r)
            results.append(r)
    return results
