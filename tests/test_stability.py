"""안정성 회귀 테스트: 중단 처리, 잠금, 설정 오류 메시지, 수렴 판정, 학습 입력 검증."""
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from airfoil_surrogate.cases import _coerce, build_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import ConfigError, load_config
from airfoil_surrogate.convergence import ConvergenceConfig, classify
from airfoil_surrogate.expr import ExpressionError, evaluate
from airfoil_surrogate.fileutil import FileLock, LockedError, write_json_atomic
from airfoil_surrogate.results import rank_geometries
from airfoil_surrogate.runner import LOCK_NAME, read_status, run_case
from airfoil_surrogate.template import render

SLOW_SOLVER = """
import sys, time
time.sleep(float(sys.argv[1]))
open("coef-rfile.out", "w").write('"x"\\n("Iteration" "cl" "cd")\\n1 0.1 0.01\\n')
"""


def make_slow_project(tmp_path, seconds=30, n=6, parallel=2):
    (tmp_path / "slow.py").write_text(SLOW_SOLVER, encoding="utf-8")
    (tmp_path / "j.jou").write_text("AOA=${aoa}\n", encoding="utf-8")
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "j.jou"
        max_parallel = {parallel}
        [solver]
        kind = "custom"
        command = ["${{python}}", "{(tmp_path / 'slow.py').as_posix()}", "{seconds}"]
        [cases]
        name = "a{{aoa}}"
        [cases.sweep]
        aoa = {list(range(n))}
    """), encoding="utf-8")
    return tmp_path / "project.toml"


# ---------------------------------------------------------------- 중단

def test_run_case_stop_event_kills_solver(tmp_path):
    cfg = load_config(make_slow_project(tmp_path, seconds=60, n=1))
    case = build_cases(cfg)[0]
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    t0 = time.monotonic()
    r = run_case(cfg, case, stop=stop)
    assert r.state == "interrupted" and time.monotonic() - t0 < 10
    assert read_status(case)["state"] == "interrupted"
    assert not (case.dir / LOCK_NAME).exists()


@pytest.mark.skipif(os.name == "nt", reason="SIGINT 전달은 POSIX에서만 시험")
def test_parallel_run_ctrl_c_stops_quickly(tmp_path):
    cfg_path = make_slow_project(tmp_path, seconds=60, n=6, parallel=2)
    code = ("import signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler)\n"
            "from airfoil_surrogate.cli import main\n"
            f"sys.exit(main(['-c', {str(cfg_path)!r}, '--no-banner', '--no-clear', 'run']))")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).parents[1],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    runs = tmp_path / "runs"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and sum(1 for _ in runs.glob("*/status.json")) < 2:
        time.sleep(0.2)
    time.sleep(1.0)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=20)   # 예전에는 모든 케이스가 끝날 때까지(수 분) 기다렸다
    assert proc.returncode == 130, out
    states = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))["state"] for p in runs.glob("*/status.json")}
    assert "done" not in states.values()
    assert len(states) == 2 and set(states.values()) == {"interrupted"}   # 대기 케이스는 시작하지 않음
    assert not list(runs.glob(f"*/{LOCK_NAME}"))


# ---------------------------------------------------------------- 잠금·원자적 쓰기

def test_case_lock_prevents_concurrent_run(tmp_path):
    cfg = load_config(make_slow_project(tmp_path, seconds=0, n=1))
    case = build_cases(cfg)[0]
    with FileLock(case.dir / LOCK_NAME):
        r = run_case(cfg, case)
        assert r.state == "skipped" and "다른 프로세스" in r.message
        with pytest.raises(LockedError):
            FileLock(case.dir / LOCK_NAME).acquire()
    assert run_case(cfg, case).state == "done"


def test_stale_lock_from_dead_process_is_replaced(tmp_path):
    import socket

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock = tmp_path / LOCK_NAME
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": dead.pid}), encoding="utf-8")
    with FileLock(lock):
        assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not lock.exists()


def test_atomic_json_and_corrupt_status(tmp_path):
    cfg = load_config(make_slow_project(tmp_path, seconds=0, n=1))
    case = build_cases(cfg)[0]
    case.dir.mkdir(parents=True)
    (case.dir / "status.json").write_text('{"state": "do', encoding="utf-8")   # 쓰다 끊긴 파일
    assert read_status(case)["state"] == "unknown"
    write_json_atomic(case.dir / "status.json", {"state": "done"})
    assert read_status(case)["state"] == "done"
    assert not list(case.dir.glob(".*.tmp"))


# ---------------------------------------------------------------- 설정 오류

def test_config_errors_are_friendly(tmp_path, capsys):
    cfg_path = make_slow_project(tmp_path)
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(text.replace('kind = "custom"', 'kind = "custom"\nprocess = 4'), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"process \(→ processes\?\)"):
        load_config(cfg_path)
    assert main(["-c", str(cfg_path), "--no-banner", "--no-clear", "plan", "--dry-run"]) == 2
    assert "알 수 없는 항목" in capsys.readouterr().err

    cfg_path.write_text("[project]\ntemplate = 'j.jou'\n[cases\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="문법 오류"):
        load_config(cfg_path)
    cfg_path.write_text(text.replace("max_parallel = 2", 'max_parallel = "2"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_parallel"):
        load_config(cfg_path)
    cfg_path.write_text(text + '\n[results]\nobjective = "max_l"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="objective"):
        load_config(cfg_path)


def test_expression_and_template_errors():
    with pytest.raises(ExpressionError, match="ZeroDivisionError"):
        evaluate("1 / aoa", {"aoa": 0})
    with pytest.raises(ExpressionError, match="ValueError"):
        evaluate("sqrt(aoa - 1)", {"aoa": 0})
    with pytest.raises(ExpressionError):
        evaluate("2 ** 10 ** 10", {})          # 예전에는 거대한 정수를 계산하느라 멈췄다
    assert evaluate("2 ** 10", {}) == 1024 and isinstance(evaluate("2 ** 10", {}), int)
    with pytest.raises(ConfigError, match="형식"):
        render("${name:.2f}", {"name": "abc"})


def test_mesh_growth_rate_validated(tmp_path):
    from airfoil_surrogate.meshing import make_job

    with pytest.raises(ConfigError, match="growth_rate"):
        make_job({"main_airfoil": "0012"}, {"growth_rate": 1.0}, tmp_path)
    with pytest.raises(ConfigError, match="형식"):
        make_job({"main_airfoil": "0012"}, {"refine": "2"}, tmp_path)


def test_csv_coerce_and_mixed_split_sort():
    assert _coerce("nan") == "nan" and _coerce("inf") == "inf" and _coerce("0012") == "0012"
    assert _coerce("1e5") == 1e5 and _coerce("3") == 3
    rows = [{"state": "done", "cl": 1.0, "cd": 0.1, "l/d": 10.0, "geo": g, "speed": s}
            for g, s in (("a", 10), ("b", None), ("c", "fast"))]
    ranking = rank_geometries(rows, ["geo", "speed"], "aoa", "max_ld", ["speed"])
    assert [e["geo"] for e in ranking] == ["a", "c", "b"]


# ---------------------------------------------------------------- 수렴 판정

def test_incomplete_runs_are_not_labels():
    cfg = ConvergenceConfig(window=100)
    flat = (np.full(500, 0.5), np.full(500, 0.02))
    assert classify(flat, cfg, {"state": "done"})["conv_state"] == "converged"
    r = classify(flat, cfg, {"state": "failed", "message": "시간 초과 (60분)"})
    assert r["conv_state"] == "incomplete" and "시간 초과" in r["conv_reason"]
    assert classify(flat, cfg, {"state": "interrupted"})["conv_state"] == "incomplete"
    short = (np.full(5, 0.5), np.full(5, 0.02))
    assert classify(short, cfg, {"state": "done"})["conv_state"] == "incomplete"
    # 발산은 실행 상태보다 우선
    assert classify((np.full(500, 50.0), np.full(500, 0.02)), cfg, {"state": "failed"})["conv_state"] == "diverged"
    with pytest.raises(ConfigError):
        ConvergenceConfig.from_dict({"window": "200"})


# ---------------------------------------------------------------- 학습

def test_train_options_and_input_checks():
    from airfoil_surrogate.learn import TrainOptions, _check_finite, _scale, feature_columns

    for bad in ({"epochs": 0}, {"n_seeds": 0}, {"lr": 0.0}, {"ood_train_fraction": 1.5}, {"hidden": []},
                {"stall_weight": 0.0}):
        with pytest.raises(ConfigError):
            TrainOptions(**bad).validate()
    TrainOptions().validate()

    rows = [{"case": "a", "x": 1.0, "cl": math.nan}, {"case": "b", "x": None, "cl": 0.1}]
    with pytest.raises(ConfigError, match="2개"):
        _check_finite(rows, ["x", "cl"], "학습")

    # 값이 하나뿐인 열(Re 한 가지)은 표준편차 1 → 다른 값으로 예측해도 입력이 폭발하지 않음
    x = np.array([[1.0, 5.3], [2.0, 5.3], [3.0, 5.3]])
    mean, std = _scale(x)
    assert std[1] == 1.0 and std[0] == pytest.approx(np.std([1, 2, 3]))

    cols = feature_columns([{"cst8_u0": 1, "cst8_l0": 1, "cst8_rms": 0, "cst8_": 0, "cst8_max_err": 0}], "cst8")
    assert cols == ["cst8_u0", "cst8_l0", "aoa", "log10_re"]


def test_mlp_single_seed_has_no_interval():
    pytest.importorskip("torch")
    from airfoil_surrogate.learn import MLPEnsemble, TrainOptions

    rng = np.random.default_rng(0)
    rows = [{"f": float(v), "aoa": float(a), "cl": 0.1 * a, "log10_cd": -2 + 0.01 * v}
            for v, a in zip(rng.normal(size=60), rng.uniform(-5, 10, 60))]
    cols = ["f", "aoa"]
    one = MLPEnsemble(TrainOptions(n_seeds=1, epochs=20, hidden=[8]))
    one.fit(rows[:40], rows[40:], cols, np.ones(40))
    p = one.predict(rows[40:])
    assert "cl_lo" not in p and np.all(np.isfinite(p["cl"]))
    two = MLPEnsemble(TrainOptions(n_seeds=2, epochs=20, hidden=[8]))
    two.fit(rows[:40], rows[40:], cols, np.ones(40))
    p = two.predict(rows[40:])
    assert np.all(p["cl_hi"] >= p["cl_lo"])
