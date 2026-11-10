import csv
import json
import textwrap
from pathlib import Path

import pytest

from airfoil_surrogate.cases import build_cases, filter_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import ConfigError, load_config
from airfoil_surrogate.results import parse_report_file
from airfoil_surrogate.template import render

FAKE_SOLVER = Path(__file__).with_name("fake_solver.py")


def make_project(tmp_path, sweep="aoa = [0, 4]", extra=""):
    (tmp_path / "journal.jou").write_text(
        "; case ${case_name}\nAOA=${aoa} MACH=${mach} FX=${flow_x:.4f}\n/solve/iterate ${iterations}\n",
        encoding="utf-8",
    )
    (tmp_path / "cases.csv").write_text("mach,iterations\n0.3,100\n0.5,200\n", encoding="utf-8")
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"

        [solver]
        kind = "custom"
        command = ["${{python}}", "{FAKE_SOLVER.as_posix()}", "${{journal}}"]

        [constants]
        iterations = 50

        [cases]
        csv = "cases.csv"
        name = "M{{mach}}_AoA{{aoa}}"
        group = "M{{mach}}"

        [cases.sweep]
        {sweep}

        [derived]
        flow_x = "cos(radians(aoa))"

        [results]
        average_last = 2
        {extra}
    """), encoding="utf-8")
    return tmp_path / "project.toml"


def test_build_cases_grouping_and_derived(tmp_path):
    cfg = load_config(make_project(tmp_path))
    cases = build_cases(cfg)
    assert [c.name for c in cases] == ["M0.3_AoA0", "M0.3_AoA4", "M0.5_AoA0", "M0.5_AoA4"]
    assert cases[0].dir == cfg.output_dir / "M0.3" / "M0.3_AoA0"
    assert cases[0].params["iterations"] == 100  # CSV 값이 constants보다 우선
    assert cases[0].params["flow_x"] == pytest.approx(1.0)
    assert [c.name for c in filter_cases(cases, ["M0.5"])] == ["M0.5_AoA0", "M0.5_AoA4"]


def test_duplicate_case_names_rejected(tmp_path):
    path = make_project(tmp_path)
    path.write_text(path.read_text().replace('name = "M{mach}_AoA{aoa}"', 'name = "M{mach}"'))
    with pytest.raises(ConfigError, match="중복"):
        build_cases(load_config(path))


def test_render_reports_missing_variables():
    assert render("a=${x:.2f}", {"x": 1}) == "a=1.00"
    with pytest.raises(ConfigError, match="missing"):
        render("${missing}", {})


def test_parse_report_file(tmp_path):
    p = tmp_path / "r.out"
    p.write_text('"r"\n"Iteration" "cl"\n("Iteration" "cl")\n1 0.5\n2 0.6\n')
    assert parse_report_file(p) == (["Iteration", "cl"], [[1.0, 0.5], [2.0, 0.6]])


def test_run_status_collect(tmp_path, capsys):
    cfg_path = make_project(tmp_path, sweep="aoa = [0, 4, 99, -99]")
    assert main(["-c", str(cfg_path), "plan", "--dry-run"]) == 0
    assert not (tmp_path / "runs").exists()

    assert main(["-c", str(cfg_path), "run", "-j", "2"]) == 1  # 실패 케이스 포함

    cfg = load_config(cfg_path)
    states = {c.name: json.loads((c.dir / "status.json").read_text())["state"] for c in build_cases(cfg)}
    assert states["M0.3_AoA4"] == "done"
    assert states["M0.3_AoA99"] == "failed"   # 로그의 "Divergence detected"
    assert states["M0.3_AoA-99"] == "failed"  # 종료 코드 3

    journal = (cfg.output_dir / "M0.5" / "M0.5_AoA4" / "run.jou").read_text()
    assert "AOA=4 MACH=0.5 FX=0.9976" in journal
    assert "/solve/iterate 200" in journal

    with open(cfg.output_dir / "summary.csv", encoding="utf-8-sig") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    # 마지막 2개 iteration 평균: 0.4 + (0.009 + 0.010) / 2
    assert float(rows["M0.3_AoA4"]["cl"]) == pytest.approx(0.4095)
    assert rows["M0.3_AoA4"]["iter_done"] == "10"

    capsys.readouterr()
    assert main(["-c", str(cfg_path), "run", "--only", "M0.3_AoA0"]) == 0
    assert "skipped" in capsys.readouterr().out  # 완료된 케이스는 재실행 안 함


def test_timeout_marks_failed(tmp_path):
    cfg_path = make_project(tmp_path, sweep="aoa = [0]")
    text = cfg_path.read_text().replace(
        '[constants]', 'timeout_minutes = 0.002\n\n[constants]'
    ).replace(f'"{FAKE_SOLVER.as_posix()}", "${{journal}}"', '"-c", "import time; time.sleep(30)"')
    cfg_path.write_text(text)
    assert main(["-c", str(cfg_path), "run", "--only", "M0.3_AoA0"]) == 1
    cfg = load_config(cfg_path)
    status = json.loads((build_cases(cfg)[0].dir / "status.json").read_text())
    assert status["state"] == "failed" and "시간 초과" in status["message"]
