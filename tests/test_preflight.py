"""Fluent 실행 전 점검(afs check)과 배치 실행 안전장치 (transcript, 이전 결과 정리, 남은 시간)."""
import json
import textwrap
from pathlib import Path

from airfoil_surrogate.cases import build_cases
from airfoil_surrogate.cli import _eta, main
from airfoil_surrogate.config import load_config
from airfoil_surrogate.preflight import ERROR, OK, TODO, WARN, run_checks
from airfoil_surrogate.runner import TRANSCRIPT_NAME, read_status, run_case

# 가짜 Fluent: 저널 줄을 transcript.txt에 "> " 프롬프트와 함께 그대로 찍고(Fluent처럼), 리포트 파일을 쓴다.
# AOA=99면 transcript에만 오류를 남긴다 (Windows -hidden 실행처럼 표준 출력은 비어 있음).
FAKE = '''
import re, sys
from pathlib import Path
journal = Path(sys.argv[1]).read_text(encoding="utf-8")
aoa = float(re.search(r"AOA=(\\S+)", journal).group(1))
lines = ["> " + ln for ln in journal.splitlines()]
if aoa >= 99:
    lines.append("Error: Divergence detected in AMG solver: x-momentum")
Path("transcript.txt").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
rows = ['"coef-rfile"', '("Iteration" "cl" "cd")'] + [f"{i} {0.1 * aoa:.6e} {0.01:.6e}" for i in range(1, 6)]
Path("coef-rfile.out").write_text("\\n".join(rows) + "\\n")
'''

MESH_SMALL = """
[mesh]
chord = 0.3
reynolds = 2e5
surface_cells = 40
farfield_radius = 8
farfield_size = 1.0
near_size = 0.03
wake_size = 0.05
wake_length = 1.0
"""


def make_project(tmp_path, journal: str, extra: str = "", sweep: str = "aoa = [0, 4]") -> Path:
    (tmp_path / "fake.py").write_text(FAKE, encoding="utf-8")
    (tmp_path / "journal.jou").write_text(journal, encoding="utf-8")
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"

        [solver]
        kind = "custom"
        command = ["${{python}}", "{(tmp_path / 'fake.py').as_posix()}", "${{journal}}"]

        [constants]
        main_airfoil = "0012"

        [cases]
        name = "a{{aoa}}"

        [cases.sweep]
        {sweep}
    """) + textwrap.dedent(extra), encoding="utf-8")
    return tmp_path / "project.toml"


def levels(items):
    return {it.level for it in items}


def titles(items, level):
    return [it.title for it in items if it.level == level]


def test_transcript_errors_and_echoed_comments(tmp_path):
    journal = ('/file/start-transcript "transcript.txt"\n'
               "; 주석에 Error: 가 있어도 (transcript에 찍혀도) 오류가 아님\n"
               "AOA=${aoa}\n")
    cfg = load_config(make_project(tmp_path, journal, sweep="aoa = [0, 99]"))
    ok, bad = build_cases(cfg)
    assert run_case(cfg, ok).state == "done"
    r = run_case(cfg, bad)
    assert r.state == "failed" and "Divergence detected" in r.message  # transcript에서 잡음

    # 다시 실행할 때 이전 결과(result.cas.h5, transcript 등)를 먼저 지운다 (덮어쓰기 확인 질문 방지)
    stale = [ok.dir / "result.cas.h5", ok.dir / "result.dat.h5", ok.dir / "img_old.png"]
    for p in stale:
        p.write_text("old")
    assert run_case(cfg, ok, force=True).state == "done"
    assert not any(p.exists() for p in stale)
    assert (ok.dir / TRANSCRIPT_NAME).is_file()


def test_status_eta():
    statuses = [{"state": "done", "elapsed_sec": 120}, {"state": "done", "elapsed_sec": 240},
                {"state": "pending"}, {"state": "pending"}, {"state": "failed"}]
    text = _eta(statuses, parallel=1)
    assert "남은 케이스 3개" in text and "3.0분" in text  # 중앙값 3분 × 3개
    assert _eta([{"state": "pending"}], 1) is None


def test_check_base_case_meshes_and_zones(tmp_path):
    journal = ('/file/start-transcript "transcript.txt"\n'
               '/file/read-case "${project_dir}/${base_case}"\n'
               '/file/replace-mesh "${mesh_file}"\n'
               '/solve/report-files/edit "coef-rfile" file-name "coef-rfile.out" q\n'
               "AOA=${aoa}\n")
    extra = """
        [[cases.set]]
        label = "farfield"
        name = "s1_a{aoa}"
        constants = { base_case = "base.cas.h5" }

        [[cases.set]]
        label = "split"
        name = "s2_a{aoa}"
        constants = { base_case = "base.cas.h5", farfield_split = true, mesh_level = 2 }
    """ + MESH_SMALL
    cfg_path = make_project(tmp_path, journal, extra, sweep="aoa = [0]")
    cfg = load_config(cfg_path)
    cases = build_cases(cfg)
    items = run_checks(cfg, cases)
    assert any("base.cas.h5" in t for t in titles(items, TODO))
    assert any("메쉬 만들기: 2개" in t for t in titles(items, TODO))
    assert any("리포트 파일 coef-rfile.out" in t for t in titles(items, OK))

    (tmp_path / "base.cas.h5").write_text("x")
    assert main(["-c", str(cfg_path), "--no-banner", "mesh", "-j", "2"]) == 0
    items = run_checks(cfg, cases)
    assert not titles(items, TODO)
    # 바깥 경계가 다른 두 세트(farfield / inlet + outlet)가 같은 베이스 케이스를 쓰면 replace-mesh가 실패한다
    assert any("경계 이름이 다름" in t for t in titles(items, ERROR))
    assert main(["-c", str(cfg_path), "--no-banner", "check"]) == 1


def test_check_journal_problems_and_paths(tmp_path):
    proj = tmp_path / "한글폴더"
    proj.mkdir()
    journal = ('/solve/report-files/edit "coef-rfile" file-name "coef.txt" q\n'
               "/display/save-picture \"img.png\"\nAOA=${aoa}\n")
    cfg = load_config(make_project(proj, journal))
    items = run_checks(cfg, build_cases(cfg))
    assert any("report_files 패턴과 안 맞음" in t for t in titles(items, ERROR))
    warn = titles(items, WARN)
    assert any("start-transcript" in t for t in warn)
    assert any("graphics = false" in t for t in warn)
    assert any("영문이 아닌 문자" in t for t in warn)
    assert ERROR in levels(items)


def test_check_uses_finished_cases_for_estimates(tmp_path):
    journal = '/file/start-transcript "transcript.txt"\nAOA=${aoa}\n'
    cfg_path = make_project(tmp_path, journal, sweep="aoa = [0, 4, 8]")
    cfg = load_config(cfg_path)
    cases = build_cases(cfg)
    run_case(cfg, cases[0])
    status = json.loads((cases[0].dir / "status.json").read_text(encoding="utf-8"))
    status["elapsed_sec"] = 300
    (cases[0].dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert read_status(cases[0])["state"] == "done"
    items = run_checks(cfg, cases)
    detail = " ".join(it.detail for it in items)
    assert "케이스당 5.0분(끝난 1개 중앙값)" in detail
    assert "끝난 케이스 폴더 크기 중앙값" in detail
