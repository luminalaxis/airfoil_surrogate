"""팀 파이프라인 명세 (§0~§9) 테스트: 유동 설정·모드, (m, p, t) 형상, LHS, 메쉬 판정, 수렴, 결과 스키마, 검증, 학습."""

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from airfoil_surrogate.cases import build_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import ConfigError, load_config
from airfoil_surrogate.convergence import ConvergenceConfig, classify
from airfoil_surrogate.cp import interpolate_cp
from airfoil_surrogate.flow import FlowConfig, retheta_t
from airfoil_surrogate.fluent_msh import Boundary, check_msh, write_msh
from airfoil_surrogate.geometry import naca4, naca4_mpt, naca_code, normalize, section_from_params
from airfoil_surrogate.results import RESULT_COLUMNS, _moment, _wind_axes
from airfoil_surrogate.sampling import (SamplingConfig, generate_cases, latin_hypercube, valid_naca_codes,
                                        validation_cases)
from airfoil_surrogate.validate import (Airfoil2VecConfig, compare_reference, marching_alphas,
                                        mesh_mode_difference, parse_polar, xfoil_script)

HERE = Path(__file__).parent
FAKE_SOLVER = HERE / "fake_naca_solver.py"
FAKE_XFOIL = HERE / "fake_xfoil.py"
EXAMPLE = HERE.parent / "examples" / "naca4" / "project.toml"

FLOW = {
    "chord": 0.1, "nu": 1.4607e-5, "turbulence_intensity": 0.01, "length_scale": 0.01,
    "production": {"reynolds": 1e5, "y_plus": 1.0, "wall_treatment": "low_re"},
    "validation": {"chord": 1.0, "reynolds": 4.66e6, "y_plus": 50, "wall_treatment": "wall_function",
                   "naca_te": "sharp"},
}


# ================================================================ §0 유동 조건·모드

def test_flow_derives_velocity_re_and_reference_values():
    f = FlowConfig.from_dict(FLOW)
    assert f.Re == pytest.approx(1e5)
    assert f.U == pytest.approx(1e5 * 1.4607e-5 / 0.1)        # 14.6 m/s
    c = f.constants()
    assert c["aref"] == pytest.approx(0.1) and c["lref"] == pytest.approx(0.1)
    assert c["velocity"] == pytest.approx(f.U) and c["mesh_tag"] == "pro-light"

    v = FlowConfig.from_dict(FLOW, {"re_mode": "validation", "mesh_mode": "fine"})
    assert v.Re == pytest.approx(4.66e6) and v.chord == 1.0 and v.naca_te == "sharp"
    assert v.constants()["aref"] == pytest.approx(1.0)        # Aref = c × span 이 모드에 따라 바뀜
    assert v.U == pytest.approx(4.66e6 * 1.4607e-5)
    assert v.tag == "validation_kOmegaSST_fine_validation"

    by_u = FlowConfig.from_dict({**FLOW, "production": {"velocity": 15.0}})
    assert by_u.Re == pytest.approx(15.0 * 0.1 / 1.4607e-5)   # U 15 m/s → Re ≈ 1.03e5


def test_turbulence_inlet_from_intensity_and_length_scale():
    f = FlowConfig.from_dict(FLOW)
    i = f.turbulence_inlet()
    k = 1.5 * (f.U * 0.01) ** 2
    assert i["k_in"] == pytest.approx(k)
    assert i["omega_in"] == pytest.approx(math.sqrt(k) / (0.09**0.25 * 0.01 * 0.1))
    assert retheta_t(1.0) == pytest.approx(1173.51 - 589.428 + 0.2196, rel=1e-6)
    lm = FlowConfig.from_dict(FLOW, {"turbulence_model": "kOmegaSSTLM"})
    assert "intermittency_in" in lm.model_journal_lines()
    assert "intermittency" not in f.model_journal_lines()


@pytest.mark.parametrize("bad, msg", [
    ({"production": {"reynolds": 1e5, "velocity": 15}}, "하나만"),
    ({"production": {"reynolds": 1e5, "y_plus": 50}}, "low_re"),
    ({"turbulence_model": "kOmegaSSTLM", "production": {"reynolds": 1e5, "y_plus": 50,
                                                        "wall_treatment": "wall_function"}}, "Transition"),
    ({"mesh_mode": "medium"}, "mesh_mode"),
])
def test_flow_rejects_inconsistent_settings(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        FlowConfig.from_dict({**FLOW, **bad})


def test_example_project_fixed_conditions():
    cfg = load_config(EXAMPLE)
    assert cfg.flow.chord == 0.1 and cfg.flow.turbulence_model == "kOmegaSST"   # 코드 100 mm, 난류 모델 고정
    assert cfg.constants["base_case"] == "base_sst.cas.h5"
    assert cfg.mesh["surface_cells"] == 120 and cfg.output_dir.name == cfg.flow.tag
    assert load_config(EXAMPLE, {"mesh_mode": "fine"}).mesh["surface_cells"] == 300
    assert load_config(EXAMPLE, {"re_mode": "validation"}).flow.chord == 0.1
    with pytest.raises(ConfigError, match="kOmegaSSTLM"):
        load_config(EXAMPLE, {"turbulence_model": "kOmegaSSTLM"})       # 다른 난류 모델은 막힘
    rows = generate_cases(SamplingConfig.from_dict(cfg.sampling))
    assert len(rows) == 300 and len({r["shape_id"] for r in rows}) == 100
    assert {r["alpha"] for r in rows} == {0.0, 8.0, 15.0}


# ================================================================ §1 형상

def test_naca_mpt_matches_code_and_handles_symmetric_p0():
    np.testing.assert_allclose(naca4_mpt(2, 4, 12), naca4("2412"))
    sym = naca4_mpt(0, 0, 12)                                  # p = 0 대칭익: 0으로 나누지 않음
    assert np.all(np.isfinite(sym))
    np.testing.assert_allclose(sym[:, 1], -sym[::-1, 1], atol=1e-12)
    assert np.all(np.isfinite(naca4_mpt(0, 4, 12)))
    with pytest.raises(ConfigError, match="p가 0"):
        naca4_mpt(2, 0, 12)
    # 실수 파라미터 (LHS): 정수 코드 사이 값도 그대로
    xy = naca4_mpt(2.37, 3.81, 11.29)
    el = section_from_params({"m": 2.37, "p": 3.81, "t": 11.29}, Path("."))
    assert len(xy) > 100 and el.xy.shape[1] == 2
    assert naca_code(2.37, 3.81, 11.29) == "2411"


def test_trailing_edge_options_and_cosine_spacing():
    sharp, blunt = naca4_mpt(2, 4, 12, te="sharp"), naca4_mpt(2, 4, 12, te="blunt")
    assert np.allclose(sharp[0], sharp[-1])
    gap = np.linalg.norm(blunt[0] - blunt[-1])
    assert gap == pytest.approx(2 * 5 * 0.12 * (0.2969 - 0.126 - 0.3516 + 0.2843 - 0.1015), rel=0.05)
    xy, le = normalize(blunt)
    dx = np.diff(xy[le:, 0])
    assert dx[0] < dx[len(dx) // 2] / 5 and dx[-1] < dx[len(dx) // 2] / 5   # 앞전·뒷전에 밀집
    assert section_from_params({"m": 2, "p": 4, "t": 12, "naca_te": "sharp"}, Path(".")).sharp_te


# ================================================================ §2 케이스 목록

def test_latin_hypercube_one_point_per_bin_and_seeded():
    u = latin_hypercube(50, 4, np.random.default_rng(1))
    for j in range(4):
        assert sorted(np.floor(u[:, j] * 50).astype(int)) == list(range(50))
    np.testing.assert_array_equal(u, latin_hypercube(50, 4, np.random.default_rng(1)))


def test_generate_cases_nested_ranges_assignees_and_reproducible():
    sc = SamplingConfig.from_dict({"n_shapes": 20, "alpha_per_shape": 4, "assignees": ["a", "b", "c", "d"]})
    rows = generate_cases(sc)
    assert rows == generate_cases(sc)                       # 시드 고정 → 같은 목록
    assert list(rows[0]) == ["case_id", "shape_id", "m", "p", "t", "alpha", "assignee"]
    assert len(rows) == 80 and len({r["case_id"] for r in rows}) == 80
    for key, (lo, hi) in (("m", (0, 6)), ("p", (2, 6)), ("t", (6, 18)), ("alpha", (0, 12))):
        assert all(lo <= r[key] <= hi for r in rows)
    by_shape = {}
    for r in rows:
        by_shape.setdefault(r["shape_id"], []).append(r)
    for rs in by_shape.values():
        assert len({r["assignee"] for r in rs}) == 1         # 형상 단위 담당 (메쉬 한 번)
        assert sorted(int(r["alpha"] // 3) for r in rs) == [0, 1, 2, 3]   # 형상마다 α 층화
    counts = {a: sum(1 for r in rows if r["assignee"] == a) for a in "abcd"}
    assert set(counts.values()) == {20}
    joint = generate_cases(SamplingConfig.from_dict({"mode": "joint", "n_cases": 30}))
    assert len({r["shape_id"] for r in joint}) == 30


def test_naca_code_grid_same_alphas_for_all_shapes():
    sc = SamplingConfig.from_dict({"mode": "naca_codes", "n_shapes": 100, "alpha_values": [0, 8, 15],
                                   "m": [0, 6], "p": [2, 6], "t": [6, 18]})
    pool = valid_naca_codes(sc)
    assert len(pool) == 13 + 6 * 5 * 13 and "0012" in pool and "2412" in pool and "2012" not in pool
    rows = generate_cases(sc)
    assert rows == generate_cases(sc)
    shapes = {}
    for r in rows:
        shapes.setdefault(r["shape_id"], []).append(r["alpha"])
    assert len(shapes) == 100 and all(sorted(a) == [0, 8, 15] for a in shapes.values())
    assert rows[0]["case_id"] == f"{rows[0]['shape_id']}_a0"
    assert all(r["p"] == 0 for r in rows if r["m"] == 0)
    fixed = generate_cases(SamplingConfig.from_dict({"mode": "naca_codes", "codes": ["0012", "2412"],
                                                     "alpha_values": [0, 8, 15]}))
    assert [r["case_id"] for r in fixed][:3] == ["naca0012_a0", "naca0012_a8", "naca0012_a15"]


def test_marching_alphas_for_xfoil():
    assert marching_alphas([0, 8, 15], 1.0) == [float(a) for a in range(16)]
    assert marching_alphas([0, 8, 15], 0) == [0.0, 8.0, 15.0]
    assert marching_alphas([-4, 2], 2.0) == [-4.0, -2.0, 2.0]


def test_validation_case_list():
    rows = validation_cases(SamplingConfig.from_dict({"validation": {"shapes": ["0012", "2412"], "alpha": [0, 4, 8]}}))
    assert len(rows) == 6 and rows[0]["case_id"] == "v_naca0012_a0" and rows[3]["m"] == 2


# ================================================================ §3 메쉬 판정

def test_mesh_quality_aspect_ratio_and_non_orthogonality(tmp_path):
    # 사각형 셀 2개: 직사각형(AR 4), 기울어진 사다리꼴(가장 긴 변 7 / 가장 짧은 변 1)
    nodes = np.array([[0, 0], [4, 0], [9, 0], [0, 1], [4, 1], [11, 1]], float)
    cells = [(0, 1, 4, 3), (1, 2, 5, 4)]
    bnd = [Boundary("wall", "wall", [(0, 1), (1, 2)]), Boundary("far", "pressure-far-field",
                                                                   [(2, 5), (5, 4), (4, 3), (3, 0)])]
    write_msh(tmp_path / "m.msh", nodes, cells, bnd)
    c = check_msh(tmp_path / "m.msh")
    assert c["max_aspect_ratio"] == pytest.approx(7.0)
    assert 0 < c["max_non_orthogonality"] < 60


# ================================================================ §4 양력·항력 방향, CM

def test_wind_axes_lift_drag_directions_follow_alpha():
    a = 8.0
    cl, cd = 0.9, 0.02
    r = math.radians(a)
    row = {"alpha": a, "cx": cd * math.cos(r) - cl * math.sin(r), "cy": cl * math.cos(r) + cd * math.sin(r)}
    _wind_axes(row, "alpha")
    assert row["cl"] == pytest.approx(cl) and row["cd"] == pytest.approx(cd)
    # 베이스 케이스에 고정 방향(α = 0) cl/cd 리포트가 남아 있으면 버그로 표시
    stale = {"alpha": a, "cx": row["cx"], "cy": row["cy"], "cl": row["cy"], "cd": row["cx"]}
    _wind_axes(stale, "alpha")
    assert stale["force_dir_check"] == "mismatch" and stale["cl"] == pytest.approx(cl)


def test_moment_transfer_to_quarter_chord():
    # 대칭익: 양력이 c/4에 작용 → CM(c/4) = 0
    row = {"cy": 0.4, "cmz": 0.25 * 0.4}
    _moment(row)
    assert row["cm"] == pytest.approx(0.0)
    # 압력 중심이 0.4c → 머리 숙임 (CM < 0)
    row = {"cy": 0.4, "cmz": 0.4 * 0.4}
    _moment(row)
    assert row["cm"] == pytest.approx(-0.15 * 0.4)


# ================================================================ §6 수렴 판정

def test_converged_flag_requires_both_criteria():
    it = np.arange(500)
    steady = (0.8 + 0.3 * np.exp(-it / 20), np.full(500, 0.02))
    osc = (0.8 + 0.05 * np.sin(it / 5), np.full(500, 0.02))
    both = ConvergenceConfig(window=100, residual_tol=1e-5, criteria="both")
    low = {"res_continuity": 3e-6, "res_k": 1e-6}
    high = {"res_continuity": 3e-4, "res_k": 1e-6}
    assert classify(steady, both, log_info=low)["converged"] is True
    r = classify(steady, both, log_info=high)
    assert r["converged"] is False and r["conv_coef_ok"] and not r["conv_res_ok"]
    assert classify(osc, both, log_info=low)["converged"] is False
    assert classify(None, both, {"state": "failed", "message": "Divergence detected"})["converged"] is False
    assert classify(steady, ConvergenceConfig(window=100), log_info=high)["converged"] is True  # either (예전 동작)


# ================================================================ 표면 Cp

def test_cp_interpolation_by_surface(tmp_path):
    xy, le = normalize(naca4("0012"))
    lines = ["nodenumber, x-coordinate, y-coordinate, pressure-coefficient"]
    for i, (x, y) in enumerate(xy):
        lines.append(f"{i}, {x * 0.1}, {y * 0.1}, {(x - 1) if y >= 0 else x / 2}")
    (tmp_path / "cp.txt").write_text("\n".join(lines))
    out = interpolate_cp(tmp_path / "cp.txt", [0.25, 0.5], xy, le, 0.1)
    assert out["cp_u_0.500"] == pytest.approx(-0.5, abs=1e-3)
    assert out["cp_l_0.250"] == pytest.approx(0.125, abs=1e-3)


# ================================================================ §5·§7 실행·결과 스키마 (가짜 솔버)

def make_project(tmp, extra=""):
    (tmp / "journal.jou").write_text(
        "; ${case_name}\nAOA=${flow_angle} RE=${reynolds} M=${m} P=${p} T=${t} CHORD=${chord} MESH=${mesh_file}\n"
        "AREF=${aref} LREF=${lref} K=${k_in} W=${omega_in}\n${model_lines}\n${cp_export}\n", encoding="utf-8")
    (tmp / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"

        [solver]
        kind = "custom"
        command = ["${{python}}", "{FAKE_SOLVER.as_posix()}", "${{journal}}"]

        [flow]
        chord = 0.1
        nu = 1.4607e-5
        [flow.production]
        reynolds = 1.0e5
        [flow.validation]
        chord = 1.0
        reynolds = 4.66e6
        y_plus = 50
        wall_treatment = "wall_function"
        naca_te = "sharp"

        [sampling]
        n_shapes = 4
        alpha_per_shape = 2
        m = [0.0, 6.0]
        assignees = ["kim", "lee"]
        [sampling.validation]
        shapes = ["0012", "2412"]
        alpha = [0, 4]

        [cases]
        csv = "cases.csv"
        csv_validation = "cases_validation.csv"
        name = "{{case_id}}"
        group = "{{shape_id}}"

        [mesh]
        farfield_shape = "box"
        box_upstream = 5
        box_downstream = 8
        box_height = 5
        [mesh.light]
        surface_cells = 40
        near_size = 0.04
        wake_size = 0.08
        box_far_size = 2.0
        [mesh.fine]
        surface_cells = 60
        near_size = 0.03
        wake_size = 0.06
        box_far_size = 2.0

        [convergence]
        window = 100
        residual_tol = 1e-5
        criteria = "both"

        [results]
        average_last = 100
        x_param = "alpha"
        {extra}

        [ml]
        shape_key = "shape_id"
        split = [0.5, 0.25, 0.25]
        train_states = ["converged"]
        require_converged = true
    """), encoding="utf-8")
    return tmp / "project.toml"


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_end_to_end_assignee_run_results_schema_merge(tmp_path, capsys):
    cfg_path = make_project(tmp_path, "cp_stations = [0.25, 0.5]\ncp_export_command = '/file/export/ascii \"cp.txt\"'")
    args = ["--no-banner", "--no-clear", "-c", str(cfg_path)]
    assert main(args + ["make-cases"]) == 0
    assert main(args + ["make-cases"]) == 0                  # 같은 내용이면 그대로 둠
    text = (tmp_path / "cases.csv").read_text()
    (tmp_path / "cases.csv").write_text(text.replace("kim", "park", 1))
    assert main(args + ["make-cases"]) == 2                  # 다르면 덮어쓰지 않음 (--force 필요)
    assert main(args + ["make-cases", "--force"]) == 0

    assert main(args + ["run", "--assignee", "kim"]) in (0, 1)
    cfg = load_config(cfg_path)
    cases = build_cases(cfg)
    mine = [c for c in cases if c.params["assignee"] == "kim"]
    assert all((c.dir / "status.json").is_file() for c in mine)
    assert not any((c.dir / "status.json").is_file() for c in cases if c.params["assignee"] == "lee")
    journal = (mine[0].dir / "run.jou").read_text()
    assert "AREF=0.1 LREF=0.1" in journal and "export/ascii" in journal

    res = _read(cfg.output_dir / "results.csv")
    assert list(res[0])[:len(RESULT_COLUMNS)] == list(RESULT_COLUMNS)
    assert {r["case_id"] for r in res} == {c.params["case_id"] for c in mine}
    by_id = {c.params["case_id"]: c for c in mine}
    for r in res:
        p = by_id[r["case_id"]].params
        assert float(r["CL"]) == pytest.approx(0.105 * (p["alpha"] + 1.05 * p["m"]), rel=1e-4)
        assert float(r["CM"]) == pytest.approx(-0.025 * p["m"], abs=1e-9)
        assert r["converged"] == str(p["m"] < 5.5)
        assert r["turb_model"] == "kOmegaSST" and r["mesh_mode"] == "light" and float(r["Re"]) == 1e5
        assert r["solver_version"] == "Fluent 2026 R1" and int(r["n_cells"]) > 100
        assert float(r["yplus_max"]) == pytest.approx(0.8) and float(r["yplus_mean"]) == pytest.approx(0.35)
        assert int(float(r["n_iter"])) == 400 and float(r["runtime_s"]) >= 0
        assert float(r["cp_u_0.500"]) == pytest.approx(-0.5, abs=0.01)
    log = _read(cfg.output_dir / "run_log.csv")
    assert {r["assignee"] for r in log} == {"kim"} and len(log) == len(mine)

    capsys.readouterr()
    assert main(args + ["run", "--assignee", "kim"]) in (0, 1)
    assert "skipped" in capsys.readouterr().out               # 완료 케이스 건너뜀
    assert len(_read(cfg.output_dir / "run_log.csv")) == len(mine)

    assert main(args + ["run", "--assignee", "lee"]) in (0, 1)
    lee_results = tmp_path / "lee.csv"
    lee_results.write_text((cfg.output_dir / "results.csv").read_text(encoding="utf-8-sig"), encoding="utf-8")
    assert main(args + ["merge-results", str(tmp_path / "all.csv"), str(cfg.output_dir / "results.csv"),
                        str(lee_results)]) == 0
    assert len(_read(tmp_path / "all.csv")) == len(cases)
    assert main(args + ["run", "--assignee", "nobody"]) == 2

    # §9 학습: (m, p, t, α) 입력, converged == False 제외
    assert main(args + ["dataset"]) == 0
    ds = _read(cfg.output_dir / "dataset" / "dataset.csv")
    assert {"m", "p", "t", "converged"} <= set(ds[0])
    split = json.loads((cfg.output_dir / "dataset" / "split.json").read_text())["assignments"]
    for r in ds:                                             # 형상 단위 분할
        assert r["split"] == split[r["shape"]]
    assert main(args + ["train", "--model", "baseline", "--features", "mpt", "--name", "mpt"]) == 0
    run = json.loads((cfg.output_dir / "ml" / "mpt" / "run.json").read_text())
    assert run["features"][:4] == ["m", "p", "t", "aoa"]
    preds = _read(cfg.output_dir / "ml" / "mpt" / "predictions.csv")
    assert all(float(r["m"]) < 5.5 for r in preds if r["conv_state"] == "converged")


def test_validation_mode_uses_own_case_list_mesh_and_folder(tmp_path):
    cfg_path = make_project(tmp_path)
    args = ["--no-banner", "--no-clear", "-c", str(cfg_path)]
    assert main(args + ["make-cases"]) == 0
    for mm in ("light", "fine"):
        assert main(args + ["--re-mode", "validation", "--mesh-mode", mm, "run"]) == 0
    v = load_config(cfg_path, {"re_mode": "validation"})
    cases = build_cases(v)
    assert {c.params["shape_id"] for c in cases} == {"naca0012", "naca2412"}
    assert cases[0].mesh.name.endswith("_val-light") and cases[0].mesh.geometry_params["naca_te"] == "sharp"
    res = _read(v.output_dir / "results.csv")
    assert float(res[0]["Re"]) == pytest.approx(4.66e6)
    fine = load_config(cfg_path, {"re_mode": "validation", "mesh_mode": "fine"})
    n_light = json.loads((cases[0].mesh.dir / "mesh_info.json").read_text())["cells"]
    n_fine = json.loads((build_cases(fine)[0].mesh.dir / "mesh_info.json").read_text())["cells"]
    assert n_fine > n_light

    # §8 1단계: 대칭 변환 — 그쪽 inverted 형상 −α의 부호를 바꾼 값이 우리 +α와 같으면 통과
    meta = []
    for r in res:
        code = r["case_id"].split("_")[1].removeprefix("naca")
        a = float(r["alpha"])
        meta.append({"airfoil": f"NACA{code}_inverted", "aoa": -a, "Re": 4.66e6,
                     "Cl": -float(r["CL"]) * 1.01, "Cd": float(r["CD"]) * 1.05, "Cm": -float(r["CM"])})
        meta.append({"airfoil": f"NACA{code}", "aoa": a, "Re": 4.66e6, "Cl": 99, "Cd": 99, "Cm": 99})
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(meta), tmp_path / "metadata.parquet")
    assert main(args + ["validate", "airfoil2vec", "--metadata", "metadata.parquet"]) == 0
    report = _read(tmp_path / "runs" / "validation" / "airfoil2vec_report.csv")
    assert len(report) == 8 and all(r["pass"] == "True" for r in report if r["CL_ref"])
    assert (tmp_path / "runs" / "validation" / "mesh_light_vs_fine.csv").is_file()


def test_airfoil2vec_symmetry_and_criteria():
    a2v = Airfoil2VecConfig()
    ours = [{"case_id": "v", "m": 2, "p": 4, "t": 12, "alpha": 4.0, "CL": 0.66, "CD": 0.011, "CM": -0.05,
             "mesh_mode": "light", "converged": True}]
    ref = {"naca2412_inverted": [{"alpha": -4.0, "cl": -0.65, "cd": 0.0105, "cm": 0.051},
                                 {"alpha": 4.0, "cl": 9.0, "cd": 9.0, "cm": 9.0}]}
    r = compare_reference(ours, ref, a2v)[0]
    assert r["CL_ref"] == pytest.approx(0.65) and r["CD_ref"] == pytest.approx(0.0105) and r["CM_ref"] == pytest.approx(-0.051)
    assert r["CL_pass"] and r["CD_pass"] and r["pass"]
    ours[0]["CD"] = 0.0125                                   # +19% → CD ±10% 기준 미통과
    assert not compare_reference(ours, ref, a2v)[0]["CD_pass"]
    d = mesh_mode_difference({"light": [{"case_id": "x", "CL": 0.5, "CD": 0.011}],
                              "fine": [{"case_id": "x", "CL": 0.52, "CD": 0.010}]})
    assert d[0]["dCL"] == pytest.approx(-0.02) and d[0]["dCD_rel"] == pytest.approx(0.1)


def test_xfoil_script_and_polar(tmp_path):
    s = xfoil_script("foil.dat", "polar.txt", 1e5, [8, 0, 4], type("X", (), {
        "panels": 200, "mach": 0, "ncrit": 9, "iterations": 100})())
    assert "VISC 100000" in s and "N 9" in s
    assert s.index("ALFA 0") < s.index("ALFA 4") < s.index("ALFA 8")
    (tmp_path / "p.txt").write_text("x\n  ------ ----\n   4.000   0.6500   0.01200   0.00500  -0.0500   0.4  0.9\n")
    assert parse_polar(tmp_path / "p.txt")[0]["CD"] == pytest.approx(0.012)


@pytest.mark.skipif(sys.platform == "win32", reason="가짜 XFOIL 실행 스크립트(#!)는 POSIX 전용")
def test_validate_xfoil_compares_turbulence_models(tmp_path):
    cfg_path = make_project(tmp_path)
    text = cfg_path.read_text().replace("[ml]", f'[validation.xfoil]\nexecutable = "{FAKE_XFOIL.as_posix()}"\n\n[ml]')
    cfg_path.write_text(text)
    args = ["--no-banner", "--no-clear", "-c", str(cfg_path)]
    assert main(args + ["make-cases"]) == 0
    assert main(args + ["--case-list", "validation", "run"]) == 0
    assert main(args + ["validate", "xfoil"]) == 0
    summary = json.loads((tmp_path / "runs" / "validation" / "xfoil_summary.json").read_text())
    assert summary["by_model"]["kOmegaSST"]["compared"] == 4
    assert "error" in summary["by_model"]["kOmegaSSTLM"]      # SST-LM 결과는 아직 없음
    assert summary["recommended"] == "kOmegaSST"
    assert len(_read(tmp_path / "runs" / "validation" / "xfoil_polars.csv")) == 4


def make_grid_project(tmp):
    """NACA 코드 격자 (형상 8개 × 0·8·15°), 코드 0.1 m, SST 고정."""
    path = make_project(tmp)
    text = path.read_text()
    text = text.replace("n_shapes = 4\nalpha_per_shape = 2\nm = [0.0, 6.0]",
                        'mode = "naca_codes"\nn_shapes = 8\nalpha_values = [0, 8, 15]\nm = [0, 6]')
    text = text.replace("[ml]", f'[validation.xfoil]\nexecutable = "{FAKE_XFOIL.as_posix()}"\n\n[ml]')
    text = text.replace("split = [0.5, 0.25, 0.25]", 'split = [0.5, 0.25, 0.25]\nmodels = ["lgbm", "mlp"]\nfeatures = "mpt"')
    path.write_text(text)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="가짜 XFOIL 실행 스크립트(#!)는 POSIX 전용")
def test_xfoil_vs_fluent_then_train_mlp_lgbm(tmp_path, capsys):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    cfg_path = make_grid_project(tmp_path)
    args = ["--no-banner", "--no-clear", "-c", str(cfg_path)]
    assert main(args + ["make-cases"]) == 0
    cfg = load_config(cfg_path)
    assert len(build_cases(cfg)) == 24

    # 1. 해석툴(XFOIL)로 해석
    assert main(args + ["xfoil"]) == 0
    xf = _read(tmp_path / "runs" / "xfoil_production_main" / "results.csv")
    assert list(xf[0])[:len(RESULT_COLUMNS)] == list(RESULT_COLUMNS)
    assert len(xf) == 24 and all(r["turb_model"] == "xfoil" and r["converged"] == "True" for r in xf)
    assert {float(r["alpha"]) for r in xf} == {0.0, 8.0, 15.0} and float(xf[0]["Re"]) == pytest.approx(1e5)

    # ANSYS(가짜 Fluent) 해석 → 2. 비교
    assert main(args + ["run"]) in (0, 1)
    capsys.readouterr()
    assert main(args + ["compare-solvers"]) == 0
    out = capsys.readouterr().out
    assert "XFOIL vs ANSYS Fluent" in out and "15°" in out
    cmp_dir = cfg.output_dir / "compare_xfoil"
    rows = _read(cmp_dir / "compare.csv")
    assert len(rows) == 24 and (cmp_dir / "parity.png").is_file()
    r = next(r for r in rows if r["both_converged"] == "True")
    assert float(r["dCL"]) == pytest.approx(float(r["CL_xfoil"]) - float(r["CL_fluent"]))
    summary = json.loads((cmp_dir / "summary.json").read_text())
    assert set(summary["by_alpha"]) == {"0", "8", "15"}

    # 3. MLP + LightGBM 학습 (정답 = ANSYS, 그리고 XFOIL)
    assert main(args + ["dataset"]) == 0
    assert main(args + ["train", "--epochs", "50", "--seeds", "2"]) == 0
    run = json.loads((cfg.output_dir / "ml" / "mpt_h64x64" / "run.json").read_text())
    assert run["features"] == ["m", "p", "t", "aoa"]          # Re 고정 → log10_re 빠짐
    assert run["options"]["models"] == ["lgbm", "mlp"]
    metrics = _read(cfg.output_dir / "ml" / "mpt_h64x64" / "metrics.csv")
    assert {m["model"] for m in metrics} == {"lgbm", "mlp"}
    assert main(args + ["dataset", "--source", "xfoil"]) == 0
    ds_x = _read(cfg.output_dir / "dataset_xfoil" / "dataset.csv")
    ds_f = _read(cfg.output_dir / "dataset" / "dataset.csv")
    assert {r["shape"]: r["split"] for r in ds_x} == {r["shape"]: r["split"] for r in ds_f}  # 같은 형상 분할
    assert main(args + ["train", "--source", "xfoil", "--epochs", "50", "--seeds", "2"]) == 0
    assert (cfg.output_dir / "ml" / "mpt_h64x64_xfoil" / "metrics.csv").is_file()
