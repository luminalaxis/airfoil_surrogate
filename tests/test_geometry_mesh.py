import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from airfoil_surrogate.cases import build_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import load_config
from airfoil_surrogate.fluent_msh import Boundary, check_msh, read_msh, write_msh
from airfoil_surrogate.geometry import build_geometry, naca4, normalize

FAKE_SOLVER = Path(__file__).with_name("fake_solver.py")


def test_naca4_normalized():
    xy, le = normalize(naca4("2412"))
    assert xy[le] == pytest.approx([0, 0], abs=1e-9)
    assert xy[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert xy[: le + 1, 1].mean() > xy[le:, 1].mean()  # 윗면 먼저 (Selig)
    assert xy[:, 1].max() - xy[:, 1].min() == pytest.approx(0.12, abs=0.01)


def test_read_dat_file(tmp_path):
    xy = naca4("0012", n=60)
    lines = ["NACA 0012 test"] + [f"{x:.6f} {y:.6f}" for x, y in xy]
    (tmp_path / "foil.dat").write_text("\n".join(lines))
    geom = build_geometry({"main_airfoil": "foil.dat"}, tmp_path)
    assert len(geom.elements) == 1
    assert geom.elements[0].xy[:, 0].max() == pytest.approx(1.0, abs=1e-3)


def test_geometry_params_only_shape_keys():
    from airfoil_surrogate.geometry import geometry_params

    params = {"main_airfoil": "2412", "chord": 0.3, "aoa": 4, "reynolds": 2e5, "family": "naca4"}
    assert geometry_params(params) == {"main_airfoil": "2412", "chord": 0.3}


def test_fluent_msh_orientation(tmp_path):
    # 2x1 사각형 2개 + 삼각형 섞인 작은 메쉬 (일부러 시계방향 셀 포함)
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [3, 0.5]], float)
    cells = [(0, 1, 4, 3), (1, 2, 5, 4)[::-1], (2, 6, 5)]
    edges = [(0, 1), (1, 2), (2, 6), (6, 5), (5, 4), (4, 3), (3, 0)]
    stats = write_msh(tmp_path / "t.msh", nodes, cells, [Boundary("wall", "wall", edges)])
    assert stats["zones"] == {"interior-fluid": 2, "wall": 7}
    check = check_msh(tmp_path / "t.msh")
    assert check["bad_cells"] == 0 and check["open_cells"] == 0
    assert check["total_area"] == pytest.approx(2.5)
    # Fluent 규칙: n0→n1 진행 방향 기준 c0 셀은 왼쪽, c1 셀은 오른쪽, 경계면은 c1 = 0
    data = read_msh(tmp_path / "t.msh")
    centroid = {i: nodes[list(c)].mean(0) for i, c in enumerate(cells, start=1)}
    for n0, n1, c0, c1, _ in data["faces"]:
        a, b = data["nodes"][n0 - 1], data["nodes"][n1 - 1]
        def side(c):  # +: 왼쪽
            d, v = b - a, centroid[c] - a
            return d[0] * v[1] - d[1] * v[0]

        assert c0 != 0 and side(c0) > 0
        if c1:
            assert side(c1) < 0


def test_orthogonal_quality_matches_fluent_definition(tmp_path):
    from airfoil_surrogate.fluent_msh import orthogonal_quality

    # 정사각형 2개: 모든 면에서 면 벡터와 (중심→면 중심), (중심→이웃 중심)이 나란함 → 1
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    edges = [(0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)]
    write_msh(tmp_path / "sq.msh", nodes, [(0, 1, 4, 3), (1, 2, 5, 4)], [Boundary("wall", "wall", edges)])
    check = check_msh(tmp_path / "sq.msh")
    assert check["min_orthogonal_quality"] == pytest.approx(1.0)
    # 직각이등변삼각형 하나: 빗변은 1, 두 변은 cos = (1/3)/|(1/6, 1/3)| = 2/√5
    tri = np.array([[0, 0], [1, 0], [0, 1]], float)
    write_msh(tmp_path / "tri.msh", tri, [(0, 1, 2)], [Boundary("wall", "wall", [(0, 1), (1, 2), (2, 0)])])
    data = read_msh(tmp_path / "tri.msh")
    oq = orthogonal_quality(data["nodes"], data["faces"], 1)
    assert oq[0] == pytest.approx(2 / 5**0.5)
    # 심하게 찌그러진 사각형은 낮게 나온다
    skew = np.array([[0, 0], [1, 0], [1.9, 0.2], [0.9, 0.2]], float)
    write_msh(tmp_path / "sk.msh", skew, [(0, 1, 2, 3)], [Boundary("wall", "wall", [(0, 1), (1, 2), (2, 3), (3, 0)])])
    assert check_msh(tmp_path / "sk.msh")["min_orthogonal_quality"] < 0.3


def test_mesh_worker_uses_fresh_process_per_mesh():
    # gmsh는 한 프로세스 안에서 앞서 만든 메쉬에 따라 같은 설정이어도 셀이 조금씩 달라진다 (지면 모드 실제 크기
    # 메쉬에서 73,452 → 73,454 → 73,406). 형상마다 새 프로세스를 써야 순서·동시 개수·분담과 관계없이 재현된다.
    import os

    from airfoil_surrogate.meshing import MeshWorker

    with MeshWorker() as w:
        pids = [w._get_pool().apply(os.getpid) for _ in range(2)]
    assert pids[0] != pids[1] and os.getpid() not in pids


def test_unassigned_boundary_rejected(tmp_path):
    nodes = np.array([[0, 0], [1, 0], [0, 1]], float)
    with pytest.raises(ValueError, match="경계"):
        write_msh(tmp_path / "t.msh", nodes, [(0, 1, 2)], [Boundary("wall", "wall", [(0, 1)])])


def make_airfoil_project(tmp_path, airfoils='["0012", "2412", "4412"]', mesh_extra=""):
    (tmp_path / "journal.jou").write_text(
        "AOA=${aoa} AF=${main_airfoil} MESH=${mesh_file}\n/solve/iterate ${iterations}\n", encoding="utf-8")
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"
        max_parallel = 2

        [solver]
        kind = "custom"
        command = ["${{python}}", "{FAKE_SOLVER.as_posix()}", "${{journal}}"]

        [constants]
        iterations = 10

        [cases]
        name = "{{main_airfoil}}_a{{aoa}}"
        group = "{{main_airfoil}}"

        [cases.sweep]
        main_airfoil = {airfoils}
        aoa = [0, 5, 10]

        [mesh]   # 테스트용 거친 메쉬
        reynolds = 1e5
        surface_cells = 60
        farfield_radius = 10
        farfield_size = 1.0
        near_size = 0.03
        wake_size = 0.05
        wake_length = 1.0
    """) + mesh_extra, encoding="utf-8")
    return tmp_path / "project.toml"


def test_airfoil_pipeline_end_to_end(tmp_path, capsys):
    cfg_path = make_airfoil_project(tmp_path)
    cfg = load_config(cfg_path)
    cases = build_cases(cfg)
    assert len(cases) == 9
    assert len({c.mesh.name for c in cases}) == 3  # 받음각만 다른 케이스는 메쉬 공유

    assert main(["-c", str(cfg_path), "run"]) == 0
    out = capsys.readouterr().out
    assert "형상 3개 중 3개 생성 필요" in out

    mesh_dir = cases[0].mesh.dir
    for f in ("mesh.msh", "mesh_info.json", "geometry.png", "mesh.png", "mesh_le.png", "mesh_domain.png"):
        assert (mesh_dir / f).is_file(), f
    info = json.loads((mesh_dir / "mesh_info.json").read_text(encoding="utf-8"))
    assert set(info["zones"]) == {"interior-fluid", "main", "farfield"}
    assert info["min_quality"] > 0.05
    check = check_msh(mesh_dir / "mesh.msh")
    assert check["bad_cells"] == 0 and check["open_cells"] == 0
    assert ("wall", "main") in check["names"].values()

    with open(cfg.output_dir / "summary.csv", encoding="utf-8-sig") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    r = rows["4412_a10"]
    assert float(r["cl"]) == pytest.approx(1.0 + 0.4, rel=1e-6)  # cx,cy → cl 회전
    assert float(r["cd"]) == pytest.approx(0.03, rel=1e-6)
    assert float(r["l/d"]) == pytest.approx(float(r["cl"]) / float(r["cd"]))

    with open(cfg.output_dir / "ranking.csv", encoding="utf-8-sig") as f:
        ranking = list(csv.DictReader(f))
    assert [row["main_airfoil"] for row in ranking] == ["4412", "2412", "0012"]  # 가상 해석에서 캠버 순
    assert ranking[0]["rank"] == "1"
    for f in ("cl_alpha.png", "cd_alpha.png", "ld_alpha.png", "polar.png"):
        assert (cfg.output_dir / "plots" / f).is_file()
    assert (cases[0].dir / "convergence.png").is_file()

    # 다시 실행하면 메쉬도 케이스도 건너뜀
    assert main(["-c", str(cfg_path), "run"]) == 0
    out = capsys.readouterr().out
    assert "형상 3개 중 0개" in out or "메쉬:" not in out


def test_mesh_failure_marks_cases_failed(tmp_path):
    (tmp_path / "broken.dat").write_text("broken\n1.0 0.0\n0.5 0.05\n0.0 0.0\n", encoding="utf-8")
    cfg_path = make_airfoil_project(tmp_path, airfoils='["broken.dat"]')
    assert main(["-c", str(cfg_path), "run"]) == 1
    cfg = load_config(cfg_path)
    status = json.loads((build_cases(cfg)[0].dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and "메쉬 실패" in status["message"]
    assert "좌표 점이 너무 적습니다" in status["message"]  # 실제 실패 사유가 남아야 함
