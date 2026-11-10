"""메쉬 독립성 확인: refine 세분화, mesh_level 메쉬 이름, Richardson 외삽·GCI."""
import math
import textwrap
from pathlib import Path

import pytest

from airfoil_surrogate.cases import build_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import ConfigError, load_config
from airfoil_surrogate.mesh_study import gci3, mesh_dependent_keys, mesh_study
from airfoil_surrogate.meshing import MeshSettings, generate_mesh, make_job

# 가짜 솔버: 셀 크기(1/refine)의 제곱에 비례하는 이산화 오차가 있는 cl, cd (정답 cl 0.5, cd 0.02)
FAKE = '''
import re, sys
from pathlib import Path
journal = Path(sys.argv[1]).read_text(encoding="utf-8")
refine = float(re.search(r"REFINE=(\\S+)", journal).group(1))
cl, cd = 0.5 + 0.03 / refine ** 2, 0.02 + 0.004 / refine ** 2
rows = ['"coef-rfile"', '("Iteration" "cl" "cd")'] + [f"{i} {cl:.8e} {cd:.8e}" for i in range(1, 6)]
Path("coef-rfile.out").write_text("\\n".join(rows) + "\\n")
'''


def test_gci3_second_order_series():
    h = (1.0, 1.5, 2.25)
    phi = tuple(1.0 + 0.003 * x**2 for x in h)  # 오차가 값보다 충분히 작은(점근 구간) 2차 수렴
    g = gci3(phi, h)
    assert g["convergence"] == "monotonic"
    assert g["p"] == pytest.approx(2.0, abs=1e-9)
    assert g["phi_ext"] == pytest.approx(1.0, abs=1e-9)
    assert g["gci_fine"] == pytest.approx(1.25 * abs(phi[1] - phi[0]) / phi[0] / (1.5**2 - 1))
    assert g["asymptotic"] == pytest.approx(1.0, rel=0.05)


def test_gci3_unequal_ratios_and_bad_cases():
    h = (1.0, 1.4, 2.3)
    phi = tuple(2.0 - 0.1 * x**1.5 for x in h)
    g = gci3(phi, h)
    assert g["p"] == pytest.approx(1.5, abs=1e-6) and g["phi_ext"] == pytest.approx(2.0, abs=1e-6)
    osc = gci3((1.0, 1.1, 0.8), (1.0, 1.5, 2.25))  # 부호가 바뀌지만 조밀할수록 차이가 줄어듦 → 진동 수렴
    assert osc["convergence"] == "oscillatory" and osc["p"] > 0
    bad = gci3((1.0, 1.3, 1.2), (1.0, 1.5, 2.25))  # 조밀할수록 차이가 커짐 → 수렴 안 함
    assert bad["convergence"] == "oscillatory-divergent" and math.isnan(bad["p"])
    assert gci3((1.0, 1.2, 1.3), (1.0, 1.5, 2.25))["convergence"] == "divergent"
    assert gci3((1.0, 1.0, 1.0), h)["convergence"] == "exact"
    assert gci3((1.0, 1.1, 1.2), (2.0, 1.5, 1.0))["convergence"] == "invalid"


def make_project(tmp_path, levels="[1, 2, 3]", extra_derived='refine = "1.5 ** (mesh_level - 2)"'):
    (tmp_path / "fake.py").write_text(FAKE, encoding="utf-8")
    (tmp_path / "journal.jou").write_text("REFINE=${refine} MESH=${mesh_file}\n", encoding="utf-8")
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"

        [solver]
        kind = "custom"
        command = ["${{python}}", "{(tmp_path / 'fake.py').as_posix()}", "${{journal}}"]

        [constants]
        main_airfoil = "0012"

        [cases]
        name = "a{{aoa}}_m{{mesh_level}}"

        [cases.sweep]
        mesh_level = {levels}
        aoa = [2]

        [derived]
        {extra_derived}

        [mesh]
        chord = 0.3
        reynolds = 2e5
        surface_cells = 40
        farfield_radius = 8
        farfield_size = 1.0
        near_size = 0.03
        wake_size = 0.05
        wake_length = 1.0

        [results]
        average_last = 2
    """), encoding="utf-8")
    return tmp_path / "project.toml"


def test_mesh_levels_get_own_meshes(tmp_path):
    cfg = load_config(make_project(tmp_path))
    cases = build_cases(cfg)
    names = [c.mesh.name for c in cases]
    assert names == ["0012_m1", "0012_m2", "0012_m3"]
    assert [round(c.mesh.settings.refine, 4) for c in cases] == [0.6667, 1.0, 1.5]
    assert {"mesh_level", "refine"} <= mesh_dependent_keys(cfg)
    # 2단계(refine 1)는 mesh_level이 없을 때와 같은 메쉬 설정 → 지문이 같다
    plain = make_job({"main_airfoil": "0012"}, cfg.mesh, tmp_path)
    assert plain.fingerprint() == cases[1].mesh.fingerprint()


def test_mesh_setting_override_without_level_is_rejected(tmp_path):
    # mesh_level 없이 케이스마다 메쉬 설정을 바꾸면 이름이 같은 메쉬를 공유해 엉뚱한 메쉬로 계산된다 → 오류
    path = make_project(tmp_path, extra_derived='surface_cells = "40 * level"')
    text = path.read_text(encoding="utf-8")
    text = text.replace("mesh_level = [1, 2, 3]", "level = [1, 2]").replace("_m{mesh_level}", "_l{level}")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="mesh_level"):
        build_cases(load_config(path))


def test_refine_scales_cells(tmp_path):
    base = dict(chord=0.3, reynolds=2e5, surface_cells=40, farfield_radius=8, farfield_size=1.0,
                near_size=0.03, wake_size=0.05, wake_length=1.0)
    cells = []
    for f in (1.0, 1.5):
        job = make_job({"main_airfoil": "0012", "mesh_level": f}, {**base, "refine": f}, tmp_path)
        info = generate_mesh(job, Path("."), plots=False)
        cells.append(info["cells"])
        assert info["first_cell_height_c"] == pytest.approx(MeshSettings(**{**base, "refine": f}).first_cell_height())
    assert 1.8 < cells[1] / cells[0] < 3.0  # 셀 크기 1/1.5 → 셀 수 약 2.25배


def test_mesh_study_end_to_end(tmp_path, capsys):
    cfg_path = make_project(tmp_path)
    assert main(["-c", str(cfg_path), "--no-banner", "run"]) == 0
    cfg = load_config(cfg_path)
    r = mesh_study(cfg, build_cases(cfg))
    by_q = {e["quantity"]: e for e in r["rows"]}
    assert set(by_q) == {"cl", "cd"}
    cl = by_q["cl"]
    assert cl["convergence"] == "monotonic" and cl["level_fine"] == 3 and cl["level_coarse"] == 1
    assert cl["phi_ext"] == pytest.approx(0.5, abs=0.01)
    assert by_q["cd"]["phi_ext"] == pytest.approx(0.02, abs=0.002)
    assert r["csv"].is_file() and r["plots"]

    assert main(["-c", str(cfg_path), "--no-banner", "gci"]) == 0
    out = capsys.readouterr().out
    assert "GCI" in out and "monotonic" in out
